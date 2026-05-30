import argparse
import json
import random
import re
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import get_datasets
from diffusion import load_pretrained_DPM
from metrics import calculate_FD, evaluation_pipeline

torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore")


def set_deterministic(seed):
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn(
            "You have chosen to seed evaluation. This will turn on the CUDNN "
            "deterministic setting, which can slow down evaluation."
        )


set_deterministic(31)


def build_model_path(model_base, with_fftloss, input_lead, target_lead):
    loss_dir = "withfftloss" if with_fftloss else "none"
    return Path(model_base) / loss_dir / f"{input_lead}to{target_lead}"


def find_checkpoint_epoch(model_path, checkpoint_epoch=None):
    if checkpoint_epoch is not None:
        return checkpoint_epoch

    epochs = []
    for checkpoint in Path(model_path).glob("RDDM_epoch*.pth"):
        match = re.search(r"RDDM_epoch(\d+)\.pth$", checkpoint.name)
        if match:
            epochs.append(int(match.group(1)))

    if not epochs:
        raise FileNotFoundError(f"No RDDM_epoch*.pth checkpoint found in {model_path}")

    return max(epochs)


def validate_leads(input_lead, target_leads):
    all_leads = [input_lead, *target_leads]
    invalid = [lead for lead in all_leads if lead < 1 or lead > 12]
    if invalid:
        raise ValueError(f"Lead numbers must be between 1 and 12: {invalid}")
    if input_lead in target_leads:
        raise ValueError("input_lead cannot also be a target lead.")


def validate_checkpoints(model_path, checkpoint_epoch, model_type):
    required = [
        model_path / f"RDDM_epoch{checkpoint_epoch}.pth",
        model_path / f"ConditionNet1_epoch{checkpoint_epoch}.pth",
    ]
    if model_type != "Naive":
        required.append(model_path / f"ConditionNet2_epoch{checkpoint_epoch}.pth")

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint file(s): " + ", ".join(missing))


def pearson_correlation(real_ecg, fake_ecg):
    real = real_ecg.reshape(real_ecg.shape[0], -1)
    fake = fake_ecg.reshape(fake_ecg.shape[0], -1)
    values = []

    for real_row, fake_row in zip(real, fake):
        real_std = np.std(real_row)
        fake_std = np.std(fake_row)
        if real_std == 0 or fake_std == 0:
            continue
        values.append(np.corrcoef(real_row, fake_row)[0, 1])

    return float(np.mean(values)) if values else float("nan")


def spectral_similarity(real_ecg, fake_ecg):
    real_mag = np.abs(np.fft.rfft(real_ecg.reshape(real_ecg.shape[0], -1), axis=-1))
    fake_mag = np.abs(np.fft.rfft(fake_ecg.reshape(fake_ecg.shape[0], -1), axis=-1))
    numerator = np.sum(real_mag * fake_mag, axis=-1)
    denominator = np.linalg.norm(real_mag, axis=-1) * np.linalg.norm(fake_mag, axis=-1)
    valid = denominator > 0
    if not np.any(valid):
        return float("nan")
    return float(np.mean(numerator[valid] / denominator[valid]))


def dtw_distance_1d(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    previous = np.full(y.shape[0] + 1, np.inf)
    current = np.full(y.shape[0] + 1, np.inf)
    previous[0] = 0.0

    for i in range(1, x.shape[0] + 1):
        current[0] = np.inf
        for j in range(1, y.shape[0] + 1):
            cost = abs(x[i - 1] - y[j - 1])
            current[j] = cost + min(current[j - 1], previous[j], previous[j - 1])
        previous, current = current, previous

    return float(previous[y.shape[0]] / (x.shape[0] + y.shape[0]))


def dtw_distance(real_ecg, fake_ecg, max_samples=100, stride=1):
    real = real_ecg.reshape(real_ecg.shape[0], -1)
    fake = fake_ecg.reshape(fake_ecg.shape[0], -1)
    sample_count = real.shape[0] if max_samples is None else min(max_samples, real.shape[0])

    if sample_count == 0:
        return float("nan"), 0

    values = []
    for real_row, fake_row in zip(real[:sample_count], fake[:sample_count]):
        values.append(dtw_distance_1d(real_row[::stride], fake_row[::stride]))

    return float(np.mean(values)), sample_count


def calculate_extra_metrics(real_ecg, fake_ecg, dtw_samples=100, dtw_stride=1):
    dtw_score, dtw_sample_count = dtw_distance(
        real_ecg,
        fake_ecg,
        max_samples=dtw_samples,
        stride=dtw_stride,
    )
    return {
        "Correlation_coefficient": pearson_correlation(real_ecg, fake_ecg),
        "DTW": dtw_score,
        "DTW_sample_count": dtw_sample_count,
        "Spectral_similarity": spectral_similarity(real_ecg, fake_ecg),
    }


def eval_diffusion(
    window_size,
    EVAL_DATASETS,
    DATA_PATH="/tf/revision/data/",
    nT=10,
    batch_size=32,
    PATH="/tf/revision/model/none/1to4/",
    device="cuda",
    input_lead=1,
    target_lead=4,
    checkpoint_epoch=None,
    with_fftcond=False,
    num_workers=64,
    max_batches=None,
    dtw_samples=100,
    dtw_stride=1,
    check_sig=False,
):
    _, dataset_test = get_datasets(
        DATA_PATH=DATA_PATH,
        datasets=EVAL_DATASETS,
        window_size=window_size,
        input_lead=input_lead,
        target_lead=target_lead,
    )

    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model_path = Path(PATH)
    model_type = "RDDMfft" if with_fftcond else "RDDM"
    checkpoint_epoch = find_checkpoint_epoch(model_path, checkpoint_epoch)
    validate_checkpoints(model_path, checkpoint_epoch, model_type)

    dpm, conditioning_network1, conditioning_network2 = load_pretrained_DPM(
        PATH=model_path,
        nT=nT,
        type=model_type,
        device=device,
        checkpoint_epoch=checkpoint_epoch,
    )

    dpm.eval()
    conditioning_network1.eval()
    conditioning_network2.eval()

    with torch.no_grad():
        fd_list = []
        fake_batches = []
        real_batches = []
        cond_batches = []
        roi_batches = []

        for batch_idx, (y_ecg, x_ecg, ecg_roi) in enumerate(tqdm(testloader)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x_ecg = x_ecg.float().to(device)
            y_ecg = y_ecg.float().to(device)
            ecg_roi = ecg_roi.float().to(device)

            generated_windows = []

            for ecg_window in torch.split(x_ecg, 128 * window_size, dim=-1):
                if ecg_window.shape[-1] != 128 * window_size:
                    ecg_window = F.pad(
                        ecg_window,
                        (0, 128 * window_size - ecg_window.shape[-1]),
                        "constant",
                        0,
                    )

                ecg_conditions1 = conditioning_network1(ecg_window)
                ecg_conditions2 = conditioning_network2(ecg_window)

                xh = dpm(
                    cond1=ecg_conditions1,
                    cond2=ecg_conditions2,
                    mode="sample",
                    window_size=128 * window_size,
                )

                generated_windows.append(xh.cpu().numpy())

            xh = np.concatenate(generated_windows, axis=-1)[:, :, :128 * window_size]
            fd = calculate_FD(y_ecg, torch.from_numpy(xh).to(device), window_size)

            fake_batches.append(xh.reshape(-1, 128 * window_size))
            real_batches.append(y_ecg.reshape(-1, 128 * window_size).cpu().numpy())
            cond_batches.append(x_ecg.reshape(-1, 128 * window_size).cpu().numpy())
            roi_batches.append(ecg_roi.reshape(-1, 128 * window_size).cpu().numpy())
            fd_list.append(fd)

            if check_sig:
                return (
                    np.concatenate(fake_batches, axis=0),
                    np.concatenate(real_batches, axis=0),
                    np.concatenate(cond_batches, axis=0),
                )

        fake_ecgs = np.concatenate(fake_batches, axis=0)
        real_ecgs = np.concatenate(real_batches, axis=0)
        mae_hr_ecg, rmse_score = evaluation_pipeline(real_ecgs, fake_ecgs)

        tracked_metrics = {
            "input_lead": input_lead,
            "target_lead": target_lead,
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_path": str(model_path),
            "model_type": model_type,
            "RMSE_score": float(rmse_score),
            "MAE_HR_ECG": float(mae_hr_ecg),
            "FD": float(sum(fd_list) / len(fd_list)),
        }
        tracked_metrics.update(
            calculate_extra_metrics(
                real_ecgs,
                fake_ecgs,
                dtw_samples=dtw_samples,
                dtw_stride=dtw_stride,
            )
        )

        return tracked_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained RDDM ECG lead translation models.")
    parser.add_argument("--data-path", default="/tf/revision/data/", help="Root directory containing dataset folders.")
    parser.add_argument("--datasets", nargs="+", default=["PTBXL"], help="Dataset folder names under data-path.")
    parser.add_argument("--model-base", default="/tf/revision/model/", help="Root directory used by train.py --model-root.")
    parser.add_argument("--input-lead", type=int, default=1, help="Condition ECG lead number.")
    parser.add_argument("--target-leads", nargs="+", type=int, default=[4], help="Target ECG lead numbers to evaluate.")
    parser.add_argument("--with-fftloss", action="store_true", help="Read checkpoints from the withfftloss directory.")
    parser.add_argument("--with-fftcond", action="store_true", help="Load FFT-aware condition networks.")
    parser.add_argument("--checkpoint-epoch", type=int, default=None, help="Checkpoint epoch. Defaults to latest in each model directory.")
    parser.add_argument("--window-size", type=int, default=5, help="Signal window size in seconds.")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=64, help="DataLoader worker count.")
    parser.add_argument("--nT", type=int, default=10, help="Diffusion denoising steps.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional quick-eval batch limit.")
    parser.add_argument("--dtw-samples", type=int, default=100, help="Number of generated samples used for DTW. Use 0 for all samples.")
    parser.add_argument("--dtw-stride", type=int, default=1, help="Stride applied before DTW to speed up evaluation.")
    parser.add_argument("--json", action="store_true", help="Print metrics as JSON.")
    return parser.parse_args()


def validate_args(args):
    validate_leads(args.input_lead, args.target_leads)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if args.window_size <= 0:
        raise ValueError("window_size must be greater than 0.")
    if args.dtw_stride <= 0:
        raise ValueError("dtw_stride must be greater than 0.")


if __name__ == "__main__":
    args = parse_args()
    validate_args(args)
    dtw_samples = None if args.dtw_samples == 0 else args.dtw_samples
    all_metrics = []

    for target_lead in args.target_leads:
        model_path = build_model_path(args.model_base, args.with_fftloss, args.input_lead, target_lead)
        print(f"\nEvaluating lead{args.input_lead} -> lead{target_lead}")
        print(f"Loading checkpoints from: {model_path}")

        tracked_metrics = eval_diffusion(
            window_size=args.window_size,
            EVAL_DATASETS=args.datasets,
            DATA_PATH=args.data_path,
            nT=args.nT,
            batch_size=args.batch_size,
            PATH=model_path,
            device=args.device,
            input_lead=args.input_lead,
            target_lead=target_lead,
            checkpoint_epoch=args.checkpoint_epoch,
            with_fftcond=args.with_fftcond,
            num_workers=args.num_workers,
            max_batches=args.max_batches,
            dtw_samples=dtw_samples,
            dtw_stride=args.dtw_stride,
        )
        all_metrics.append(tracked_metrics)

        if not args.json:
            print(
                f"lead{args.input_lead}->lead{target_lead}: "
                f"RMSE={tracked_metrics['RMSE_score']}, "
                f"FD={tracked_metrics['FD']}, "
                f"MAE_HR_ECG={tracked_metrics['MAE_HR_ECG']}, "
                f"Corr={tracked_metrics['Correlation_coefficient']}, "
                f"DTW={tracked_metrics['DTW']}, "
                f"SpectralSim={tracked_metrics['Spectral_similarity']}"
            )

    if args.json:
        print(json.dumps({"results": all_metrics}, indent=2))
