import argparse
from pathlib import Path
import torch
import wandb
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from model import DiffusionUNetCrossAttention, ConditionNet, ConditionNetWithFFT
from diffusion import RDDM
from data import get_datasets
import torch.nn as nn
from metrics import *
from lr_scheduler import CosineAnnealingLRWarmup

from torch.utils.data import DataLoader

def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
        warnings.warn('You have chosen to seed training. '
              'This will turn on the CUDNN deterministic setting, '
              'which can slow down your training considerably! '
              'You may see unexpected behavior when restarting '
              'from checkpoints.')

set_deterministic(31)

def train_rddm(config, resume_epoch=None):

    n_epoch = config["n_epoch"]
    device = config["device"]
    batch_size = config["batch_size"]
    nT = config["nT"]
    num_heads = config["attention_heads"]
    cond_mask = config["cond_mask"]
    alpha1 = config["alpha1"]
    alpha2 = config["alpha2"]
    alphafft = config["alphafft"]
    PATH = config["PATH"]
    with_fftloss = config["with_fftloss"]
    with_fftcond = config["with_fftcond"]
    sampling_fate = config["sampling_rate"]
    cutoff_freq = config["cutoff_freq"]
    input_lead = config["input_lead"]
    target_lead = config["target_lead"]
    data_path = config["DATA_PATH"]
    datasets = config["datasets"]
    window_size = config["window_size"]
    num_workers = config["num_workers"]
    save_every = config["save_every"]

    Path(PATH).mkdir(parents=True, exist_ok=True)

    if config["use_wandb"]:
        wandb.init(
            project=config["wandb_project"],
            id=config["wandb_run_id"],
            config=config
        )

    dataset_train, _ = get_datasets(
        DATA_PATH=data_path,
        datasets=datasets,
        window_size=window_size,
        input_lead=input_lead,
        target_lead=target_lead
    )

    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    rddm = RDDM(
        eps_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        region_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        betas=(1e-4, 0.2), 
        n_T=nT
    )

    if with_fftcond :
        
        Conditioning_network1 = ConditionNetWithFFT(device=device).to(device)
        Conditioning_network2 = ConditionNetWithFFT(device=device).to(device)
    else:
        Conditioning_network1 = ConditionNet().to(device)
        Conditioning_network2 = ConditionNet().to(device)
    rddm.to(device)

    optim = torch.optim.AdamW([*rddm.parameters(), *Conditioning_network1.parameters(), *Conditioning_network2.parameters()], lr=1e-4)

    rddm = nn.DataParallel(rddm)
    Conditioning_network1 = nn.DataParallel(Conditioning_network1)
    Conditioning_network2 = nn.DataParallel(Conditioning_network2)

    scheduler = CosineAnnealingLRWarmup(optim, 20, n_epoch)


    if resume_epoch is not None:
        checkpoint_rddm = f"{PATH}/RDDM_epoch{resume_epoch}.pth"
        checkpoint_cond1 = f"{PATH}/ConditionNet1_epoch{resume_epoch}.pth"
        checkpoint_cond2 = f"{PATH}/ConditionNet2_epoch{resume_epoch}.pth"

        print(f"Resuming from epoch {resume_epoch} checkpoints...")
        rddm.load_state_dict(torch.load(checkpoint_rddm, map_location=device))
        Conditioning_network1.load_state_dict(torch.load(checkpoint_cond1, map_location=device))
        Conditioning_network2.load_state_dict(torch.load(checkpoint_cond2, map_location=device))

    
    
    for i in range(n_epoch):
        print(f"\n****************** Epoch - {i} *******************\n\n")

        rddm.train()
        Conditioning_network1.train()
        Conditioning_network2.train()
        pbar = tqdm(dataloader)

        for y_ecg, x_ecg, ecg_roi in pbar:
            
            ## Train Diffusion
            optim.zero_grad() 
            x_ecg = x_ecg.float().to(device)
            y_ecg = y_ecg.float().to(device)
            ecg_roi = ecg_roi.float().to(device)

            ecg_conditions1 = Conditioning_network1(x_ecg)
            ecg_conditions2 = Conditioning_network2(x_ecg)

            ddpm_loss, region_loss, pred_signal = rddm(x=y_ecg, cond1=ecg_conditions1, cond2=ecg_conditions2, patch_labels=ecg_roi)

            fft_loss = 0
            if with_fftloss:
                fft_loss = compute_fft_loss(pred_signal, y_ecg)

            ddpm_loss = alpha1 * ddpm_loss
            region_loss = alpha2 * region_loss
            fft_loss = alphafft * fft_loss
            
            loss = ddpm_loss + region_loss + fft_loss

            loss.mean().backward()
            
            optim.step()

            pbar.set_description(f"loss: {loss.mean().item():.4f}")

            if with_fftloss and config["use_wandb"]:
                wandb.log({
                    "DDPM_loss": ddpm_loss.mean().item(),
                    "Region_loss": region_loss.mean().item(),
                    "fft_loss" : fft_loss.mean().item()
                })
            elif config["use_wandb"]:
                wandb.log({
                    "DDPM_loss": ddpm_loss.mean().item(),
                    "Region_loss": region_loss.mean().item(),
                })

        scheduler.step()

        if i % save_every == 0:
            torch.save(rddm.module.state_dict(), f"{PATH}/RDDM_epoch{i}.pth")
            torch.save(Conditioning_network1.module.state_dict(), f"{PATH}/ConditionNet1_epoch{i}.pth")
            torch.save(Conditioning_network2.module.state_dict(), f"{PATH}/ConditionNet2_epoch{i}.pth")

    if config["use_wandb"]:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Train RDDM ECG lead translation models.")
    parser.add_argument("--data-path", default="/tf/revision/data/", help="Root directory containing dataset folders.")
    parser.add_argument("--datasets", nargs="+", default=["PTBXL"], help="Dataset folder names under data-path.")
    parser.add_argument("--model-root", default="/tf/revision/model/", help="Root directory for model checkpoints.")
    parser.add_argument("--input-lead", type=int, default=1, help="Condition ECG lead number.")
    parser.add_argument("--target-leads", nargs="+", type=int, default=[4], help="Target ECG lead numbers to train sequentially.")
    parser.add_argument("--window-size", type=int, default=5, help="Signal window size in seconds.")
    parser.add_argument("--epochs", type=int, default=121, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=128, help="DataLoader worker count.")
    parser.add_argument("--nT", type=int, default=10, help="Diffusion denoising steps.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--attention-heads", type=int, default=8, help="Cross-attention head count.")
    parser.add_argument("--cond-mask", type=float, default=0.0)
    parser.add_argument("--alpha1", type=float, default=1, help="DDPM loss weight.")
    parser.add_argument("--alpha2", type=float, default=100, help="Region loss weight.")
    parser.add_argument("--alphafft", type=float, default=0.1, help="FFT loss weight.")
    parser.add_argument("--with-fftloss", action="store_true", help="Use FFT loss and save under withfftloss.")
    parser.add_argument("--with-fftcond", action="store_true", help="Use ConditionNetWithFFT.")
    parser.add_argument("--sampling-rate", type=int, default=128)
    parser.add_argument("--cutoff-freq", type=float, default=30.0)
    parser.add_argument("--save-every", type=int, default=30, help="Checkpoint save interval in epochs.")
    parser.add_argument("--resume-epoch", type=int, default=None, help="Resume from this checkpoint epoch for every selected lead.")
    parser.add_argument("--wandb-project", default="RDDM")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable wandb logging.")
    return parser.parse_args()


def validate_leads(input_lead, target_leads):
    all_leads = [input_lead, *target_leads]
    invalid = [lead for lead in all_leads if lead < 1 or lead > 12]
    if invalid:
        raise ValueError(f"Lead numbers must be between 1 and 12: {invalid}")
    if input_lead in target_leads:
        raise ValueError("input_lead cannot also be a target lead.")


def validate_args(args):
    validate_leads(args.input_lead, args.target_leads)
    if args.save_every <= 0:
        raise ValueError("save_every must be greater than 0.")
    if args.epochs <= 0:
        raise ValueError("epochs must be greater than 0.")


def build_output_path(model_root, with_fftloss, input_lead, target_lead):
    loss_dir = "withfftloss" if with_fftloss else "none"
    return str(Path(model_root) / loss_dir / f"{input_lead}to{target_lead}")


def build_config(args, target_lead):
    output_path = build_output_path(args.model_root, args.with_fftloss, args.input_lead, target_lead)
    config = {
        "n_epoch": args.epochs,
        "batch_size": args.batch_size,
        "nT": args.nT,
        "device": args.device,
        "attention_heads": args.attention_heads,
        "cond_mask": args.cond_mask,
        "alpha1": args.alpha1,
        "alpha2": args.alpha2,
        "alphafft": args.alphafft,
        "PATH": output_path,
        "with_fftloss": args.with_fftloss,
        "with_fftcond": args.with_fftcond,
        "sampling_rate": args.sampling_rate,
        "cutoff_freq": args.cutoff_freq,
        "DATA_PATH": args.data_path,
        "datasets": args.datasets,
        "window_size": args.window_size,
        "input_lead": args.input_lead,
        "target_lead": target_lead,
        "num_workers": args.num_workers,
        "save_every": args.save_every,
        "use_wandb": not args.disable_wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_id": f"ECG{args.input_lead}TOECG{target_lead}",
    }
    return config


if __name__ == "__main__":
    args = parse_args()
    validate_args(args)

    for target_lead in args.target_leads:
        config = build_config(args, target_lead)
        print(f"\nTraining lead{args.input_lead} -> lead{target_lead}")
        print(f"Saving checkpoints to: {config['PATH']}")
        train_rddm(config, resume_epoch=args.resume_epoch)
