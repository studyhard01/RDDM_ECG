# train_rddm.py

import os
import torch
import wandb
import random
from tqdm.auto import tqdm
import warnings
import numpy as np
from model import DiffusionUNetCrossAttention, ConditionNet
from diffusion import RDDM
from data import get_datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader
from metrics import compute_fft_loss
from lr_scheduler import CosineAnnealingLRWarmup

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
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting.')

def compute_cosine_similarity_loss(pred_signal, target_signal):
    cos_sim = F.cosine_similarity(pred_signal, target_signal, dim=-1).mean()
    return 1 - cos_sim

def train_rddm(config, resume_epoch=-1):
    set_deterministic(config.get("seed", 42))

    # 경로 자동 설정
    save_dir = f'./saves/{config["exp_name"]}/'
    os.makedirs(save_dir, exist_ok=True)
    config["PATH"] = save_dir

    wandb.init(
        project="RDDM_4lead",
        name=config["exp_name"],
        config=config
    )

    dataset_train, _ = get_datasets()
    dataloader = DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)

    rddm = RDDM(
        eps_model=DiffusionUNetCrossAttention(512, 1, config["device"], num_heads=config["attention_heads"]),
        region_model=DiffusionUNetCrossAttention(512, 1, config["device"], num_heads=config["attention_heads"]),
        betas=(1e-4, 0.2),
        n_T=config["nT"]
    ).to(config["device"])

    Conditioning_network1 = ConditionNet().to(config["device"])
    Conditioning_network2 = ConditionNet().to(config["device"])

    optim = torch.optim.AdamW(
        [*rddm.parameters(), *Conditioning_network1.parameters(), *Conditioning_network2.parameters()],
        lr=1e-4
    )
    scheduler = CosineAnnealingLRWarmup(optim, 20, config["n_epoch"])

    # Resume
    if resume_epoch > 0:
        rddm.load_state_dict(torch.load(f"{save_dir}/RDDM_epoch{resume_epoch}.pth"))
        Conditioning_network1.load_state_dict(torch.load(f"{save_dir}/ConditionNet1_epoch{resume_epoch}.pth"))
        Conditioning_network2.load_state_dict(torch.load(f"{save_dir}/ConditionNet2_epoch{resume_epoch}.pth"))
    resume_epoch += 1

    for epoch in range(resume_epoch, config["n_epoch"]):
        print(f"\n********** Epoch {epoch} **********\n")
        rddm.train()
        Conditioning_network1.train()
        Conditioning_network2.train()

        pbar = tqdm(dataloader, ncols=100)

        for y_ecg, x_ecg, ecg_roi in pbar:
            optim.zero_grad()
            x_ecg = x_ecg.float().to(config["device"])
            y_ecg = y_ecg.float().to(config["device"])
            ecg_roi = ecg_roi.float().to(config["device"])

            cond1 = Conditioning_network1(x_ecg)
            cond2 = Conditioning_network2(x_ecg)

            ddpm_loss, region_loss, pred_signal = rddm(
                x=y_ecg, cond1=cond1, cond2=cond2, patch_labels=ecg_roi
            )

            fft_loss = compute_fft_loss(pred_signal, y_ecg) if config["with_fftloss"] else 0
            cossim_loss = compute_cosine_similarity_loss(pred_signal, y_ecg) if config["with_cossimloss"] else 0

            loss = (
                config["alpha1"] * ddpm_loss +
                config["alpha2"] * region_loss +
                config["alphafft"] * fft_loss +
                config["alphacos"] * cossim_loss
            )

            loss.mean().backward()
            optim.step()
            pbar.set_description(f"loss: {loss.mean().item():.4f}")

            log_dict = {
                "DDPM_loss": ddpm_loss.mean().item(),
                "Region_loss": region_loss.mean().item(),
            }
            if config["with_fftloss"]:
                log_dict["FFT_loss"] = fft_loss.mean().item()
            if config["with_cossimloss"]:
                log_dict["CosineSim_loss"] = cossim_loss.mean().item()

            wandb.log(log_dict)

        scheduler.step()

        if epoch % 50 == 0:
            torch.save(rddm.state_dict(), f"{save_dir}/RDDM_epoch{epoch}.pth")
            torch.save(Conditioning_network1.state_dict(), f"{save_dir}/ConditionNet1_epoch{epoch}.pth")
            torch.save(Conditioning_network2.state_dict(), f"{save_dir}/ConditionNet2_epoch{epoch}.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, default="default_exp")
    parser.add_argument("--n_epoch", type=int, default=501)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nT", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--alpha1", type=float, default=100)
    parser.add_argument("--alpha2", type=float, default=1)
    parser.add_argument("--alphafft", type=float, default=0.1)
    parser.add_argument("--alphacos", type=float, default=10)
    parser.add_argument("--with_fftloss", action="store_true")
    parser.add_argument("--with_cossimloss", action="store_true")
    parser.add_argument("--sampling_rate", type=int, default=128)
    parser.add_argument("--cutoff_freq", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train_rddm(vars(args))