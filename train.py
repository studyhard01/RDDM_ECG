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

from torch.utils.data import Dataset, DataLoader

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

    wandb.init(
        project="RDDM",
        id=f"ECG1TOECG4",
        config=config
    )

    dataset_train, _ = get_datasets()

    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=128)

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

            if with_fftloss :
                wandb.log({
                    "DDPM_loss": ddpm_loss.mean().item(),
                    "Region_loss": region_loss.mean().item(),
                    "fft_loss" : fft_loss.mean().item()
                })
            else :
                wandb.log({
                    "DDPM_loss": ddpm_loss.mean().item(),
                    "Region_loss": region_loss.mean().item(),
                })

        scheduler.step()

        if i % 30 == 0:
            torch.save(rddm.module.state_dict(), f"{PATH}/RDDM_epoch{i}.pth")
            torch.save(Conditioning_network1.module.state_dict(), f"{PATH}/ConditionNet1_epoch{i}.pth")
            torch.save(Conditioning_network2.module.state_dict(), f"{PATH}/ConditionNet2_epoch{i}.pth")

                
if __name__ == "__main__":

    config = {
        "n_epoch": 121,
        "batch_size": 32,
        "nT":10,
        "device": "cuda",
        "attention_heads": 8,
        "cond_mask": 0.0,
        "alpha1": 100,
        "alpha2": 1,
        "alphafft" : 0.1 , 
        "PATH": "/tf/revision/model/none/1to4",
        "with_fftloss" : False ,
        "sampling_rate" : 128 ,
        "cutoff_freq" : 30.0  ,
        "with_fftcond" : False
    }

    train_rddm(config)
