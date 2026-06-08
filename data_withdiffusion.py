import torch
torch.autograd.set_detect_anomaly(True)
import random
from tqdm import tqdm
import warnings
from metrics import *
warnings.filterwarnings("ignore")
import numpy as np
from diffusion import load_pretrained_DPM
import matplotlib.pyplot as plt
import torch.nn.functional as F
from data import get_datasets
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import sklearn.preprocessing as skp

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

def get_datasets(
    DATA_PATH = "/tf/hsh/ECG_capstone/data/", 
    #datasets=["BIDMC", "CAPNO", "DALIA", "MIMIC-AFib", "WESAD"],
    datasets=["CPSC2018"],
    window_size=10, lead_num=12
    ):

    ecg_train_list = []
    ppg_train_list = []
    ecg_test_list = []
    ppg_test_list = []
    y_train_list = []
    y_test_list = []
    
    for dataset in datasets:
        print(dataset)
        if dataset != "CPSC2018/RDDM" and dataset != "physionet2017/RDDM"  :
            ecg_train = np.load(DATA_PATH + dataset + f"/lead{lead_num}_train.npy", allow_pickle=True).reshape(-1, 128*window_size)
            ppg_train = np.load(DATA_PATH + dataset + f"/lead1_train.npy", allow_pickle=True).reshape(-1, 128*window_size)
            y_train = np.load(DATA_PATH + dataset + f"/y_train.npy", allow_pickle=True)
        
        ecg_test = np.load(DATA_PATH + dataset + f"/lead{lead_num}_test.npy", allow_pickle=True).reshape(-1, 128*window_size)
        ppg_test = np.load(DATA_PATH + dataset + f"/lead1_test.npy", allow_pickle=True).reshape(-1, 128*window_size)
        y_test = np.load(DATA_PATH + dataset + f"/y_test.npy", allow_pickle=True) - 1
        print(max(y_test), min(y_test))

        if dataset != "CPSC2018/RDDM" and dataset != "physionet2017/RDDM" :
            
            ecg_train_list.append(ecg_train)
            ppg_train_list.append(ppg_train)
            y_train_list.append(y_train)
        
        ecg_test_list.append(ecg_test)
        ppg_test_list.append(ppg_test)
        y_test_list.append(y_test)
        
    if datasets[0] != "CPSC2018/RDDM" and datasets[0] != "physionet2017/RDDM" :
        ecg_train = np.nan_to_num(np.concatenate(ecg_train_list).astype("float32"))
        ppg_train = np.nan_to_num(np.concatenate(ppg_train_list).astype("float32"))

    ecg_test = np.nan_to_num(np.concatenate(ecg_test_list).astype("float32"))
    ppg_test = np.nan_to_num(np.concatenate(ppg_test_list).astype("float32"))
    
    if datasets[0] != "CPSC2018/RDDM" and datasets[0] != "physionet2017/RDDM" :
        dataset_train = ECGDataset(
            ecg_train,
            ppg_train,
            np.array(y_train_list[0])
        )
    dataset_test = ECGDataset(
        ecg_test,
        ppg_test,
        np.array(y_test_list[0])
    )

    return None, dataset_test

class ECGDataset():
    
    def __init__(self, ecg_data, ppg_data, y_data=None):
        self.ecg_data = ecg_data
        self.ppg_data = ppg_data
        self.y_data = y_data

    def __getitem__(self, index):

        ecg = self.ecg_data[index]
        ppg = self.ppg_data[index]
        y = self.y_data[index]
        
        window_size = ecg.shape[-1]

        ppg = nk.ppg_clean(ppg.reshape(window_size), sampling_rate=128)
        ecg = nk.ecg_clean(ecg.reshape(window_size), sampling_rate=128, method="pantompkins1985")
        _, info = nk.ecg_peaks(ecg, sampling_rate=128, method="pantompkins1985", correct_artifacts=True, show=False)

        # Create a numpy array for ROI regions with the same shape as ECG
        ecg_roi_array = np.zeros_like(ecg.reshape(1, window_size))

        # Iterate through ECG R peaks and set values to 1 within the ROI regions
        roi_size = 32
        for peak in info["ECG_R_Peaks"]:
            roi_start = max(0, peak - roi_size // 2)
            roi_end = min(roi_start + roi_size, window_size)
            ecg_roi_array[0, roi_start:roi_end] = 1

        return ecg.reshape(1, window_size).copy(), ppg.reshape(1, window_size).copy(), ecg_roi_array.copy(), y.copy() #, ppg_cwt.copy()

    def __len__(self):
        return len(self.ecg_data)

def get_dataset_withdiffusion(MODEL_PATH = "/tf/hsh/ECG_capstone/ECG2ECG_FINAL/LEAD1TO", DATA_PATH = "/tf/hsh/ECG_capstone/data/", lead_num=[2], only_one = False) :
    
    set_deterministic(31)
    
    for i in range(len(lead_num)) :
        _, dataset_test = get_datasets(DATA_PATH = DATA_PATH, datasets=["physionet2017/RDDM"], window_size=10, lead_num = lead_num[i])
        
        testloader = DataLoader(dataset_test, batch_size=16, shuffle=True, num_workers=64)
        
        dpm, Conditioning_network1, Conditioning_network2 = load_pretrained_DPM(
                PATH=MODEL_PATH + str(lead_num[i]) + '/',
                nT=10,
                type="RDDM",
                device="cuda")
            
        dpm = nn.DataParallel(dpm)
        Conditioning_network1 = nn.DataParallel(Conditioning_network1)
        Conditioning_network2 = nn.DataParallel(Conditioning_network2)
        
        dpm.eval()
        Conditioning_network1.eval()
        Conditioning_network2.eval()
        
        window_size = 10
        device="cuda"
        with torch.no_grad():
            
            fd_list = []
            fake_ecgs = np.zeros((1, 128*window_size))
            real_ppgs = np.zeros((1, 128*window_size))
            y_datas = np.array([0])
        
            for y_ecg, x_ppg, ecg_roi, y_data in tqdm(testloader):
                x_ppg = x_ppg.float().to(device)
                y_ecg = y_ecg.float().to(device)
        
                generated_windows = []
        
                for ppg_window in torch.split(x_ppg, 128*5, dim=-1):
                    
                    if ppg_window.shape[-1] != 128*5:
                        ppg_window = F.pad(ppg_window, (0, 128*5 - ppg_window.shape[-1]), "constant", 0)
        
                    ppg_conditions1 = Conditioning_network1(ppg_window)
                    ppg_conditions2 = Conditioning_network2(ppg_window)
        
                    xh = dpm(
                        cond1=ppg_conditions1, 
                        cond2=ppg_conditions2, 
                        mode="sample", 
                        window_size=128*5
                    )
                        
                    generated_windows.append(xh.cpu().numpy())
        
                xh = np.concatenate(generated_windows, axis=-1)[:, :, :128*window_size]
        
                fake_ecgs = np.concatenate((fake_ecgs, xh.reshape(-1, 128*window_size))) # fake y (만들어진 lead 2)
                real_ppgs = np.concatenate((real_ppgs, x_ppg.reshape(-1, 128*window_size).cpu().numpy())) # real x (lead 1)
                try :
                    y_datas = np.concatenate((y_datas, y_data.argmax(dim=1).numpy()))
                except :
                    y_datas = np.concatenate((y_datas, y_data.numpy()))
                
        if not only_one : 
            fake_ecgs_tensor = torch.tensor(fake_ecgs[1:], dtype=torch.float32)
        
        real_ppgs_tensor = torch.tensor(real_ppgs[1:], dtype=torch.float32)
        labels_tensor = torch.tensor(y_datas[1:], dtype=torch.float32)  # [N, 5]
        
        #assert fake_ecgs_tensor.shape == real_ppgs_tensor.shape
        
        if not only_one : 
            if i == 0:
                combined_data = torch.stack([real_ppgs_tensor, fake_ecgs_tensor], dim=1)  # [N, 2, 1280]
            else:
                fake_ecgs_tensor = fake_ecgs_tensor.unsqueeze(1)  # [N, 1, 1280]
                combined_data = torch.cat([combined_data, fake_ecgs_tensor], dim=1)       # [N, 기존+1, 1280]
        else :
            combined_data = torch.stack([real_ppgs_tensor], dim=1)
        
        print('----data setting with diffusion 완료----')
    
    dataset = TensorDataset(combined_data, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    batch_size=16
    N = len(dataset)
    train_len = int(N * 0.6)
    val_len = int(N * 0.2)
    test_len = N - train_len - val_len
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
    
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


def get_dataset_withdiffusion_naive(MODEL_PATH = "/tf/hsh/ECG_capstone/ECG2ECG_FINAL/LEAD1TO", DATA_PATH = "/tf/hsh/ECG_capstone/data/", lead_num=[2], only_one = False) :
    
    set_deterministic(31)
    
    for i in range(len(lead_num)) :
        _, dataset_test = get_datasets(DATA_PATH = DATA_PATH, datasets=[""], window_size=10, lead_num = lead_num[i])
        
        testloader = DataLoader(dataset_test, batch_size=16, shuffle=True, num_workers=64)
        
        dpm, Conditioning_network1, Conditioning_network2 = load_pretrained_DPM(
                PATH=MODEL_PATH + str(lead_num[i]) + '/',
                nT=10,
                type="naive",
                device="cuda")
            
        dpm = nn.DataParallel(dpm)
        Conditioning_network1 = nn.DataParallel(Conditioning_network1)
        
        dpm.eval()
        Conditioning_network1.eval()
        
        window_size = 10
        device="cuda"
        with torch.no_grad():
            
            fd_list = []
            fake_ecgs = np.zeros((1, 128*window_size))
            real_ppgs = np.zeros((1, 128*window_size))
            y_datas = np.array([0])
        
            for y_ecg, x_ppg, ecg_roi, y_data in tqdm(testloader):
                x_ppg = x_ppg.float().to(device)
                y_ecg = y_ecg.float().to(device)
        
                generated_windows = []
        
                for ppg_window in torch.split(x_ppg, 128*5, dim=-1):
                    
                    if ppg_window.shape[-1] != 128*5:
                        ppg_window = F.pad(ppg_window, (0, 128*5 - ppg_window.shape[-1]), "constant", 0)
        
                    ppg_conditions1 = Conditioning_network1(ppg_window)
        
                    xh = dpm(
                        cond=ppg_conditions1,
                        mode="sample", 
                        window_size=128*5
                    )
                        
                    generated_windows.append(xh.cpu().numpy())
        
                xh = np.concatenate(generated_windows, axis=-1)[:, :, :128*window_size]
        
                fake_ecgs = np.concatenate((fake_ecgs, xh.reshape(-1, 128*window_size))) # fake y (만들어진 lead 2)
                real_ppgs = np.concatenate((real_ppgs, x_ppg.reshape(-1, 128*window_size).cpu().numpy())) # real x (lead 1)
                try :
                    y_datas = np.concatenate((y_datas, y_data.argmax(dim=1).numpy()))
                except :
                    y_datas = np.concatenate((y_datas, y_data.numpy()))
                
        if not only_one : 
            fake_ecgs_tensor = torch.tensor(fake_ecgs[1:], dtype=torch.float32)
        
        real_ppgs_tensor = torch.tensor(real_ppgs[1:], dtype=torch.float32)
        labels_tensor = torch.tensor(y_datas[1:], dtype=torch.float32)  # [N, 5]
        
        #assert fake_ecgs_tensor.shape == real_ppgs_tensor.shape
        
        if not only_one : 
            if i == 0:
                combined_data = torch.stack([real_ppgs_tensor, fake_ecgs_tensor], dim=1)  # [N, 2, 1280]
            else:
                fake_ecgs_tensor = fake_ecgs_tensor.unsqueeze(1)  # [N, 1, 1280]
                combined_data = torch.cat([combined_data, fake_ecgs_tensor], dim=1)       # [N, 기존+1, 1280]
        else :
            combined_data = torch.stack([real_ppgs_tensor], dim=1)
        
        print('----data setting with diffusion 완료----')
    
    dataset = TensorDataset(combined_data, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    batch_size=16
    N = len(dataset)
    train_len = int(N * 0.6)
    val_len = int(N * 0.2)
    test_len = N - train_len - val_len
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
    
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader