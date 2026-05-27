from pathlib import Path
import numpy as np
import neurokit2 as nk
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, ecgall_data, ecgcond_data):
        self.ecgall_data = ecgall_data
        self.ecgcond_data = ecgcond_data

    def __getitem__(self, index):

        ecgall = self.ecgall_data[index]
        ecgcond = self.ecgcond_data[index]
        
        window_size = ecgall.shape[-1]

        ecgall = ecgall.reshape(window_size)
        ecgcond = ecgcond.reshape(window_size)

        _, info = nk.ecg_peaks(ecgall, sampling_rate=128, method="pantompkins1985", correct_artifacts=True, show=False)

        # Create a numpy array for ROI regions with the same shape as ECG
        ecg_roi_array = np.zeros_like(ecgall.reshape(1, window_size))

        # Iterate through ECG R peaks and set values to 1 within the ROI regions
        roi_size = 32
        for peak in info["ECG_R_Peaks"]:
            roi_start = max(0, peak - roi_size // 2)
            roi_end = min(roi_start + roi_size, window_size)
            ecg_roi_array[0, roi_start:roi_end] = 1

        return ecgall.reshape(1, window_size).copy(), ecgcond.reshape(1, window_size).copy(), ecg_roi_array.copy() #, ppg_cwt.copy()

    def __len__(self):
        return len(self.ecgall_data)

def get_datasets(
    DATA_PATH = "/tf/revision/data/",
    datasets=["PTBXL"],
    window_size=5,
    input_lead=1,
    target_lead=4
    ):

    ecgall_train_list = []
    ecgcond_train_list = []
    ecgall_test_list = []
    ecgcond_test_list = []
    
    
    for dataset in datasets:
        dataset_path = Path(DATA_PATH) / dataset
        
        ecgall_train = np.load(dataset_path / f"lead{target_lead}_train.npy", allow_pickle=True).reshape(-1, 128*window_size)
        ecgcond_train = np.load(dataset_path / f"lead{input_lead}_train.npy", allow_pickle=True).reshape(-1, 128*window_size)
        
        ecgall_test = np.load(dataset_path / f"lead{target_lead}_test.npy", allow_pickle=True).reshape(-1, 128*window_size)
        ecgcond_test = np.load(dataset_path / f"lead{input_lead}_test.npy", allow_pickle=True).reshape(-1, 128*window_size)

        ecgall_train_list.append(ecgall_train)
        ecgcond_train_list.append(ecgcond_train)
        ecgall_test_list.append(ecgall_test)
        ecgcond_test_list.append(ecgcond_test)

    ecgall_train = np.nan_to_num(np.concatenate(ecgall_train_list).astype("float32"))
    ecgcond_train = np.nan_to_num(np.concatenate(ecgcond_train_list).astype("float32"))

    ecgall_test = np.nan_to_num(np.concatenate(ecgall_test_list).astype("float32"))
    ecgcond_test = np.nan_to_num(np.concatenate(ecgcond_test_list).astype("float32"))

    dataset_train = ECGDataset(
        ecgall_train,
        ecgcond_train
    )
    dataset_test = ECGDataset(
        ecgall_test,
        ecgcond_test
    )

    return dataset_train, dataset_test
