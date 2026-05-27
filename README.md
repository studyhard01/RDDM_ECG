# RDDM_ECG

Region-Disentangled Diffusion Model(RDDM)을 ECG lead 변환 실험에 맞게 수정한 프로젝트입니다. 단일 ECG Lead I(`lead1`)을 조건으로 다른 ECG lead를 생성하고, 생성된 lead를 Lead I와 결합해 5-class 심혈관 질환 분류 성능을 확인하는 흐름을 다룹니다.

기반 논문 구조는 RDDM의 PPG-to-ECG translation이지만, 이 저장소의 실험은 ECG-to-ECG multi-lead generation입니다. 프로젝트 내부 PDF(`Region-Disentangled Diffusion과 주파수 정보 융합.pdf`)의 실험 구조는 다음과 같습니다.

- PTB-XL 12-lead ECG를 128 Hz로 다운샘플링합니다.
- Lead I만 입력으로 사용하고, target lead는 개별 모델로 생성합니다.
- 실험상 주요 생성/분류 대상 lead는 Lead II, aVR, aVL, aVF, V5, V6입니다.
- 생성 모델은 baseline RDDM과 FFT loss를 추가한 제안 방법을 비교합니다.
- 생성 성능은 RMSE, FD로 평가하고, 질환 분류는 Accuracy, macro F1, ROC-AUC로 평가합니다.
- 분류 입력은 10초 길이 기준이며, 생성 신호는 5초 단위로 생성한 뒤 이어 붙여 10초 입력으로 사용합니다.

## Dataset

코드는 `.npy` 파일을 직접 읽습니다. 기본 경로는 스크립트마다 다르므로 실행 환경에 맞게 수정해야 합니다.

생성 모델 학습/평가용 `data.py` 기본 구조:

```text
/tf/revision/data/
└── PTBXL/
    ├── lead1_train.npy
    ├── lead1_test.npy
    ├── ...
    ├── lead12_train.npy
    └── lead12_test.npy
```

분류용 `data_withdiffusion.py` 기본 구조:

```text
/tf/hsh/ECG_capstone/data/
├── lead1_train.npy
├── lead1_test.npy
├── ...
├── lead12_train.npy
├── lead12_test.npy
├── y_train.npy
└── y_test.npy
```

Lead 번호는 일반 12-lead ECG 순서를 따르는 것으로 사용합니다.

| File lead | ECG lead |
| --- | --- |
| `lead1` | I |
| `lead2` | II |
| `lead3` | III |
| `lead4` | aVR |
| `lead5` | aVL |
| `lead6` | aVF |
| `lead7` | V1 |
| `lead8` | V2 |
| `lead9` | V3 |
| `lead10` | V4 |
| `lead11` | V5 |
| `lead12` | V6 |

현재 `data.py`는 조건 lead를 `lead1`, target lead를 `lead4`로 읽습니다. 다른 target lead를 학습하려면 `data.py`의 `lead4_train.npy`, `lead4_test.npy` 부분을 원하는 target lead 파일명으로 바꾸고, `train.py`의 저장 경로와 wandb run id도 같이 구분해야 합니다.

## File Guide

| File | Role |
| --- | --- |
| `README.md` | 프로젝트 구조, 파일 역할, 실험 실행 순서를 설명합니다. |
| `LICENSE` | 원본/프로젝트 라이선스 파일입니다. |
| `Region-Disentangled Diffusion과 주파수 정보 융합.pdf` | 실험 배경과 최종 실험 구성, 생성/분류 결과를 설명하는 문서입니다. |
| `data.py` | 생성 모델 학습/평가용 PTB-XL lead pair dataset을 만듭니다. 기본은 `lead1 -> lead4`입니다. target lead에서 R-peak를 찾고, peak 주변 32 sample을 ROI mask로 만듭니다. 현재 버전은 로드된 ECG에 `ecg_clean`을 적용하지 않습니다. |
| `train.py` | RDDM 또는 FFT 조건/FFT loss 변형 모델을 학습합니다. `data.py`의 train dataset을 사용하고 checkpoint를 저장합니다. |
| `diffusion.py` | DDPM/RDDM forward process, reverse sampling, pretrained checkpoint loader를 정의합니다. `RDDM`, `RDDMfft`, `Naive` 타입 로딩을 담당합니다. |
| `model.py` | 모델 정의 모음입니다. RDDM용 1D U-Net, cross-attention U-Net, condition network, FFT condition network, ST-MEM 기반 분류 보조 모델이 들어 있습니다. |
| `metrics.py` | FFT loss, FD, RMSE, heart-rate MAE 계산에 필요한 지표 함수를 제공합니다. |
| `lr_scheduler.py` | warmup을 포함한 cosine annealing learning-rate scheduler입니다. |
| `std_eval.py` | 학습된 RDDM/Naive checkpoint를 로드해 생성 성능을 평가합니다. RMSE, FD, MAE_HR_ECG를 계산합니다. |
| `std_eval.sh` | SLURM 환경에서 `std_eval.py`를 실행하기 위한 예시 submit script입니다. 경로는 현재 클러스터 예시값이므로 실행 환경에 맞게 수정해야 합니다. |
| `data_withdiffusion.py` | 학습된 diffusion 모델로 target lead를 생성하고, Lead I와 생성 lead를 결합한 classification dataloader를 만듭니다. `only_one=True`이면 Lead I baseline만 사용합니다. |
| `RDDM_classification.ipynb` | 생성 lead를 이용한 1D CNN/2D CNN/ST-MEM 기반 질환 분류 실험 notebook입니다. 현재 실행 흐름은 주로 2D CNN 분류를 사용합니다. |
| `RDDM_visualization.ipynb` | 학습된 모델로 생성한 ECG와 실제 target ECG, 입력 Lead I를 시각화하는 notebook입니다. |
| `.ipynb_checkpoints/RDDM_visualization-checkpoint.ipynb` | Jupyter 자동 checkpoint입니다. 실험 실행에 직접 사용할 필요는 없습니다. |

## Experiment Flow

### 1. 데이터 준비

PTB-XL 데이터를 128 Hz, 5초 또는 10초 window로 전처리한 `.npy` 파일을 준비합니다.

생성 모델 학습은 5초 window를 기본으로 사용합니다.

```python
dataset_train, dataset_test = get_datasets(
    DATA_PATH="/tf/revision/data/",
    datasets=["PTBXL"],
    window_size=5,
)
```

분류 실험은 10초 입력을 기본으로 사용하며, `data_withdiffusion.py`는 10초 신호를 5초 chunk로 나누어 생성한 뒤 다시 이어 붙입니다.

### 2. Target lead 선택

현재 기본 코드는 `lead1 -> lead4(aVR)`입니다.

PDF 실험의 6개 주요 target lead를 재현하려면 target별로 개별 모델을 학습합니다.

| Experiment | Input | Target file |
| --- | --- | --- |
| Lead II generation | `lead1` | `lead2` |
| aVR generation | `lead1` | `lead4` |
| aVL generation | `lead1` | `lead5` |
| aVF generation | `lead1` | `lead6` |
| V5 generation | `lead1` | `lead11` |
| V6 generation | `lead1` | `lead12` |

각 target마다 `data.py`의 target filename과 `train.py`의 `PATH`, wandb `id`를 target lead에 맞게 바꿔서 실행합니다. 예를 들어 Lead II 모델은 `lead4_train.npy`를 `lead2_train.npy`로, `lead4_test.npy`를 `lead2_test.npy`로 바꿉니다.

### 3. RDDM baseline 학습

`train.py`의 config를 baseline RDDM에 맞춥니다.

```python
config = {
    "n_epoch": 181,
    "batch_size": 32,
    "nT": 10,
    "device": "cuda",
    "attention_heads": 8,
    "cond_mask": 0.0,
    "alpha1": 100,
    "alpha2": 1,
    "alphafft": 0.1,
    "PATH": "/tf/revision/model/none/1to4",
    "with_fftloss": False,
    "sampling_rate": 128,
    "cutoff_freq": 30.0,
    "with_fftcond": False,
}
```

실행:

```bash
python train.py
```

학습은 다음 세 모델을 저장합니다.

```text
RDDM_epoch{epoch}.pth
ConditionNet1_epoch{epoch}.pth
ConditionNet2_epoch{epoch}.pth
```

현재 저장 주기는 `i % 30 == 0`입니다.

### 4. FFT loss / FFT condition 모델 학습

PDF의 제안 방법은 FFT loss를 추가해 생성 신호와 실제 신호의 주파수 magnitude 차이를 줄이는 방식입니다.

`train.py`에서 다음 값을 켭니다.

```python
"with_fftloss": True,
"with_fftcond": True,
```

loss는 코드 기준으로 다음 가중합입니다.

```text
loss = 100 * DDPM_loss + 1 * Region_loss + 0.1 * FFT_loss
```

`with_fftcond=True`이면 `ConditionNetWithFFT`를 사용해 time-domain 조건과 rFFT magnitude 조건을 함께 구성합니다.

### 5. 생성 성능 평가

`std_eval.py`는 checkpoint를 로드해 생성 ECG를 만들고 RMSE, FD, MAE_HR_ECG를 계산합니다.

```bash
python std_eval.py
```

SLURM 환경에서는 `std_eval.sh`를 환경에 맞게 고친 뒤 제출합니다.

```bash
sbatch std_eval.sh
```

주의할 점:

- `diffusion.py`의 `load_pretrained_DPM()`은 현재 RDDM/RDDMfft에서 `RDDM_epoch500.pth`, `ConditionNet1_epoch500.pth`, `ConditionNet2_epoch500.pth`를 찾습니다.
- `train.py` 기본 config는 `n_epoch=181`이고 저장 주기는 30 epoch라서 기본 실행만으로는 `epoch500` checkpoint가 만들어지지 않습니다.
- 평가 전에 `diffusion.py`의 checkpoint epoch 또는 `train.py`의 epoch/save 정책을 실제 파일에 맞춰야 합니다.
- FFT condition 모델을 평가하려면 `std_eval.py`의 `type="RDDM"` 호출을 `type="RDDMfft"`로 바꿔야 합니다.

### 6. 생성 lead 시각화

`RDDM_visualization.ipynb`를 실행하면 생성 lead, 실제 target lead, 입력 Lead I를 함께 그려볼 수 있습니다.

Notebook 내부 기본 흐름:

1. `eval_diffusion()`으로 checkpoint를 로드합니다.
2. test set에서 ECG를 생성합니다.
3. `fake_ecgs`, `real_ecg2`, `real_ecg1`를 plot합니다.

시각화 notebook은 분석용 보조 파일이므로, 정량 평가는 `std_eval.py`를 우선 사용합니다.

### 7. 생성 lead로 분류 dataloader 만들기

`data_withdiffusion.py`의 `get_dataset_withdiffusion()`을 사용합니다.

Lead I baseline:

```python
from data_withdiffusion import get_dataset_withdiffusion

train_loader, val_loader, test_loader = get_dataset_withdiffusion(
    MODEL_PATH="/cap/RDDM-main/hsh/ECG2ECG_FINAL/LEAD1TO",
    DATA_PATH="/cap/RDDM-main/datasets/",
    only_one=True,
)
```

Lead I와 생성 lead를 함께 사용하는 실험:

```python
train_loader, val_loader, test_loader = get_dataset_withdiffusion(
    MODEL_PATH="/cap/RDDM-main/hsh/ECG2ECG_FINAL/LEAD1TO",
    DATA_PATH="/cap/RDDM-main/datasets/",
    lead_num=[2],
    only_one=False,
)
```

여러 생성 lead를 붙이는 실험:

```python
train_loader, val_loader, test_loader = get_dataset_withdiffusion(
    MODEL_PATH="/cap/RDDM-main/hsh/ECG2ECG_FINAL/LEAD1TO",
    DATA_PATH="/cap/RDDM-main/datasets/",
    lead_num=[2, 4, 5, 6, 11, 12],
    only_one=False,
)
```

`MODEL_PATH + str(lead_num) + "/"` 아래에 각 lead별 checkpoint가 있어야 합니다.

### 8. 질환 분류 실험

`RDDM_classification.ipynb`를 실행합니다.

주요 흐름:

1. `get_dataset_withdiffusion()`으로 train/val/test loader를 만듭니다.
2. `ECG_CNN` 2D CNN 모델을 정의합니다.
3. CrossEntropyLoss와 Adam(`lr=1e-4`)으로 학습합니다.
4. 최대 50 epoch 동안 학습하고, validation loss 기준 early stopping(`patience=7`)을 적용합니다.
5. test set에서 Accuracy, macro F1, ROC-AUC를 출력합니다.

`data_withdiffusion.py`는 전체 dataset을 6:2:2 비율로 나눕니다.

```text
train: 60%
validation: 20%
test: 20%
batch size: 16
```

## Current Defaults And Things To Check

- `data.py`는 현재 `lead1 -> lead4`로 고정되어 있습니다. 여러 lead 실험은 target filename을 바꿔 반복 실행해야 합니다.
- `data.py`는 raw ECG를 그대로 사용하고, ROI mask 생성을 위해 `nk.ecg_peaks()`만 호출합니다.
- `data_withdiffusion.py` 내부 변수명은 `ppg`로 남아 있지만, 이 프로젝트에서는 실제로 Lead I ECG 조건 신호를 의미합니다.
- `train.py` 기본값은 FFT loss와 FFT condition이 꺼져 있습니다.
- `diffusion.py`의 load epoch와 실제 저장 epoch가 다를 수 있으므로 평가 전 checkpoint 이름을 확인해야 합니다.
- `std_eval.py` 기본 `eval_diffusion()`은 `type="RDDM"`을 사용합니다. FFT condition checkpoint는 `type="RDDMfft"`로 평가해야 합니다.
- notebook에는 실행 결과와 실험 흔적이 남아 있습니다. 재현 실험 전에는 첫 셀부터 경로와 `only_one`, `lead_num`을 확인하세요.

## Suggested Reproduction Order

1. PTB-XL `.npy` 파일을 준비합니다.
2. `data.py`에서 target lead를 정합니다.
3. `train.py`에서 baseline RDDM config를 설정하고 학습합니다.
4. `train.py`에서 FFT loss/FFT condition config를 설정하고 같은 target lead를 다시 학습합니다.
5. target lead마다 2-4단계를 반복합니다.
6. `diffusion.py`와 `std_eval.py`의 checkpoint epoch/type/path를 실제 파일에 맞춥니다.
7. `std_eval.py`로 RMSE, FD, MAE_HR_ECG를 계산합니다.
8. `RDDM_visualization.ipynb`로 생성 파형을 확인합니다.
9. `data_withdiffusion.py`로 Lead I baseline 또는 Lead I + generated lead dataloader를 만듭니다.
10. `RDDM_classification.ipynb`로 2D CNN 분류 성능을 평가합니다.

## Notes

이 저장소에는 `requirements.txt`가 없습니다. 코드 import 기준으로 최소한 PyTorch, NumPy, tqdm, wandb, neurokit2, scikit-learn, biosppy, torchmetrics, similaritymeasures, matplotlib, tensorboard, Jupyter 환경이 필요합니다.
