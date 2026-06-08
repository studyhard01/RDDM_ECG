import os
import torch
import matplotlib.pyplot as plt
from data import get_datasets
from model import ConditionNet
from diffusion import load_pretrained_DPM

# 설정
SAVE_DIR = "./visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)

# 구성 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
nT = 10
model_path = "./saves/lead6_cossim/"  # 학습된 모델이 저장된 경로
num_samples = 16  # 시각화할 샘플 수
window_size = 128 * 5  # 5초 기준
batch_size = 16

# 데이터셋 불러오기
dataset_train, dataset_test = get_datasets()
dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# 사전 학습된 모델 로드
dpm, cond_net1, cond_net2 = load_pretrained_DPM(model_path, nT=nT, type="RDDM", device=device)

# 시각화용 배치 가져오기
x_targets, x_conditions, _ = next(iter(dataloader))
x_targets = x_targets.float().to(device)
x_conditions = x_conditions.float().to(device)

# 조건 네트워크를 통해 조건 인코딩 생성
with torch.no_grad():
    cond1 = cond_net1(x_conditions)
    cond2 = cond_net2(x_conditions)

# 샘플링
with torch.no_grad():
    generated = dpm(x=None, cond1=cond1, cond2=cond2, mode="sample")

# numpy로 변환
x_targets = x_targets.cpu().numpy()
generated = generated.cpu().numpy()

# 시각화
plt.figure(figsize=(20, 12))
for i in range(num_samples):
    plt.subplot(4, 4, i + 1)
    plt.plot(x_targets[i].flatten(), label="Target", linewidth=1.5)
    plt.plot(generated[i].flatten(), label="Generated", linestyle='--', linewidth=1.5)
    plt.title(f"Sample {i+1}")
    plt.xticks([])
    plt.yticks([])
    if i == 0:
        plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "lead6_cossim_400epoch.png"))
print(f"✅ 시각화 결과 저장 완료")
