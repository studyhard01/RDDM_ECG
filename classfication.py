import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter

from data_withdiffusion import get_dataset_withdiffusion

# 하이퍼파라미터 설정
NUM_CLASSES = 4
WINDOW_SIZE = 1280  # 10초 * 128Hz

# ✅ Dataset 로드
train_loader, val_loader, test_loader = get_dataset_withdiffusion(
    MODEL_PATH='/tf/hsh/SW_ECG/RDDM_ECG/saves/lead6_cossim',
    DATA_PATH='/tf/hsh/SW_ECG/single_lead_data/',
    only_one=False,
    lead_num=[6]
)

# ✅ 모델 정의
class ECG_CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 5), stride=1, padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 10))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 10, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, leads, time)
        x = self.conv(x)
        return self.classifier(x)

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc="Training", leave=False)
    
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        #output = model(x)
        output = model(x[:,:,:1260])
        loss = criterion(output, y.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader.dataset)

# ✅ 평가 함수 with 리포트
def evaluate_with_report(model, loader, criterion, device, phase="Eval", class_names=None):
    model.eval()
    total_loss = 0
    preds_all, targets_all, probs_all = [], [], []
    loop = tqdm(loader, desc=phase, leave=False)

    with torch.no_grad():
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            x = x[:, :, :WINDOW_SIZE]  # 슬라이싱
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item() * x.size(0)

            preds_all.append(output.cpu())
            targets_all.append(y.cpu())
            probs_all.append(torch.softmax(output, dim=1).cpu())

    preds = torch.cat(preds_all).argmax(dim=1)
    targets = torch.cat(targets_all)
    probs = torch.cat(probs_all)

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    try:
        auc = roc_auc_score(targets, probs, multi_class='ovr')
    except:
        auc = float('nan')

    report_str = classification_report(targets, preds, target_names=class_names if class_names else None)
    print(report_str)

    # 클래스별 정확도
    class_accs = {}
    for i in range(NUM_CLASSES):
        idx = (targets == i)
        class_accs[class_names[i] if class_names else str(i)] = accuracy_score(targets[idx], preds[idx]) if idx.sum() > 0 else 0.0

    return total_loss / len(loader.dataset), acc, f1, auc, report_str, class_accs

# ✅ EarlyStopping 클래스
class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
            if self.verbose:
                print("✅ Validation loss improved.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️ No improvement. Patience {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# ✅ 시드 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ✅ 학습 실행
def model_test(is_cnn=True, epochs=51):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECG_CNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter(log_dir="runs/exp_cnn")
    early_stopper = EarlyStopping(patience=7)

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, criterion, device, phase="Val")

        early_stopper(val_loss, model)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        writer.add_scalar("AUC/val", val_auc, epoch)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

        if early_stopper.early_stop:
            print("🛑 Early stopping triggered.")
            break

    # ✅ 모델 저장 및 테스트
    torch.save(early_stopper.best_model_state, './saves/best_model.pt')
    model.load_state_dict(torch.load('best_model.pt'))

    class_names = ["NORM", "AF", "I-AVB", "LBBB"]  # 클래스 수에 맞게 조정
    test_loss, test_acc, test_f1, test_auc, report_str, class_accs = evaluate_with_report(
        model, test_loader, criterion, device, phase="Test", class_names=class_names
    )

    print(f"\n✅ [FINAL TEST] Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}")
    print("Class-wise accuracy:", class_accs)

# 🚀 실행
model_test()