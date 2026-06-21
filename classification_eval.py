import argparse
import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data_withdiffusion import get_dataset_withdiffusion


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
            nn.AdaptiveMaxPool2d((1, 10)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 10, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.classifier(x)


class EarlyStopping:
    def __init__(self, patience=7, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_dataset(dataset, batch_size, seed):
    dataset_len = len(dataset)
    train_len = int(dataset_len * 0.6)
    val_len = int(dataset_len * 0.2)
    test_len = dataset_len - train_len - val_len
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=generator,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, quiet=False):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc="Training", leave=False, disable=quiet)
    for x, y in loop:
        x = x.to(device)
        y = y.to(device).long()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, phase="Eval", quiet=False):
    model.eval()
    total_loss = 0.0
    logits_all = []
    targets_all = []
    probs_all = []
    loop = tqdm(loader, desc=phase, leave=False, disable=quiet)
    with torch.no_grad():
        for x, y in loop:
            x = x.to(device)
            y = y.to(device).long()
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item() * x.size(0)
            logits_all.append(output.cpu())
            targets_all.append(y.cpu())
            probs_all.append(torch.softmax(output, dim=1).cpu())
            loop.set_postfix(loss=loss.item())

    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    probs = torch.cat(probs_all)
    preds = logits.argmax(dim=1)

    acc = accuracy_score(targets.numpy(), preds.numpy())
    f1 = f1_score(targets.numpy(), preds.numpy(), average="macro")
    try:
        auc = roc_auc_score(targets.numpy(), probs.numpy(), multi_class="ovr")
    except ValueError:
        auc = float("nan")

    return total_loss / len(loader.dataset), acc, f1, auc


def run_single_experiment(dataset, args, run_idx):
    seed = args.seed + run_idx
    set_seed(seed)
    train_loader, val_loader, test_loader = split_dataset(dataset, args.batch_size, seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = ECG_CNN(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early_stopper = EarlyStopping(patience=args.patience)

    best_epoch = 0
    best_val_acc = float("nan")
    best_val_f1 = float("nan")
    best_val_auc = float("nan")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, quiet=args.quiet)
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, criterion, device, phase="Val", quiet=args.quiet)
        early_stopper(val_loss, model)
        if val_loss <= early_stopper.best_loss:
            best_epoch = epoch
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_val_auc = val_auc
        if not args.quiet:
            print(
                f"[Run {run_idx + 1}/{args.repeats} Epoch {epoch}] "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_auc={val_auc:.4f}"
            )
        if early_stopper.early_stop:
            break

    if early_stopper.best_model_state is not None:
        model.load_state_dict(early_stopper.best_model_state)

    test_loss, test_acc, test_f1, test_auc = evaluate(model, test_loader, criterion, device, phase="Test", quiet=args.quiet)
    return {
        "row_type": "run",
        "run": run_idx + 1,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "best_val_auc": best_val_auc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "test_roc_auc": test_auc,
    }


def summarize(rows, metric):
    values = np.array([row[metric] for row in rows], dtype=float)
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0


def write_results_csv(rows, args):
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_type",
        "run",
        "seed",
        "best_epoch",
        "best_val_acc",
        "best_val_f1",
        "best_val_auc",
        "test_loss",
        "test_acc",
        "test_macro_f1",
        "test_roc_auc",
        "input_lead",
        "target_leads",
        "model_base",
        "with_fftloss",
        "with_fftcond",
        "checkpoint_epoch",
        "only_one",
    ]

    enriched_rows = []
    for row in rows:
        enriched = dict(row)
        enriched.update(common_metadata(args))
        enriched_rows.append(enriched)

    for metric in ["test_loss", "test_acc", "test_macro_f1", "test_roc_auc"]:
        mean, std = summarize(rows, metric)
        summary_row = {
            "row_type": f"{metric}_summary",
            "run": "",
            "seed": "",
            "best_epoch": "",
            "best_val_acc": "",
            "best_val_f1": "",
            "best_val_auc": "",
            "test_loss": mean if metric == "test_loss" else "",
            "test_acc": mean if metric == "test_acc" else "",
            "test_macro_f1": mean if metric == "test_macro_f1" else "",
            "test_roc_auc": mean if metric == "test_roc_auc" else "",
            **common_metadata(args),
        }
        summary_row[f"{metric}_std"] = std
        enriched_rows.append(summary_row)

    extra_fields = sorted({key for row in enriched_rows for key in row.keys()} - set(fieldnames))
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra_fields)
        writer.writeheader()
        writer.writerows(enriched_rows)

    return output_path


def common_metadata(args):
    return {
        "input_lead": args.input_lead,
        "target_leads": " ".join(str(lead) for lead in args.target_leads),
        "model_base": args.model_base,
        "with_fftloss": args.with_fftloss,
        "with_fftcond": args.with_fftcond,
        "checkpoint_epoch": args.checkpoint_epoch,
        "only_one": args.only_one,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run repeated ECG classification experiments with generated leads.")
    parser.add_argument("--data-path", default="/tf/revision/data/", help="Root directory containing dataset folders.")
    parser.add_argument("--datasets", nargs="+", default=["PTBXL"], help="Dataset folder names under data-path.")
    parser.add_argument("--model-base", default="/tf/revision/model/", help="Root directory used by train.py --model-root.")
    parser.add_argument("--input-lead", type=int, default=1, help="Condition ECG lead number.")
    parser.add_argument("--target-leads", nargs="+", type=int, default=[2, 4, 5, 6, 11, 12], help="Generated target leads to use.")
    parser.add_argument("--with-fftloss", action="store_true", help="Read checkpoints from the withfftloss directory.")
    parser.add_argument("--with-fftcond", action="store_true", help="Load FFT-aware condition networks.")
    parser.add_argument("--checkpoint-epoch", type=int, default=120, help="Diffusion checkpoint epoch.")
    parser.add_argument("--only-one", action="store_true", help="Use only the input lead baseline.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeated classification experiments.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed. Run i uses seed + i.")
    parser.add_argument("--epochs", type=int, default=50, help="Max classifier training epochs per repeat.")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for generation and classification.")
    parser.add_argument("--num-workers", type=int, default=64, help="DataLoader worker count for generation.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Classifier learning rate.")
    parser.add_argument("--num-classes", type=int, default=5, help="Number of disease classes.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--output-csv", default="classification_results.csv", help="CSV file for repeated experiment results.")
    parser.add_argument("--quiet", action="store_true", help="Disable tqdm and per-epoch logs.")
    return parser.parse_args()


def validate_args(args):
    if args.repeats <= 0:
        raise ValueError("repeats must be greater than 0.")
    if args.epochs <= 0:
        raise ValueError("epochs must be greater than 0.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    leads = [args.input_lead, *args.target_leads]
    invalid = [lead for lead in leads if lead < 1 or lead > 12]
    if invalid:
        raise ValueError(f"Lead numbers must be between 1 and 12: {invalid}")
    if not args.only_one and args.input_lead in args.target_leads:
        raise ValueError("input_lead cannot also be a generated target lead.")


if __name__ == "__main__":
    args = parse_args()
    validate_args(args)
    print("Building classification dataset...")
    dataset = get_dataset_withdiffusion(
        DATA_PATH=args.data_path,
        datasets=args.datasets,
        lead_num=args.target_leads,
        only_one=args.only_one,
        model_base=args.model_base,
        input_lead=args.input_lead,
        with_fftloss=args.with_fftloss,
        with_fftcond=args.with_fftcond,
        checkpoint_epoch=args.checkpoint_epoch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        return_dataset=True,
    )
    print(f"Dataset size: {len(dataset)}")

    rows = []
    for run_idx in range(args.repeats):
        print(f"\nStarting classification repeat {run_idx + 1}/{args.repeats}")
        row = run_single_experiment(dataset, args, run_idx)
        rows.append(row)
        print(
            f"[Run {row['run']}] "
            f"test_acc={row['test_acc']:.4f} "
            f"test_macro_f1={row['test_macro_f1']:.4f} "
            f"test_roc_auc={row['test_roc_auc']:.4f}"
        )

    output_path = write_results_csv(rows, args)
    print(f"\nSaved repeated classification results to: {output_path}")
    for metric in ["test_acc", "test_macro_f1", "test_roc_auc"]:
        mean, std = summarize(rows, metric)
        print(f"{metric}: mean={mean:.4f}, std={std:.4f}")
