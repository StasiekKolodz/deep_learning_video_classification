import argparse
import random

from dataclasses import dataclass
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import course datasets (adjust names if they differ in your `datasets.py`)
from datasets import FrameVideoDataset, FlowVideoDataset 
from temporal_cnn import TemporalCNN


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 112

transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

training_transform = transforms.Compose([
    transforms.RandomResizedCrop(112, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])



@dataclass
class Args:
    data_root: str
    split_train: str = "train"
    split_val: str = "val"
    batch_size: int = 8
    epochs: int = 5
    lr: float = 1e-3
    num_workers: int = 4
    num_classes: int = 10
    num_frames: int = 10  # expected T per sample
    img_size: int = 112
    mode: str = "perframe"  # perframe | late | early | 3d
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "checkpoints"  # directory to save model checkpoints
    plot_dir: str = "plots"  # directory to save training plots
    save_plots: bool = True  # whether to save training plots


    
# -----------------------------
# Train / Eval
# -----------------------------

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()



def run_epoch(model, loader, opt, device, mode: str, train: bool):
    model.train(train)
    total_loss, total_acc, n = 0.0, 0.0, 0
    
    with torch.set_grad_enabled(train):
        for batch in loader:
            flows, y = batch
            # Data loaders provide the correct shape for each mode:
            # - perframe/late: [B,T,C,H,W]
            x = flows.to(device)
            y = y.to(device)

            if train:
                if opt is None:
                    raise ValueError("Optimizer must be provided when train=True")
                opt.zero_grad()

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            if train:
                loss.backward()
                opt.step()

            total_loss += loss.item() * y.size(0)
            total_acc += accuracy(logits, y) * y.size(0)
            n += y.size(0)

    return total_loss / n, total_acc / n


def plot_training_history(history: dict, save_path: str, mode: str):
    """Plot and save training history (loss and accuracy curves)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'Training and Validation Loss - {mode.upper()}', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
    axes[1].plot(epochs, history['val_acc'], 'r-s', label='Val Accuracy', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'Training and Validation Accuracy - {mode.upper()}', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  → Training plot saved to {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/dtu/datasets1/02516/ufc10", help="Root of provided dataset (frames/videos + CSVs)")
    p.add_argument("--mode", default="spatial", choices=["spatial", "temporal"], help="Which baseline to train/eval")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--img_size", type=int, default=112)
    p.add_argument("--num_frames", type=int, default=10)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    p.add_argument("--plot_dir", type=str, default="plots", help="Directory to save training plots")
    p.add_argument("--no_plots", action="store_true", help="Disable saving training plots")
    args_ns = p.parse_args()

    a = Args(
        data_root=args_ns.data_root,
        batch_size=args_ns.batch_size,
        epochs=args_ns.epochs,
        lr=args_ns.lr,
        num_workers=args_ns.num_workers,
        num_frames=args_ns.num_frames,
        img_size=args_ns.img_size,
        num_classes=args_ns.num_classes,
        mode=args_ns.mode,
        save_dir=args_ns.save_dir,
        plot_dir=args_ns.plot_dir,
        save_plots=not args_ns.no_plots,
    )

    device = torch.device(a.device)
    print(f"Using device: {device}")


    # Model
    if a.mode == "temporal":
        print("SPATIAL DATASET")
        ds_train = FlowVideoDataset(root_dir=a.data_root, split="train",
                                 transform=transform)
        ds_val = FlowVideoDataset(root_dir=a.data_root, split="val",
                                 transform=transform)
        print("Training samples:", len(ds_train))
        print("Validation samples:", len(ds_val))
        train_loader = DataLoader(ds_train, batch_size=a.batch_size, shuffle=True,
                                  num_workers=a.num_workers, pin_memory=True)
        val_loader = DataLoader(ds_val, batch_size=a.batch_size, shuffle=False,
                                num_workers=a.num_workers, pin_memory=True)
        model = TemporalCNN(a.num_classes)


    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)

    # Create save directories
    from pathlib import Path
    save_dir = Path(a.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if a.save_plots:
        plot_dir = Path(a.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

    # Track training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Train
    best_val_acc = 0.0
    for epoch in range(1, a.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, device, a.mode, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, None, device, a.mode, train=False)
        print(f"Epoch {epoch:02d}: train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        
        # Update history
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'train_loss': tr_loss,
            'train_acc': tr_acc,
            'val_loss': va_loss,
            'val_acc': va_acc,
            'mode': a.mode,
            'num_classes': a.num_classes,
            'num_frames': a.num_frames,
            'img_size': a.img_size,
            'history': history,  # Save full training history
        }
        
        # Save last checkpoint
        last_path = save_dir / f"{a.mode}_last.pth"
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_path = save_dir / f"{a.mode}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"  → Saved best model (val_acc: {va_acc:.3f})")

    # Save training plots
    if a.save_plots:
        plot_path = plot_dir / f"{a.mode}_training_history.png"
        plot_training_history(history, str(plot_path), a.mode)

    print(f"\n✓ Training complete! Best val accuracy: {best_val_acc:.3f}")
    print(f"  Models saved in: {save_dir}")
    print(f"    - {a.mode}_best.pth (best validation accuracy)")
    print(f"    - {a.mode}_last.pth (final epoch)")
    if a.save_plots:
        print(f"  Training plot saved in: {plot_dir}")
        print(f"    - {a.mode}_training_history.png")


if __name__ == "__main__":
    main()