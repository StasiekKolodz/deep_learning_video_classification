#!/usr/bin/env python3
"""
Project 4.1 — Minimal Video Classification Baselines (PyTorch)

Covers the four required model families as simply as possible, using the course-provided
`datasets.py` loaders (frames/videos, with `stackframes` controlling return shape).

Models implemented (select via --mode):
  1) perframe  — aggregation of per-frame 2D CNN predictions (avg logits over T)
  2) late      — late fusion with shared 2D backbone over frames + mean-pool features → linear head
  3) early     — early fusion: stack T RGB frames along channels (3*T) and pass to a 2D CNN
  4) 3d        — simple 3D CNN (R3D‑18) over [C,T,H,W]

Keeps everything tiny & readable so you can expand later.

Assumptions about `datasets.py`:
  - Uses `FrameVideoDataset` class which loads video frames from the UCF101 dataset.
  - When called with `stack_frames=True`, samples are returned as a 4D tensor [C, T, H, W].
  - When `stack_frames=False`, samples return a list of T tensors [C, H, W] (one per frame).
  - Dataset expects `root_dir` parameter pointing to the UCF101 data directory.
"""

import argparse
from dataclasses import dataclass
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import models, transforms

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import course datasets (adjust names if they differ in your `datasets.py`)
from datasets import FrameVideoDataset  # noqa: F401  # type: ignore


# -----------------------------
# Config & small utilities
# -----------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def tf_2d(img_size: int = 112):
    """Basic transform for 2D models."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def tf_3d(img_size: int = 112):
    """Same as 2D; frames are normalized one-by-one before stacking."""
    return tf_2d(img_size)


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
# Data loading helpers
# -----------------------------

def build_dataloaders(a: Args):
    """Create train/val loaders according to the selected mode.
    - perframe/late use stack_frames=False (list-of-frames) and a custom collate that stacks to [B,T,C,H,W]
    - early/3d use stack_frames=True to get [C,T,H,W] per sample; default collation gives [B,C,T,H,W]
    """
    is_early_or_3d = a.mode in {"early", "3d"}

    # Use FrameVideoDataset for both cases; switch stack_frames accordingly.
    ds_train = FrameVideoDataset(root_dir=a.data_root, split=a.split_train,
                                 transform=tf_3d(a.img_size), stack_frames=is_early_or_3d)
    ds_val = FrameVideoDataset(root_dir=a.data_root, split=a.split_val,
                               transform=tf_3d(a.img_size), stack_frames=is_early_or_3d)

    if is_early_or_3d:
        # Samples are [C,T,H,W]; default collation gives [B,C,T,H,W]
        train_loader = DataLoader(ds_train, batch_size=a.batch_size, shuffle=True,
                                  num_workers=a.num_workers, pin_memory=True)
        val_loader = DataLoader(ds_val, batch_size=a.batch_size, shuffle=False,
                                num_workers=a.num_workers, pin_memory=True)
    else:
        # Samples are list of T frames [C,H,W]; collate to [B,T,C,H,W]
        def collate_frames_list(batch):
            # Each item in batch is a tuple (frames_list, label)
            # where frames_list is a list of T tensors each [C,H,W]
            frames_batched = []
            labels_batched = []
            for frames_list, label in batch:
                # Stack the list of frames into [T,C,H,W]
                frames_tensor = torch.stack(frames_list, dim=0)
                frames_batched.append(frames_tensor)
                labels_batched.append(label)
            
            # Stack into [B,T,C,H,W]
            frames = torch.stack(frames_batched, dim=0)
            labels = torch.tensor(labels_batched, dtype=torch.long)
            return frames, labels

        train_loader = DataLoader(ds_train, batch_size=a.batch_size, shuffle=True,
                                  num_workers=a.num_workers, pin_memory=True,
                                  collate_fn=collate_frames_list)
        val_loader = DataLoader(ds_val, batch_size=a.batch_size, shuffle=False,
                                num_workers=a.num_workers, pin_memory=True,
                                collate_fn=collate_frames_list)

    return train_loader, val_loader


# -----------------------------
# Models
# -----------------------------

class ResNet18Backbone(nn.Module):
    """ResNet-18 backbone returning pooled features (no final classifier)."""
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the FC; keep everything up to avgpool
        self.stem = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
        )
        self.pool = m.avgpool  # outputs [B,512,1,1]
        self.out_dim = 512

    def forward(self, x):  # x: [N,3,H,W]
        x = self.stem(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x  # [N,512]


class PerFrameAggregationModel(nn.Module):
    """Apply 2D CNN independently to each frame, average logits across time."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = ResNet18Backbone()
        self.head = nn.Linear(self.backbone.out_dim, num_classes)

    def forward(self, frames_btchw):  # [B,T,C,H,W]
        B, T, C, H, W = frames_btchw.shape
        x = frames_btchw.view(B * T, C, H, W)
        feat = self.backbone(x)              # [B*T, 512]
        logits = self.head(feat)             # [B*T, K]
        logits = logits.view(B, T, -1).mean(dim=1)  # avg over time → [B,K]
        return logits


class LateFusionModel(nn.Module):
    """Shared 2D backbone per frame → mean-pool features over time → linear head."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = ResNet18Backbone()
        self.head = nn.Linear(self.backbone.out_dim, num_classes)

    def forward(self, frames_btchw):  # [B,T,C,H,W]
        B, T, C, H, W = frames_btchw.shape
        x = frames_btchw.view(B * T, C, H, W)
        feat = self.backbone(x).view(B, T, -1)  # [B,T,512]
        feat_vid = feat.mean(dim=1)             # [B,512]
        return self.head(feat_vid)              # [B,K]


class EarlyFusion2DModel(nn.Module):
    """Stack frames along channels (3*T) and feed to a 2D CNN (random init first conv)."""
    def __init__(self, num_frames: int, num_classes: int):
        super().__init__()
        m = models.resnet18(weights=None)  # first conv will change, so skip pretrained to keep it simple
        in_ch = 3 * num_frames
        m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net = m
        self.net.fc = nn.Linear(512, num_classes)

    def forward(self, frames_bcthw):  # [B,C,T,H,W]
        B, C, T, H, W = frames_bcthw.shape
        x = frames_bcthw.view(B, C * T, H, W)  # [B, 3*T, H, W]
        return self.net(x)


class R3D18Model(nn.Module):
    """3D CNN over [C,T,H,W]."""
    def __init__(self, num_classes: int):
        super().__init__()
        try:
            m = models.video.r3d_18(weights=None) 
        except Exception:
            # Fallback for older torchvision namespaces
            m = models.video.r3d_18(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.net = m

    def forward(self, x_bcthw):  # [B,C,T,H,W]
        return self.net(x_bcthw)


# -----------------------------
# Train / Eval
# -----------------------------

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


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


def run_epoch(model, loader, opt, device, mode: str, train: bool):
    model.train(train)
    total_loss, total_acc, n = 0.0, 0.0, 0
    
    with torch.set_grad_enabled(train):
        for batch in loader:
            frames, y = batch
            # Data loaders provide the correct shape for each mode:
            # - perframe/late: [B,T,C,H,W]
            # - early/3d: [B,C,T,H,W]
            x = frames.to(device)
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


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/dtu/datasets1/02516/ufc10", help="Root of provided dataset (frames/videos + CSVs)")
    p.add_argument("--mode", default="perframe", choices=["perframe", "late", "early", "3d"], help="Which baseline to train/eval")
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

    # Data
    train_loader, val_loader = build_dataloaders(a)

    # Model
    if a.mode == "perframe":
        model = PerFrameAggregationModel(a.num_classes)
    elif a.mode == "late":
        model = LateFusionModel(a.num_classes)
    elif a.mode == "early":
        model = EarlyFusion2DModel(a.num_frames, a.num_classes)
    else:  # 3d
        model = R3D18Model(a.num_classes)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

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
