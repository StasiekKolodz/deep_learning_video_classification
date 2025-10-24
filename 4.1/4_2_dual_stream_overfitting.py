import argparse
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
from datasets import FrameImageDataset, FlowVideoDataset 


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 112

transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

training_transform = transforms.Compose([
    transforms.RandomResizedCrop(100, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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


class SpatialCNN(nn.Module):
    def __init__(self, num_classes=101, dropout_p=0.6, freeze_until_layer=2):
        super().__init__()

        # Load pretrained ResNet-18
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Optionally freeze early layers (to reduce overfitting)
        for name, param in self.resnet.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False

        # Replace classifier head
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
    

def modify_resnet_for_flow(model, num_in_channels=20):
    # Get the pretrained conv1 weights
    old_weights = model.conv1.weight.data  # shape [64, 3, 7, 7]
    # Average across RGB channels
    mean_weights = old_weights.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
    # Replicate across flow channels (x,y for T frames)
    new_weights = mean_weights.repeat(1, num_in_channels, 1, 1)  # [64, 20, 7, 7]

    # Replace conv1 with new layer
    model.conv1 = nn.Conv2d(num_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.data = new_weights

    return model

class TemporalCNN(nn.Module):
    def __init__(self, num_classes=101, num_in_channels=20):
        super().__init__()
        self.resnet = models.resnet101(pretrained=True)
        self.resnet = modify_resnet_for_flow(self.resnet, num_in_channels)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    
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
            frames, y = batch
            # Data loaders provide the correct shape for each mode:
            # - perframe/late: [B,T,C,H,W]
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
    if a.mode == "spatial":
        print("SPATIAL DATASET")
        ds_train = FrameImageDataset(root_dir=a.data_root, split="train",
                                 transform=training_transform)
        ds_val = FrameImageDataset(root_dir=a.data_root, split="val",
                                 transform=transform)
        print("Training samples:", len(ds_train))
        print("Validation samples:", len(ds_val))
        train_loader = DataLoader(ds_train, batch_size=a.batch_size, shuffle=True,
                                  num_workers=a.num_workers, pin_memory=True)
        val_loader = DataLoader(ds_val, batch_size=a.batch_size, shuffle=False,
                                num_workers=a.num_workers, pin_memory=True)
        model = SpatialCNN(a.num_classes)
    elif a.mode == "temporal":

        model = TemporalCNN(a.num_classes)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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
        
    #     # Update history
    #     history['train_loss'].append(tr_loss)
    #     history['train_acc'].append(tr_acc)
    #     history['val_loss'].append(va_loss)
    #     history['val_acc'].append(va_acc)

    #     # Save checkpoint
    #     checkpoint = {
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': opt.state_dict(),
    #         'train_loss': tr_loss,
    #         'train_acc': tr_acc,
    #         'val_loss': va_loss,
    #         'val_acc': va_acc,
    #         'mode': a.mode,
    #         'num_classes': a.num_classes,
    #         'num_frames': a.num_frames,
    #         'img_size': a.img_size,
    #         'history': history,  # Save full training history
    #     }
        
    #     # Save last checkpoint
    #     last_path = save_dir / f"{a.mode}_last.pth"
    #     torch.save(checkpoint, last_path)
        
    #     # Save best checkpoint
    #     if va_acc > best_val_acc:
    #         best_val_acc = va_acc
    #         best_path = save_dir / f"{a.mode}_best.pth"
    #         torch.save(checkpoint, best_path)
    #         print(f"  → Saved best model (val_acc: {va_acc:.3f})")

    # # Save training plots
    # if a.save_plots:
    #     plot_path = plot_dir / f"{a.mode}_training_history.png"
    #     plot_training_history(history, str(plot_path), a.mode)

    # print(f"\n✓ Training complete! Best val accuracy: {best_val_acc:.3f}")
    # print(f"  Models saved in: {save_dir}")
    # print(f"    - {a.mode}_best.pth (best validation accuracy)")
    # print(f"    - {a.mode}_last.pth (final epoch)")
    # if a.save_plots:
    #     print(f"  Training plot saved in: {plot_dir}")
    #     print(f"    - {a.mode}_training_history.png")


if __name__ == "__main__":
    main()