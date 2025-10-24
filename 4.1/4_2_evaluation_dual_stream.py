#!/usr/bin/env python3
"""
Project 4.1 — Model Evaluation Script

Evaluates trained video classification models on test set.
Computes detailed metrics including:
  - Overall accuracy
  - Per-class accuracy
  - Confusion matrix
  - Top-k accuracy

Usage:
    python 4_1_evaluation.py --checkpoint path/to/model.pth --mode perframe --data_root ucf101/ufc10
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict

from torchvision import models, transforms

# Import dataset
from datasets import FrameVideoCustomDataset, FlowVideoDataset
from spatial_cnn import SpatialCNN
from temporal_cnn import TemporalCNN

# Import model definitions from training script
# Note: This assumes 4_1_training.py is in the same directory
# If you get import errors, make sure both files are in the same folder
from importlib import import_module

# Dynamic import to handle the filename with dots
import importlib.util
spec = importlib.util.spec_from_file_location("training_module", "4_1_training.py")
training_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_module)

ResNet18Backbone = training_module.ResNet18Backbone
PerFrameAggregationModel = training_module.PerFrameAggregationModel
LateFusionModel = training_module.LateFusionModel
EarlyFusion2DModel = training_module.EarlyFusion2DModel
R3D18Model = training_module.R3D18Model
tf_3d = training_module.tf_3d

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 112

transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


# -----------------------------
# Config
# -----------------------------

@dataclass
class EvalArgs:
    checkpoint: str
    data_root: str
    split_test: str = "test"
    batch_size: int = 8
    num_workers: int = 4
    num_classes: int = 10
    num_frames: int = 10
    img_size: int = 112
    mode: str = "perframe"  # perframe | late | early | 3d
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    class_names: List[str] = None  # Will be loaded from data if available


# -----------------------------
# Data loading
# -----------------------------

def build_test_loader(a: EvalArgs):
    """Create test loader according to the selected mode."""
    is_early_or_3d = a.mode in {"early", "3d"}

    ds_test = FrameVideoDataset(
        root_dir=a.data_root, 
        split=a.split_test,
        transform=tf_3d(a.img_size), 
        stack_frames=is_early_or_3d
    )

    if is_early_or_3d:
        # Samples are [C,T,H,W]; default collation gives [B,C,T,H,W]
        test_loader = DataLoader(
            ds_test, 
            batch_size=a.batch_size, 
            shuffle=False,
            num_workers=a.num_workers, 
            pin_memory=True
        )
    else:
        # Samples are list of T frames [C,H,W]; collate to [B,T,C,H,W]
        def collate_frames_list(batch):
            frames_batched = []
            labels_batched = []
            for frames_list, label in batch:
                frames_tensor = torch.stack(frames_list, dim=0)
                frames_batched.append(frames_tensor)
                labels_batched.append(label)
            
            frames = torch.stack(frames_batched, dim=0)
            labels = torch.tensor(labels_batched, dtype=torch.long)
            return frames, labels

        test_loader = DataLoader(
            ds_test, 
            batch_size=a.batch_size, 
            shuffle=False,
            num_workers=a.num_workers, 
            pin_memory=True,
            collate_fn=collate_frames_list
        )

    return test_loader, ds_test


# -----------------------------
# Model loading
# -----------------------------

def load_model(checkpoint_path: str, mode: str, num_classes: int, num_frames: int, device: str):
    """Load trained model from checkpoint."""
    # Create model
    if mode == "temporal":
        model = TemporalCNN(num_classes)
    elif mode == "spatial":
        model = SpatialCNN(num_classes)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {checkpoint_path}")
    return model


# -----------------------------
# Evaluation metrics
# -----------------------------

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm


def compute_per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """Compute per-class accuracy from confusion matrix."""
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc)  # Handle division by zero
    return per_class_acc


def compute_topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Compute top-k accuracy."""
    with torch.no_grad():
        _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.item()


# -----------------------------
# Main evaluation loop
# -----------------------------

@torch.no_grad()
def evaluate_model(model, loader, device, num_classes: int, mode) -> Dict:
    """Run full evaluation on test set."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_logits = []
    
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_samples = 0
    
    print("\nEvaluating on test set...")
    
    for batch_idx, (frames, targets) in enumerate(loader):
        frames = frames.to(device)
        targets = targets.to(device)

        if mode == "spatial":
            batch_logits = []
            for f in frames:  # iterate over videos in batch
                # Stack all frames of this video
                print("frames shape", f.shape) # [num_frames, C, H, W]
                logits = model(f)  # [num_frames, num_classes]
                logits = logits.mean(dim=0)  # average over frames -> [num_classes]
                batch_logits.append(logits)
            x = torch.stack(batch_logits)  # [B, num_classes]
        
        # Forward pass
        logits = x if mode == "spatial" else model(frames)
        loss = F.cross_entropy(logits, targets)
        
        # Top-1 predictions
        preds = logits.argmax(dim=1)
        
        # Accumulate
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_logits.append(logits.cpu())
        
        # Metrics
        total_loss += loss.item() * targets.size(0)
        total_correct_top1 += (preds == targets).sum().item()
        total_correct_top5 += compute_topk_accuracy(logits, targets, k=5)
        total_samples += targets.size(0)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {total_samples} samples...")
    
    # Convert to numpy
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_logits = torch.cat(all_logits, dim=0)
    
    # Compute metrics
    avg_loss = total_loss / total_samples
    top1_acc = total_correct_top1 / total_samples
    top5_acc = total_correct_top5 / total_samples
    
    # Confusion matrix and per-class accuracy
    cm = compute_confusion_matrix(all_targets, all_preds, num_classes)
    per_class_acc = compute_per_class_accuracy(cm)
    
    results = {
        'loss': avg_loss,
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_acc,
        'predictions': all_preds,
        'targets': all_targets,
        'logits': all_logits,
        'total_samples': total_samples
    }
    
    return results


# -----------------------------
# Results display
# -----------------------------

def print_results(results: Dict, class_names: List[str] = None):
    """Pretty print evaluation results."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n📊 Overall Metrics:")
    print(f"  • Test Loss:        {results['loss']:.4f}")
    print(f"  • Top-1 Accuracy:   {results['top1_accuracy']*100:.2f}%")
    print(f"  • Top-5 Accuracy:   {results['top5_accuracy']*100:.2f}%")
    print(f"  • Total Samples:    {results['total_samples']}")
    
    # Per-class accuracy
    print(f"\n📈 Per-Class Accuracy:")
    per_class_acc = results['per_class_accuracy']
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(per_class_acc))]
    
    for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
        print(f"  {i:2d}. {name:25s}: {acc*100:5.2f}%")
    
    print(f"\n  Mean per-class accuracy: {per_class_acc.mean()*100:.2f}%")
    
    # Confusion matrix summary
    cm = results['confusion_matrix']
    print(f"\n🔢 Confusion Matrix Shape: {cm.shape}")
    print(f"  Diagonal sum (correct): {np.trace(cm)}")
    print(f"  Off-diagonal sum (errors): {cm.sum() - np.trace(cm)}")
    
    # Most confused pairs
    print(f"\n❌ Top-5 Most Confused Pairs:")
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    
    confused_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm_no_diag[i, j] > 0:
                confused_pairs.append((i, j, cm_no_diag[i, j]))
    
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for rank, (true_idx, pred_idx, count) in enumerate(confused_pairs[:5], 1):
        true_name = class_names[true_idx] if class_names else f"Class {true_idx}"
        pred_name = class_names[pred_idx] if class_names else f"Class {pred_idx}"
        print(f"  {rank}. {true_name} → {pred_name}: {count} times")
    
    print("\n" + "="*70)


def save_results(results: Dict, output_path: str):
    """Save detailed results to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy archive
    np.savez(
        output_path,
        confusion_matrix=results['confusion_matrix'],
        per_class_accuracy=results['per_class_accuracy'],
        predictions=results['predictions'],
        targets=results['targets'],
        logits=results['logits'].numpy(),
        loss=results['loss'],
        top1_accuracy=results['top1_accuracy'],
        top5_accuracy=results['top5_accuracy']
    )
    
    print(f"\n✓ Results saved to {output_path}")


def print_confusion_matrix(cm: np.ndarray, class_names: List[str] = None, max_width: int = 80):
    """Print confusion matrix in a readable format."""
    n_classes = len(cm)
    
    if class_names is None:
        class_names = [f"C{i}" for i in range(n_classes)]
    
    # Truncate class names if needed
    max_name_len = max(3, (max_width - 10) // (n_classes + 1))
    class_names_short = [name[:max_name_len] for name in class_names]
    
    print("\n📊 Confusion Matrix:")
    print("    " + " ".join(f"{name:>5s}" for name in class_names_short))
    print("    " + "-" * (6 * n_classes))
    
    for i, row in enumerate(cm):
        row_str = " ".join(f"{val:5d}" for val in row)
        print(f"{class_names_short[i]:>3s} {row_str}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained video classification model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data_root", default="/dtu/datasets1/02516/ufc10", help="Root of dataset")
    parser.add_argument("--mode", required=True, choices=["temporal", "spatial"], 
                       help="Model architecture type")
    parser.add_argument("--split_test", default="test", help="Test split name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--img_size", type=int, default=112, help="Image size")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--output", default=None, help="Output path for results (.npz)")
    parser.add_argument("--show_cm", action="store_true", help="Print confusion matrix")
    
    args = parser.parse_args()
    
    # Create eval args
    eval_args = EvalArgs(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        split_test=args.split_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        img_size=args.img_size,
        mode=args.mode,
    )
    
    device = torch.device(eval_args.device)
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading test data from {eval_args.data_root}...")

        # Model
    if eval_args.mode == "spatial":
        print("SPATIAL DATASET")
        ds_test = FrameVideoCustomDataset(root_dir=eval_args.data_root, split="test",
                                    transform=transform, stack_frames=False)
        print("Training samples:", len(ds_test))

        test_loader = DataLoader(ds_test, batch_size=eval_args.batch_size, shuffle=True,
                                    num_workers=eval_args.num_workers, pin_memory=True)

    elif eval_args.mode == "temporal":
        print("TEMPORAL DATASET")
        ds_test = FlowVideoDataset(root_dir=eval_args.data_root, split="test",
                                    transform=transform)
        print("Training samples:", len(ds_test))

        test_loader = DataLoader(ds_test, batch_size=eval_args.batch_size, shuffle=True,
                                    num_workers=eval_args.num_workers, pin_memory=True)
    
    # Try to get class names from dataset
    class_names = None
    if hasattr(ds_test, 'df') and 'class_name' in ds_test.df.columns:
        # Get unique class names sorted by label
        df_classes = ds_test.df[['label', 'class_name']].drop_duplicates().sort_values('label')
        class_names = df_classes['class_name'].tolist()
    
    # Load model
    print(f"\nLoading {eval_args.mode} model...")
    model = load_model(
        eval_args.checkpoint, 
        eval_args.mode, 
        eval_args.num_classes, 
        eval_args.num_frames,
        device
    )
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, eval_args.num_classes, eval_args.mode)
    
    # Display results
    print_results(results, class_names)
    
    # Optionally print confusion matrix
    if args.show_cm:
        print_confusion_matrix(results['confusion_matrix'], class_names)
    
    # Save results
    if args.output:
        save_results(results, args.output)
    else:
        # Auto-generate output filename
        checkpoint_name = Path(args.checkpoint).stem
        output_path = f"evaluations/results_{checkpoint_name}_{args.mode}.npz"
        save_results(results, output_path)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
