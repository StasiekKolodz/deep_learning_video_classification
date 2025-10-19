#!/usr/bin/env python3
"""
Visualize evaluation results from saved .npz files.

Usage:
    python visualize_results.py results_perframe_best.npz
    python visualize_results.py results_*.npz  # Compare multiple models
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict


def load_results(npz_path: str) -> Dict:
    """Load results from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        'confusion_matrix': data['confusion_matrix'],
        'per_class_accuracy': data['per_class_accuracy'],
        'predictions': data['predictions'],
        'targets': data['targets'],
        'loss': float(data['loss']),
        'top1_accuracy': float(data['top1_accuracy']),
        'top5_accuracy': float(data['top5_accuracy']),
    }


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix", 
                         class_names: List[str] = None, save_path: str = None):
    """Plot confusion matrix as heatmap."""
    plt.figure(figsize=(10, 8))
    
    if class_names is None:
        class_names = [f"C{i}" for i in range(len(cm))]
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Plot
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
    else:
        plt.show()


def plot_per_class_accuracy(per_class_acc: np.ndarray, title: str = "Per-Class Accuracy",
                           class_names: List[str] = None, save_path: str = None):
    """Plot per-class accuracy as bar chart."""
    plt.figure(figsize=(12, 6))
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(per_class_acc))]
    
    x = np.arange(len(per_class_acc))
    bars = plt.bar(x, per_class_acc * 100, color='steelblue', alpha=0.8)
    
    # Color bars by performance
    for i, (bar, acc) in enumerate(zip(bars, per_class_acc)):
        if acc >= 0.8:
            bar.set_color('green')
        elif acc >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.axhline(y=per_class_acc.mean() * 100, color='red', linestyle='--', 
                label=f'Mean: {per_class_acc.mean()*100:.2f}%', linewidth=2)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved per-class accuracy to {save_path}")
    else:
        plt.show()


def plot_model_comparison(results_dict: Dict[str, Dict], save_path: str = None):
    """Compare multiple models."""
    models = list(results_dict.keys())
    top1_accs = [results_dict[m]['top1_accuracy'] * 100 for m in models]
    top5_accs = [results_dict[m]['top5_accuracy'] * 100 for m in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Top-1 accuracy comparison
    x = np.arange(len(models))
    bars1 = ax1.bar(x, top1_accs, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax1.set_title('Top-1 Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, top1_accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Top-5 accuracy comparison
    bars2 = ax2.bar(x, top5_accs, color='coral', alpha=0.8)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Top-5 Accuracy (%)', fontsize=12)
    ax2.set_title('Top-5 Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars2, top5_accs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved model comparison to {save_path}")
    else:
        plt.show()


def plot_per_class_comparison(results_dict: Dict[str, Dict], class_names: List[str] = None,
                              save_path: str = None):
    """Compare per-class accuracy across models."""
    models = list(results_dict.keys())
    num_classes = len(results_dict[models[0]]['per_class_accuracy'])
    
    if class_names is None:
        class_names = [f"C{i}" for i in range(num_classes)]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(num_classes)
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        per_class_acc = results_dict[model]['per_class_accuracy'] * 100
        offset = (i - len(models)/2) * width + width/2
        ax.bar(x + offset, per_class_acc, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved per-class comparison to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("results", nargs="+", help="Path(s) to .npz result file(s)")
    parser.add_argument("--save", action="store_true", help="Save plots instead of displaying")
    parser.add_argument("--output_dir", default="plots", help="Directory to save plots")
    parser.add_argument("--class_names", nargs="+", default=None, help="Class names")
    
    args = parser.parse_args()
    
    # Load results
    results_dict = {}
    for result_path in args.results:
        path = Path(result_path)
        if not path.exists():
            print(f"Warning: {result_path} not found, skipping...")
            continue
        
        model_name = path.stem.replace("results_", "")
        results_dict[model_name] = load_results(result_path)
        print(f"✓ Loaded {result_path}")
    
    if not results_dict:
        print("Error: No valid result files found!")
        return
    
    # Create output directory if saving
    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving plots to {output_dir}/")
    
    # Single model visualization
    if len(results_dict) == 1:
        model_name = list(results_dict.keys())[0]
        results = results_dict[model_name]
        
        print(f"\nVisualizing results for: {model_name}")
        print(f"  Top-1 Accuracy: {results['top1_accuracy']*100:.2f}%")
        print(f"  Top-5 Accuracy: {results['top5_accuracy']*100:.2f}%")
        
        # Confusion matrix
        save_path = output_dir / f"{model_name}_confusion_matrix.png" if args.save else None
        plot_confusion_matrix(
            results['confusion_matrix'],
            title=f"Confusion Matrix - {model_name}",
            class_names=args.class_names,
            save_path=save_path
        )
        
        # Per-class accuracy
        save_path = output_dir / f"{model_name}_per_class_accuracy.png" if args.save else None
        plot_per_class_accuracy(
            results['per_class_accuracy'],
            title=f"Per-Class Accuracy - {model_name}",
            class_names=args.class_names,
            save_path=save_path
        )
    
    # Multi-model comparison
    else:
        print(f"\nComparing {len(results_dict)} models:")
        for model_name, results in results_dict.items():
            print(f"  • {model_name}: Top-1 {results['top1_accuracy']*100:.2f}%, "
                  f"Top-5 {results['top5_accuracy']*100:.2f}%")
        
        # Model comparison
        save_path = output_dir / "model_comparison.png" if args.save else None
        plot_model_comparison(results_dict, save_path=save_path)
        
        # Per-class comparison
        save_path = output_dir / "per_class_comparison.png" if args.save else None
        plot_per_class_comparison(results_dict, class_names=args.class_names, save_path=save_path)
        
        # Individual confusion matrices
        if args.save:
            for model_name, results in results_dict.items():
                save_path = output_dir / f"{model_name}_confusion_matrix.png"
                plot_confusion_matrix(
                    results['confusion_matrix'],
                    title=f"Confusion Matrix - {model_name}",
                    class_names=args.class_names,
                    save_path=save_path
                )
    
    if not args.save:
        print("\nNote: Use --save flag to save plots instead of displaying them")
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
