#!/usr/bin/env python3
"""
Print Best Training and Test Accuracies

This script reads checkpoint files and evaluation results to display
the best accuracies achieved by each model in a markdown table format.

Usage:
    python print_results.py                           # Auto-detect all models
    python print_results.py --markdown                # Output as markdown table
    python print_results.py --models perframe late    # Specific models only
    python print_results.py --summary_only            # Skip detailed output
    python print_results.py --checkpoint_dir checkpoints --eval_dir evaluations
"""

import argparse
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def load_checkpoint_info(checkpoint_path: Path) -> Optional[Dict]:
    """Load training information from checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            return {
                'mode': checkpoint.get('mode', 'unknown'),
                'epoch': checkpoint.get('epoch', 0),
                'train_acc': checkpoint.get('train_acc', 0.0),
                'val_acc': checkpoint.get('val_acc', 0.0),
                'train_loss': checkpoint.get('train_loss', 0.0),
                'val_loss': checkpoint.get('val_loss', 0.0),
                'history': checkpoint.get('history', None)
            }
        return None
    except Exception as e:
        print(f"Warning: Could not load {checkpoint_path}: {e}")
        return None


def load_evaluation_results(eval_path: Path) -> Optional[Dict]:
    """Load evaluation results from .npz file."""
    try:
        data = np.load(eval_path, allow_pickle=True)
        return {
            'test_acc': float(data['top1_accuracy']),
            'test_acc_top5': float(data['top5_accuracy']),
            'test_loss': float(data['loss']),
            'per_class_acc': data['per_class_accuracy'],
        }
    except Exception as e:
        print(f"Warning: Could not load {eval_path}: {e}")
        return None


def find_best_from_history(history: Optional[Dict]) -> Tuple[float, float]:
    """Extract best training and validation accuracy from history."""
    if history is None:
        return 0.0, 0.0
    
    train_accs = history.get('train_acc', [])
    val_accs = history.get('val_acc', [])
    
    best_train = max(train_accs) if train_accs else 0.0
    best_val = max(val_accs) if val_accs else 0.0
    
    return best_train, best_val


def print_model_results(model_name: str, checkpoint_info: Optional[Dict], 
                       eval_info: Optional[Dict], show_details: bool = False):
    """Print results for a single model."""
    print(f"\n{'='*70}")
    print(f"  {model_name.upper()} MODEL")
    print(f"{'='*70}")
    
    if checkpoint_info is None and eval_info is None:
        print("  ⚠ No results found")
        return
    
    # Training results
    if checkpoint_info:
        print(f"\n📊 Training Results:")
        
        # Best from history if available
        if checkpoint_info['history']:
            best_train, best_val = find_best_from_history(checkpoint_info['history'])
            print(f"  • Best Training Accuracy:   {best_train*100:6.2f}%")
            print(f"  • Best Validation Accuracy: {best_val*100:6.2f}%")
        else:
            print(f"  • Final Training Accuracy:   {checkpoint_info['train_acc']*100:6.2f}%")
            print(f"  • Final Validation Accuracy: {checkpoint_info['val_acc']*100:6.2f}%")
        
        print(f"  • Epochs Trained: {checkpoint_info['epoch']}")
        
        if show_details:
            print(f"  • Final Training Loss:   {checkpoint_info['train_loss']:.4f}")
            print(f"  • Final Validation Loss: {checkpoint_info['val_loss']:.4f}")
    
    # Test results
    if eval_info:
        print(f"\n🎯 Test Results:")
        print(f"  • Test Accuracy (Top-1): {eval_info['test_acc']*100:6.2f}%")
        print(f"  • Test Accuracy (Top-5): {eval_info['test_acc_top5']*100:6.2f}%")
        
        if show_details:
            print(f"  • Test Loss: {eval_info['test_loss']:.4f}")
            print(f"  • Mean Per-Class Accuracy: {eval_info['per_class_acc'].mean()*100:6.2f}%")


def print_markdown_table(results: Dict[str, Dict]):
    """Print results in markdown table format comparing train vs test."""
    print("\n## Results Summary\n")
    
    # Check if we have any results
    if not results:
        print("No results found.\n")
        return
    
    # Markdown table header
    print("| Model | Best Train Acc | Best Val Acc | Test Acc (Top-1) | Test Acc (Top-5) | Val Loss | Test Loss |")
    print("|-------|----------------|--------------|------------------|------------------|----------|-----------|")
    
    # Sort by test accuracy (if available), otherwise by validation accuracy
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get('test_acc', x[1].get('best_val_acc', 0.0)),
        reverse=True
    )
    
    for model_name, data in sorted_models:
        # Get metrics
        train_acc = data.get('best_train_acc', 0.0) * 100
        val_acc = data.get('best_val_acc', 0.0) * 100
        test_acc = data.get('test_acc', 0.0) * 100
        test_acc_top5 = data.get('test_acc_top5', 0.0) * 100
        
        # Get losses
        checkpoint = data.get('checkpoint', {})
        val_loss = checkpoint.get('val_loss', 0.0) if checkpoint else 0.0
        test_loss = data.get('test_loss', 0.0)
        
        # Format values
        train_str = f"{train_acc:.2f}%" if train_acc > 0 else "N/A"
        val_str = f"{val_acc:.2f}%" if val_acc > 0 else "N/A"
        test_str = f"{test_acc:.2f}%" if test_acc > 0 else "N/A"
        test5_str = f"{test_acc_top5:.2f}%" if test_acc_top5 > 0 else "N/A"
        val_loss_str = f"{val_loss:.4f}" if val_loss > 0 else "N/A"
        test_loss_str = f"{test_loss:.4f}" if test_loss > 0 else "N/A"
        
        # Print row
        print(f"| {model_name:<5} | {train_str:>14} | {val_str:>12} | {test_str:>16} | {test5_str:>16} | {val_loss_str:>8} | {test_loss_str:>9} |")
    
    print()
    
    # Add best model indicator
    if results:
        best_model = max(results.items(), 
                        key=lambda x: x[1].get('test_acc', 0.0))
        if best_model[1].get('test_acc', 0.0) > 0:
            print(f"**Best Model:** `{best_model[0]}` with Test Accuracy: `{best_model[1]['test_acc']*100:.2f}%`\n")


def print_summary_table(results: Dict[str, Dict]):
    """Print a summary table comparing all models."""
    print(f"\n{'='*90}")
    print(f"  SUMMARY - ALL MODELS")
    print(f"{'='*90}\n")
    
    # Header
    print(f"{'Model':<15} {'Best Train':>12} {'Best Val':>12} {'Test (Top-1)':>15} {'Test (Top-5)':>15}")
    print(f"{'-'*90}")
    
    # Sort by test accuracy (if available)
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get('test_acc', 0.0),
        reverse=True
    )
    
    for model_name, data in sorted_models:
        train_acc = data.get('best_train_acc', 0.0) * 100
        val_acc = data.get('best_val_acc', 0.0) * 100
        test_acc = data.get('test_acc', 0.0) * 100
        test_acc_top5 = data.get('test_acc_top5', 0.0) * 100
        
        train_str = f"{train_acc:6.2f}%" if train_acc > 0 else "N/A"
        val_str = f"{val_acc:6.2f}%" if val_acc > 0 else "N/A"
        test_str = f"{test_acc:6.2f}%" if test_acc > 0 else "N/A"
        test5_str = f"{test_acc_top5:6.2f}%" if test_acc_top5 > 0 else "N/A"
        
        print(f"{model_name:<15} {train_str:>12} {val_str:>12} {test_str:>15} {test5_str:>15}")
    
    print(f"{'-'*90}")


def main():
    parser = argparse.ArgumentParser(description="Print best training and test accuracies")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directory containing model checkpoints")
    parser.add_argument("--eval_dir", type=str, default="evaluations",
                       help="Directory containing evaluation results (.npz files)")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Models to check (default: auto-detect from checkpoint_dir)")
    parser.add_argument("--details", action="store_true",
                       help="Show detailed results (losses, per-class accuracy)")
    parser.add_argument("--summary_only", action="store_true",
                       help="Only show summary table")
    parser.add_argument("--markdown", action="store_true",
                       help="Output results as markdown table")
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    eval_dir = Path(args.eval_dir)
    
    # Check if directories exist
    if not checkpoint_dir.exists():
        print(f"⚠ Checkpoint directory not found: {checkpoint_dir}")
        print(f"  Please train models first or specify correct path with --checkpoint_dir")
        return
    
    # Auto-detect models if not specified
    if args.models is None:
        # Find all *_best.pth files
        checkpoint_files = list(checkpoint_dir.glob("*_best.pth"))
        args.models = [f.stem.replace("_best", "") for f in checkpoint_files]
        
        if not args.models:
            print(f"⚠ No checkpoint files found in {checkpoint_dir}")
            print(f"  Looking for files matching pattern: *_best.pth")
            return
        
        print(f"Found {len(args.models)} models: {', '.join(args.models)}\n")
    
    # Collect results
    all_results = {}
    
    for model_name in args.models:
        # Load checkpoint (best model)
        checkpoint_path = checkpoint_dir / f"{model_name}_best.pth"
        checkpoint_info = None
        if checkpoint_path.exists():
            checkpoint_info = load_checkpoint_info(checkpoint_path)
        
        # Load evaluation results
        eval_path = eval_dir / f"results_{model_name}_best.npz"
        eval_info = None
        if eval_path.exists():
            eval_info = load_evaluation_results(eval_path)
        
        # Store combined results
        if checkpoint_info or eval_info:
            all_results[model_name] = {}
            
            if checkpoint_info:
                best_train, best_val = find_best_from_history(checkpoint_info.get('history'))
                all_results[model_name]['best_train_acc'] = best_train
                all_results[model_name]['best_val_acc'] = best_val
                all_results[model_name]['checkpoint'] = checkpoint_info
            
            if eval_info:
                all_results[model_name].update(eval_info)
        
        # Print individual results if not summary only
        if not args.summary_only and not args.markdown:
            print_model_results(model_name, checkpoint_info, eval_info, args.details)
    
    # Print markdown table or summary table
    if all_results:
        if args.markdown:
            print_markdown_table(all_results)
        else:
            print_summary_table(all_results)
    else:
        print("\n⚠ No results found!")
        print(f"  Checkpoint dir: {checkpoint_dir}")
        print(f"  Evaluation dir: {eval_dir}")
        print(f"\nPlease ensure you have:")
        print(f"  1. Trained models (checkpoints/*.pth)")
        print(f"  2. Run evaluation (evaluations/results_*.npz)")
    
    # Print best overall model
    if all_results:
        best_model = max(all_results.items(), 
                        key=lambda x: x[1].get('test_acc', 0.0))
        
        print(f"\n🏆 Best Model: {best_model[0].upper()}")
        print(f"   Test Accuracy: {best_model[1].get('test_acc', 0.0)*100:.2f}%")
    
    print(f"\n{'='*90}\n")


if __name__ == "__main__":
    main()
