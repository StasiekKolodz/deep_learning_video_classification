#!/usr/bin/env python3
"""
Quick script to compare multiple trained models on the test set.

Usage:
    python compare_models.py --data_root ucf101/ufc10 --checkpoint_dir checkpoints
"""

import argparse
from pathlib import Path
import torch
import subprocess
import sys


def find_checkpoints(checkpoint_dir: Path, pattern: str = "*_best.pth"):
    """Find all checkpoint files matching pattern."""
    checkpoints = list(checkpoint_dir.glob(pattern))
    return sorted(checkpoints)


def extract_mode_from_checkpoint(checkpoint_path: Path) -> str:
    """Extract mode from checkpoint filename."""
    # Assumes format: {mode}_best.pth or {mode}_last.pth
    name = checkpoint_path.stem
    mode = name.split('_')[0]
    return mode


def run_evaluation(checkpoint_path: Path, data_root: str, mode: str = None):
    """Run evaluation script for a single checkpoint."""
    if mode is None:
        mode = extract_mode_from_checkpoint(checkpoint_path)
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {checkpoint_path.name} (mode: {mode})")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable,  # Use same Python interpreter
        "4_1_evaluation.py",
        "--checkpoint", str(checkpoint_path),
        "--data_root", data_root,
        "--mode", mode,
        "--output", f"results_{checkpoint_path.stem}.npz"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {checkpoint_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Compare multiple trained models")
    parser.add_argument("--data_root", required=True, help="Root of dataset")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Directory with checkpoints")
    parser.add_argument("--pattern", default="*_best.pth", help="Checkpoint filename pattern")
    parser.add_argument("--modes", nargs="+", default=None, 
                       help="Specific modes to evaluate (default: all found)")
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory '{checkpoint_dir}' does not exist!")
        return
    
    # Find checkpoints
    checkpoints = find_checkpoints(checkpoint_dir, args.pattern)
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir} matching pattern '{args.pattern}'")
        return
    
    print(f"Found {len(checkpoints)} checkpoint(s):")
    for cp in checkpoints:
        mode = extract_mode_from_checkpoint(cp)
        print(f"  • {cp.name} (mode: {mode})")
    
    # Filter by modes if specified
    if args.modes:
        checkpoints = [cp for cp in checkpoints 
                      if extract_mode_from_checkpoint(cp) in args.modes]
        print(f"\nFiltered to {len(checkpoints)} checkpoint(s) matching modes: {args.modes}")
    
    # Evaluate each checkpoint
    results = {}
    for checkpoint in checkpoints:
        mode = extract_mode_from_checkpoint(checkpoint)
        success = run_evaluation(checkpoint, args.data_root, mode)
        results[checkpoint.name] = success
    
    # Summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    
    for name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {name:30s}: {status}")
    
    print(f"\n✓ All evaluations complete!")
    print(f"  Results saved as: results_*.npz")


if __name__ == "__main__":
    main()
