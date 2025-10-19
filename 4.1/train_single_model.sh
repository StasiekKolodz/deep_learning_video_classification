#!/bin/bash

################################################################################
# Quick Training Script for Single Model
#
# Usage:
#   bash train_single_model.sh perframe
#   bash train_single_model.sh late --epochs 10 --batch_size 16
################################################################################

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Check if model mode is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: Model mode not specified${NC}"
    echo "Usage: $0 <mode> [options]"
    echo "  mode: perframe, late, early, or 3d"
    echo ""
    echo "Example: $0 late --epochs 10"
    exit 1
fi

MODE=$1
shift  # Remove first argument (mode)

# Validate mode
if [[ ! "$MODE" =~ ^(perframe|late|early|3d)$ ]]; then
    echo -e "${RED}Error: Invalid mode '$MODE'${NC}"
    echo "Valid modes: perframe, late, early, 3d"
    exit 1
fi

# Default settings
DATA_ROOT="/dtu/datasets1/02516/ufc10"
EPOCHS=5
BATCH_SIZE=8

echo -e "${BLUE}Training ${MODE} model...${NC}"
echo ""

# Run training
python 4_1_training.py \
    --data_root "${DATA_ROOT}" \
    --mode "${MODE}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    "$@"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate: python 4_1_evaluation.py --checkpoint checkpoints/${MODE}_best.pth --mode ${MODE} --data_root ${DATA_ROOT}"
    echo "  2. Visualize: python visualize_results.py results_${MODE}_best.npz --save"
else
    echo -e "${RED}✗ Training failed${NC}"
    exit 1
fi
