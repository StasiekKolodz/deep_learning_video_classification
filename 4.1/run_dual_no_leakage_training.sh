#!/bin/bash

################################################################################
# Complete Video Classification Workflow
# 
# This script automates:
#   1. Training all 4 models (perframe, late, early, 3d)
#   2. Evaluating trained models on test set
#   3. Visualizing results (confusion matrices, accuracy plots)
#
# Usage:
#   bash run_complete_workflow.sh
#   bash run_complete_workflow.sh --data_root path/to/data --epochs 10
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default parameters
DATA_ROOT="/dtu/datasets1/02516/ucf101_noleakage"
EPOCHS=10
BATCH_SIZE=64
LR=0.0001
NUM_WORKERS=4
IMG_SIZE=112
NUM_FRAMES=10
NUM_CLASSES=10
CHECKPOINT_DIR="no_leakage/checkpoints"
PLOT_DIR="no_leakage/plots"
RESULTS_DIR="no_leakage/evaluations"

# Models to train
MODELS=("temporal" "spatial")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --models)
            IFS=',' read -ra MODELS <<< "$2"
            shift 2
            ;;
        --skip_training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip_evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        --skip_visualization)
            SKIP_VISUALIZATION=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data_root PATH          Path to dataset (default: ucf101/ufc10)"
            echo "  --epochs N                Number of training epochs (default: 5)"
            echo "  --batch_size N            Batch size (default: 8)"
            echo "  --lr FLOAT                Learning rate (default: 0.001)"
            echo "  --num_workers N           DataLoader workers (default: 4)"
            echo "  --models MODEL1,MODEL2    Comma-separated list of models (default: perframe,late,early,3d)"
            echo "  --skip_training           Skip training phase"
            echo "  --skip_evaluation         Skip evaluation phase"
            echo "  --skip_visualization      Skip visualization phase"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --data_root ucf101/ufc10 --epochs 10"
            echo "  $0 --models perframe,late --skip_training"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     Video Classification - Complete Workflow Script       ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Data Root:       ${GREEN}${DATA_ROOT}${NC}"
echo -e "  Epochs:          ${GREEN}${EPOCHS}${NC}"
echo -e "  Batch Size:      ${GREEN}${BATCH_SIZE}${NC}"
echo -e "  Learning Rate:   ${GREEN}${LR}${NC}"
echo -e "  Models:          ${GREEN}${MODELS[*]}${NC}"
echo -e "  Checkpoint Dir:  ${GREEN}${CHECKPOINT_DIR}${NC}"
echo -e "  Plot Dir:        ${GREEN}${PLOT_DIR}${NC}"
echo -e "  Results Dir:     ${GREEN}${RESULTS_DIR}${NC}"
echo ""

# Create directories
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${PLOT_DIR}"
mkdir -p "${RESULTS_DIR}"

# Check if data directory exists
if [ ! -d "${DATA_ROOT}" ]; then
    echo -e "${RED}Error: Data directory '${DATA_ROOT}' not found!${NC}"
    echo -e "${YELLOW}Please download the dataset or specify correct path with --data_root${NC}"
    exit 1
fi

# Function to print section header
print_section() {
    echo ""
    echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║  $1${NC}"
    echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Function to print step
print_step() {
    echo -e "${CYAN}▶ $1${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Start time
START_TIME=$(date +%s)

################################################################################
# Phase 1: Training
################################################################################


for mode in "${MODELS[@]}"; do
    print_step "Training ${mode} model..."
    
    python 4_2_dual_stream.py \
        --data_root "${DATA_ROOT}" \
        --mode "${mode}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --num_workers "${NUM_WORKERS}" \
        --img_size "${IMG_SIZE}" \
        --num_frames "${NUM_FRAMES}" \
        --num_classes "${NUM_CLASSES}" \
        --save_dir "${CHECKPOINT_DIR}" \
        --plot_dir "${PLOT_DIR}"
    
    if [ $? -eq 0 ]; then
        print_success "Successfully trained ${mode} model"
    else
        print_error "Failed to train ${mode} model"
        exit 1
    fi
    echo ""
done

print_success "Training phase completed!"
else
print_warning "Skipping training phase"
fi

################################################################################
# Phase 2: Evaluation
################################################################################

# if [ "${SKIP_EVALUATION}" != true ]; then
#     print_section "PHASE 2: EVALUATING MODELS"
    
#     RESULT_FILES=()
    
#     for mode in "${MODELS[@]}"; do
#         checkpoint_path="${CHECKPOINT_DIR}/${mode}_best.pth"
        
#         if [ ! -f "${checkpoint_path}" ]; then
#             print_warning "Checkpoint not found: ${checkpoint_path}, skipping evaluation for ${mode}"
#             continue
#         fi
        
#         print_step "Evaluating ${mode} model..."
        
#         result_file="${RESULTS_DIR}/results_${mode}_best.npz"
        
#         python 4_1_evaluation.py \
#             --checkpoint "${checkpoint_path}" \
#             --data_root "${DATA_ROOT}" \
#             --mode "${mode}" \
#             --batch_size "${BATCH_SIZE}" \
#             --num_workers "${NUM_WORKERS}" \
#             --img_size "${IMG_SIZE}" \
#             --num_frames "${NUM_FRAMES}" \
#             --num_classes "${NUM_CLASSES}" \
#             --output "${result_file}" \
#             --show_cm
        
#         if [ $? -eq 0 ]; then
#             print_success "Successfully evaluated ${mode} model"
#             RESULT_FILES+=("${result_file}")
#         else
#             print_error "Failed to evaluate ${mode} model"
#             exit 1
#         fi
#         echo ""
#     done
    
#     print_success "Evaluation phase completed!"
# else
#     print_warning "Skipping evaluation phase"
#     # Find existing result files for visualization
#     RESULT_FILES=()
#     for mode in "${MODELS[@]}"; do
#         result_file="${RESULTS_DIR}/results_${mode}_best.npz"
#         if [ -f "${result_file}" ]; then
#             RESULT_FILES+=("${result_file}")
#         fi
#     done
# fi

# ################################################################################
# # Phase 3: Visualization
# ################################################################################

# if [ "${SKIP_VISUALIZATION}" != true ]; then
#     print_section "PHASE 3: VISUALIZING RESULTS"
    
#     if [ ${#RESULT_FILES[@]} -eq 0 ]; then
#         print_warning "No result files found for visualization"
#     else
#         print_step "Generating visualizations for ${#RESULT_FILES[@]} models..."
        
#         python visualize_results.py \
#             "${RESULT_FILES[@]}" \
#             --save \
#             --output_dir "${PLOT_DIR}"
        
#         if [ $? -eq 0 ]; then
#             print_success "Successfully generated visualizations"
#         else
#             print_error "Failed to generate visualizations"
#             exit 1
#         fi
#     fi
    
#     print_success "Visualization phase completed!"
# else
#     print_warning "Skipping visualization phase"
# fi

# ################################################################################
# # Summary
# ################################################################################

# END_TIME=$(date +%s)
# ELAPSED_TIME=$((END_TIME - START_TIME))
# HOURS=$((ELAPSED_TIME / 3600))
# MINUTES=$(((ELAPSED_TIME % 3600) / 60))
# SECONDS=$((ELAPSED_TIME % 60))

# print_section "WORKFLOW COMPLETED"

# echo -e "${GREEN}All tasks completed successfully!${NC}"
# echo ""
# echo -e "${BLUE}Time elapsed: ${HOURS}h ${MINUTES}m ${SECONDS}s${NC}"
# echo ""
# echo -e "${BLUE}Generated files:${NC}"
# echo -e "  Checkpoints:  ${GREEN}${CHECKPOINT_DIR}/${NC}"
# for mode in "${MODELS[@]}"; do
#     if [ -f "${CHECKPOINT_DIR}/${mode}_best.pth" ]; then
#         echo -e "    • ${mode}_best.pth"
#         echo -e "    • ${mode}_last.pth"
#     fi
# done
# echo ""
# echo -e "  Plots:        ${GREEN}${PLOT_DIR}/${NC}"
# if [ -d "${PLOT_DIR}" ]; then
#     for mode in "${MODELS[@]}"; do
#         if [ -f "${PLOT_DIR}/${mode}_training_history.png" ]; then
#             echo -e "    • ${mode}_training_history.png"
#         fi
#     done
#     if [ -f "${PLOT_DIR}/model_comparison.png" ]; then
#         echo -e "    • model_comparison.png"
#         echo -e "    • per_class_comparison.png"
#     fi
# fi
# echo ""
# echo -e "  Results:      ${GREEN}${RESULTS_DIR}/${NC}"
# if [ -d "${RESULTS_DIR}" ]; then
#     for mode in "${MODELS[@]}"; do
#         if [ -f "${RESULTS_DIR}/results_${mode}_best.npz" ]; then
#             echo -e "    • results_${mode}_best.npz"
#         fi
#     done
# fi
# echo ""

# # Print performance summary if results exist
# if [ ${#RESULT_FILES[@]} -gt 0 ]; then
#     echo -e "${BLUE}Model Performance Summary:${NC}"
#     echo ""
    
#     for mode in "${MODELS[@]}"; do
#         result_file="${RESULTS_DIR}/results_${mode}_best.npz"
#         if [ -f "${result_file}" ]; then
#             # Extract accuracy using Python
#             accuracy=$(python -c "import numpy as np; d=np.load('${result_file}'); print(f\"{d['top1_accuracy']*100:.2f}\")")
#             echo -e "  ${mode}:$(printf '%12s' ' ')${GREEN}${accuracy}%${NC} (Top-1 Accuracy)"
#         fi
#     done
#     echo ""
# fi

# echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
# echo -e "${CYAN}║               Workflow Execution Complete! 🎉              ║${NC}"
# echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
