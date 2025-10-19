#!/bin/bash
# Advanced setup script with CUDA detection and PyTorch installation options

set -e

echo "========================================="
echo "Video Classification - Advanced Setup"
echo "========================================="
echo ""

VENV_NAME="venv"
PYTHON_VERSION="python3"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check Python
if ! command -v $PYTHON_VERSION &> /dev/null; then
    print_error "Python 3 not found!"
    exit 1
fi

PYTHON_VER=$($PYTHON_VERSION --version 2>&1 | awk '{print $2}')
print_info "Found Python version: $PYTHON_VER"

# Check CUDA availability
print_info "Checking for CUDA..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_AVAILABLE=true
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    print_info "CUDA detected: Version $CUDA_VERSION"
    nvidia-smi --query-gpu=name --format=csv,noheader | while read gpu; do
        print_info "GPU: $gpu"
    done
else
    CUDA_AVAILABLE=false
    print_warning "CUDA not detected. Will install CPU-only version."
fi

# Create virtual environment
if [ -d "$VENV_NAME" ]; then
    print_warning "Virtual environment exists."
    read -p "Remove and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_NAME"
    else
        print_info "Using existing environment."
        source "$VENV_NAME/bin/activate"
        pip install --upgrade pip
        
        # Detect CUDA and install PyTorch accordingly
        if [ "$CUDA_AVAILABLE" = true ]; then
            print_info "Installing PyTorch with CUDA support..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        else
            print_info "Installing PyTorch (CPU-only)..."
            pip install torch torchvision
        fi
        
        print_info "Installing other dependencies..."
        pip install pandas pillow numpy matplotlib seaborn tqdm
        
        echo ""
        print_info "Setup complete!"
        exit 0
    fi
fi

print_info "Creating virtual environment: $VENV_NAME"
$PYTHON_VERSION -m venv "$VENV_NAME"

print_info "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

print_info "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch based on CUDA availability
if [ "$CUDA_AVAILABLE" = true ]; then
    echo ""
    print_info "Installing PyTorch with CUDA support..."
    print_info "This may take a few minutes..."
    
    # Try to detect CUDA version and install appropriate PyTorch
    if [[ "$CUDA_VERSION" == "12."* ]]; then
        print_info "Installing for CUDA 12.x..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == "11."* ]]; then
        print_info "Installing for CUDA 11.x..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        print_warning "Unknown CUDA version. Installing default PyTorch..."
        pip install torch torchvision
    fi
else
    print_info "Installing PyTorch (CPU-only)..."
    pip install torch torchvision
fi

# Install other dependencies
print_info "Installing other dependencies..."
pip install pandas pillow numpy matplotlib seaborn tqdm

# Verify installation
echo ""
print_info "Verifying installation..."
python << EOF
import torch
import torchvision
import pandas
import numpy
import matplotlib
import seaborn

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ torchvision: {torchvision.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"✓ pandas: {pandas.__version__}")
print(f"✓ numpy: {numpy.__version__}")
print(f"✓ matplotlib: {matplotlib.__version__}")
print(f"✓ seaborn: {seaborn.__version__}")
EOF

# Create directories
print_info "Creating project directories..."
mkdir -p checkpoints plots

echo ""
echo "========================================="
print_info "Setup Complete!"
echo "========================================="
echo ""
echo "To activate: source $VENV_NAME/bin/activate"
echo "To deactivate: deactivate"
echo ""
echo "Quick start:"
echo "  1. source $VENV_NAME/bin/activate"
echo "  2. python 4_1_training.py --data_root ucf101/ufc10 --mode late"
echo ""
print_info "Ready to train! 🚀"
