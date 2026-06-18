#!/bin/bash
# =========================================================================
# Portable Startup Script for SSN_Tools.py (Linux/macOS)
# =========================================================================

# Move to the directory containing this script
cd "$(dirname "$0")"

# 1. Locate uv executable
if command -v uv &> /dev/null; then
    UV_EXE="uv"
elif [ -f "$HOME/.local/bin/uv" ]; then
    UV_EXE="$HOME/.local/bin/uv"
else
    echo "uv package manager not found. Installing it automatically..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV_EXE="$HOME/.local/bin/uv"
fi

# 2. Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating isolated local virtual environment (.venv)..."
    "$UV_EXE" venv --python 3.11
fi

# 3. Detect GPU type using python script
echo "Detecting hardware configuration..."
GPU_TYPE=$("$UV_EXE" run --quiet python Detect_GPU.py)

# 4. Resolve dependencies based on GPU Type
echo "Detected platform/GPU type: $GPU_TYPE"
echo ""

if [ "$GPU_TYPE" = "NVIDIA" ]; then
    echo "NVIDIA GPU detected. Syncing with PyTorch CUDA 13.0 support..."
    "$UV_EXE" pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
elif [ "$GPU_TYPE" = "INTEL" ]; then
    echo "Intel Arc/GPU detected. Syncing with PyTorch XPU (oneAPI/SYCL) support..."
    "$UV_EXE" pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/xpu --index-strategy unsafe-best-match
elif [ "$GPU_TYPE" = "AMD" ]; then
    echo "AMD GPU detected on Linux. Syncing with PyTorch ROCm 6.1 support..."
    "$UV_EXE" pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm6.1 --index-strategy unsafe-best-match
elif [ "$GPU_TYPE" = "MPS" ]; then
    echo "Apple Silicon detected. Syncing with macOS MPS support..."
    "$UV_EXE" pip install -r requirements.txt
else
    echo "No dedicated GPU detected. Syncing with CPU-only PyTorch..."
    "$UV_EXE" pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
fi
echo ""

# 5. Run the tools
echo "Starting SSN_Tools..."
"$UV_EXE" run SSN_Tools.py
