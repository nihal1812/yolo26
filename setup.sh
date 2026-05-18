#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "========================================"
echo " Jetson YOLO26 environment setup"
echo " Project: ${PROJECT_DIR}"
echo "========================================"

cd "${PROJECT_DIR}"

echo ""
echo "==> System info"

ARCH="$(uname -m)"
echo "Architecture: ${ARCH}"

if [[ "${ARCH}" != "aarch64" ]]; then
    echo "WARNING: This script is intended for Jetson/aarch64."
fi

if [[ -f /etc/nv_tegra_release ]]; then
    cat /etc/nv_tegra_release
else
    echo "WARNING: /etc/nv_tegra_release not found."
fi

lsb_release -a || true

echo ""
echo "==> Checking CUDA"

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo "WARNING: nvcc not found in PATH."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
else
    echo "nvidia-smi not available. This can be normal on some Jetson setups."
fi

echo ""
echo "==> Updating apt package lists"
sudo apt update

echo ""
echo "==> Installing system packages"
sudo apt install -y \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    pkg-config \
    python3-dev \
    python3-venv \
    python3-pip \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    libgl1 \
    libglib2.0-0 \
    python3-gi \
    python3-gi-cairo \
    python3-gst-1.0 \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    redis-server

echo ""
echo "==> Enabling Redis service"
if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl enable redis-server || true
    sudo systemctl start redis-server || true
fi

echo ""
echo "==> Installing cuDSS if missing"

if dpkg -l | grep -q "^ii  cudss "; then
    echo "cudss already installed."
else
    CUDA_KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
    CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/${CUDA_KEYRING_DEB}"

    if [[ ! -f "${PROJECT_DIR}/${CUDA_KEYRING_DEB}" ]]; then
        echo "Downloading CUDA keyring..."
        wget -O "${PROJECT_DIR}/${CUDA_KEYRING_DEB}" "${CUDA_KEYRING_URL}"
    else
        echo "CUDA keyring already exists."
    fi

    sudo dpkg -i "${PROJECT_DIR}/${CUDA_KEYRING_DEB}" || true
    sudo apt update
    sudo apt install -y cudss
fi

echo ""
echo "==> Creating Python virtual environment"

if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}" --system-site-packages
else
    echo "Virtual environment already exists: ${VENV_DIR}"
fi

echo ""
echo "==> Activating virtual environment"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo ""
echo "==> Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo ""
echo "==> Checking existing Torch/CUDA stack before installing requirements"

python - <<'PY'
import sys

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: torch is installed but CUDA is not available.")
except Exception as exc:
    print("WARNING: torch check failed:", exc)
    print("On Jetson, install the NVIDIA-compatible PyTorch stack before running GPU inference.")
PY

echo ""
echo "==> Installing Python requirements"

if [[ -f "${PROJECT_DIR}/requirements.txt" ]]; then
    python -m pip install --no-cache-dir -r "${PROJECT_DIR}/requirements.txt"
else
    echo "WARNING: requirements.txt not found. Skipping Python package installation."
fi

echo ""
echo "==> Verifying project dependencies"

python - <<'PY'
import importlib.util

packages = [
    "numpy",
    "yaml",
    "redis",
    "msgpack",
    "ultralytics",
    "cv2",
    "PIL",
    "tqdm",
    "matplotlib",
    "psutil",
    "requests",
    "zmq",
    "boto3",
    "botocore",
    "openai",
    "torch",
    "torchvision",
]

missing = []

for pkg in packages:
    found = importlib.util.find_spec(pkg)
    status = "OK" if found else "MISSING"
    print(f"{pkg:15s}: {status}")
    if not found:
        missing.append(pkg)

print("")
if missing:
    print("Missing packages:", ", ".join(missing))
    raise SystemExit(1)
else:
    print("All checked packages are available.")
PY

echo ""
echo "==> Verifying Torch + CUDA"

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("ERROR: CUDA is not available in torch.")
PY

echo ""
echo "==> Verifying OpenCV"

python - <<'PY'
import cv2
print("opencv:", cv2.__version__)
PY

echo ""
echo "========================================"
echo " Setup complete"
echo " Activate with:"
echo " source venv/bin/activate"
echo "========================================"
