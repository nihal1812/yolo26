#!/bin/bash

echo "======================================"
echo " AI SERVICE HEALTH CHECK"
echo "======================================"
echo ""

echo "Date:"
date
echo ""

echo "Hostname:"
hostname
echo ""

echo "Uptime:"
uptime
echo ""

echo "======================================"
echo "1. SERVICE STATUS"
echo "======================================"
systemctl status ai.service --no-pager
echo ""

echo "Is Active:"
systemctl is-active ai.service
echo ""

echo "Is Enabled:"
systemctl is-enabled ai.service
echo ""

echo "Restart Count:"
systemctl show ai.service -p NRestarts
echo ""

echo "Service File:"
systemctl cat ai.service
echo ""

echo "======================================"
echo "2. RECENT SERVICE LOGS"
echo "======================================"
journalctl -u ai.service -n 100 --no-pager
echo ""

echo "======================================"
echo "3. SERVICE ERRORS ONLY"
echo "======================================"
journalctl -u ai.service -p err -b --no-pager
echo ""

echo "======================================"
echo "4. ERROR KEYWORD SEARCH"
echo "======================================"
journalctl -u ai.service -b --no-pager | grep -iE "error|exception|traceback|failed|cuda|camera|memory|killed|segmentation|timeout" || echo "No obvious error keywords found."
echo ""

echo "======================================"
echo "5. SYSTEM ERRORS"
echo "======================================"
journalctl -p err -b --no-pager
echo ""

echo "======================================"
echo "6. KERNEL ERRORS / OOM / CAMERA / CUDA"
echo "======================================"
dmesg -T | grep -iE "error|fail|warn|cuda|usb|camera|nvargus|oom|killed|memory" || echo "No obvious kernel issues found."
echo ""

echo "======================================"
echo "7. MEMORY"
echo "======================================"
free -h
echo ""

echo "======================================"
echo "8. DISK"
echo "======================================"
df -h
echo ""

echo "Journal Disk Usage:"
journalctl --disk-usage
echo ""

echo "======================================"
echo "9. TOP PROCESSES"
echo "======================================"
ps aux --sort=-%mem | head -15
echo ""

echo "======================================"
echo "10. CAMERA DEVICES"
echo "======================================"
ls /dev/video* 2>/dev/null || echo "No /dev/video devices found."
echo ""

if command -v v4l2-ctl >/dev/null 2>&1; then
    v4l2-ctl --list-devices
else
    echo "v4l2-ctl not installed. Install with: sudo apt install v4l-utils"
fi
echo ""

echo "======================================"
echo "11. NVARGUS STATUS"
echo "======================================"
systemctl status nvargus-daemon --no-pager 2>/dev/null || echo "nvargus-daemon not found or not used."
echo ""

echo "======================================"
echo "12. JETSON VERSION"
echo "======================================"
cat /etc/nv_tegra_release 2>/dev/null || echo "Could not read Jetson L4T version."
echo ""

echo "======================================"
echo "13. CUDA CHECK"
echo "======================================"
nvcc --version 2>/dev/null || echo "nvcc not found."
echo ""

echo "======================================"
echo "14. PYTHON PACKAGE CHECKS"
echo "======================================"

python3 - << 'EOF'
packages = ["cv2", "numpy", "torch", "tensorrt"]
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, "__version__", "version unknown")
        print(f"{pkg}: OK - {version}")
    except Exception as e:
        print(f"{pkg}: ERROR - {e}")

try:
    import torch
    print("torch.cuda.is_available():", torch.cuda.is_available())
except Exception as e:
    print("PyTorch CUDA check failed:", e)
EOF

echo ""

echo "======================================"
echo "15. POWER MODE"
echo "======================================"
sudo nvpmodel -q 2>/dev/null || echo "Could not check nvpmodel."
echo ""

echo "======================================"
echo "16. JETSON CLOCKS"
echo "======================================"
sudo jetson_clocks --show 2>/dev/null || echo "Could not check jetson_clocks."
echo ""

echo "======================================"
echo "HEALTH CHECK COMPLETE"
echo "======================================"
