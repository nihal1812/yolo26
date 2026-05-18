#!/bin/bash

set -u

SERVICE_NAME="${SERVICE_NAME:-ai.service}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

section() {
    echo ""
    echo "======================================"
    echo "$1"
    echo "======================================"
}

safe_run() {
    "$@" 2>/dev/null || true
}

section "AI SERVICE HEALTH CHECK"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Project: ${PROJECT_DIR}"
echo "Uptime: $(uptime)"

section "1. SERVICE STATUS"
safe_run systemctl status "${SERVICE_NAME}" --no-pager
echo ""
echo "Is Active:"
safe_run systemctl is-active "${SERVICE_NAME}"
echo ""
echo "Is Enabled:"
safe_run systemctl is-enabled "${SERVICE_NAME}"
echo ""
echo "Restart Count:"
safe_run systemctl show "${SERVICE_NAME}" -p NRestarts
echo ""
echo "Selected Unit Properties:"
safe_run systemctl show "${SERVICE_NAME}" \
    -p WorkingDirectory \
    -p ExecStart \
    -p Restart \
    -p KillMode \
    -p TimeoutStopUSec \
    -p User \
    -p Group \
    -p NRestarts \
    -p ActiveState \
    -p SubState

section "2. CHILD PROCESS LIST"
ps -eo pid,ppid,pgid,sid,stat,%cpu,%mem,cmd --sort=ppid | awk '
    /src\/main.py|perception.py|reid_pipeline.py|brain.py|ui_pipeline.py|s3_feedback_poller.py|trainer_node.py|model_node.py|policy_node.py|python/ {
        print
    }
' || true

section "3. RECENT SERVICE LOGS"
safe_run journalctl -u "${SERVICE_NAME}" -n 150 --no-pager

section "4. RECENT ERRORS AND RESTART SIGNALS"
safe_run journalctl -u "${SERVICE_NAME}" -b --no-pager | grep -iE "error|exception|traceback|failed|cuda|camera|memory|killed|segmentation|timeout|stale|restart|shutdown|oom" || echo "No obvious service error keywords found."

section "5. PER-CAMERA RECENT ACTIVITY"
for cam in cam1 cam2 cam3 cam4 cam5 cam6; do
    echo "--- ${cam} ---"
    safe_run journalctl -u "${SERVICE_NAME}" -b --no-pager | grep -E "\\[perception\\]\\[${cam}\\]|\\[rtsp_stream\\].*${cam}|\\[bbox_overlay\\].*${cam}|\\[reid_node\\].*${cam}" | tail -20 || echo "No recent ${cam} activity found."
done

section "6. DISK USAGE"
df -h
echo ""
echo "Runtime directories:"
for path in \
    "${PROJECT_DIR}/logs" \
    "${PROJECT_DIR}/clips_cache" \
    "${PROJECT_DIR}/clips_cache/alert_overlay" \
    "${PROJECT_DIR}/models" \
    "${PROJECT_DIR}/src/models" \
    "/tmp/zono_clips" \
    "/home/yahboom/zono/yolo26/logs"; do
    if [[ -e "${path}" ]]; then
        du -sh "${path}" 2>/dev/null || true
    else
        echo "missing: ${path}"
    fi
done
echo ""
echo "Journal Disk Usage:"
safe_run journalctl --disk-usage

section "7. MEMORY AND TOP PROCESSES"
free -h
echo ""
ps aux --sort=-%mem | head -20

section "8. REDIS"
if command -v redis-cli >/dev/null 2>&1; then
    redis-cli ping 2>/dev/null || echo "Redis ping failed."
    redis-cli info server clients memory stats 2>/dev/null | grep -E "redis_version|connected_clients|used_memory_human|total_commands_processed|rejected_connections" || true
else
    echo "redis-cli not installed."
fi

section "9. GPU / CUDA / JETSON"
cat /etc/nv_tegra_release 2>/dev/null || echo "Could not read Jetson L4T version."
echo ""
nvcc --version 2>/dev/null || echo "nvcc not found."
echo ""
if command -v tegrastats >/dev/null 2>&1; then
    timeout 3 tegrastats 2>/dev/null || true
else
    echo "tegrastats not found."
fi
echo ""
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi 2>/dev/null || true
else
    echo "nvidia-smi not available. This can be normal on Jetson."
fi

section "10. PYTHON PACKAGE AND CUDA CHECKS"
python3 - << 'EOF'
packages = ["cv2", "numpy", "torch", "tensorrt", "zmq", "redis"]
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
    if torch.cuda.is_available():
        print("torch.cuda.device_count():", torch.cuda.device_count())
        print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
        print("torch.cuda.memory_allocated_mb:", torch.cuda.memory_allocated(0) / (1024 * 1024))
        print("torch.cuda.memory_reserved_mb:", torch.cuda.memory_reserved(0) / (1024 * 1024))
except Exception as e:
    print("PyTorch CUDA check failed:", e)
EOF

section "11. CAMERA DEVICES / NVARGUS"
ls /dev/video* 2>/dev/null || echo "No /dev/video devices found."
echo ""
if command -v v4l2-ctl >/dev/null 2>&1; then
    v4l2-ctl --list-devices 2>/dev/null || true
else
    echo "v4l2-ctl not installed. Install with: sudo apt install v4l-utils"
fi
echo ""
safe_run systemctl status nvargus-daemon --no-pager || echo "nvargus-daemon not found or not used."

section "12. KERNEL ERRORS / OOM / CAMERA / CUDA"
safe_run dmesg -T | grep -iE "error|fail|warn|cuda|usb|camera|nvargus|oom|killed|memory" || echo "No obvious kernel issues found or dmesg unavailable."

section "HEALTH CHECK COMPLETE"
