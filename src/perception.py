#!/usr/bin/env python3
import sys
import time
import yaml
import signal
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run():
    # NOTE: Your original line is a bit unusual: HERE / "C:\\..."
    # Keeping your intent: use an absolute Windows path directly.
    cfg_path = r"C:\Users\nihal\yolo26\config\config.yaml"
    cfg = load_cfg(cfg_path)

    py = sys.executable

    rtsp_cmd = [py, str(HERE / "rtsp_stream.py"), "--config", cfg_path]
    pose_cmd = [py, str(HERE / "pose_track.py"), "--config", cfg_path]
    seg_cmd  = [py, str(HERE / "seg_track.py"),  "--config", cfg_path]
    feat_cmd = [py, str(HERE / "feature_builder.py"), "--config", cfg_path]

    procs = []
    try:
        print("[perception] starting rtsp_stream...")
        procs.append(subprocess.Popen(rtsp_cmd, cwd=str(HERE)))

        time.sleep(0.8)  # ensure PUB is up before SUBs

        print("[perception] starting pose_track...")
        procs.append(subprocess.Popen(pose_cmd, cwd=str(HERE)))

        print("[perception] starting seg_track...")
        procs.append(subprocess.Popen(seg_cmd, cwd=str(HERE)))

        time.sleep(0.6)  # ensure pose/seg PUB sockets are up

        print("[perception] starting feature_builder...")
        procs.append(subprocess.Popen(feat_cmd, cwd=str(HERE)))

        print("[perception] all processes started. Ctrl+C to stop.")
        while True:
            for p in procs:
                if p.poll() is not None:
                    raise RuntimeError(f"Process exited unexpectedly: pid={p.pid} code={p.returncode}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[perception] stopping...")
    except Exception as e:
        print(f"[perception] error: {e}")
    finally:
        for p in procs:
            try:
                if p.poll() is None:
                    p.send_signal(signal.SIGINT)
            except Exception:
                pass

        t0 = time.time()
        while time.time() - t0 < 3.0:
            if all(p.poll() is not None for p in procs):
                break
            time.sleep(0.1)

        for p in procs:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass

        print("[perception] done.")

if __name__ == "__main__":
    run()
