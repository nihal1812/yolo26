#!/usr/bin/env python3
import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path

from config_utils import load_cfg, get_active_cams, get_brain_args, format_cam_dict

HERE = Path(__file__).resolve().parent


def popen_node(py_exe, script_path: Path, cfg_path: str, cam_id: str, extra_args=None):
    cmd = [py_exe, str(script_path), "--config", cfg_path, "--cam_id", cam_id]
    if extra_args:
        cmd.extend(extra_args)
    print(f"[perception] exec: {' '.join(map(str, cmd))}")
    return subprocess.Popen(cmd, cwd=str(HERE))


def terminate_proc(p, grace_s=3.0):
    if p is None:
        return
    try:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
    except Exception:
        return

    t0 = time.time()
    while time.time() - t0 < grace_s:
        if p.poll() is not None:
            return
        time.sleep(0.1)

    try:
        if p.poll() is None:
            p.kill()
    except Exception:
        pass


def terminate_all(procs, grace_s=3.0):
    for p in procs:
        terminate_proc(p, grace_s=grace_s)


def dict_to_cli_args(d: dict):
    args = []
    for k, v in (d or {}).items():
        key = f"--{k}"
        if isinstance(v, bool):
            if v:
                args.append(key)
        elif v is not None:
            args.extend([key, str(v)])
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--restart_delay_s", type=float, default=2.0)
    args = ap.parse_args()

    cfg_path = args.config
    cfg = load_cfg(cfg_path)
    active_cams = get_active_cams(cfg)

    py = sys.executable
    fb_args_cfg = get_brain_args(cfg, "feature_builder_args")

    try:
        print(f"[perception] config={cfg_path}")
        print(f"[perception] active_cams={active_cams}")

        proc_specs = {}
        procs = {}

        for cam_id in active_cams:
            feat_cmd_extra = dict_to_cli_args(format_cam_dict(fb_args_cfg, cam_id))

            proc_specs[(cam_id, "rtsp_stream")] = {
                "script": HERE / "rtsp_stream.py",
                "extra_args": [],
            }
            proc_specs[(cam_id, "pose_track")] = {
                "script": HERE / "pose_track.py",
                "extra_args": [],
            }
            proc_specs[(cam_id, "seg_track")] = {
                "script": HERE / "seg_track.py",
                "extra_args": [],
            }
            proc_specs[(cam_id, "feature_builder")] = {
                "script": HERE / "feature_builder.py",
                "extra_args": feat_cmd_extra,
            }

        def start_one(cam_id, role):
            spec = proc_specs[(cam_id, role)]
            print(f"[perception] starting {role} for {cam_id} ...")
            p = popen_node(py, spec["script"], cfg_path, cam_id, spec["extra_args"])
            procs[(cam_id, role)] = p
            return p

        # initial boot
        for cam_id in active_cams:
            start_one(cam_id, "rtsp_stream")
            time.sleep(0.4)

            start_one(cam_id, "pose_track")
            start_one(cam_id, "seg_track")
            time.sleep(0.4)

            start_one(cam_id, "feature_builder")
            time.sleep(0.4)

        print("[perception] all camera pipelines started. Ctrl+C to stop.")

        while True:
            for key, p in list(procs.items()):
                rc = p.poll()
                if rc is None:
                    continue

                cam_id, role = key
                print(f"[perception] child exited: cam={cam_id} role={role} code={rc}")

                terminate_proc(p)
                time.sleep(args.restart_delay_s)

                # If RTSP restarts, dependent nodes for that camera should also restart
                if role == "rtsp_stream":
                    for dep_role in ["pose_track", "seg_track", "feature_builder"]:
                        dep = procs.get((cam_id, dep_role))
                        if dep is not None:
                            print(f"[perception] stopping dependent node cam={cam_id} role={dep_role}")
                            terminate_proc(dep)
                            procs.pop((cam_id, dep_role), None)

                    start_one(cam_id, "rtsp_stream")
                    time.sleep(0.5)
                    start_one(cam_id, "pose_track")
                    start_one(cam_id, "seg_track")
                    time.sleep(0.5)
                    start_one(cam_id, "feature_builder")
                    time.sleep(0.3)

                elif role in ("pose_track", "seg_track"):
                    # feature_builder depends on both streams; restart it too
                    dep = procs.get((cam_id, "feature_builder"))
                    if dep is not None:
                        print(f"[perception] stopping dependent node cam={cam_id} role=feature_builder")
                        terminate_proc(dep)
                        procs.pop((cam_id, "feature_builder"), None)

                    start_one(cam_id, role)
                    time.sleep(0.5)
                    start_one(cam_id, "feature_builder")
                    time.sleep(0.3)

                elif role == "feature_builder":
                    start_one(cam_id, "feature_builder")
                    time.sleep(0.3)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[perception] stopping...")
    except Exception as e:
        print(f"[perception] error: {e}")
    finally:
        terminate_all(list(procs.values()))
        print("[perception] done.")


if __name__ == "__main__":
    main()