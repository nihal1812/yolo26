#!/usr/bin/env python3
"""
main.py (top-level orchestrator)

Starts pipelines in dependency order.

Recommended order:
  1) perception.py
  2) reid_pipeline.py
  3) brain.py
  4) ui_pipeline.py
"""

import os
import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path

from config_utils import load_cfg

HERE = Path(__file__).resolve().parent

DEFAULT_PIPELINE_ORDER = ["perception", "reid_pipeline", "brain", "ui_pipeline"]
DOWNSTREAM_OF_PERCEPTION = ["reid_pipeline", "brain", "ui_pipeline"]


def popen_node(py_exe, script_path: Path, cfg_path: str, extra_args=None):
    cmd = [py_exe, str(script_path), "--config", cfg_path]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[main] exec: {' '.join(map(str, cmd))}")

    # start_new_session=True creates a separate process group for each pipeline.
    # This lets main.py stop the whole pipeline tree cleanly, including
    # multiprocessing children and worker subprocesses.
    return subprocess.Popen(
        cmd,
        cwd=str(HERE),
        start_new_session=True,
    )


def terminate_proc(p, grace_s=3.0):
    if p is None:
        return

    try:
        if p.poll() is not None:
            return
    except Exception:
        return

    # First try graceful shutdown of the whole process group.
    try:
        pgid = os.getpgid(p.pid)
        print(f"[main] terminating process group pgid={pgid} pid={p.pid}")
        os.killpg(pgid, signal.SIGTERM)
    except Exception as e:
        print(f"[main] process-group SIGTERM failed for pid={getattr(p, 'pid', None)}: {e}")
        try:
            p.send_signal(signal.SIGTERM)
        except Exception:
            return

    t0 = time.time()
    while time.time() - t0 < grace_s:
        try:
            if p.poll() is not None:
                return
        except Exception:
            return
        time.sleep(0.1)

    # Force kill the whole process group if graceful shutdown did not finish.
    try:
        pgid = os.getpgid(p.pid)
        print(f"[main] killing process group pgid={pgid} pid={p.pid}")
        os.killpg(pgid, signal.SIGKILL)
    except Exception as e:
        print(f"[main] process-group SIGKILL failed for pid={getattr(p, 'pid', None)}: {e}")
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass


def terminate_named_proc(procs, name, grace_s=3.0):
    p = procs.get(name)
    if p is None:
        return
    terminate_proc(p, grace_s=grace_s)
    procs.pop(name, None)


def terminate_all(procs, grace_s=3.0):
    for name in list(procs.keys())[::-1]:
        terminate_named_proc(procs, name, grace_s=grace_s)


def clean_startup_order(raw_order, enabled_names):
    enabled = set(enabled_names)

    order = []
    for name in raw_order:
        if name in enabled and name not in order:
            order.append(name)

    for name in DEFAULT_PIPELINE_ORDER:
        if name in enabled and name not in order:
            order.append(name)

    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    ap.add_argument("--no-perception", action="store_true")
    ap.add_argument("--no-brain", action="store_true")
    ap.add_argument("--no-reid-pipeline", action="store_true")
    ap.add_argument("--no-ui", action="store_true")

    ap.add_argument("--no-alert-sender", action="store_true")
    ap.add_argument("--no-clip-writer", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--incident-leader-cam", default=None)

    ap.add_argument("--reid-device", default=None)

    ap.add_argument("--restart_delay_s", type=float, default=2.0)
    ap.add_argument("--max_restarts_per_pipeline", type=int, default=20)
    ap.add_argument("--restart_backoff_step_s", type=float, default=1.0)
    ap.add_argument("--restart_backoff_cap_s", type=float, default=20.0)

    ap.add_argument("--perception_warmup_s", type=float, default=8.0)
    ap.add_argument("--between_pipeline_start_s", type=float, default=1.5)
    ap.add_argument("--downstream_restart_delay_s", type=float, default=4.0)

    args = ap.parse_args()

    cfg = load_cfg(args.config) or {}
    main_cfg = cfg.get("main", {})

    args.perception_warmup_s = float(main_cfg.get("perception_warmup_s", args.perception_warmup_s))
    args.between_pipeline_start_s = float(main_cfg.get("between_pipeline_start_s", args.between_pipeline_start_s))
    args.downstream_restart_delay_s = float(main_cfg.get("downstream_restart_delay_s", args.downstream_restart_delay_s))

    args.restart_delay_s = float(main_cfg.get("restart_delay_s", args.restart_delay_s))
    args.max_restarts_per_pipeline = int(main_cfg.get("max_restarts_per_pipeline", args.max_restarts_per_pipeline))
    args.restart_backoff_step_s = float(main_cfg.get("restart_backoff_step_s", args.restart_backoff_step_s))
    args.restart_backoff_cap_s = float(main_cfg.get("restart_backoff_cap_s", args.restart_backoff_cap_s))

    raw_startup_order = main_cfg.get("startup_order", DEFAULT_PIPELINE_ORDER)
    if not isinstance(raw_startup_order, list):
        raw_startup_order = DEFAULT_PIPELINE_ORDER

    py = sys.executable

    scripts = {
        "perception": HERE / "perception.py",
        "reid_pipeline": HERE / "reid_pipeline.py",
        "brain": HERE / "brain.py",
        "ui_pipeline": HERE / "ui_pipeline.py",
    }

    proc_specs = {}
    procs = {}
    restart_counts = {}
    shutting_down = False

    def request_shutdown(signum=None, _frame=None):
        nonlocal shutting_down
        if not shutting_down:
            sig_name = signal.Signals(signum).name if signum is not None else "manual"
            print(f"[main] shutdown requested ({sig_name})")
        shutting_down = True

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    def register_proc(name: str, script_path: Path, extra_args=None):
        proc_specs[name] = {
            "script": script_path,
            "extra_args": list(extra_args) if extra_args else [],
        }
        restart_counts[name] = 0

    if not args.no_perception:
        register_proc("perception", scripts["perception"], [])

    if not args.no_reid_pipeline:
        reid_extra = []
        if args.reid_device:
            reid_extra.extend(["--device", str(args.reid_device)])
        register_proc("reid_pipeline", scripts["reid_pipeline"], reid_extra)

    if not args.no_brain:
        register_proc("brain", scripts["brain"], [])

    if not args.no_ui:
        ui_extra = []
        if args.no_alert_sender:
            ui_extra.append("--no-alert-sender")
        if args.no_clip_writer:
            ui_extra.append("--no-clip-writer")
        if args.no_overlay:
            ui_extra.append("--no-overlay")
        if args.no_feedback:
            ui_extra.append("--no-feedback")
        if args.incident_leader_cam:
            ui_extra.extend(["--incident-leader-cam", str(args.incident_leader_cam)])
        register_proc("ui_pipeline", scripts["ui_pipeline"], ui_extra)

    if not proc_specs:
        raise RuntimeError("[main] Nothing to start; all pipelines disabled.")

    startup_order = clean_startup_order(raw_startup_order, proc_specs.keys())

    def start_one(name: str):
        spec = proc_specs[name]
        print(f"[main] starting {name} ...")
        p = popen_node(py, spec["script"], args.config, extra_args=spec["extra_args"])
        procs[name] = p
        return p

    def restart_delay_for(name: str):
        count = int(restart_counts.get(name, 0))
        return min(
            args.restart_delay_s + max(0, count - 1) * args.restart_backoff_step_s,
            args.restart_backoff_cap_s,
        )

    def wait_with_interrupt_check(seconds: float):
        t0 = time.time()
        while not shutting_down and time.time() - t0 < seconds:
            time.sleep(0.1)

    def start_initial_pipelines():
        print("[main] starting pipelines in dependency order...")

        for name in startup_order:
            if shutting_down:
                break
            start_one(name)

            if name == "perception":
                print(f"[main] waiting {args.perception_warmup_s:.1f}s for perception warmup...")
                wait_with_interrupt_check(args.perception_warmup_s)
            else:
                wait_with_interrupt_check(args.between_pipeline_start_s)

        print("[main] all pipelines started. Ctrl+C to stop.")

    def restart_perception_and_downstream():
        print("[main] perception is upstream; stopping downstream pipelines...")

        for name in DOWNSTREAM_OF_PERCEPTION:
            if name in procs:
                print(f"[main] stopping downstream {name} ...")
                terminate_named_proc(procs, name)

        delay = restart_delay_for("perception")
        print(f"[main] restarting perception in {delay:.1f}s ...")
        wait_with_interrupt_check(delay)
        if shutting_down:
            return
        start_one("perception")

        print(f"[main] waiting {args.perception_warmup_s:.1f}s for perception warmup...")
        wait_with_interrupt_check(args.perception_warmup_s)
        if shutting_down:
            return

        for name in startup_order:
            if name == "perception":
                continue
            if name in proc_specs:
                if name in procs:
                    terminate_named_proc(procs, name)
                print(f"[main] restarting downstream {name} ...")
                wait_with_interrupt_check(args.downstream_restart_delay_s)
                if shutting_down:
                    return
                start_one(name)
                wait_with_interrupt_check(args.between_pipeline_start_s)
                if shutting_down:
                    return

    print(f"[main] config={args.config}")
    print(f"[main] enabled_pipelines={list(proc_specs.keys())}")
    print(f"[main] startup_order={startup_order}")

    try:
        start_initial_pipelines()

        while not shutting_down:
            for name in list(procs.keys()):
                p = procs.get(name)
                if p is None:
                    continue

                rc = p.poll()
                if rc is None:
                    continue

                print(f"[main] child exited: pipeline={name} code={rc}")
                procs.pop(name, None)

                if shutting_down:
                    continue

                restart_counts[name] = int(restart_counts.get(name, 0)) + 1
                if restart_counts[name] > args.max_restarts_per_pipeline:
                    raise RuntimeError(
                        f"[main] pipeline {name} exceeded max restarts "
                        f"({args.max_restarts_per_pipeline})"
                    )

                if name == "perception":
                    restart_perception_and_downstream()
                else:
                    if "perception" in proc_specs:
                        perception_proc = procs.get("perception")
                        if perception_proc is None or perception_proc.poll() is not None:
                            print(f"[main] delaying restart of {name} because perception is not healthy.")
                            continue

                    delay = restart_delay_for(name)
                    print(f"[main] restarting pipeline {name} in {delay:.1f}s ...")
                    wait_with_interrupt_check(delay)
                    if shutting_down:
                        break
                    start_one(name)
                    wait_with_interrupt_check(args.between_pipeline_start_s)

            time.sleep(0.5)

    except KeyboardInterrupt:
        request_shutdown()
    except Exception as e:
        shutting_down = True
        print(f"[main] error: {e}")
    finally:
        shutting_down = True
        terminate_all(procs)
        print("[main] done.")


if __name__ == "__main__":
    main()
