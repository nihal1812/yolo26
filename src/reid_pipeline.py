#!/usr/bin/env python3
"""
reid_pipeline.py

Starts:
  - one reid_node.py per active camera
  - one identity_stitcher.py globally
  - one identity_enricher.py globally
  - one incident_builder.py globally

Behavior:
  - starts subprocesses
  - monitors children
  - restarts failed children
  - clean shutdown on Ctrl+C

Usage:
  python3 reid_pipeline.py --config config.yaml
"""

import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path

from config_utils import load_cfg, get_active_cams

HERE = Path(__file__).resolve().parent


def popen_node(py_exe, script_path: Path, cfg_path: str, extra_args=None):
    if not script_path.exists():
        raise FileNotFoundError(f"Node script not found: {script_path}")

    cmd = [py_exe, str(script_path), "--config", cfg_path]
    if extra_args:
        cmd.extend(extra_args)

    print("[reid_pipeline] CMD:", " ".join(map(str, cmd)))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    ap.add_argument("--no-reid", action="store_true")
    ap.add_argument("--no-stitcher", action="store_true")
    ap.add_argument("--no-enricher", action="store_true")
    ap.add_argument("--no-incident-builder", action="store_true")

    ap.add_argument("--reid-args", nargs="*", default=None)
    ap.add_argument("--stitcher-args", nargs="*", default=None)
    ap.add_argument("--enricher-args", nargs="*", default=None)
    ap.add_argument("--incident-builder-args", nargs="*", default=None)

    ap.add_argument("--restart_delay_s", type=float, default=2.0)
    ap.add_argument("--max_restarts_per_child", type=int, default=50)
    ap.add_argument("--restart_backoff_step_s", type=float, default=1.0)
    ap.add_argument("--restart_backoff_cap_s", type=float, default=15.0)

    args = ap.parse_args()

    cfg = load_cfg(args.config)
    active_cams = get_active_cams(cfg)

    py = sys.executable

    reid_node = HERE / "reid_node.py"
    identity_stitcher = HERE / "identity_stitcher.py"
    identity_enricher = HERE / "identity_enricher.py"
    incident_builder = HERE / "incident_builder.py"

    proc_specs = {}
    procs = {}
    restart_counts = {}

    def register_proc(name, script_path: Path, extra_args=None):
        proc_specs[name] = {
            "script": script_path,
            "extra_args": list(extra_args) if extra_args else [],
        }
        restart_counts[name] = 0

    if not args.no_enricher:
        register_proc(
            "identity_enricher",
            identity_enricher,
            args.enricher_args or [],
        )

    if not args.no_stitcher:
        register_proc(
            "identity_stitcher",
            identity_stitcher,
            args.stitcher_args or [],
        )

    if not args.no_incident_builder:
        register_proc(
            "incident_builder",
            incident_builder,
            args.incident_builder_args or [],
        )

    if not args.no_reid:
        for cam_id in active_cams:
            extra = ["--cam_id", cam_id]
            if args.reid_args:
                extra.extend(args.reid_args)
            register_proc(f"reid_node:{cam_id}", reid_node, extra)

    if not proc_specs:
        raise RuntimeError("Nothing to start (all ReID pipeline components disabled).")

    def start_one(name: str):
        spec = proc_specs[name]
        print(f"[reid_pipeline] starting {name} ...")
        p = popen_node(py, spec["script"], args.config, spec["extra_args"])
        procs[name] = p
        return p

    def dependent_names_for(name: str):
        deps = []
        # If stitcher dies, enricher and incident builder may keep running but with stale/no updates.
        # Restart them too for a clean chain.
        if name == "identity_stitcher":
            if "identity_enricher" in procs:
                deps.append("identity_enricher")
            if "incident_builder" in procs:
                deps.append("incident_builder")

        # If enricher dies, incident builder should restart because it consumes enriched streams.
        elif name == "identity_enricher":
            if "incident_builder" in procs:
                deps.append("incident_builder")

        return deps

    try:
        print(f"[reid_pipeline] active_cams={active_cams}")

        # Start in dependency order
        if "identity_enricher" in proc_specs:
            start_one("identity_enricher")
            time.sleep(0.5)

        if "identity_stitcher" in proc_specs:
            start_one("identity_stitcher")
            time.sleep(0.5)

        if "incident_builder" in proc_specs:
            start_one("incident_builder")
            time.sleep(0.5)

        for cam_id in active_cams:
            name = f"reid_node:{cam_id}"
            if name in proc_specs:
                start_one(name)
                time.sleep(0.3)

        print("[reid_pipeline] all processes started. Ctrl+C to stop.")

        while True:
            for name, p in list(procs.items()):
                rc = p.poll()
                if rc is None:
                    continue

                print(f"[reid_pipeline] child exited: name={name} code={rc}")

                terminate_proc(p)
                procs.pop(name, None)

                restart_counts[name] = int(restart_counts.get(name, 0)) + 1
                if restart_counts[name] > args.max_restarts_per_child:
                    raise RuntimeError(
                        f"[reid_pipeline] child {name} exceeded max restarts "
                        f"({args.max_restarts_per_child})"
                    )

                deps = dependent_names_for(name)
                for dep in deps:
                    dep_proc = procs.get(dep)
                    if dep_proc is not None:
                        print(f"[reid_pipeline] stopping dependent child: {dep}")
                        terminate_proc(dep_proc)
                        procs.pop(dep, None)

                delay = min(
                    args.restart_delay_s + (restart_counts[name] - 1) * args.restart_backoff_step_s,
                    args.restart_backoff_cap_s,
                )
                print(f"[reid_pipeline] restarting {name} in {delay:.1f}s ...")
                time.sleep(delay)
                start_one(name)
                time.sleep(0.5)

                for dep in deps:
                    if dep in proc_specs:
                        dep_delay = min(
                            args.restart_delay_s + (restart_counts.get(dep, 0)) * args.restart_backoff_step_s,
                            args.restart_backoff_cap_s,
                        )
                        print(f"[reid_pipeline] restarting dependent child {dep} in {dep_delay:.1f}s ...")
                        time.sleep(dep_delay)
                        start_one(dep)
                        time.sleep(0.5)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[reid_pipeline] stopping...")
    except Exception as e:
        print(f"[reid_pipeline] error: {e}")
    finally:
        terminate_all(list(procs.values()))
        print("[reid_pipeline] done.")


if __name__ == "__main__":
    main()