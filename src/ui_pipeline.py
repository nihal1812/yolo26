#!/usr/bin/env python3
"""
ui_pipeline.py

Scalable UI pipeline.

Behavior:
- one global s3_feedback_poller process
- one per-camera UI worker process

Each per-camera worker can run:
    - local alert sender stage
    - optional incident alert sender stage (leader cam only)
    - bbox overlay stage
    - annotated alert clip writer inside bbox_overlay

Update:
- bbox_overlay writes annotated MP4 clips on alert
- ui_pipeline caches the annotated MP4 clip_ref in alert_sender
- alert_sender attaches matching clipPath to alerts using event_id
"""

import sys
import time
import json
import signal
import argparse
import subprocess
from pathlib import Path

import zmq

from config_utils import (
    load_cfg,
    get_active_cams,
    get_zmq_endpoint,
    local_connect_addr,
)
from alert_sender import AlertSenderStage
from bbox_overlay import BBoxOverlayStage

HERE = Path(__file__).resolve().parent


def popen_node(py_exe, script_path: Path, cfg_path: str, extra_args=None):
    if not script_path.exists():
        raise FileNotFoundError(f"Node script not found: {script_path}")

    cmd = [py_exe, str(script_path), "--config", cfg_path]
    if extra_args:
        cmd.extend(extra_args)

    print("[ui_pipeline] CMD:", " ".join(map(str, cmd)))
    return subprocess.Popen(cmd, cwd=str(HERE))


def terminate_proc(p, grace_s=3.0):
    if p is None:
        return

    try:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
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


def make_sub_socket(ctx, connect_addr, topic, rcvhwm=1000, latest_only=False):
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.RCVHWM, int(rcvhwm))

    # multipart PUB/SUB stream -> do not use CONFLATE
    _ = latest_only

    sub.connect(connect_addr)
    sub.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
    return sub


class UICamWorker:
    def __init__(
        self,
        cfg_path: str,
        cam_id: str,
        *,
        incident_leader_cam: str,
        enable_local_alerts: bool = True,
        enable_overlay: bool = True,
        enable_incident_sender: bool = True,
    ):
        self.cfg_path = cfg_path
        self.cfg = load_cfg(cfg_path)
        self.cam_id = str(cam_id)
        self.leader_cam = str(incident_leader_cam)

        self.enable_local_alerts = bool(enable_local_alerts)
        self.enable_overlay = bool(enable_overlay)
        self.enable_incident_sender = bool(enable_incident_sender) and (
            self.cam_id == self.leader_cam
        )

        self.ctx = zmq.Context.instance()
        self.poller = zmq.Poller()

        self.subscribers = {}
        self.socket_roles = {}

        self.local_alert_stage = None
        self.incident_alert_stage = None
        self.overlay_stage = None

        self._running = True
        self._closed = False

        self._build()

    def stop(self):
        self._running = False

    def _register_subscriber(
        self,
        role: str,
        cam_id: str,
        zmq_key: str,
        latest_only=False,
        default_rcvhwm=1000,
    ):
        if role in self.subscribers:
            print(f"[ui_cam_worker][{self.cam_id}] subscriber already registered role={role}")
            return

        ep = get_zmq_endpoint(self.cfg, cam_id, zmq_key)
        connect_addr = local_connect_addr(ep["bind"])
        topic = ep["topic"]
        rcvhwm = int(ep.get("rcvhwm", default_rcvhwm))
        latest_only = bool(ep.get("latest_only", latest_only))

        sub = make_sub_socket(
            self.ctx,
            connect_addr=connect_addr,
            topic=topic,
            rcvhwm=rcvhwm,
            latest_only=latest_only,
        )

        self.poller.register(sub, zmq.POLLIN)
        self.subscribers[role] = sub
        self.socket_roles[sub] = role

        print(f"[ui_cam_worker][{self.cam_id}] SUB {role} {connect_addr} topic={topic}")

    def _build(self):
        print(f"[ui_cam_worker] cam_id={self.cam_id}")
        print(f"[ui_cam_worker] leader_cam={self.leader_cam}")

        if self.enable_local_alerts:
            try:
                self.local_alert_stage = AlertSenderStage(
                    self.cfg,
                    self.cam_id,
                    source_mode="local",
                    external_mode=True,
                )
                self._register_subscriber(
                    "local_alerts",
                    self.cam_id,
                    "alerts_enriched",
                    latest_only=False,
                    default_rcvhwm=1000,
                )
            except Exception as e:
                print(f"[ui_cam_worker][{self.cam_id}] local alert sender disabled: {e}")
                self.local_alert_stage = None

        if self.enable_incident_sender:
            try:
                self.incident_alert_stage = AlertSenderStage(
                    self.cfg,
                    self.cam_id,
                    source_mode="incident",
                    incident_leader_cam=self.leader_cam,
                    external_mode=True,
                )
                self._register_subscriber(
                    "incident_alerts",
                    self.cam_id,
                    "incident_alerts",
                    latest_only=False,
                    default_rcvhwm=1000,
                )
            except Exception as e:
                print(f"[ui_cam_worker][{self.cam_id}] incident alert sender disabled: {e}")
                self.incident_alert_stage = None

        if self.enable_overlay:
            try:
                overlay_cfg = self.cfg.get("live_view", {}).get("overlay", {})

                self.overlay_stage = BBoxOverlayStage(
                    self.cfg,
                    self.cam_id,
                    frame_buffer=int(overlay_cfg.get("frame_buffer", 120)),
                    pose_buffer=int(overlay_cfg.get("pose_buffer", 240)),
                    ema_alpha=float(overlay_cfg.get("ema_alpha", 0.30)),
                    state_ttl_s=float(overlay_cfg.get("state_ttl_s", 4.0)),
                    decision_freshness_s=float(
                        overlay_cfg.get("decision_freshness_s", 4.0)
                    ),
                    track_memory_ttl_s=float(
                        overlay_cfg.get("track_memory_ttl_s", 4.0)
                    ),
                    iou_match_thr=float(overlay_cfg.get("iou_match_thr", 0.20)),
                    iou_loose_thr=float(overlay_cfg.get("iou_loose_thr", 0.08)),
                    send_every_n_frames=int(
                        overlay_cfg.get("send_every_n_frames", 1)
                    ),
                    pose_match_max_delta=int(
                        overlay_cfg.get("pose_match_max_delta", 8)
                    ),
                )

                self._register_subscriber(
                    "video",
                    self.cam_id,
                    "video_ui",
                    latest_only=False,
                    default_rcvhwm=2,
                )
                self._register_subscriber(
                    "pose",
                    self.cam_id,
                    "pose_features",
                    latest_only=False,
                    default_rcvhwm=2,
                )
                self._register_subscriber(
                    "decisions",
                    self.cam_id,
                    "decisions_enriched",
                    latest_only=False,
                    default_rcvhwm=1000,
                )
            except Exception as e:
                print(f"[ui_cam_worker][{self.cam_id}] overlay disabled: {e}")
                self.overlay_stage = None

    
    def _recv_json_payload(self, sock):
        parts = sock.recv_multipart()
        if len(parts) != 3:
            return None, None

        _topic_b, header_b, payload_b = parts

        try:
            header = json.loads(header_b.decode("utf-8"))
        except Exception:
            header = None

        try:
            payload = json.loads(payload_b.decode("utf-8"))
        except Exception:
            payload = None

        return header, payload

    def _recv_video_payload(self, sock):
        parts = sock.recv_multipart()
        if len(parts) != 3:
            return None, None

        _topic_b, header_b, enc = parts

        try:
            header = json.loads(header_b.decode("utf-8"))
        except Exception:
            header = {}

        return header, enc


    def _build_overlay_clip_for_alert(self, stage, payload: dict) -> bool:
        """
        Synchronously build an annotated MP4 clip from bbox_overlay's bounded rendered-frame cache.
        The generated clip_ref is cached in alert_sender before the webhook is sent.
        """
        if stage is None or not isinstance(payload, dict):
            return False

        if self.overlay_stage is None:
            return False

        if not hasattr(self.overlay_stage, "build_alert_clip_ref"):
            return False

        try:
            clip_ref = self.overlay_stage.build_alert_clip_ref(payload)
        except Exception as e:
            print(f"[ui_cam_worker][{self.cam_id}] overlay alert clip error: {e}")
            return False

        if not isinstance(clip_ref, dict):
            return False

        try:
            stage.handle_clip_ref_obj(clip_ref)
            return True
        except Exception as e:
            print(f"[ui_cam_worker][{self.cam_id}] failed to cache overlay clip_ref: {e}")
            return False

    def _queue_or_send_alert(self, kind: str, payload: dict):
        if not isinstance(payload, dict):
            return

        if kind == "local":
            stage = self.local_alert_stage
        else:
            stage = self.incident_alert_stage

        if stage is None:
            return

        # Build the annotated MP4 only when an alert arrives.
        # No continuous clip writing, no unbounded frame dumping.
        self._build_overlay_clip_for_alert(stage, payload)

        # Send immediately. If clip creation failed, alert still goes out without blocking.
        stage.handle_alert_obj(payload)


    def close(self):
        if self._closed:
            return
        self._closed = True

        try:
            if self.overlay_stage is not None and hasattr(self.overlay_stage, "close"):
                self.overlay_stage.close()
        except Exception:
            pass

        try:
            if self.local_alert_stage is not None and hasattr(self.local_alert_stage, "close"):
                self.local_alert_stage.close()
        except Exception:
            pass

        try:
            if self.incident_alert_stage is not None and hasattr(self.incident_alert_stage, "close"):
                self.incident_alert_stage.close()
        except Exception:
            pass

        for sock in list(self.subscribers.values()):
            try:
                self.poller.unregister(sock)
            except Exception:
                pass
            try:
                sock.close(0)
            except Exception:
                pass

        self.subscribers.clear()
        self.socket_roles.clear()

    def run_forever(self):
        try:
            while self._running:
                try:
                    events = dict(self.poller.poll(timeout=100))
                except KeyboardInterrupt:
                    break
                except zmq.ZMQError as e:
                    if getattr(e, "errno", None) == zmq.EINTR:
                        break
                    if not self._running:
                        break
                    raise

                if not self._running:
                    break

                role_priority = {
                    "video": 1,
                    "pose": 2,
                    "decisions": 3,
                    "local_alerts": 4,
                    "incident_alerts": 5,
                }

                ready_socks = sorted(
                    list(events.keys()),
                    key=lambda s: role_priority.get(self.socket_roles.get(s), 99),
                )

                for sock in ready_socks:
                    role = self.socket_roles.get(sock)
                    if role is None:
                        continue

                    try:
                        if role == "video" and self.overlay_stage is not None:
                            header, enc = self._recv_video_payload(sock)
                            if enc is not None:
                                self.overlay_stage.ingest_video_msg(header or {}, enc)

                        elif role == "pose" and self.overlay_stage is not None:
                            _header, payload = self._recv_json_payload(sock)
                            if isinstance(payload, dict):
                                self.overlay_stage.ingest_pose_msg(payload)

                        elif role == "decisions" and self.overlay_stage is not None:
                            _header, payload = self._recv_json_payload(sock)
                            if isinstance(payload, dict):
                                self.overlay_stage.ingest_decision_msg(payload)

                        elif role == "local_alerts" and self.local_alert_stage is not None:
                            _header, payload = self._recv_json_payload(sock)
                            if isinstance(payload, dict):
                                self._queue_or_send_alert("local", payload)

                        elif role == "incident_alerts" and self.incident_alert_stage is not None:
                            _header, payload = self._recv_json_payload(sock)
                            if isinstance(payload, dict):
                                self._queue_or_send_alert("incident", payload)

                    except Exception as e:
                        print(f"[ui_cam_worker][{self.cam_id}] socket handler error ({role}): {e}")


                if self.overlay_stage is not None:
                    try:
                        self.overlay_stage.maybe_send_overlay()
                    except Exception as e:
                        print(f"[ui_cam_worker][{self.cam_id}] overlay error: {e}")

        finally:
            self.close()
            print(f"[ui_cam_worker][{self.cam_id}] stopped cleanly.")


def worker_main(args):
    worker = UICamWorker(
        cfg_path=args.config,
        cam_id=args.worker_cam_id,
        incident_leader_cam=args.incident_leader_cam,
        enable_local_alerts=not args.no_alert_sender,
        enable_overlay=not args.no_overlay,
        enable_incident_sender=not args.no_alert_sender,
    )

    def _handle_stop_signal(signum, frame):
        worker.stop()

    signal.signal(signal.SIGINT, _handle_stop_signal)
    signal.signal(signal.SIGTERM, _handle_stop_signal)

    try:
        worker.run_forever()
    except KeyboardInterrupt:
        worker.stop()
    finally:
        worker.stop()


def supervisor_main(args):
    cfg = load_cfg(args.config)
    active_cams = get_active_cams(cfg)
    if not active_cams:
        raise RuntimeError("No active cameras configured.")

    ui_cfg = cfg.get("ui_pipeline", {})
    wh_enabled = bool(ui_cfg.get("webhooks", {}).get("enabled", False))

    leader_cam = args.incident_leader_cam or active_cams[0]
    py = sys.executable
    # Use an absolute config path for child processes because they run with cwd=src.
    cfg_path_abs = str(Path(args.config).resolve())

    s3_feedback_poller = HERE / "s3_feedback_poller.py"
    this_script = HERE / "ui_pipeline.py"

    proc_specs = {}
    procs = {}
    restart_counts = {}
    shutting_down = False

    def register_proc(name: str, script_path: Path, extra_args=None):
        proc_specs[name] = {
            "script": script_path,
            "extra_args": list(extra_args) if extra_args else [],
        }
        restart_counts[name] = 0

    if not args.no_s3_feedback:
        register_proc("s3_feedback_poller", s3_feedback_poller, args.s3_feedback_args or [])

    for cam_id in active_cams:
        extra = [
            "--worker-mode",
            "--worker-cam-id",
            cam_id,
            "--incident-leader-cam",
            leader_cam,
        ]

        if args.no_alert_sender:
            extra.append("--no-alert-sender")
        if args.no_overlay:
            extra.append("--no-overlay")

        register_proc(f"ui_cam_worker:{cam_id}", this_script, extra)

    if not proc_specs:
        raise RuntimeError("Nothing to start (all UI components disabled).")

    def start_one(name: str):
        spec = proc_specs[name]
        print(f"[ui_pipeline] starting {name} ...")
        p = popen_node(py, spec["script"], cfg_path_abs, spec["extra_args"])
        procs[name] = p
        return p

    try:
        print(f"[ui_pipeline] active_cams={active_cams}")
        print(f"[ui_pipeline] incident_leader_cam={leader_cam}")
        print(f"[ui_pipeline] webhooks_enabled={wh_enabled}")

        if "s3_feedback_poller" in proc_specs:
            start_one("s3_feedback_poller")
            time.sleep(0.3)

        for cam_id in active_cams:
            name = f"ui_cam_worker:{cam_id}"
            if name in proc_specs:
                start_one(name)
                time.sleep(0.3)

        print("[ui_pipeline] all processes started. Ctrl+C to stop.")

        while True:
            for name, p in list(procs.items()):
                rc = p.poll()
                if rc is None:
                    continue

                print(f"[ui_pipeline] child exited: name={name} code={rc}")
                procs.pop(name, None)

                if shutting_down:
                    continue

                restart_counts[name] = int(restart_counts.get(name, 0)) + 1
                if restart_counts[name] > args.max_restarts_per_child:
                    raise RuntimeError(
                        f"[ui_pipeline] child {name} exceeded max restarts "
                        f"({args.max_restarts_per_child})"
                    )

                delay = min(
                    args.restart_delay_s
                    + (restart_counts[name] - 1) * args.restart_backoff_step_s,
                    args.restart_backoff_cap_s,
                )
                print(f"[ui_pipeline] restarting {name} in {delay:.1f}s ...")
                time.sleep(delay)
                start_one(name)
                time.sleep(0.2)

            time.sleep(0.5)

    except KeyboardInterrupt:
        shutting_down = True
        print("\n[ui_pipeline] stopping...")
    except Exception as e:
        shutting_down = True
        print(f"[ui_pipeline] error: {e}")
    finally:
        shutting_down = True
        terminate_all(list(procs.values()))
        print("[ui_pipeline] done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    ap.add_argument("--worker-mode", action="store_true")
    ap.add_argument("--worker-cam-id", default=None)

    ap.add_argument("--no-alert-sender", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    # Backward-compatible alias: --no-feedback now means do not start s3_feedback_poller.
    ap.add_argument("--no-s3-feedback", "--no-feedback", dest="no_s3_feedback", action="store_true")

    ap.add_argument("--incident-leader-cam", default=None)

    ap.add_argument("--restart_delay_s", type=float, default=2.0)
    ap.add_argument("--max_restarts_per_child", type=int, default=50)
    ap.add_argument("--restart_backoff_step_s", type=float, default=1.0)
    ap.add_argument("--restart_backoff_cap_s", type=float, default=15.0)

    # Optional args passed to s3_feedback_poller. Put this last if used.
    ap.add_argument("--s3-feedback-args", "--feedback-args", dest="s3_feedback_args", nargs=argparse.REMAINDER, default=None)

    args = ap.parse_args()

    if args.worker_mode:
        if not args.worker_cam_id:
            raise RuntimeError("--worker-cam-id required in --worker-mode")
        worker_main(args)
    else:
        supervisor_main(args)


if __name__ == "__main__":
    main()
