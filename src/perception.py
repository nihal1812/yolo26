#!/usr/bin/env python3
import time
import signal
import argparse
import traceback
import threading

from config_utils import load_cfg, get_active_cams, get_brain_args, format_cam_dict

from rtsp_stream import RtspStreamRuntime
from pose_track import PoseTracker
from seg_track import SegTracker
from feature_builder import FeatureBuilder

STOP_EVENT = threading.Event()


def dict_to_cli_namespace(d: dict):
    ns = argparse.Namespace()
    for k, v in (d or {}).items():
        setattr(ns, k, v)
    return ns


class CameraRuntime(threading.Thread):
    """
    Integrated per-camera perception runtime.

    Flow:
      RTSP ingest -> decode once in rtsp_stream
                 -> pose
                 -> seg
                 -> feature_builder
    """

    def __init__(self, cfg_path: str, cam_id: str, debug: bool = False):
        super().__init__(daemon=True)
        self.cfg_path = cfg_path
        self.cam_id = str(cam_id)
        self.debug = bool(debug)

        self.cfg = load_cfg(cfg_path)
        perception_cfg = self.cfg.get("perception", {}) if isinstance(self.cfg, dict) else {}
        fb_args_cfg = get_brain_args(self.cfg, "feature_builder_args")
        fb_args = dict_to_cli_namespace(format_cam_dict(fb_args_cfg, self.cam_id))

        for k, v in {
            "T": 16,
            "clip_h": 112,
            "clip_w": 112,
            "sigma": 2.5,
            "kp_conf_thr": 0.3,
            "contact_use_poly": True,
            "contact_dist_px": 25.0,
            "vis_hist": 30,
            "comotion_window": 10,
            "obj_switch_margin_ratio": 0.80,
            "obj_switch_confirm_frames": 3,
            "disappear_miss_frames": 12,
            "disappear_contact_min_frames": 5,
            "disappear_window_frames": 40,
            "person_state_ttl_frames": 120,
            "object_state_ttl_frames": 240,
            "emit_heuristic_theft_score": False,
            "max_join_buf": 512,
            "clip_dir": "/tmp/zono_clips",
            "clip_codec": "npz",
            "clip_retention_s": 1800,
            "clip_cleanup_interval_s": 60.0,
            "publish_frame_scalars": True,
            "publish_clip_events": True,
            "publish_scalars_clip_events": True,
        }.items():
            if not hasattr(fb_args, k):
                setattr(fb_args, k, v)

        if hasattr(fb_args, "publish_frame_scalars_redis") and not hasattr(fb_args, "publish_frame_scalars"):
            setattr(fb_args, "publish_frame_scalars", bool(getattr(fb_args, "publish_frame_scalars_redis")))

        self.fb_args = fb_args

        self.rtsp = None
        self.pose = None
        self.seg = None
        self.fb = None

        self.frames_seen = 0
        self.last_ok_ts = None
        self.resources_built = False

        self.consecutive_failures = 0
        self.base_restart_delay_s = float(perception_cfg.get("camera_runtime_restart_delay_s", 2.0))
        self.restart_backoff_step_s = float(perception_cfg.get("camera_runtime_restart_backoff_step_s", 1.0))
        self.restart_backoff_cap_s = float(perception_cfg.get("camera_runtime_restart_backoff_cap_s", 10.0))

        self.startup_warmup_s = float(perception_cfg.get("camera_startup_warmup_s", 8.0))
        self.stale_frame_timeout_s = float(perception_cfg.get("stale_frame_timeout_s", 20.0))
        self.health_log_every_s = float(perception_cfg.get("camera_health_log_every_s", 15.0))
        self._last_health_log_t = 0.0

    def _log(self, stage, msg):
        print(f"[perception][{self.cam_id}][{stage}] {msg}")

    def _restart_delay(self):
        return min(
            self.base_restart_delay_s + max(0, self.consecutive_failures - 1) * self.restart_backoff_step_s,
            self.restart_backoff_cap_s,
        )

    def last_frame_age_s(self):
        if self.last_ok_ts is None:
            return None
        return max(0.0, time.time() - float(self.last_ok_ts))

    def _maybe_log_health(self, force=False):
        now = time.time()
        if not force and (now - self._last_health_log_t) < self.health_log_every_s:
            return
        self._last_health_log_t = now

        age = self.last_frame_age_s()
        age_txt = "none" if age is None else f"{age:.1f}s"
        self._log(
            "health",
            f"frames_seen={self.frames_seen} last_frame_age={age_txt} "
            f"consecutive_failures={self.consecutive_failures}",
        )

    def _safe_shutdown_obj(self, name, obj):
        if obj is None:
            return

        try:
            if hasattr(obj, "stop"):
                obj.stop()
        except Exception as e:
            self._log("cleanup", f"error stopping {name}: {e}")

        try:
            if hasattr(obj, "close"):
                obj.close()
        except Exception as e:
            self._log("cleanup", f"error closing {name}: {e}")

    def build_resources(self):
        self._log("init", "building resources...")
        self.resources_built = False

        self.rtsp = RtspStreamRuntime(
            cfg_path=self.cfg_path,
            cam_id=self.cam_id,
            publish_external=True,
            decoded_queue_size=2,
            debug=self.debug,
        )
        self.rtsp.start()

        self.pose = PoseTracker(
            cfg_path=self.cfg_path,
            cam_id=self.cam_id,
            publish_external=True,
            debug=self.debug,
        )

        self.seg = SegTracker(
            cfg_path=self.cfg_path,
            cam_id=self.cam_id,
            publish_external=True,
            debug=self.debug,
        )

        self.fb = FeatureBuilder(
            cfg_path=self.cfg_path,
            cam_id=self.cam_id,
            args=self.fb_args,
            debug=self.debug,
        )

        self.resources_built = True
        self._log("init", "resources ready")

    def close_resources(self):
        for name, obj in [
            ("feature_builder", self.fb),
            ("seg", self.seg),
            ("pose", self.pose),
            ("rtsp", self.rtsp),
        ]:
            self._safe_shutdown_obj(name, obj)

        self.fb = None
        self.seg = None
        self.pose = None
        self.rtsp = None
        self.resources_built = False

    def run(self):
        while not STOP_EVENT.is_set():
            try:
                self.build_resources()
                self._log("run", "camera runtime started")

                startup_deadline = time.time() + self.startup_warmup_s

                while not STOP_EVENT.is_set():
                    item = self.rtsp.read_frame(timeout=1.0)
                    if item is None:
                        # During initial GStreamer warmup, avoid noisy timeout logs.
                        if self.frames_seen == 0 and time.time() < startup_deadline:
                            continue

                        stale_reference = self.last_ok_ts or startup_deadline
                        stale_age = time.time() - stale_reference
                        if self.stale_frame_timeout_s > 0 and stale_age >= self.stale_frame_timeout_s:
                            raise RuntimeError(
                                f"decoded frame stale for {stale_age:.1f}s "
                                f"(timeout={self.stale_frame_timeout_s:.1f}s)"
                            )

                        self._log("rtsp", "decoded frame timeout")
                        self._maybe_log_health()
                        continue

                    header, frame = item
                    frame_id = header.get("frame_id", -1)

                    if self.debug and self.frames_seen % 50 == 0:
                        self._log("frame", f"frame_id={frame_id} shape={frame.shape}")

                    _pose_header, pose_payload = self.pose.process_frame(frame, header)
                    _seg_header, seg_payload = self.seg.process_frame(frame, header)
                    self.fb.process_pair(pose_payload, seg_payload)

                    self.frames_seen += 1
                    self.last_ok_ts = time.time()
                    self.consecutive_failures = 0

                    if self.debug and self.frames_seen % 50 == 0:
                        self._log(
                            "ok",
                            f"processed={self.frames_seen} "
                            f"frame_id={frame_id} "
                            f"pose_people={len(pose_payload.get('people', []))} "
                            f"seg_instances={len(seg_payload.get('instances', []))}",
                        )
                    self._maybe_log_health()

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.consecutive_failures += 1
                self._log("error", f"{e}")
                traceback.print_exc()

                self.close_resources()

                if not STOP_EVENT.is_set():
                    delay = self._restart_delay()
                    self._log("restart", f"rebuilding camera runtime in {delay:.1f}s ...")
                    time.sleep(delay)
            finally:
                self.close_resources()

        self._log("done", "camera runtime stopped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    STOP_EVENT.clear()

    cfg_path = args.config
    cfg = load_cfg(cfg_path)
    active_cams = get_active_cams(cfg)
    perception_cfg = cfg.get("perception", {}) if isinstance(cfg, dict) else {}

    thread_restart_delay_s = float(perception_cfg.get("camera_thread_restart_delay_s", 2.0))
    thread_restart_backoff_step_s = float(perception_cfg.get("camera_thread_restart_backoff_step_s", 1.0))
    thread_restart_backoff_cap_s = float(perception_cfg.get("camera_thread_restart_backoff_cap_s", 30.0))

    workers = []
    worker_restart_counts = {}
    next_restart_at = {}

    def _thread_restart_delay(count: int) -> float:
        return min(
            thread_restart_delay_s + max(0, int(count) - 1) * thread_restart_backoff_step_s,
            thread_restart_backoff_cap_s,
        )

    def _start_worker(cam_id: str):
        w = CameraRuntime(cfg_path=cfg_path, cam_id=cam_id, debug=args.debug)
        print(f"[perception] starting integrated worker for {cam_id} ...")
        w.start()
        return w

    def _handle_sigint(_sig, _frame):
        STOP_EVENT.set()

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    try:
        print(f"[perception] config={cfg_path}")
        print(f"[perception] active_cams={active_cams}")
        print("[perception] integrated mode: one worker per camera")

        for cam_id in active_cams:
            w = _start_worker(cam_id)
            workers.append(w)
            time.sleep(0.4)

        print("[perception] all camera runtimes started. Ctrl+C to stop.")

        while not STOP_EVENT.is_set():
            for idx, w in enumerate(list(workers)):
                if not w.is_alive() and not STOP_EVENT.is_set():
                    cam_id = w.cam_id
                    now = time.time()
                    due = float(next_restart_at.get(cam_id, 0.0))
                    if now < due:
                        continue

                    try:
                        w.join(timeout=0.1)
                    except Exception:
                        pass

                    worker_restart_counts[cam_id] = int(worker_restart_counts.get(cam_id, 0)) + 1
                    count = worker_restart_counts[cam_id]
                    delay = _thread_restart_delay(count)
                    next_restart_at[cam_id] = now + delay

                    print(
                        f"[perception] worker thread died unexpectedly for cam={cam_id}; "
                        f"restart_count={count} restarting_in={delay:.1f}s"
                    )

                    t0 = time.time()
                    while not STOP_EVENT.is_set() and (time.time() - t0) < delay:
                        time.sleep(0.1)

                    if STOP_EVENT.is_set():
                        break

                    workers[idx] = _start_worker(cam_id)
                    print(
                        f"[perception] worker restarted for cam={cam_id}; "
                        f"restart_count={count}"
                    )
            time.sleep(0.5)

    except KeyboardInterrupt:
        STOP_EVENT.set()
        print("\n[perception] stopping...")
    except Exception as e:
        STOP_EVENT.set()
        print(f"[perception] error: {e}")
        traceback.print_exc()
    finally:
        STOP_EVENT.set()
        for w in workers:
            try:
                w.join(timeout=3.0)
            except Exception:
                pass
        print("[perception] done.")


if __name__ == "__main__":
    main()
