#!/usr/bin/env python3
import argparse
import json
import time
from collections import deque
from pathlib import Path

import zmq
import numpy as np
import cv2
import requests

from config_utils import load_cfg, get_zmq_endpoint, local_connect_addr


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def clamp01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def iou_xyxy(a, b):
    if a is None or b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    denom = area_a + area_b - inter + 1e-6
    return float(inter / denom)


def resolve_video_ui_endpoint(cfg, cam_id: str):
    return get_zmq_endpoint(cfg, cam_id, "video_ui")


def resolve_decisions_endpoint(cfg, cam_id: str):
    try:
        return get_zmq_endpoint(cfg, cam_id, "decisions_enriched")
    except Exception:
        return get_zmq_endpoint(cfg, cam_id, "decisions")


def suspicion_to_color(s):
    s = clamp01(s)
    if s < 0.30:
        return (0, 255, 0)
    elif s < 0.60:
        return (0, 255, 255)
    elif s < 0.80:
        return (0, 165, 255)
    return (0, 0, 255)


def post_overlay_with_retry(url, meta, jpg_bytes, timeout_s, retries=2):
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(
                url,
                data={"meta": json.dumps(meta)},
                files={"image": ("overlay.jpg", jpg_bytes, "image/jpeg")},
                timeout=timeout_s,
            )
            ok = 200 <= r.status_code < 300
            if ok:
                return True, r.status_code, None
            last_err = f"status={r.status_code}"
        except Exception as e:
            last_err = str(e)

        if attempt < retries - 1:
            time.sleep(0.3 * (attempt + 1))

    return False, None, last_err



def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def safe_int(v, default=-1):
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def safe_float(v, default=0.0):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default



def re_safe_filename(s: str) -> str:
    s = str(s)
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:180] or f"event_{time.time_ns()}"


def alert_event_id(alert: dict, cam_id: str) -> str:
    event_id = alert.get("event_id")
    if event_id:
        return str(event_id)

    person_track_id = safe_int(alert.get("person_track_id", alert.get("trackId", -1)), -1)
    frame_id_end = safe_int(alert.get("frame_id_end", alert.get("frame_id", -1)), -1)
    stamp_ns_end = safe_int(alert.get("stamp_ns_end", alert.get("stamp_ns", time.time_ns())), time.time_ns())
    return f"{cam_id}:{person_track_id}:{frame_id_end}:{stamp_ns_end}"


def make_sub_socket(ctx, connect_addr, topic, rcvhwm=2, latest_only=False):
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.RCVHWM, int(rcvhwm))

    # Do not use CONFLATE on multipart streams.
    _ = latest_only

    sub.connect(connect_addr)
    sub.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
    return sub


class BBoxOverlayStage:
    def __init__(
        self,
        cfg,
        cam_id: str,
        *,
        frame_buffer: int = 120,
        pose_buffer: int = 240,
        ema_alpha: float = 0.30,
        state_ttl_s: float = 4.0,
        decision_freshness_s: float = 4.0,
        track_memory_ttl_s: float = 4.0,
        iou_match_thr: float = 0.20,
        iou_loose_thr: float = 0.08,
        send_every_n_frames: int = 1,
        pose_match_max_delta: int = 8,
    ):
        self.cfg = cfg
        self.cam_id = str(cam_id)

        overlay_cfg = cfg.get("live_view", {}).get("overlay", {})
        frame_buffer = int(overlay_cfg.get("frame_buffer", frame_buffer))
        pose_buffer = int(overlay_cfg.get("pose_buffer", pose_buffer))
        ema_alpha = float(overlay_cfg.get("ema_alpha", ema_alpha))
        state_ttl_s = float(overlay_cfg.get("state_ttl_s", state_ttl_s))
        decision_freshness_s = float(overlay_cfg.get("decision_freshness_s", decision_freshness_s))
        track_memory_ttl_s = float(overlay_cfg.get("track_memory_ttl_s", track_memory_ttl_s))
        iou_match_thr = float(overlay_cfg.get("iou_match_thr", iou_match_thr))
        iou_loose_thr = float(overlay_cfg.get("iou_loose_thr", iou_loose_thr))
        send_every_n_frames = int(overlay_cfg.get("send_every_n_frames", send_every_n_frames))
        pose_match_max_delta = int(overlay_cfg.get("pose_match_max_delta", pose_match_max_delta))

        ui_cfg = cfg.get("ui_pipeline", {})
        wh = ui_cfg.get("webhooks", {})

        self.overlay_url = wh.get("overlay_url")
        self.timeout_s = float(wh.get("timeout_s", 3.0))
        self.webhook_enabled = bool(wh.get("enabled", False))

        # Detections are now sent through ZMQ, not HTTP POST.
        self.detections_url = None
        self.detections_enabled = True

        lv = cfg.get("live_view", {})
        lv_publish = lv.get("publish", {})
        lv_zmq = lv.get("zmq", {})
        cam_live = lv_zmq.get(self.cam_id, {})

        self.live_publish_enabled = bool(lv.get("enabled", False)) and bool(lv_publish.get("enabled", False))
        self.live_bind = cam_live.get("bind")
        self.live_topic = cam_live.get("topic", f"live.overlay.{self.cam_id}")
        self.live_sndhwm = int(cam_live.get("sndhwm", lv_publish.get("sndhwm", 3)))
        self.jpeg_quality = int(lv_publish.get("jpeg_quality", 85))

        clips_cfg = ui_cfg.get("clips", {}) or {}
        alert_clip_cfg = ui_cfg.get("alert_clip_overlay", {}) or {}

        self.alert_clip_enabled = bool(alert_clip_cfg.get("enabled", True))
        out_dir = (
            alert_clip_cfg.get("out_dir")
            or clips_cfg.get("out_dir")
            or "clips_cache/alert_overlay"
        )
        self.alert_clip_out_dir = Path(out_dir)
        if not self.alert_clip_out_dir.is_absolute():
            self.alert_clip_out_dir = project_root() / self.alert_clip_out_dir

        self.alert_clip_fps = float(alert_clip_cfg.get("fps", 8.0))
        self.alert_clip_pre_frames = int(alert_clip_cfg.get("pre_frames", 72))
        self.alert_clip_post_frames = int(alert_clip_cfg.get("post_frames", 0))
        self.alert_clip_max_frames = int(alert_clip_cfg.get("max_frames", 120))
        self.alert_clip_retention_hours = float(
            alert_clip_cfg.get(
                "retention_hours",
                clips_cfg.get("retention_hours", 48),
            )
        )
        self.alert_clip_max_total_mb = float(alert_clip_cfg.get("max_total_mb", 2048))
        self.alert_clip_cleanup_interval_s = float(alert_clip_cfg.get("cleanup_interval_s", 300.0))
        self._last_alert_clip_cleanup_s = 0.0
        self.alert_clip_buffer_frames = int(
            alert_clip_cfg.get(
                "buffer_frames",
                max(int(frame_buffer), self.alert_clip_pre_frames + self.alert_clip_post_frames + 30),
            )
        )

        self.ctx = zmq.Context.instance()

        self.pub = None
        self.det_pub = None

        if self.live_publish_enabled:
            if not self.live_bind:
                raise RuntimeError(
                    f"[bbox_overlay][{self.cam_id}] live_view enabled but live_view.zmq.{self.cam_id}.bind missing"
                )

            self.pub = self.ctx.socket(zmq.PUB)
            self.pub.setsockopt(zmq.LINGER, 0)
            self.pub.setsockopt(zmq.SNDHWM, self.live_sndhwm)
            self.pub.bind(self.live_bind)

        # New ZMQ detections publisher.
        cam_num = int(self.cam_id.replace("cam", ""))
        self.det_bind = f"tcp://*:{5800 + cam_num}"
        self.det_topic = f"bbox_detections.{self.cam_id}"

        self.det_pub = self.ctx.socket(zmq.PUB)
        self.det_pub.setsockopt(zmq.LINGER, 0)
        self.det_pub.setsockopt(zmq.SNDHWM, 3)
        self.det_pub.bind(self.det_bind)

        self.frame_buf = {}
        self.pose_buf = {}
        self.frame_fifo = deque(maxlen=int(frame_buffer))
        self.pose_fifo = deque(maxlen=int(pose_buffer))

        self.render_states = {}
        self.local_track_memory = {}

        self.sent_frame_count = 0
        self.live_publish_count = 0
        self.detection_publish_count = 0
        self.webhook_publish_count = 0
        self.alert_clip_write_count = 0

        self.ema_alpha = float(ema_alpha)
        self.state_ttl_s = float(state_ttl_s)
        self.decision_freshness_s = float(decision_freshness_s)
        self.track_memory_ttl_s = float(track_memory_ttl_s)
        self.iou_match_thr = float(iou_match_thr)
        self.iou_loose_thr = float(iou_loose_thr)
        self.send_every_n_frames = max(1, int(send_every_n_frames))
        self.pose_match_max_delta = int(pose_match_max_delta)

        self.last_rendered_frame_fid = -1

        # Bounded rolling cache of already-annotated JPEG frames.
        # This is intentionally small and memory-safe. MP4 clips are written only on alerts.
        self.rendered_frame_fifo = deque(maxlen=max(1, int(self.alert_clip_buffer_frames)))

        self._last_wait_log_t = 0.0
        self._last_sync_log_t = 0.0
        self._last_render_log_t = 0.0
        self._last_people0_log_t = 0.0
        self._last_publish_log_t = 0.0
        self._last_det_publish_log_t = 0.0

        print(f"[bbox_overlay] cam_id={self.cam_id}")
        print(f"[bbox_overlay] webhook_enabled={self.webhook_enabled} overlay_url={self.overlay_url}")
        print(f"[bbox_overlay] live_publish_enabled={self.live_publish_enabled}")
        print(f"[bbox_overlay] PUB detections {self.det_bind} topic={self.det_topic}")
        print(f"[bbox_overlay] pose_match_max_delta={self.pose_match_max_delta}")
        print(f"[bbox_overlay] frame_buffer={self.frame_fifo.maxlen} pose_buffer={self.pose_fifo.maxlen}")
        print(f"[bbox_overlay] decision_freshness_s={self.decision_freshness_s} track_memory_ttl_s={self.track_memory_ttl_s}")

        if self.live_publish_enabled:
            print(f"[bbox_overlay] PUB live {self.live_bind} topic={self.live_topic}")

        print(
            f"[bbox_overlay] alert_clip_enabled={self.alert_clip_enabled} "
            f"out_dir={self.alert_clip_out_dir} "
            f"buffer_frames={self.alert_clip_buffer_frames} "
            f"pre={self.alert_clip_pre_frames} post={self.alert_clip_post_frames} "
            f"max={self.alert_clip_max_frames} fps={self.alert_clip_fps} "
            f"retention_hours={self.alert_clip_retention_hours} "
            f"max_total_mb={self.alert_clip_max_total_mb}"
        )

        self._cleanup_alert_clip_files(force=True)

    def close(self):
        try:
            if self.pub is not None:
                self.pub.close(0)
        except Exception:
            pass
        self.pub = None

        try:
            if self.det_pub is not None:
                self.det_pub.close(0)
        except Exception:
            pass
        self.det_pub = None

    def _rate_limited_wait_log(self, msg: str, every_s: float = 2.0):
        now = time.time()
        if (now - self._last_wait_log_t) >= every_s:
            print(f"[bbox_overlay] cam={self.cam_id} {msg}")
            self._last_wait_log_t = now

    def _rate_limited_info_log(self, attr_name: str, msg: str, every_s: float = 1.0):
        now = time.time()
        last_t = getattr(self, attr_name, 0.0)
        if (now - last_t) >= every_s:
            print(f"[bbox_overlay] cam={self.cam_id} {msg}")
            setattr(self, attr_name, now)

    def put_buf(self, buf, fifo, fid, obj):
        if fid in buf:
            return

        maxlen = int(fifo.maxlen or 0)

        if maxlen > 0 and len(fifo) >= maxlen:
            old = fifo.popleft()
            buf.pop(old, None)

        buf[fid] = obj
        fifo.append(fid)

        self.check_buffer_health()

    def check_buffer_health(self):
        if len(self.frame_buf) > int(self.frame_fifo.maxlen or 0) + 5:
            print(
                f"[bbox_overlay][WARN] cam={self.cam_id} "
                f"frame_buf leak? size={len(self.frame_buf)} max={self.frame_fifo.maxlen}"
            )

        if len(self.pose_buf) > int(self.pose_fifo.maxlen or 0) + 5:
            print(
                f"[bbox_overlay][WARN] cam={self.cam_id} "
                f"pose_buf leak? size={len(self.pose_buf)} max={self.pose_fifo.maxlen}"
            )

    def find_nearest(self, buf, fid, max_delta=3):
        if fid in buf:
            return fid, buf[fid]

        for d in range(1, max_delta + 1):
            if (fid - d) in buf:
                return fid - d, buf[fid - d]
            if (fid + d) in buf:
                return fid + d, buf[fid + d]

        return None, None

    def find_latest_aligned_pair(self):
        if not self.frame_fifo or not self.pose_buf:
            return None, None, None, None

        frame_ids = list(self.frame_fifo)

        for frame_fid in reversed(frame_ids):
            frame_fid = int(frame_fid)

            frame_item = self.frame_buf.get(frame_fid)
            if frame_item is None:
                continue

            pose_fid, pose = self.find_nearest(
                self.pose_buf,
                frame_fid,
                max_delta=self.pose_match_max_delta,
            )

            if pose is not None:
                return frame_fid, frame_item, int(pose_fid), pose

        return None, None, None, None

    def state_id_from_decision(self, dec):
        gid = dec.get("global_person_id", None)
        pid = int(dec.get("person_track_id", -1))

        if gid is not None:
            try:
                return f"gid:{int(gid)}"
            except Exception:
                pass

        return f"pid:{pid}"

    def prune_state(self):
        now = time.time()

        dead_states = []
        for sid, st in self.render_states.items():
            t_dec = float(st.get("decision_updated_at", st.get("updated_at", 0.0)))
            if (now - t_dec) > self.state_ttl_s:
                dead_states.append(sid)

        for sid in dead_states:
            self.render_states.pop(sid, None)

        dead_tracks = []
        for pid, tm in self.local_track_memory.items():
            if (now - tm.get("updated_at", 0.0)) > self.track_memory_ttl_s:
                dead_tracks.append(pid)

        for pid in dead_tracks:
            self.local_track_memory.pop(pid, None)

    def update_render_state_from_decision(self, dec):
        pid = int(dec.get("person_track_id", -1))
        if pid < 0:
            return

        raw_score = dec.get("score", None)
        S = dec.get("S", None)
        gid = dec.get("global_person_id", None)

        suspicion = S if S is not None else raw_score
        suspicion = clamp01(0.0 if suspicion is None else suspicion)

        sid = self.state_id_from_decision(dec)
        prev = self.render_states.get(sid, None)

        if prev is None:
            display_s = suspicion
            last_bbox = None
            last_matched_pid = pid
        else:
            old = float(prev.get("display_s", suspicion))
            display_s = (1.0 - self.ema_alpha) * old + self.ema_alpha * suspicion
            last_bbox = prev.get("last_bbox", None)
            last_matched_pid = prev.get("last_matched_pid", pid)

        now = time.time()
        self.render_states[sid] = {
            "state_id": sid,
            "display_s": float(display_s),
            "raw_score": float(raw_score) if raw_score is not None else None,
            "S": float(S) if S is not None else None,
            "global_person_id": int(gid) if gid is not None else None,
            "pid_hint": int(pid),
            "event_id": dec.get("event_id"),
            "frame_id_end": int(dec.get("frame_id_end", -1)),
            "stamp_ns_end": int(dec.get("stamp_ns_end", 0)),
            "will_alert": bool(dec.get("will_alert", False)),
            "identity_enriched": bool(dec.get("identity_enriched", gid is not None)),
            "updated_at": now,
            "decision_updated_at": now,
            "last_rendered_at": prev.get("last_rendered_at") if isinstance(prev, dict) else None,
            "last_bbox": last_bbox,
            "last_matched_pid": last_matched_pid,
            "match_source": "decision_update",
        }

        self.local_track_memory[pid] = {
            "state_id": sid,
            "updated_at": now,
        }

    def is_state_fresh(self, st):
        now = time.time()
        t = st.get("decision_updated_at", st.get("updated_at", 0.0))
        return (now - t) <= self.decision_freshness_s

    def score_state_match(self, person_bbox, pid, st):
        score = -1e9
        reasons = []

        if not self.is_state_fresh(st):
            return score, reasons

        if st.get("pid_hint", None) == pid:
            score += 1000.0
            reasons.append("pid_hint")

        if st.get("last_matched_pid", None) == pid:
            score += 800.0
            reasons.append("last_matched_pid")

        last_bbox = st.get("last_bbox", None)
        iou = iou_xyxy(person_bbox, last_bbox) if last_bbox is not None else 0.0

        score += 25.0 * iou

        if iou >= self.iou_match_thr:
            reasons.append(f"iou_strong:{iou:.3f}")
        elif iou >= self.iou_loose_thr:
            reasons.append(f"iou_loose:{iou:.3f}")

        age = max(0.0, time.time() - float(st.get("decision_updated_at", st.get("updated_at", time.time()))))
        freshness_bonus = max(0.0, 2.0 - age)
        score += freshness_bonus
        reasons.append(f"fresh:{freshness_bonus:.2f}")

        return score, reasons

    def match_person_to_state(self, person):
        pid = int(person.get("track_id", -1))
        pb = person.get("bbox_xyxy", None)

        if pid < 0 or pb is None:
            return None, "none"

        mem = self.local_track_memory.get(pid, None)
        if mem is not None:
            sid = mem.get("state_id")
            st = self.render_states.get(sid)

            if st is not None and self.is_state_fresh(st):
                iou = iou_xyxy(pb, st.get("last_bbox", None))
                if st.get("pid_hint") == pid or st.get("last_matched_pid") == pid or iou >= self.iou_loose_thr:
                    return sid, "track_memory"

        for sid, st in self.render_states.items():
            if not self.is_state_fresh(st):
                continue
            if st.get("pid_hint", None) == pid:
                return sid, "pid_hint"

        best_sid = None
        best_score = -1e9
        best_iou = 0.0

        for sid, st in self.render_states.items():
            sc, _ = self.score_state_match(pb, pid, st)
            iou = iou_xyxy(pb, st.get("last_bbox", None))
            if sc > best_score:
                best_score = sc
                best_sid = sid
                best_iou = iou

        if best_sid is not None and best_iou >= self.iou_match_thr:
            return best_sid, "iou_fallback"

        return None, "unmatched"

    def ingest_video_msg(self, header: dict, enc: bytes):
        try:
            fid = int(header.get("frame_id", 0))
        except Exception:
            fid = 0

        if enc is None or len(enc) == 0:
            if fid > 0 and fid % 50 == 0:
                print(f"[bbox_overlay] cam={self.cam_id} empty video_ui payload fid={fid}")
            return

        arr = np.frombuffer(enc, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            if fid > 0 and fid % 50 == 0:
                print(f"[bbox_overlay] cam={self.cam_id} jpeg decode failed fid={fid} bytes={len(enc)}")
            return

        if fid > 0:
            stamp_ns = int(header.get("stamp_ns", time.time_ns()))
            self.put_buf(self.frame_buf, self.frame_fifo, fid, (stamp_ns, frame))

            if fid % 50 == 0:
                print(f"[bbox_overlay] cam={self.cam_id} buffered video_ui fid={fid} shape={frame.shape}")

    def ingest_pose_msg(self, pose: dict):
        try:
            fid = int(pose.get("frame_id", 0))
            if fid > 0:
                self.put_buf(self.pose_buf, self.pose_fifo, fid, pose)

                if fid % 50 == 0:
                    n_people = len(pose.get("people", []))
                    print(f"[bbox_overlay] cam={self.cam_id} buffered pose fid={fid} people={n_people}")

        except Exception as e:
            print(f"[bbox_overlay] cam={self.cam_id} pose ingest error: {e}")

    def ingest_decision_msg(self, dec: dict):
        try:
            self.update_render_state_from_decision(dec)
        except Exception as e:
            print(f"[bbox_overlay] cam={self.cam_id} decision ingest error: {e}")


    def _cache_rendered_frame(self, frame_fid: int, stamp_ns: int, jpg_bytes: bytes, meta: dict, detections: list):
        if not self.alert_clip_enabled:
            return
        if not jpg_bytes:
            return

        try:
            item = {
                "frame_id": int(frame_fid),
                "stamp_ns": int(stamp_ns),
                "jpg": bytes(jpg_bytes),
                "meta": dict(meta or {}),
                "detections": list(detections or []),
                "cached_at_s": time.time(),
            }
            self.rendered_frame_fifo.append(item)
        except Exception as e:
            print(f"[bbox_overlay] cam={self.cam_id} rendered frame cache error: {e}")

    def _iter_alert_clip_files(self):
        if not self.alert_clip_out_dir.exists():
            return []

        files = []
        for path in self.alert_clip_out_dir.glob("*"):
            try:
                if not path.is_file():
                    continue
                name = path.name
                if not (name.endswith("_overlay.mp4") or name.endswith(".tmp.mp4")):
                    continue
                st = path.stat()
                files.append((path, st.st_mtime, st.st_size))
            except FileNotFoundError:
                continue
            except Exception:
                continue
        return files

    def _cleanup_alert_clip_files(self, force=False):
        now = time.time()
        if not force and (now - self._last_alert_clip_cleanup_s) < self.alert_clip_cleanup_interval_s:
            return
        self._last_alert_clip_cleanup_s = now

        files = self._iter_alert_clip_files()
        if not files:
            if force:
                print(
                    f"[bbox_overlay] cam={self.cam_id} alert clip cleanup "
                    f"dir={self.alert_clip_out_dir} files=0 size_mb=0.0 removed=0"
                )
            return

        removed = 0
        removed_bytes = 0
        retention_s = max(0.0, self.alert_clip_retention_hours * 3600.0)

        kept = []
        for path, mtime, size in files:
            age_s = now - mtime
            remove_for_age = retention_s > 0 and age_s > retention_s
            remove_tmp = path.name.endswith(".tmp.mp4") and age_s > 600.0
            if remove_for_age or remove_tmp:
                try:
                    path.unlink()
                    removed += 1
                    removed_bytes += int(size)
                    continue
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"[bbox_overlay] cam={self.cam_id} alert clip cleanup unlink failed path={path}: {e}")
            kept.append((path, mtime, size))

        max_total_bytes = int(max(0.0, self.alert_clip_max_total_mb) * 1024 * 1024)
        total_bytes = sum(int(size) for _path, _mtime, size in kept)
        if max_total_bytes > 0 and total_bytes > max_total_bytes:
            for path, _mtime, size in sorted(kept, key=lambda x: x[1]):
                if total_bytes <= max_total_bytes:
                    break
                try:
                    path.unlink()
                    total_bytes -= int(size)
                    removed += 1
                    removed_bytes += int(size)
                except FileNotFoundError:
                    total_bytes -= int(size)
                except Exception as e:
                    print(f"[bbox_overlay] cam={self.cam_id} alert clip cleanup size unlink failed path={path}: {e}")

        current_files = self._iter_alert_clip_files()
        current_bytes = sum(int(size) for _path, _mtime, size in current_files)
        print(
            f"[bbox_overlay] cam={self.cam_id} alert clip cleanup "
            f"dir={self.alert_clip_out_dir} files={len(current_files)} "
            f"size_mb={current_bytes / (1024 * 1024):.1f} "
            f"removed={removed} removed_mb={removed_bytes / (1024 * 1024):.1f}"
        )

    def _maybe_cleanup_alert_clip_files(self):
        self._cleanup_alert_clip_files(force=False)

    def _select_alert_clip_frames(self, alert: dict):
        frames = list(self.rendered_frame_fifo)
        if not frames:
            return []

        target_fid = safe_int(
            alert.get("frame_id_end", alert.get("frame_id", alert.get("frameId", -1))),
            -1,
        )

        if target_fid > 0:
            best_i = min(
                range(len(frames)),
                key=lambda i: abs(int(frames[i].get("frame_id", -1)) - target_fid),
            )
        else:
            best_i = len(frames) - 1

        start_i = max(0, best_i - max(0, int(self.alert_clip_pre_frames)))
        end_i = min(len(frames), best_i + max(0, int(self.alert_clip_post_frames)) + 1)

        selected = frames[start_i:end_i]

        if len(selected) > int(self.alert_clip_max_frames):
            selected = selected[-int(self.alert_clip_max_frames):]

        return selected

    def build_alert_clip_ref(self, alert: dict):
        """
        Build an annotated MP4 clip for an alert from the already-rendered overlay frame cache.

        This does not run for every frame. It only writes to disk when ui_pipeline calls it
        after receiving a policy alert.
        """
        if not self.alert_clip_enabled:
            return None
        if not isinstance(alert, dict):
            return None

        frames = self._select_alert_clip_frames(alert)
        if not frames:
            print(
                f"[bbox_overlay] cam={self.cam_id} no rendered frames available for alert "
                f"event_id={alert.get('event_id')}"
            )
            return None

        try:
            self.alert_clip_out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[bbox_overlay] cam={self.cam_id} could not create alert clip dir: {e}")
            return None

        event_id = alert_event_id(alert, self.cam_id)
        person_track_id = safe_int(alert.get("person_track_id", alert.get("trackId", -1)), -1)
        global_person_id = alert.get("global_person_id")
        frame_id_end = safe_int(alert.get("frame_id_end", alert.get("frame_id", -1)), -1)
        stamp_ns_end = safe_int(alert.get("stamp_ns_end", alert.get("stamp_ns", time.time_ns())), time.time_ns())

        safe_event = re_safe_filename(event_id)
        out_path = self.alert_clip_out_dir / f"{self.cam_id}_{safe_event}_overlay.mp4"

        decoded_frames = []
        width = None
        height = None

        for item in frames:
            jpg = item.get("jpg")
            if not jpg:
                continue
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            if width is None:
                width, height = int(w), int(h)
            elif int(w) != width or int(h) != height:
                frame = cv2.resize(frame, (width, height))
            decoded_frames.append(frame)

        if not decoded_frames or width is None or height is None:
            print(f"[bbox_overlay] cam={self.cam_id} failed to decode frames for alert clip event_id={event_id}")
            return None

        tmp_path = out_path.with_suffix(".tmp.mp4")
        writer = None

        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(tmp_path), fourcc, float(self.alert_clip_fps), (width, height))

            if not writer.isOpened():
                print(f"[bbox_overlay] cam={self.cam_id} VideoWriter open failed path={tmp_path}")
                return None

            for frame in decoded_frames:
                writer.write(frame)

            writer.release()
            writer = None

            tmp_path.replace(out_path)

            self.alert_clip_write_count += 1

            first_frame_id = int(frames[0].get("frame_id", -1))
            last_frame_id = int(frames[-1].get("frame_id", -1))

            clip_ref = {
                "type": "clip_ref",
                "event_id": event_id,
                "cam_id": self.cam_id,
                "cameraId": self.cam_id,
                "person_track_id": person_track_id,
                "trackId": person_track_id,
                "global_person_id": global_person_id,

                "frame_id_start": first_frame_id,
                "frame_id_end": frame_id_end if frame_id_end > 0 else last_frame_id,
                "rendered_frame_id_start": first_frame_id,
                "rendered_frame_id_end": last_frame_id,
                "stamp_ns_end": stamp_ns_end,

                "clipPath": str(out_path),
                "local_clip_path": str(out_path),
                "clip_filename": out_path.name,
                "clipUrl": "",
                "clip_url": "",
                "storage_status": "local_ready",
                "format": "mp4",
                "codec": "mp4v",
                "annotated": True,
                "bbox_overlay": True,
                "source": "bbox_overlay",
                "num_frames": len(decoded_frames),
                "fps": float(self.alert_clip_fps),
                "width": width,
                "height": height,

                "score": alert.get("score"),
                "score_fused": alert.get("score_fused"),
                "score_cnn": alert.get("score_cnn"),
                "score_mlp": alert.get("score_mlp"),
                "suspicion": alert.get("suspicion", alert.get("score")),
                "object_track_id": alert.get("object_track_id", -1),
                "object_class_id": alert.get("object_class_id", -1),
                "model_version": alert.get("model_version", "unknown"),
                "created_at_s": time.time(),
            }

            print(
                f"[bbox_overlay] cam={self.cam_id} wrote alert overlay clip "
                f"event_id={event_id} frames={len(decoded_frames)} path={out_path}"
            )

            self._maybe_cleanup_alert_clip_files()
            return clip_ref

        except Exception as e:
            print(f"[bbox_overlay] cam={self.cam_id} alert clip write failed event_id={event_id}: {e}")
            return None

        finally:
            try:
                if writer is not None:
                    writer.release()
            except Exception:
                pass
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    def _publish_live(self, meta, jpg_bytes, frame_fid):
        if not self.live_publish_enabled or self.pub is None:
            return False

        try:
            self.pub.send_multipart(
                [
                    self.live_topic.encode("utf-8"),
                    json.dumps(meta).encode("utf-8"),
                    jpg_bytes,
                ],
                flags=zmq.NOBLOCK,
            )

            self.live_publish_count += 1
            self._rate_limited_info_log(
                "_last_publish_log_t",
                f"live publish ok frame={frame_fid} total_live_publishes={self.live_publish_count}",
                every_s=2.0,
            )
            return True

        except zmq.Again:
            print(f"[bbox_overlay] cam={self.cam_id} live publish dropped frame={frame_fid} HWM")
            return False

        except Exception as e:
            print(f"[bbox_overlay] cam={self.cam_id} live publish error frame={frame_fid}: {e}")
            return False

    def _publish_detections(self, payload):
        if self.det_pub is None:
            return False

        try:
            self.det_pub.send_multipart(
                [
                    self.det_topic.encode("utf-8"),
                    json.dumps(payload).encode("utf-8"),
                ],
                flags=zmq.NOBLOCK,
            )

            self.detection_publish_count += 1
            self._rate_limited_info_log(
                "_last_det_publish_log_t",
                f"detections publish ok total={self.detection_publish_count} topic={self.det_topic}",
                every_s=2.0,
            )
            return True

        except zmq.Again:
            print(f"[bbox_overlay] cam={self.cam_id} detections publish dropped HWM")
            return False

        except Exception as e:
            print(f"[bbox_overlay] cam={self.cam_id} detections publish error: {e}")
            return False

    def maybe_send_overlay(self):
        self.prune_state()
        self._maybe_cleanup_alert_clip_files()

        if not self.frame_buf or not self.pose_buf:
            return False

        frame_fid, frame_item, pose_fid, pose = self.find_latest_aligned_pair()

        if frame_item is None or pose is None:
            return False

        if int(frame_fid) <= int(self.last_rendered_frame_fid):
            return False

        self.last_rendered_frame_fid = int(frame_fid)

        _, frame = frame_item
        out = frame.copy()

        people = pose.get("people", [])
        detections = []

        for person in people:
            pid = int(person.get("track_id", -1))
            if pid < 0:
                continue

            pb = person.get("bbox_xyxy", None)
            if not pb:
                continue

            sid, match_source = self.match_person_to_state(person)
            st = self.render_states.get(sid) if sid is not None else None

            display_s = float(st.get("display_s", 0.0)) if st else 0.0
            raw_score = st.get("raw_score", None) if st else None
            gid = st.get("global_person_id", None) if st else None

            color = suspicion_to_color(display_s)

            x1, y1, x2, y2 = [int(v) for v in pb]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            gid_txt = "na" if gid is None else str(int(gid))
            disp_txt = f"id:{pid} gid:{gid_txt} sus:{display_s:.2f}"

            cv2.putText(
                out,
                disp_txt,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2,
            )

            detections.append({
                "type": "bbox_detections",
                "event_id": st.get("event_id") if st else None,
                "cam_id": self.cam_id,
                "person_track_id": pid,
                "global_person_id": gid,
                "frame_id": int(frame_fid),
                "stamp_ns": int(st.get("stamp_ns_end", 0)) if st else int(time.time_ns()),
                "bbox_xyxy": pb,
                "frame_w": int(out.shape[1]),
                "frame_h": int(out.shape[0]),
                "confidence": float(person.get("conf", raw_score or 0.0)),
                "score": float(raw_score) if raw_score is not None else None,
                "suspicion": float(display_s),
                "label": "person",
                "match_source": match_source,
            })

        ok, jpg = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return False

        jpg_bytes = jpg.tobytes()

        meta = {
            "cam_id": self.cam_id,
            "frame_id": int(frame_fid),
            "people_count": len(people),
        }

        frame_stamp_ns = int(frame_item[0]) if isinstance(frame_item, tuple) and len(frame_item) >= 1 else int(time.time_ns())
        self._cache_rendered_frame(
            int(frame_fid),
            frame_stamp_ns,
            jpg_bytes,
            meta,
            detections,
        )

        live_ok = self._publish_live(meta, jpg_bytes, frame_fid)

        if self.webhook_enabled and self.overlay_url:
            post_overlay_with_retry(
                self.overlay_url,
                meta,
                jpg_bytes,
                self.timeout_s,
                retries=2,
            )

        det_ok = False
        if len(detections) > 0:
            payload = {
                "cameraId": self.cam_id,
                "frameId": int(frame_fid),
                "stampNs": int(time.time_ns()),
                "frameWidth": int(out.shape[1]),
                "frameHeight": int(out.shape[0]),
                "detections": detections,
            }

            det_ok = self._publish_detections(payload)

        return live_ok or det_ok or self.webhook_enabled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cam_id", required=True)
    ap.add_argument("--frame_buffer", type=int, default=120)
    ap.add_argument("--pose_buffer", type=int, default=240)

    ap.add_argument("--ema_alpha", type=float, default=0.30)
    ap.add_argument("--state_ttl_s", type=float, default=4.0)
    ap.add_argument("--decision_freshness_s", type=float, default=4.0)
    ap.add_argument("--track_memory_ttl_s", type=float, default=4.0)
    ap.add_argument("--iou_match_thr", type=float, default=0.20)
    ap.add_argument("--iou_loose_thr", type=float, default=0.08)

    ap.add_argument("--send_every_n_frames", type=int, default=1)
    ap.add_argument("--pose_match_max_delta", type=int, default=8)
    ap.add_argument("--poll_ms", type=int, default=10)

    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cam_id = str(args.cam_id)

    overlay_cfg = cfg.get("live_view", {}).get("overlay", {})
    args.pose_match_max_delta = int(overlay_cfg.get("pose_match_max_delta", args.pose_match_max_delta))
    args.frame_buffer = int(overlay_cfg.get("frame_buffer", args.frame_buffer))
    args.pose_buffer = int(overlay_cfg.get("pose_buffer", args.pose_buffer))
    args.ema_alpha = float(overlay_cfg.get("ema_alpha", args.ema_alpha))
    args.state_ttl_s = float(overlay_cfg.get("state_ttl_s", args.state_ttl_s))
    args.decision_freshness_s = float(overlay_cfg.get("decision_freshness_s", args.decision_freshness_s))
    args.track_memory_ttl_s = float(overlay_cfg.get("track_memory_ttl_s", args.track_memory_ttl_s))
    args.iou_match_thr = float(overlay_cfg.get("iou_match_thr", args.iou_match_thr))
    args.iou_loose_thr = float(overlay_cfg.get("iou_loose_thr", args.iou_loose_thr))
    args.send_every_n_frames = int(overlay_cfg.get("send_every_n_frames", args.send_every_n_frames))
    args.poll_ms = int(overlay_cfg.get("poll_ms", args.poll_ms))

    video_cfg = resolve_video_ui_endpoint(cfg, cam_id)
    pose_cfg = get_zmq_endpoint(cfg, cam_id, "pose_features")
    dec_cfg = resolve_decisions_endpoint(cfg, cam_id)

    ctx = zmq.Context.instance()

    video_connect = local_connect_addr(video_cfg["bind"])
    video_topic = video_cfg["topic"]

    pose_connect = local_connect_addr(pose_cfg["bind"])
    pose_topic = pose_cfg["topic"]

    dec_connect = local_connect_addr(dec_cfg["bind"])
    dec_topic = dec_cfg["topic"]

    sub_v = make_sub_socket(
        ctx,
        video_connect,
        video_topic,
        rcvhwm=int(video_cfg.get("rcvhwm", 2)),
        latest_only=bool(video_cfg.get("latest_only", False)),
    )

    sub_p = make_sub_socket(
        ctx,
        pose_connect,
        pose_topic,
        rcvhwm=int(pose_cfg.get("rcvhwm", 2)),
        latest_only=bool(pose_cfg.get("latest_only", False)),
    )

    sub_d = make_sub_socket(
        ctx,
        dec_connect,
        dec_topic,
        rcvhwm=int(dec_cfg.get("rcvhwm", 1000)),
        latest_only=bool(dec_cfg.get("latest_only", False)),
    )

    poller = zmq.Poller()
    poller.register(sub_v, zmq.POLLIN)
    poller.register(sub_p, zmq.POLLIN)
    poller.register(sub_d, zmq.POLLIN)

    stage = BBoxOverlayStage(
        cfg,
        cam_id,
        frame_buffer=args.frame_buffer,
        pose_buffer=args.pose_buffer,
        ema_alpha=args.ema_alpha,
        state_ttl_s=args.state_ttl_s,
        decision_freshness_s=args.decision_freshness_s,
        track_memory_ttl_s=args.track_memory_ttl_s,
        iou_match_thr=args.iou_match_thr,
        iou_loose_thr=args.iou_loose_thr,
        send_every_n_frames=args.send_every_n_frames,
        pose_match_max_delta=args.pose_match_max_delta,
    )

    print(f"[bbox_overlay] SUB video_ui   {video_connect} topic={video_topic}")
    print(f"[bbox_overlay] SUB pose       {pose_connect} topic={pose_topic}")
    print(f"[bbox_overlay] SUB decisions  {dec_connect} topic={dec_topic}")

    try:
        while True:
            events = dict(poller.poll(timeout=args.poll_ms))

            if sub_v in events:
                try:
                    _, header_b, enc = sub_v.recv_multipart()
                    try:
                        header = json.loads(header_b.decode("utf-8"))
                    except Exception:
                        header = {}
                    stage.ingest_video_msg(header, enc)
                except Exception as e:
                    print(f"[bbox_overlay] cam={cam_id} video_ui recv error: {e}")

            if sub_p in events:
                try:
                    _, _, payload_b = sub_p.recv_multipart()
                    try:
                        pose = json.loads(payload_b.decode("utf-8"))
                        stage.ingest_pose_msg(pose)
                    except Exception as e:
                        print(f"[bbox_overlay] cam={cam_id} pose recv parse error: {e}")
                except Exception as e:
                    print(f"[bbox_overlay] cam={cam_id} pose recv error: {e}")

            if sub_d in events:
                try:
                    _, _, payload_b = sub_d.recv_multipart()
                    try:
                        dec = json.loads(payload_b.decode("utf-8"))
                        stage.ingest_decision_msg(dec)
                    except Exception as e:
                        print(f"[bbox_overlay] cam={cam_id} decision recv parse error: {e}")
                except Exception as e:
                    print(f"[bbox_overlay] cam={cam_id} decision recv error: {e}")

            stage.maybe_send_overlay()

    except KeyboardInterrupt:
        print("\n[bbox_overlay] stopping...")
    finally:
        stage.close()
        for s in [sub_v, sub_p, sub_d]:
            try:
                s.close(0)
            except Exception:
                pass


if __name__ == "__main__":
    main()
