#!/usr/bin/env python3
"""
reid_node.py

In-process per-camera ReID stage.

Responsibilities:
- subscribe to video / pose_features / seg_features for one camera
- decode frames
- generate ReID embeddings from local tracked people
- publish reid_embeddings:{cam}
- return produced embedding objects to unified pipeline

Updated:
- OSNet-only ReID via torchreid
- removed ResNet fallback
- fixed unbounded buffer memory leak caused by deque(maxlen=...) + dict mismatch
- config-driven bounded frame/pose/seg buffers
- torch.inference_mode() for embedding inference
- safer cleanup of decoded frame buffers
"""

import gc
import json
import queue
import time
from collections import deque
from typing import Optional, Dict, Tuple, List

import zmq
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import transforms

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from config_utils import get_zmq_endpoint, local_connect_addr

Gst.init(None)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < eps:
        return x
    return x / n


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def safe_bool(v, default=False):
    if isinstance(v, bool):
        return bool(v)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def polygon_to_mask(poly_xy, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if not poly_xy or len(poly_xy) < 3:
        return mask
    pts = np.array(poly_xy, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


class GstDecoder:
    def __init__(self, codec="h264", max_queue=2):
        codec = str(codec).lower()
        if codec == "h265":
            parse = "h265parse"
            dec = "avdec_h265"
            caps = "video/x-h265,stream-format=byte-stream,alignment=au"
        else:
            parse = "h264parse"
            dec = "avdec_h264"
            caps = "video/x-h264,stream-format=byte-stream,alignment=au"

        pipeline_str = f"""
            appsrc name=src is-live=true format=time block=false caps="{caps}" !
            {parse} !
            {dec} !
            videoconvert !
            video/x-raw,format=BGR !
            appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true enable-last-sample=false
        """

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsrc = self.pipeline.get_by_name("src")
        self.appsink = self.pipeline.get_by_name("sink")
        self._q = queue.Queue(maxsize=max_queue)
        self._closed = False

        if self.appsrc is None or self.appsink is None:
            raise RuntimeError("GstDecoder failed to create appsrc/appsink")

        self.appsink.connect("new-sample", self._on_sample)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GstDecoder failed to enter PLAYING state")

    def _on_sample(self, sink):
        if self._closed:
            return Gst.FlowReturn.OK

        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        caps = sample.get_caps()
        if caps is None or caps.get_size() == 0:
            return Gst.FlowReturn.OK

        s = caps.get_structure(0)
        w = int(s.get_value("width"))
        h = int(s.get_value("height"))

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK

        try:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3))
            frame = frame.copy()

            try:
                self._q.put_nowait(frame)
            except queue.Full:
                try:
                    old = self._q.get_nowait()
                    del old
                except queue.Empty:
                    pass

                try:
                    self._q.put_nowait(frame)
                except queue.Full:
                    pass
        finally:
            buf.unmap(mapinfo)

        return Gst.FlowReturn.OK

    def push(self, encoded: bytes):
        if self._closed:
            return

        gstbuf = Gst.Buffer.new_allocate(None, len(encoded), None)
        gstbuf.fill(0, encoded)
        gstbuf.pts = Gst.CLOCK_TIME_NONE
        gstbuf.dts = Gst.CLOCK_TIME_NONE

        ret = self.appsrc.emit("push-buffer", gstbuf)
        if ret != Gst.FlowReturn.OK:
            raise RuntimeError(f"GstDecoder push-buffer failed: {ret}")

    def pop(self, timeout=0.05):
        if self._closed:
            return None
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self._closed = True

        try:
            while True:
                old = self._q.get_nowait()
                del old
        except queue.Empty:
            pass

        try:
            self.appsrc.emit("end-of-stream")
        except Exception:
            pass

        try:
            self.pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass


class OSNetReID(nn.Module):
    """
    OSNet person-ReID model using torchreid.

    Config:
      reid:
        model_type: "osnet"
        model_name: "osnet_x1_0"
        pretrained: true
    """

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        pretrained: bool = True,
    ):
        super().__init__()

        try:
            import torchreid
        except Exception as e:
            raise RuntimeError(
                "torchreid is required for OSNet. Install with: pip install torchreid"
            ) from e

        self.backbone = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=pretrained,
        )

    def forward(self, x):
        z = self.backbone(x)

        if isinstance(z, (tuple, list)):
            z = z[0]

        z = torch.nn.functional.normalize(z, dim=1)
        return z


def build_preprocess():
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def make_sub_socket(ctx, connect_addr, topic, rcvhwm=1000, latest_only=False):
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.RCVHWM, int(rcvhwm))
    if latest_only:
        sub.setsockopt(zmq.CONFLATE, 1)
    sub.connect(connect_addr)
    sub.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
    return sub


def compute_crop_box(frame_h: int, frame_w: int, bbox_xyxy, pad_frac: float = 0.04):
    if bbox_xyxy is None or len(bbox_xyxy) != 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    padx = pad_frac * bw
    pady = pad_frac * bh

    x1 = int(clamp(round(x1 - padx), 0, frame_w - 1))
    y1 = int(clamp(round(y1 - pady), 0, frame_h - 1))
    x2 = int(clamp(round(x2 + padx), 0, frame_w - 1))
    y2 = int(clamp(round(y2 + pady), 0, frame_h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def crop_person(frame: np.ndarray, crop_box) -> Optional[np.ndarray]:
    if frame is None or crop_box is None:
        return None

    x1, y1, x2, y2 = crop_box
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


def apply_seg_mask_to_crop(
    crop_bgr: np.ndarray,
    crop_box,
    seg_instances: List[dict],
    person_track_id: int,
    frame_h: int,
    frame_w: int,
) -> np.ndarray:
    if crop_bgr is None or crop_bgr.size == 0 or crop_box is None:
        return crop_bgr

    match_poly = None

    for inst in seg_instances or []:
        try:
            tid = int(inst.get("track_id", -1))
            cid = int(inst.get("class_id", -1))
        except Exception:
            continue

        if tid == person_track_id and cid == 0:
            poly = inst.get("mask_poly_xy", None)
            if poly and len(poly) >= 3:
                match_poly = poly
                break

    if match_poly is None:
        return crop_bgr

    full_mask = polygon_to_mask(match_poly, frame_h, frame_w)

    x1, y1, x2, y2 = crop_box
    crop_mask = full_mask[y1:y2, x1:x2]

    if crop_mask.shape[:2] != crop_bgr.shape[:2]:
        return crop_bgr

    masked = crop_bgr.copy()
    masked[crop_mask == 0] = 0
    return masked


def crop_quality_ok(crop: np.ndarray, min_nonzero_ratio: float = 0.08) -> bool:
    if crop is None or crop.size == 0:
        return False

    h, w = crop.shape[:2]
    if h < 2 or w < 2:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    nz = float(np.count_nonzero(gray > 4))
    ratio = nz / max(1.0, float(h * w))
    return ratio >= min_nonzero_ratio


class ReIDNode:
    def __init__(
        self,
        cfg: dict,
        cam_id: str,
        publisher,
        device: str = "cuda:0",
        emb_dim: int = 512,
        latest_only: bool = False,
        max_frame_delta: int = 3,
        use_seg_mask: bool = False,
        min_bbox_h: int = 80,
        min_bbox_w: int = 30,
        pad_frac: float = 0.05,
        use_flip_test: bool = True,
        max_pose_backlog_per_step: int = 6,
        drop_old_pose_backlog: bool = True,
        debug_reid: bool = False,
    ):
        self.cfg = cfg
        self.cam_id = str(cam_id)
        self.publisher = publisher

        r_cfg = cfg.get("reid", {}) if isinstance(cfg, dict) else {}
        force_cfg = safe_bool(r_cfg.get("force_config_values", True), True)

        self.model_type = str(r_cfg.get("model_type", "osnet")).lower()
        self.model_name = str(r_cfg.get("model_name", "osnet_x1_0"))
        self.pretrained = safe_bool(r_cfg.get("pretrained", True), True)
        self.health_log_every_s = float(r_cfg.get("health_log_every_s", 10.0))

        if force_cfg:
            device = r_cfg.get("device", device)
            emb_dim = int(r_cfg.get("emb_dim", emb_dim))
            latest_only = safe_bool(r_cfg.get("latest_only", latest_only), latest_only)
            max_frame_delta = int(r_cfg.get("max_frame_delta", max_frame_delta))
            use_seg_mask = safe_bool(r_cfg.get("use_seg_mask", use_seg_mask), use_seg_mask)
            min_bbox_h = int(r_cfg.get("min_bbox_h", min_bbox_h))
            min_bbox_w = int(r_cfg.get("min_bbox_w", min_bbox_w))
            pad_frac = float(r_cfg.get("pad_frac", pad_frac))
            use_flip_test = safe_bool(r_cfg.get("use_flip_test", use_flip_test), use_flip_test)
            max_pose_backlog_per_step = int(r_cfg.get("max_pose_backlog_per_step", max_pose_backlog_per_step))
            drop_old_pose_backlog = safe_bool(
                r_cfg.get("drop_old_pose_backlog", drop_old_pose_backlog),
                drop_old_pose_backlog,
            )
            debug_reid = safe_bool(r_cfg.get("debug_reid", debug_reid), debug_reid)

        self.latest_only = bool(latest_only)
        self.max_frame_delta = int(max_frame_delta)
        self.use_seg_mask = bool(use_seg_mask)
        self.min_bbox_h = int(min_bbox_h)
        self.min_bbox_w = int(min_bbox_w)
        self.pad_frac = float(pad_frac)
        self.use_flip_test = bool(use_flip_test)
        self.max_pose_backlog_per_step = int(max_pose_backlog_per_step)
        self.drop_old_pose_backlog = bool(drop_old_pose_backlog)
        self.debug_reid = bool(debug_reid)

        # Conservative bounded buffers.
        # These can be overridden from config:
        # reid:
        #   frame_buffer_max: 60
        #   pose_buffer_max: 120
        #   seg_buffer_max: 120
        #   processed_pose_max: 300
        self.frame_buffer_max = int(r_cfg.get("frame_buffer_max", 60))
        self.pose_buffer_max = int(r_cfg.get("pose_buffer_max", 120))
        self.seg_buffer_max = int(r_cfg.get("seg_buffer_max", 120))
        self.processed_pose_max = int(r_cfg.get("processed_pose_max", 300))
        self.gc_every_embeddings = int(r_cfg.get("gc_every_embeddings", 250))

        cam_cfg = cfg["cams"][self.cam_id]
        codec = cam_cfg.get("rtsp", {}).get("codec", "h264")

        video_cfg = get_zmq_endpoint(cfg, self.cam_id, "video")
        pose_cfg = get_zmq_endpoint(cfg, self.cam_id, "pose_features")
        seg_cfg = get_zmq_endpoint(cfg, self.cam_id, "seg_features")

        self.video_connect = local_connect_addr(video_cfg["bind"])
        self.video_topic = video_cfg["topic"]
        self.pose_connect = local_connect_addr(pose_cfg["bind"])
        self.pose_topic = pose_cfg["topic"]
        self.seg_connect = local_connect_addr(seg_cfg["bind"])
        self.seg_topic = seg_cfg["topic"]

        self.ctx = zmq.Context.instance()

        self.sub_v = make_sub_socket(
            self.ctx,
            self.video_connect,
            self.video_topic,
            latest_only=self.latest_only,
        )
        self.sub_p = make_sub_socket(
            self.ctx,
            self.pose_connect,
            self.pose_topic,
            latest_only=self.latest_only,
        )
        self.sub_s = make_sub_socket(
            self.ctx,
            self.seg_connect,
            self.seg_topic,
            latest_only=self.latest_only,
        )

        self.decoder = GstDecoder(codec=codec, max_queue=2)

        self.device = torch.device(
            device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
        )

        if self.model_type != "osnet":
            print(
                f"[reid_node] WARNING cam={self.cam_id}: model_type={self.model_type} requested, "
                f"but this build is OSNet-only. Falling back to OSNet."
            )

        self.model = OSNetReID(
            model_name=self.model_name,
            pretrained=self.pretrained,
        ).to(self.device)

        self.model.eval()
        self.preprocess = build_preprocess()

        self.frame_buf: Dict[int, Tuple[int, np.ndarray]] = {}
        self.pose_buf: Dict[int, dict] = {}
        self.seg_buf: Dict[int, dict] = {}

        self.frame_fifo = deque()
        self.pose_fifo = deque()
        self.seg_fifo = deque()

        self.processed_pose_fids = set()
        self.processed_pose_fifo = deque()

        self.stats = {
            "pose_frames_seen": 0,
            "people_seen": 0,
            "embeddings_produced": 0,
            "small_bbox": 0,
            "bad_bbox": 0,
            "bad_crop": 0,
            "bad_quality": 0,
            "last_health_log_s": time.time(),
        }

        print(f"[reid_node] cam_id={self.cam_id}")
        print(f"[reid_node] SUB video {self.video_connect} topic={self.video_topic}")
        print(f"[reid_node] SUB pose  {self.pose_connect} topic={self.pose_topic}")
        print(f"[reid_node] SUB seg   {self.seg_connect} topic={self.seg_topic}")
        print(
            f"[reid_node] device={self.device} model_type=osnet "
            f"model_name={self.model_name} pretrained={self.pretrained} "
            f"emb_dim={emb_dim} latest_only={self.latest_only} "
            f"use_seg_mask={self.use_seg_mask} flip_test={self.use_flip_test} "
            f"max_frame_delta={self.max_frame_delta} backlog={self.max_pose_backlog_per_step} "
            f"buffers(frame={self.frame_buffer_max}, pose={self.pose_buffer_max}, "
            f"seg={self.seg_buffer_max}, processed={self.processed_pose_max})"
        )

    def _maybe_log_health(self):
        now = time.time()
        if (now - self.stats["last_health_log_s"]) < self.health_log_every_s:
            return

        people = max(1, self.stats["people_seen"])
        emb_rate = 100.0 * self.stats["embeddings_produced"] / people

        print(
            f"[reid_node_health] cam={self.cam_id} "
            f"pose_frames={self.stats['pose_frames_seen']} "
            f"people_seen={self.stats['people_seen']} "
            f"embeddings={self.stats['embeddings_produced']} "
            f"embedding_rate={emb_rate:.1f}% "
            f"small_bbox={self.stats['small_bbox']} "
            f"bad_bbox={self.stats['bad_bbox']} "
            f"bad_crop={self.stats['bad_crop']} "
            f"bad_quality={self.stats['bad_quality']} "
            f"bufs(frame={len(self.frame_buf)}, pose={len(self.pose_buf)}, "
            f"seg={len(self.seg_buf)}, processed={len(self.processed_pose_fids)})"
        )

        self.stats["last_health_log_s"] = now

    def sockets(self):
        return [self.sub_v, self.sub_p, self.sub_s]

    def put_buf(self, buf, fifo, fid, obj, max_items: int):
        """
        Store an object in a bounded dict+fifo pair.

        Important:
        Do NOT use deque(maxlen=...) here, because deque(maxlen=...) silently drops
        the oldest key on append, while the paired dict keeps the old payload.
        That caused frame_buf / pose_buf / seg_buf to grow forever.
        """
        fid = int(fid)

        if fid in buf:
            buf[fid] = obj
            return

        while len(fifo) >= int(max_items):
            old = fifo.popleft()
            old_obj = buf.pop(old, None)
            del old_obj

        buf[fid] = obj
        fifo.append(fid)

    def find_nearest(self, buf, fid, max_delta=3):
        fid = int(fid)

        if fid in buf:
            return fid, buf[fid]

        for d in range(1, int(max_delta) + 1):
            if (fid - d) in buf:
                return fid - d, buf[fid - d]
            if (fid + d) in buf:
                return fid + d, buf[fid + d]

        return None, None

    def handle_socket_event(self, sock) -> List[dict]:
        produced = []

        if sock is self.sub_v:
            parts = self.sub_v.recv_multipart()
            if len(parts) != 3:
                return produced

            _, header_b, enc = parts

            try:
                header = json.loads(header_b.decode("utf-8"))
            except Exception:
                header = {}

            fid = int(header.get("frame_id", 0))
            stamp_ns = int(header.get("stamp_ns", time.time_ns()))

            try:
                self.decoder.push(enc)
                frame = self.decoder.pop(timeout=0.05)
            except Exception:
                frame = None

            if frame is not None and fid >= 0:
                self.put_buf(
                    self.frame_buf,
                    self.frame_fifo,
                    fid,
                    (stamp_ns, frame),
                    self.frame_buffer_max,
                )

        elif sock is self.sub_p:
            parts = self.sub_p.recv_multipart()
            if len(parts) != 3:
                return produced

            _, _, payload_b = parts

            try:
                pose = json.loads(payload_b.decode("utf-8"))
                fid = int(pose.get("frame_id", 0))
                if fid >= 0:
                    self.put_buf(
                        self.pose_buf,
                        self.pose_fifo,
                        fid,
                        pose,
                        self.pose_buffer_max,
                    )
            except Exception:
                pass

        elif sock is self.sub_s:
            parts = self.sub_s.recv_multipart()
            if len(parts) != 3:
                return produced

            _, _, payload_b = parts

            try:
                seg = json.loads(payload_b.decode("utf-8"))
                fid = int(seg.get("frame_id", 0))
                if fid >= 0:
                    self.put_buf(
                        self.seg_buf,
                        self.seg_fifo,
                        fid,
                        seg,
                        self.seg_buffer_max,
                    )
            except Exception:
                pass

        produced.extend(self._try_process_pose_backlog())
        return produced

    def _embed_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        x = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            if self.use_flip_test:
                x_flip = torch.flip(x, dims=[3])
                x_cat = torch.cat([x, x_flip], dim=0)
                z = self.model(x_cat)
                z_np = z.detach().cpu().numpy().astype(np.float32)
                emb = l2_normalize(np.mean(z_np, axis=0))

                del x_flip
                del x_cat
                del z
                del z_np
            else:
                z = self.model(x)
                emb = z[0].detach().cpu().numpy().astype(np.float32)
                emb = l2_normalize(emb)

                del z

        del x
        del pil_img
        del crop_rgb

        return emb

    def _mark_pose_processed(self, fid: int):
        fid = int(fid)

        if fid in self.processed_pose_fids:
            return

        while len(self.processed_pose_fifo) >= self.processed_pose_max:
            old = self.processed_pose_fifo.popleft()
            self.processed_pose_fids.discard(old)

        self.processed_pose_fids.add(fid)
        self.processed_pose_fifo.append(fid)

    def _process_one_pose_fid(self, pose_fid_target: int) -> List[dict]:
        produced = []

        pose_fid, pose = self.find_nearest(self.pose_buf, pose_fid_target, max_delta=0)

        frame_fid, frame_item = self.find_nearest(
            self.frame_buf,
            pose_fid_target,
            max_delta=self.max_frame_delta,
        )

        seg_fid, seg = self.find_nearest(
            self.seg_buf,
            pose_fid_target,
            max_delta=self.max_frame_delta,
        )

        if pose is None or frame_item is None:
            return produced

        stamp_ns_frame, frame = frame_item
        frame_h, frame_w = frame.shape[:2]

        seg_instances = seg.get("instances", []) if isinstance(seg, dict) else []
        people = pose.get("people", [])

        self.stats["pose_frames_seen"] += 1
        self.stats["people_seen"] += len(people)

        for person in people:
            crop = None
            emb = None

            try:
                try:
                    pid = int(person.get("track_id", -1))
                except Exception:
                    pid = -1

                if pid < 0:
                    continue

                bbox = person.get("bbox_xyxy", None)
                if not bbox or len(bbox) != 4:
                    self.stats["bad_bbox"] += 1
                    continue

                x1, y1, x2, y2 = [float(v) for v in bbox]
                bw = int(max(0.0, x2 - x1))
                bh = int(max(0.0, y2 - y1))

                if bh < self.min_bbox_h or bw < self.min_bbox_w:
                    self.stats["small_bbox"] += 1
                    continue

                crop_box = compute_crop_box(frame_h, frame_w, bbox, pad_frac=self.pad_frac)
                if crop_box is None:
                    self.stats["bad_bbox"] += 1
                    continue

                crop = crop_person(frame, crop_box)
                if crop is None:
                    self.stats["bad_crop"] += 1
                    continue

                if self.use_seg_mask:
                    crop = apply_seg_mask_to_crop(
                        crop_bgr=crop,
                        crop_box=crop_box,
                        seg_instances=seg_instances,
                        person_track_id=pid,
                        frame_h=frame_h,
                        frame_w=frame_w,
                    )

                if not crop_quality_ok(crop):
                    self.stats["bad_quality"] += 1
                    continue

                emb = self._embed_crop(crop)
                self.stats["embeddings_produced"] += 1

                event_id = f"{self.cam_id}:{pid}:{pose_fid}:{stamp_ns_frame}"

                match_keys = [
                    event_id,
                    f"{self.cam_id}:{pid}:{pose_fid}",
                ]

                if frame_fid is not None:
                    match_keys.append(f"{self.cam_id}:{pid}:{frame_fid}")

                out = {
                    "type": "reid_embedding",
                    "event_id": event_id,
                    "cam_id": self.cam_id,
                    "person_track_id": int(pid),
                    "frame_id": int(pose_fid),
                    "stamp_ns": int(stamp_ns_frame),
                    "embedding_dim": int(emb.shape[0]),
                    "embedding": emb.tolist(),
                    "bbox_xyxy": [float(v) for v in bbox],
                    "seg_frame_id": int(seg_fid) if seg_fid is not None else None,
                    "video_frame_id": int(frame_fid) if frame_fid is not None else None,
                    "match_keys": sorted(set(match_keys)),
                }

                self.publisher.publish_reid_embedding(self.cam_id, out)
                produced.append(out)

            finally:
                if crop is not None:
                    del crop
                if emb is not None:
                    del emb

        if self.gc_every_embeddings > 0:
            total_emb = int(self.stats["embeddings_produced"])
            if total_emb > 0 and total_emb % self.gc_every_embeddings == 0:
                gc.collect()

        self._maybe_log_health()
        return produced

    def _try_process_pose_backlog(self) -> List[dict]:
        produced = []

        if not self.pose_fifo or not self.frame_fifo:
            return produced

        pending = [
            int(fid)
            for fid in list(self.pose_fifo)
            if int(fid) not in self.processed_pose_fids
        ]

        if not pending:
            return produced

        pending = sorted(set(pending))

        if self.drop_old_pose_backlog and len(pending) > self.max_pose_backlog_per_step:
            dropped = pending[:-self.max_pose_backlog_per_step]
            pending = pending[-self.max_pose_backlog_per_step:]

            for fid in dropped:
                self._mark_pose_processed(fid)

            if self.debug_reid:
                print(
                    f"[reid_node] cam={self.cam_id} dropped_old_pose_backlog={len(dropped)} "
                    f"processing={pending}"
                )
        else:
            pending = pending[:self.max_pose_backlog_per_step]

        for fid in pending:
            out = self._process_one_pose_fid(fid)

            self._mark_pose_processed(fid)

            if out:
                produced.extend(out)

        if produced and self.debug_reid:
            print(
                f"[reid_node] cam={self.cam_id} produced_embeddings={len(produced)} "
                f"processed_pose_fids={pending}"
            )

        return produced

    def close(self):
        try:
            self.decoder.close()
        except Exception:
            pass

        self.frame_buf.clear()
        self.pose_buf.clear()
        self.seg_buf.clear()
        self.frame_fifo.clear()
        self.pose_fifo.clear()
        self.seg_fifo.clear()
        self.processed_pose_fids.clear()
        self.processed_pose_fifo.clear()

        for s in [self.sub_v, self.sub_p, self.sub_s]:
            try:
                s.close(0)
            except Exception:
                pass

        try:
            del self.model
        except Exception:
            pass

        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
