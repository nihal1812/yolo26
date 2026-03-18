#!/usr/bin/env python3
"""
reid_node.py

Consumes:
  - video ZMQ stream for one camera
  - pose_features ZMQ stream for one camera
  - optionally seg_features ZMQ stream for one camera

Produces:
  - reid_embeddings:{cam}

Purpose:
- generate person ReID embeddings directly from tracked people
- uses pose bboxes for crop extraction
- optionally uses segmentation mask to reduce background

Why this design:
- no need for reid_crops:{cam}
- integrates directly into your existing perception pipeline
"""

import argparse
import json
import queue
import time
from collections import deque
from typing import Optional, Dict, Tuple, List

import redis
import zmq
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import models, transforms

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from config_utils import load_cfg, get_stream, get_zmq_endpoint, local_connect_addr

Gst.init(None)


def b2s(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < eps:
        return x
    return x / n


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def polygon_to_mask(poly_xy, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if not poly_xy or len(poly_xy) < 3:
        return mask
    pts = np.array(poly_xy, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


class GstDecoder:
    def __init__(self, codec="h264", max_queue=4):
        codec = str(codec).lower()
        if codec == "h265":
            parse = "h265parse"
            dec = "avdec_h265"
        else:
            parse = "h264parse"
            dec = "avdec_h264"

        pipeline_str = f"""
            appsrc name=src is-live=true format=time do-timestamp=true !
            {parse} !
            {dec} !
            videoconvert !
            video/x-raw,format=BGR !
            appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
        """
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsrc = self.pipeline.get_by_name("src")
        self.appsink = self.pipeline.get_by_name("sink")
        self._q = queue.Queue(maxsize=max_queue)
        self._closed = False

        self.appsink.connect("new-sample", self._on_sample)
        self.pipeline.set_state(Gst.State.PLAYING)

    def _on_sample(self, sink):
        if self._closed:
            return Gst.FlowReturn.OK

        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        w = int(s.get_value("width"))
        h = int(s.get_value("height"))

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if ok:
            try:
                frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3))
                try:
                    self._q.put_nowait(frame.copy())
                except queue.Full:
                    try:
                        _ = self._q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._q.put_nowait(frame.copy())
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
        self.appsrc.emit("push-buffer", gstbuf)

    def pop(self, timeout=0.2):
        if self._closed:
            return None
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self._closed = True
        try:
            self.pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass


class ResNetReID(nn.Module):
    """
    Generic appearance embedding model using ResNet50 backbone.
    Replace later with OSNet / FastReID for better store performance.
    """
    def __init__(self, emb_dim: int = 512):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Linear(feat_dim, emb_dim)

    def forward(self, x):
        z = self.backbone(x)
        z = self.proj(z)
        z = torch.nn.functional.normalize(z, dim=1)
        return z


def build_preprocess():
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def make_sub_socket(ctx, connect_addr, topic, rcvhwm=1000, latest_only=False):
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, int(rcvhwm))
    if latest_only:
        sub.setsockopt(zmq.CONFLATE, 1)
    sub.connect(connect_addr)
    sub.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
    return sub


def crop_person(frame: np.ndarray, bbox_xyxy, pad_frac: float = 0.05) -> Optional[np.ndarray]:
    if frame is None or bbox_xyxy is None or len(bbox_xyxy) != 4:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    padx = pad_frac * bw
    pady = pad_frac * bh

    x1 = int(clamp(round(x1 - padx), 0, w - 1))
    y1 = int(clamp(round(y1 - pady), 0, h - 1))
    x2 = int(clamp(round(x2 + padx), 0, w - 1))
    y2 = int(clamp(round(y2 + pady), 0, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def apply_seg_mask_to_crop(
    crop_bgr: np.ndarray,
    person_bbox_xyxy,
    seg_instances: List[dict],
    person_track_id: int,
    frame_h: int,
    frame_w: int
) -> np.ndarray:
    """
    Try to find matching segmentation instance for this person by track_id and class_id==0.
    If found, apply foreground mask within crop.
    """
    if crop_bgr is None or crop_bgr.size == 0:
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
    x1, y1, x2, y2 = [int(v) for v in person_bbox_xyxy]
    x1 = clamp(x1, 0, frame_w - 1)
    x2 = clamp(x2, 0, frame_w - 1)
    y1 = clamp(y1, 0, frame_h - 1)
    y2 = clamp(y2, 0, frame_h - 1)

    if x2 <= x1 or y2 <= y1:
        return crop_bgr

    crop_mask = full_mask[y1:y2, x1:x2]
    if crop_mask.shape[:2] != crop_bgr.shape[:2]:
        return crop_bgr

    masked = crop_bgr.copy()
    masked[crop_mask == 0] = 0
    return masked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cam_id", required=True)

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--emb_dim", type=int, default=512)

    ap.add_argument("--block_ms", type=int, default=50)
    ap.add_argument("--redis_maxlen", type=int, default=50000)
    ap.add_argument("--latest_only", action="store_true")
    ap.add_argument("--max_frame_delta", type=int, default=3)

    ap.add_argument("--use_seg_mask", action="store_true")
    ap.add_argument("--min_bbox_h", type=int, default=80)
    ap.add_argument("--min_bbox_w", type=int, default=30)
    ap.add_argument("--pad_frac", type=float, default=0.05)

    ap.add_argument("--embeddings_stream", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cam_id = str(args.cam_id)

    cam_cfg = cfg["cams"][cam_id]
    codec = cam_cfg.get("rtsp", {}).get("codec", "h264")

    r_cfg = cfg.get("redis", {})
    rdb = redis.Redis(
        host=r_cfg.get("host", "127.0.0.1"),
        port=int(r_cfg.get("port", 6379)),
        db=int(r_cfg.get("db", 0)),
        password=r_cfg.get("password", None),
        decode_responses=False,
    )
    rdb.ping()

    try:
        embeddings_stream = args.embeddings_stream or get_stream(cfg, "reid_embeddings", cam_id)
    except Exception:
        embeddings_stream = f"reid_embeddings:{cam_id}"

    video_cfg = get_zmq_endpoint(cfg, cam_id, "video")
    pose_cfg = get_zmq_endpoint(cfg, cam_id, "pose_features")
    seg_cfg = get_zmq_endpoint(cfg, cam_id, "seg_features")

    video_connect = local_connect_addr(video_cfg["bind"])
    video_topic = video_cfg["topic"]

    pose_connect = local_connect_addr(pose_cfg["bind"])
    pose_topic = pose_cfg["topic"]

    seg_connect = local_connect_addr(seg_cfg["bind"])
    seg_topic = seg_cfg["topic"]

    ctx = zmq.Context.instance()

    sub_v = make_sub_socket(ctx, video_connect, video_topic, rcvhwm=1000, latest_only=args.latest_only)
    sub_p = make_sub_socket(ctx, pose_connect, pose_topic, rcvhwm=1000, latest_only=args.latest_only)
    sub_s = make_sub_socket(ctx, seg_connect, seg_topic, rcvhwm=1000, latest_only=args.latest_only)

    poller = zmq.Poller()
    poller.register(sub_v, zmq.POLLIN)
    poller.register(sub_p, zmq.POLLIN)
    poller.register(sub_s, zmq.POLLIN)

    decoder = GstDecoder(codec=codec, max_queue=4)

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    model = ResNetReID(emb_dim=args.emb_dim).to(device)
    model.eval()
    preprocess = build_preprocess()

    frame_buf: Dict[int, Tuple[int, np.ndarray]] = {}
    pose_buf: Dict[int, dict] = {}
    seg_buf: Dict[int, dict] = {}

    frame_fifo = deque(maxlen=120)
    pose_fifo = deque(maxlen=240)
    seg_fifo = deque(maxlen=240)

    print(f"[reid_node] cam_id={cam_id}")
    print(f"[reid_node] SUB video {video_connect} topic={video_topic}")
    print(f"[reid_node] SUB pose  {pose_connect} topic={pose_topic}")
    print(f"[reid_node] SUB seg   {seg_connect} topic={seg_topic}")
    print(f"[reid_node] embeddings_stream={embeddings_stream}")
    print(f"[reid_node] device={device} emb_dim={args.emb_dim} use_seg_mask={args.use_seg_mask}")

    def put_buf(buf, fifo, fid, obj):
        if fid in buf:
            return
        buf[fid] = obj
        fifo.append(fid)
        while len(fifo) > fifo.maxlen:
            old = fifo.popleft()
            buf.pop(old, None)

    def find_nearest(buf, fid, max_delta=3):
        if fid in buf:
            return fid, buf[fid]
        for d in range(1, max_delta + 1):
            if (fid - d) in buf:
                return fid - d, buf[fid - d]
            if (fid + d) in buf:
                return fid + d, buf[fid + d]
        return None, None

    with torch.no_grad():
        try:
            while True:
                events = dict(poller.poll(timeout=args.block_ms))

                if sub_v in events:
                    _, header_b, enc = sub_v.recv_multipart()
                    try:
                        header = json.loads(header_b.decode("utf-8"))
                    except Exception:
                        header = {}

                    fid = int(header.get("frame_id", 0))
                    stamp_ns = int(header.get("stamp_ns", time.time_ns()))

                    decoder.push(enc)
                    frame = decoder.pop(timeout=0.05)
                    if frame is not None and fid > 0:
                        put_buf(frame_buf, frame_fifo, fid, (stamp_ns, frame))

                if sub_p in events:
                    _, _, payload_b = sub_p.recv_multipart()
                    try:
                        pose = json.loads(payload_b.decode("utf-8"))
                        fid = int(pose.get("frame_id", 0))
                        if fid > 0:
                            put_buf(pose_buf, pose_fifo, fid, pose)
                    except Exception:
                        pass

                if sub_s in events:
                    _, _, payload_b = sub_s.recv_multipart()
                    try:
                        seg = json.loads(payload_b.decode("utf-8"))
                        fid = int(seg.get("frame_id", 0))
                        if fid > 0:
                            put_buf(seg_buf, seg_fifo, fid, seg)
                    except Exception:
                        pass

                if not pose_fifo or not frame_fifo:
                    continue

                latest_pose_fid = pose_fifo[-1]
                pose_fid, pose = find_nearest(pose_buf, latest_pose_fid, max_delta=0)
                frame_fid, frame_item = find_nearest(frame_buf, latest_pose_fid, max_delta=args.max_frame_delta)
                seg_fid, seg = find_nearest(seg_buf, latest_pose_fid, max_delta=args.max_frame_delta)

                if pose is None or frame_item is None:
                    continue

                stamp_ns_frame, frame = frame_item
                frame_h, frame_w = frame.shape[:2]
                seg_instances = seg.get("instances", []) if isinstance(seg, dict) else []

                people = pose.get("people", [])
                for person in people:
                    try:
                        pid = int(person.get("track_id", -1))
                    except Exception:
                        pid = -1
                    if pid < 0:
                        continue

                    bbox = person.get("bbox_xyxy", None)
                    if not bbox or len(bbox) != 4:
                        continue

                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    bw = int(max(0.0, x2 - x1))
                    bh = int(max(0.0, y2 - y1))

                    if bh < args.min_bbox_h or bw < args.min_bbox_w:
                        continue

                    crop = crop_person(frame, bbox, pad_frac=args.pad_frac)
                    if crop is None:
                        continue

                    if args.use_seg_mask:
                        crop = apply_seg_mask_to_crop(
                            crop_bgr=crop,
                            person_bbox_xyxy=bbox,
                            seg_instances=seg_instances,
                            person_track_id=pid,
                            frame_h=frame_h,
                            frame_w=frame_w,
                        )

                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(crop_rgb)

                    x = preprocess(pil_img).unsqueeze(0).to(device)
                    emb = model(x)[0].detach().cpu().numpy().astype(np.float32)
                    emb = l2_normalize(emb)

                    event_id = f"{cam_id}:{pid}:{pose_fid}:{stamp_ns_frame}"

                    out = {
                        "type": "reid_embedding",
                        "event_id": event_id,
                        "cam_id": cam_id,
                        "person_track_id": int(pid),
                        "frame_id": int(pose_fid),
                        "stamp_ns": int(stamp_ns_frame),
                        "embedding_dim": int(emb.shape[0]),
                        "embedding": emb.tolist(),
                        "bbox_xyxy": [float(v) for v in bbox],
                        "seg_frame_id": int(seg_fid) if seg_fid is not None else None,
                    }

                    rdb.xadd(
                        embeddings_stream,
                        {
                            "event_id": str(event_id),
                            "cam_id": str(cam_id),
                            "person_track_id": str(pid),
                            "frame_id": str(pose_fid),
                            "stamp_ns": str(stamp_ns_frame),
                            "json": json.dumps(out),
                        },
                        maxlen=args.redis_maxlen,
                        approximate=True,
                    )

                    print(
                        f"[reid_node] cam={cam_id} pid={pid} "
                        f"pose_fid={pose_fid} frame_fid={frame_fid} emb_dim={emb.shape[0]}"
                    )

        except KeyboardInterrupt:
            print("\n[reid_node] stopping...")
        finally:
            decoder.close()
            sub_v.close(0)
            sub_p.close(0)
            sub_s.close(0)