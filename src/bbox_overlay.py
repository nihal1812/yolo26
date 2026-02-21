#!/usr/bin/env python3
import argparse, json, time
from collections import deque

import yaml, redis, zmq
import numpy as np
import cv2
import requests

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def b2s(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)

def parse_xread(streams):
    out = []
    for _s, msgs in streams:
        for mid, fields in msgs:
            out.append((b2s(mid), fields))
    return out

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (0.5*(x1+x2), 0.5*(y1+y2))

class GstDecoder:
    def __init__(self, codec="h264"):
        codec = codec.lower()
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
        self._latest = None
        self.appsink.connect("new-sample", self._on_sample)
        self.pipeline.set_state(Gst.State.PLAYING)

    def _on_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        w = s.get_value("width")
        h = s.get_value("height")

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if ok:
            try:
                frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3))
                self._latest = frame.copy()
            finally:
                buf.unmap(mapinfo)
        return Gst.FlowReturn.OK

    def push(self, encoded: bytes):
        gstbuf = Gst.Buffer.new_allocate(None, len(encoded), None)
        gstbuf.fill(0, encoded)
        self.appsrc.emit("push-buffer", gstbuf)

    def get_latest(self):
        return None if self._latest is None else self._latest.copy()

    def close(self):
        self.pipeline.set_state(Gst.State.NULL)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frame_buffer", type=int, default=60)    # ~2 sec at 30fps
    ap.add_argument("--pose_buffer", type=int, default=120)
    ap.add_argument("--seg_buffer", type=int, default=120)
    ap.add_argument("--alerts_block_ms", type=int, default=500)
    ap.add_argument("--alerts_count", type=int, default=50)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cam_id = cfg["system"]["cam_id"]
    codec = cfg.get("rtsp", {}).get("codec", "h264")

    # Redis alerts
    r_cfg = cfg.get("redis", {})
    rdb = redis.Redis(
        host=r_cfg.get("host", "127.0.0.1"),
        port=int(r_cfg.get("port", 6379)),
        db=int(r_cfg.get("db", 0)),
        password=r_cfg.get("password", None),
    )
    rdb.ping()
    alerts_stream = r_cfg.get("alerts_stream", f"alerts:{cam_id}")

    # Webhook
    p3 = cfg.get("pipeline3", {})
    wh = p3.get("webhooks", {})
    overlay_url = wh.get("overlay_url")
    timeout_s = float(wh.get("timeout_s", 3.0))
    if not overlay_url:
        raise RuntimeError("pipeline3.webhooks.overlay_url missing in config")

    # ZMQ video + pose + seg (subscribe)
    zcfg = cfg["zmq"]
    ctx = zmq.Context.instance()

    video_connect = zcfg["video"]["bind"].replace("*", "127.0.0.1")
    video_topic = zcfg["video"]["topic"]

    pose_connect = zcfg["pose_features"]["bind"].replace("*", "127.0.0.1")
    pose_topic = zcfg["pose_features"]["topic"]

    seg_connect = zcfg["seg_features"]["bind"].replace("*", "127.0.0.1")
    seg_topic = zcfg["seg_features"]["topic"]

    sub_v = ctx.socket(zmq.SUB)
    sub_v.connect(video_connect)
    sub_v.setsockopt(zmq.SUBSCRIBE, video_topic.encode())

    sub_p = ctx.socket(zmq.SUB)
    sub_p.connect(pose_connect)
    sub_p.setsockopt(zmq.SUBSCRIBE, pose_topic.encode())

    sub_s = ctx.socket(zmq.SUB)
    sub_s.connect(seg_connect)
    sub_s.setsockopt(zmq.SUBSCRIBE, seg_topic.encode())

    poller = zmq.Poller()
    poller.register(sub_v, zmq.POLLIN)
    poller.register(sub_p, zmq.POLLIN)
    poller.register(sub_s, zmq.POLLIN)

    decoder = GstDecoder(codec=codec)

    # buffers by frame_id
    frame_buf = {}   # fid -> (stamp_ns, frame_bgr)
    pose_buf = {}
    seg_buf = {}
    frame_fifo = deque(maxlen=args.frame_buffer)
    pose_fifo = deque(maxlen=args.pose_buffer)
    seg_fifo = deque(maxlen=args.seg_buffer)

    def put_buf(buf, fifo, fid, obj):
        if fid in buf:
            return
        buf[fid] = obj
        fifo.append(fid)
        while len(fifo) > fifo.maxlen:
            old = fifo.popleft()
            buf.pop(old, None)

    def find_nearest(buf, fid, max_delta=3):
        # exact match preferred, else +/- small window
        if fid in buf:
            return fid, buf[fid]
        for d in range(1, max_delta+1):
            if (fid-d) in buf:
                return fid-d, buf[fid-d]
            if (fid+d) in buf:
                return fid+d, buf[fid+d]
        return None, None

    last_alert_id = "0-0"

    print(f"[bbox_overlay] SUB video {video_connect} topic={video_topic}")
    print(f"[bbox_overlay] SUB pose  {pose_connect} topic={pose_topic}")
    print(f"[bbox_overlay] SUB seg   {seg_connect} topic={seg_topic}")
    print(f"[bbox_overlay] alerts={alerts_stream}")
    print(f"[bbox_overlay] POST {overlay_url}")

    while True:
        # 1) Pull ZMQ messages (non-blocking-ish)
        events = dict(poller.poll(timeout=20))
        if sub_v in events:
            _, header_b, enc = sub_v.recv_multipart()
            try:
                header = json.loads(header_b.decode())
            except Exception:
                header = {}
            fid = int(header.get("frame_id", 0))
            stamp_ns = int(header.get("stamp_ns", time.time_ns()))
            decoder.push(enc)
            frame = decoder.get_latest()
            if frame is not None and fid > 0:
                put_buf(frame_buf, frame_fifo, fid, (stamp_ns, frame))

        if sub_p in events:
            _, _, payload_b = sub_p.recv_multipart()
            try:
                pose = json.loads(payload_b.decode())
                fid = int(pose.get("frame_id", 0))
                if fid > 0:
                    put_buf(pose_buf, pose_fifo, fid, pose)
            except Exception:
                pass

        if sub_s in events:
            _, _, payload_b = sub_s.recv_multipart()
            try:
                seg = json.loads(payload_b.decode())
                fid = int(seg.get("frame_id", 0))
                if fid > 0:
                    put_buf(seg_buf, seg_fifo, fid, seg)
            except Exception:
                pass

        # 2) Read alerts from Redis
        streams = rdb.xread({alerts_stream: last_alert_id}, block=args.alerts_block_ms, count=args.alerts_count)
        if not streams:
            continue

        for mid, fields in parse_xread(streams):
            last_alert_id = mid
            js = fields.get(b"json", b"{}")
            try:
                alert = json.loads(b2s(js))
            except Exception:
                continue

            fid_end = int(alert.get("frame_id_end", -1))
            pid = int(alert.get("person_track_id", -1))
            score = alert.get("score", None)

            if fid_end < 0 or pid < 0:
                continue

            _, frame_item = find_nearest(frame_buf, fid_end, max_delta=3)
            _, pose = find_nearest(pose_buf, fid_end, max_delta=3)
            _, seg  = find_nearest(seg_buf, fid_end, max_delta=3)

            if frame_item is None or pose is None:
                continue

            _, frame = frame_item
            out = frame.copy()

            # find person bbox in pose
            pb = None
            kp = None
            for person in pose.get("people", []):
                if int(person.get("track_id", -1)) == pid:
                    pb = person.get("bbox_xyxy", None)
                    kp = person.get("keypoints_xy", None)
                    break

            if not pb:
                continue

            x1, y1, x2, y2 = [int(v) for v in pb]
            x1 = clamp(x1, 0, out.shape[1]-1); x2 = clamp(x2, 0, out.shape[1]-1)
            y1 = clamp(y1, 0, out.shape[0]-1); y2 = clamp(y2, 0, out.shape[0]-1)

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(out, f"pid={pid} score={score}", (x1, max(0, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # draw keypoints (simple dots)
            if isinstance(kp, list) and len(kp) > 0:
                for pt in kp:
                    if not pt or len(pt) < 2:
                        continue
                    px, py = int(pt[0]), int(pt[1])
                    if 0 <= px < out.shape[1] and 0 <= py < out.shape[0]:
                        cv2.circle(out, (px, py), 2, (0, 255, 0), -1)

            # draw a "likely object" bbox from seg (nearest non-person bbox to person center)
            if seg is not None:
                pcx, pcy = bbox_center(pb)
                best = None
                for inst in seg.get("instances", []):
                    cid = int(inst.get("class_id", -1))
                    if cid == 0:
                        continue
                    bb = inst.get("bbox_xyxy", None)
                    if not bb:
                        continue
                    ocx, ocy = bbox_center(bb)
                    d2 = (ocx - pcx)**2 + (ocy - pcy)**2
                    if best is None or d2 < best[0]:
                        best = (d2, inst)
                if best is not None:
                    inst = best[1]
                    bb = inst.get("bbox_xyxy", None)
                    if bb:
                        ox1, oy1, ox2, oy2 = [int(v) for v in bb]
                        ox1 = clamp(ox1, 0, out.shape[1]-1); ox2 = clamp(ox2, 0, out.shape[1]-1)
                        oy1 = clamp(oy1, 0, out.shape[0]-1); oy2 = clamp(oy2, 0, out.shape[0]-1)
                        cv2.rectangle(out, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                        cv2.putText(out, f"obj_id={inst.get('track_id',-1)} cls={inst.get('class_id',-1)}",
                                    (ox1, max(0, oy1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            # encode JPEG and send to webhook
            ok, jpg = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                continue

            meta = {
                "cam_id": alert.get("cam_id", cam_id),
                "person_track_id": pid,
                "frame_id_end": fid_end,
                "stamp_ns_end": alert.get("stamp_ns_end", 0),
                "score": score,
                "model_version": alert.get("model_version", "unknown"),
                "reason": alert.get("reason", {}),
            }

            try:
                r = requests.post(
                    overlay_url,
                    data={"meta": json.dumps(meta)},
                    files={"image": ("overlay.jpg", jpg.tobytes(), "image/jpeg")},
                    timeout=timeout_s
                )
                print(f"[bbox_overlay] sent overlay mid={mid} status={r.status_code}")
            except Exception as e:
                print(f"[bbox_overlay] webhook error mid={mid}: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[bbox_overlay] stopping...")
