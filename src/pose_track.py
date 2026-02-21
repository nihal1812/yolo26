#!/usr/bin/env python3
import time
import json
import argparse
import yaml
import threading
import queue

import zmq
import numpy as np
from ultralytics import YOLO

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class GstDecoder:
    """
    Queue-based decoder:
    - push(encoded) enqueues compressed bytes into appsrc
    - appsink callback decodes and pushes the *next* decoded frame into a queue
    - pop(timeout) returns the next decoded frame (not 'latest')
    """
    def __init__(self, codec="h264", max_queue=2):
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

        self._q = queue.Queue(maxsize=max_queue)
        self._closed = False

        self.appsink.connect("new-sample", self._on_sample)
        self.pipeline.set_state(Gst.State.PLAYING)

    def _on_sample(self, sink):
        if self._closed:
            return Gst.FlowReturn.OK

        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        w = int(s.get_value("width"))
        h = int(s.get_value("height"))

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if ok:
            try:
                frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3))
                # Put newest frame; drop oldest if queue full
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    cam_id = cfg["system"]["cam_id"]
    video_connect = cfg["zmq"]["video"]["bind"].replace("*", "127.0.0.1")  # local default
    video_topic = cfg["zmq"]["video"]["topic"]

    pose_bind = cfg["zmq"]["pose_features"]["bind"]
    pose_topic = cfg["zmq"]["pose_features"]["topic"]

    latest_only = bool(cfg["runtime"].get("latest_only", True))
    rcvhwm = int(cfg["runtime"].get("rcvhwm", 3))
    log_every_n = int(cfg["runtime"].get("log_every_n", 50))

    codec = cfg["rtsp"].get("codec", "h264")

    tracker = cfg["models"].get("tracker", "botsort.yaml")
    imgsz = int(cfg["models"].get("imgsz", 640))
    conf = float(cfg["models"].get("conf", 0.25))
    iou = float(cfg["models"].get("iou", 0.7))
    model_path = cfg["models"]["pose"]["model"]

    ctx = zmq.Context.instance()

    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, rcvhwm)
    if latest_only:
        sub.setsockopt(zmq.CONFLATE, 1)
    sub.connect(video_connect)
    sub.setsockopt(zmq.SUBSCRIBE, video_topic.encode())

    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, int(cfg["zmq"]["pose_features"].get("sndhwm", 1000)))
    pub.bind(pose_bind)

    decoder = GstDecoder(codec=codec, max_queue=2)
    model = YOLO(model_path)

    print(f"[pose_track] SUB video {video_connect} topic={video_topic} (latest_only={latest_only})")
    print(f"[pose_track] PUB pose  {pose_bind} topic={pose_topic}")
    print(f"[pose_track] model={model_path} tracker={tracker}")

    n = 0
    try:
        while True:
            topic_b, header_b, enc = sub.recv_multipart()
            header = json.loads(header_b.decode())

            cam = header.get("cam_id", cam_id)
            frame_id = int(header.get("frame_id", 0))
            stamp_ns = int(header.get("stamp_ns", time.time_ns()))

            # push encoded and pop the next decoded frame
            decoder.push(enc)
            frame = decoder.pop(timeout=0.2)
            if frame is None:
                continue

            frame_h, frame_w = frame.shape[:2]

            res = model.track(
                source=frame,
                persist=True,
                tracker=tracker,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False,
            )[0]

            payload = {
                "type": "pose_track",
                "cam_id": cam,
                "frame_id": frame_id,
                "stamp_ns": stamp_ns,
                "frame_w": int(frame_w),
                "frame_h": int(frame_h),
                "people": []
            }

            boxes = res.boxes
            kps = res.keypoints

            if boxes is not None and len(boxes) > 0 and kps is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((len(xyxy),), np.float32)
                ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None

                kp_xy = kps.xy.cpu().numpy() if getattr(kps, "xy", None) is not None else None
                kp_cf = kps.conf.cpu().numpy() if getattr(kps, "conf", None) is not None else None

                for i in range(len(xyxy)):
                    tid = int(ids[i]) if ids is not None else -1
                    x1, y1, x2, y2 = map(float, xyxy[i])

                    person = {
                        "track_id": tid,
                        "conf": float(confs[i]),
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "keypoints_xy": [],
                        "keypoints_conf": []
                    }
                    if kp_xy is not None and i < kp_xy.shape[0]:
                        person["keypoints_xy"] = np.round(kp_xy[i], 2).tolist()
                    if kp_cf is not None and i < kp_cf.shape[0]:
                        person["keypoints_conf"] = np.round(kp_cf[i], 3).tolist()

                    payload["people"].append(person)

            feat_header = {
                "cam_id": cam,
                "frame_id": frame_id,
                "stamp_ns": stamp_ns,
                "frame_w": int(frame_w),
                "frame_h": int(frame_h),
                "type": "pose_track",
            }

            pub.send_multipart([
                pose_topic.encode(),
                json.dumps(feat_header).encode(),
                json.dumps(payload).encode(),
            ])

            n += 1
            if log_every_n and (n % log_every_n == 0):
                print(f"[pose_track] published {n} messages (last frame_id={frame_id})")

    except KeyboardInterrupt:
        print("\n[pose_track] stopping...")
    finally:
        decoder.close()
        sub.close()
        pub.close()


if __name__ == "__main__":
    main()
