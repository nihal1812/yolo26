#!/usr/bin/env python3
import time
import json
import argparse
import traceback

import zmq
import numpy as np
from ultralytics import YOLO

from config_utils import (
    load_cfg,
    get_zmq_endpoint,
    get_model_cfg,
    get_runtime,
    local_connect_addr,
)


def make_sub_socket(ctx, video_connect, video_topic, rcvhwm):
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.RCVHWM, rcvhwm)
    sub.connect(video_connect)
    sub.setsockopt(zmq.SUBSCRIBE, video_topic.encode("utf-8"))
    return sub


class PoseTracker:
    """
    Transport-free pose processor.
    """

    def __init__(self, cfg_path: str, cam_id: str, publish_external: bool = True, debug: bool = False):
        self.cfg_path = cfg_path
        self.cam_id = str(cam_id)
        self.publish_external = bool(publish_external)
        self.debug = bool(debug)

        cfg = load_cfg(cfg_path)
        pose_cfg = get_zmq_endpoint(cfg, self.cam_id, "pose_features")
        models_cfg = get_model_cfg(cfg)

        self.pose_bind = pose_cfg["bind"]
        self.pose_topic = pose_cfg["topic"]
        self.pose_topic_b = self.pose_topic.encode("utf-8")
        self.sndhwm = int(pose_cfg.get("sndhwm", 1000))

        self.tracker = models_cfg.get("tracker", "botsort.yaml")
        self.imgsz = int(models_cfg.get("imgsz", 640))
        self.conf = float(models_cfg.get("conf", 0.25))
        self.iou = float(models_cfg.get("iou", 0.7))
        self.model_path = models_cfg.get("pose", {})["model"]

        self.ctx = zmq.Context.instance()
        self.pub = None
        self.pub_drop_count = 0

        if self.publish_external:
            self.pub = self.ctx.socket(zmq.PUB)
            self.pub.setsockopt(zmq.LINGER, 0)
            self.pub.setsockopt(zmq.SNDHWM, self.sndhwm)
            self.pub.bind(self.pose_bind)

        self.model = YOLO(self.model_path)
        self.n = 0

        print(f"[pose_track] cam_id={self.cam_id}")
        if self.publish_external:
            print(f"[pose_track] PUB pose {self.pose_bind} topic={self.pose_topic}")
        else:
            print(f"[pose_track] external pose publish disabled for {self.cam_id}")
        print(f"[pose_track] model={self.model_path} tracker={self.tracker}")

    def process_frame(self, frame, header: dict):
        cam = header.get("cam_id", self.cam_id)
        frame_id = int(header.get("frame_id", 0))
        stamp_ns = int(header.get("stamp_ns", time.time_ns()))

        frame_h, frame_w = frame.shape[:2]

        res = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )[0]

        payload = {
            "type": "pose_track",
            "cam_id": cam,
            "frame_id": frame_id,
            "stamp_ns": stamp_ns,
            "frame_w": int(frame_w),
            "frame_h": int(frame_h),
            "people": [],
        }

        boxes = res.boxes
        kps = res.keypoints

        if boxes is not None and len(boxes) > 0 and kps is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((len(xyxy),), dtype=np.float32)
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
                    "keypoints_conf": [],
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

        if self.pub is not None:
            try:
                self.pub.send_multipart(
                    [
                        self.pose_topic_b,
                        json.dumps(feat_header, separators=(",", ":")).encode("utf-8"),
                        json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                    ],
                    flags=zmq.NOBLOCK,
                )
            except zmq.Again:
                self.pub_drop_count += 1
                if self.debug and (self.pub_drop_count % 50 == 0):
                    print(f"[pose_track][{self.cam_id}] PUB backpressure drops={self.pub_drop_count}")

        self.n += 1
        if self.debug and (self.n % 50 == 0):
            print(f"[pose_track][{self.cam_id}] published={self.n} last_frame_id={frame_id}")

        return feat_header, payload

    def close(self):
        try:
            if self.pub is not None:
                self.pub.close(0)
        except Exception:
            pass
        self.pub = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cam_id", required=True)
    ap.add_argument("--max_consecutive_errors", type=int, default=10)
    ap.add_argument("--restart_delay_s", type=float, default=2.0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cam_id = str(args.cam_id)

    video_cfg = get_zmq_endpoint(cfg, cam_id, "video")
    video_connect = local_connect_addr(video_cfg["bind"])
    video_topic = video_cfg["topic"]
    rcvhwm = int(get_runtime(cfg, "rcvhwm", 3))

    print(f"[pose_track] SUB video {video_connect} topic={video_topic}")

    ctx = zmq.Context.instance()

    while True:
        sub = None
        tracker = None
        consecutive_errors = 0

        try:
            sub = make_sub_socket(ctx, video_connect, video_topic, rcvhwm)
            tracker = PoseTracker(
                cfg_path=args.config,
                cam_id=cam_id,
                publish_external=True,
                debug=args.debug,
            )

            print(f"[pose_track] cam={cam_id} pipeline ready")

            while True:
                try:
                    parts = sub.recv_multipart()
                    if len(parts) != 3:
                        raise RuntimeError(f"[pose_track] cam={cam_id} expected 3-part multipart, got {len(parts)}")

                    _, header_b, _enc = parts
                    _ = json.loads(header_b.decode("utf-8"))

                    # Standalone mode cannot decode anymore in this version.
                    # Integrated perception mode is the target runtime.
                    consecutive_errors = 0

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    consecutive_errors += 1
                    print(f"[pose_track] cam={cam_id} frame error ({consecutive_errors}/{args.max_consecutive_errors}): {e}")
                    if consecutive_errors <= 2:
                        traceback.print_exc()

                    if consecutive_errors >= args.max_consecutive_errors:
                        raise RuntimeError(
                            f"[pose_track] cam={cam_id} too many consecutive errors, rebuilding node resources"
                        ) from e

                    time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n[pose_track] stopping...")
            break
        except Exception as e:
            print(f"[pose_track] cam={cam_id} restartable failure: {e}")
            time.sleep(args.restart_delay_s)
        finally:
            try:
                if sub is not None:
                    sub.close(0)
            except Exception:
                pass
            try:
                if tracker is not None:
                    tracker.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
