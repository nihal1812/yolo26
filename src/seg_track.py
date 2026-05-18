#!/usr/bin/env python3
import time
import json
import queue
import threading
import argparse
import traceback

import zmq
import numpy as np
from ultralytics import YOLO

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from config_utils import (
    load_cfg,
    get_zmq_endpoint,
    get_model_cfg,
    get_runtime,
    local_connect_addr,
)

Gst.init(None)


def polygon_area(poly_xy: np.ndarray) -> float:
    if poly_xy is None or len(poly_xy) < 3:
        return 0.0
    x = poly_xy[:, 0]
    y = poly_xy[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def make_sub_socket(ctx, video_connect, video_topic, rcvhwm):
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.RCVHWM, rcvhwm)
    sub.connect(video_connect)
    sub.setsockopt(zmq.SUBSCRIBE, video_topic.encode("utf-8"))
    return sub


def build_decode_pipeline(codec: str):
    """
    appsrc(encoded AU) -> parse -> decode -> videoconvert -> BGR appsink
    """
    codec = str(codec).lower()
    if codec == "h265":
        parse = "h265parse"
        dec = "nvv4l2decoder"
        caps = "video/x-h265,stream-format=byte-stream,alignment=au"
    else:
        parse = "h264parse"
        dec = "nvv4l2decoder"
        caps = "video/x-h264,stream-format=byte-stream,alignment=au"

    pipeline_str = f"""
        appsrc name=src is-live=true format=time block=false caps="{caps}" !
        {parse} !
        {dec} !
        nvvidconv !
        video/x-raw,format=BGRx !
        videoconvert !
        video/x-raw,format=BGR !
        appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true enable-last-sample=false
    """
    return Gst.parse_launch(pipeline_str)


class EncodedVideoDecoder:
    """
    Decodes encoded H264/H265 access units coming from rtsp_stream ZMQ publisher.
    Output is BGR numpy frames.
    """

    def __init__(self, codec: str, decoded_queue_size: int = 2, debug: bool = False):
        self.codec = str(codec).lower()
        self.decoded_queue_size = int(decoded_queue_size)
        self.debug = bool(debug)

        self.pipeline = None
        self.appsrc = None
        self.appsink = None
        self.bus = None
        self.loop = None
        self.loop_thread = None
        self.running = False
        self.stopping = False

        self.frame_q = queue.Queue(maxsize=self.decoded_queue_size)
        self.err_q = queue.Queue(maxsize=8)
        self.n = 0

    def _clear_queue(self, q):
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def _push_latest_frame(self, frame):
        try:
            self.frame_q.put_nowait(frame)
        except queue.Full:
            try:
                _ = self.frame_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_q.put_nowait(frame)
            except queue.Full:
                pass

    def _push_error(self, src, err):
        try:
            self.err_q.put_nowait((src, err))
        except queue.Full:
            try:
                _ = self.err_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.err_q.put_nowait((src, err))
            except queue.Full:
                pass

    def _loop_runner(self):
        try:
            self.loop.run()
        except Exception as e:
            self._push_error("glib_loop", e)

    def start(self):
        self._clear_queue(self.frame_q)
        self._clear_queue(self.err_q)

        self.pipeline = build_decode_pipeline(self.codec)
        self.appsrc = self.pipeline.get_by_name("src")
        self.appsink = self.pipeline.get_by_name("sink")

        if self.appsrc is None or self.appsink is None:
            raise RuntimeError("decoder appsrc/appsink not found")

        self.loop = GLib.MainLoop()
        self.running = True
        self.stopping = False

        def on_new_sample(sink):
            if self.stopping:
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
                frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3)).copy()
                self._push_latest_frame(frame)
                self.n += 1

                if self.debug and self.n % 50 == 0:
                    print(f"[seg_track][decoder] decoded frames={self.n} shape={frame.shape}")
            finally:
                buf.unmap(mapinfo)

            return Gst.FlowReturn.OK

        def on_bus_message(_bus, message):
            mtype = message.type
            if mtype == Gst.MessageType.ERROR:
                err, dbg = message.parse_error()
                self._push_error("gst_error", RuntimeError(f"{err} debug={dbg}"))
                if self.loop is not None and self.loop.is_running():
                    self.loop.quit()
            elif mtype == Gst.MessageType.EOS:
                self._push_error("gst_eos", RuntimeError("decoder EOS"))
                if self.loop is not None and self.loop.is_running():
                    self.loop.quit()
            return True

        self.appsink.connect("new-sample", on_new_sample)

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", on_bus_message)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("decoder failed to set pipeline to PLAYING")

        self.loop_thread = threading.Thread(target=self._loop_runner, daemon=True)
        self.loop_thread.start()

    def push_encoded(self, enc_bytes: bytes):
        if not self.running:
            raise RuntimeError("decoder not running")

        if not self.err_q.empty():
            src, err = self.err_q.get_nowait()
            raise RuntimeError(f"[decoder] {src}: {err}") from err

        buf = Gst.Buffer.new_allocate(None, len(enc_bytes), None)
        buf.fill(0, enc_bytes)
        buf.pts = Gst.CLOCK_TIME_NONE
        buf.dts = Gst.CLOCK_TIME_NONE

        ret = self.appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            raise RuntimeError(f"decoder push-buffer failed: {ret}")

    def read_frame(self, timeout=0.5):
        if not self.running:
            return None

        if not self.err_q.empty():
            src, err = self.err_q.get_nowait()
            raise RuntimeError(f"[decoder] {src}: {err}") from err

        try:
            return self.frame_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.stopping = True
        self.running = False

        try:
            if self.appsrc is not None:
                self.appsrc.emit("end-of-stream")
        except Exception:
            pass

        try:
            if self.loop is not None and self.loop.is_running():
                self.loop.quit()
        except Exception:
            pass

        try:
            if self.bus is not None:
                self.bus.remove_signal_watch()
        except Exception:
            pass

        try:
            if self.pipeline is not None:
                self.pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass

        try:
            if self.loop_thread is not None and self.loop_thread.is_alive():
                self.loop_thread.join(timeout=1.0)
        except Exception:
            pass

        self.pipeline = None
        self.appsrc = None
        self.appsink = None
        self.bus = None
        self.loop = None
        self.loop_thread = None

        self._clear_queue(self.frame_q)
        self._clear_queue(self.err_q)


class SegTracker:
    """
    Transport-free segmentation processor.
    Can be used with:
      - direct decoded BGR frames via process_frame()
      - standalone subscriber mode via main()
    """

    def __init__(self, cfg_path: str, cam_id: str, publish_external: bool = True, debug: bool = False):
        self.cfg_path = cfg_path
        self.cam_id = str(cam_id)
        self.publish_external = bool(publish_external)
        self.debug = bool(debug)

        cfg = load_cfg(cfg_path)
        seg_cfg = get_zmq_endpoint(cfg, self.cam_id, "seg_features")
        models_cfg = get_model_cfg(cfg)

        self.seg_bind = seg_cfg["bind"]
        self.seg_topic = seg_cfg["topic"]
        self.seg_topic_b = self.seg_topic.encode("utf-8")
        self.sndhwm = int(seg_cfg.get("sndhwm", 1000))

        self.tracker = models_cfg.get("tracker", "botsort.yaml")
        self.imgsz = int(models_cfg.get("imgsz", 640))
        self.conf = float(models_cfg.get("conf", 0.25))
        self.iou = float(models_cfg.get("iou", 0.7))
        self.model_path = models_cfg.get("seg", {})["model"]
        self.poly_stride = int(models_cfg.get("seg", {}).get("poly_stride", 6))

        self.ctx = zmq.Context.instance()
        self.pub = None
        self.pub_drop_count = 0

        if self.publish_external:
            self.pub = self.ctx.socket(zmq.PUB)
            self.pub.setsockopt(zmq.LINGER, 0)
            self.pub.setsockopt(zmq.SNDHWM, self.sndhwm)
            self.pub.bind(self.seg_bind)

        self.model = YOLO(self.model_path)
        self.n = 0

        print(f"[seg_track] cam_id={self.cam_id}")
        if self.publish_external:
            print(f"[seg_track] PUB seg {self.seg_bind} topic={self.seg_topic}")
        else:
            print(f"[seg_track] external seg publish disabled for {self.cam_id}")
        print(f"[seg_track] model={self.model_path} tracker={self.tracker}")

    def process_frame(self, frame, header: dict):
        if frame is None:
            raise RuntimeError("process_frame received None frame")

        cam = header.get("cam_id", self.cam_id)
        frame_id = int(header.get("frame_id", 0))
        stamp_ns = int(header.get("stamp_ns", time.time_ns()))
        t_seg_ns = time.time_ns()

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
            "type": "seg_track",
            "cam_id": cam,
            "frame_id": frame_id,
            "stamp_ns": stamp_ns,
            "t_capture_ns": stamp_ns,
            "t_seg_ns": t_seg_ns,
            "frame_w": int(frame_w),
            "frame_h": int(frame_h),
            "instances": [],
        }

        boxes = res.boxes
        masks = res.masks

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((len(xyxy),), dtype=np.float32)
            clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((len(xyxy),), dtype=int)
            ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None
            polys = masks.xy if (masks is not None and getattr(masks, "xy", None) is not None) else None

            for i in range(len(xyxy)):
                tid = int(ids[i]) if ids is not None else -1
                x1, y1, x2, y2 = map(float, xyxy[i])

                inst = {
                    "track_id": tid,
                    "class_id": int(clss[i]),
                    "conf": float(confs[i]),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "mask_poly_xy": [],
                    "mask_area_px": 0.0,
                }

                if polys is not None and i < len(polys) and polys[i] is not None and len(polys[i]) >= 3:
                    poly = np.array(polys[i], dtype=np.float32)
                    if self.poly_stride > 1:
                        poly_ds = poly[:: self.poly_stride]
                        poly = poly_ds if len(poly_ds) >= 3 else poly
                    inst["mask_poly_xy"] = np.round(poly, 2).tolist()
                    inst["mask_area_px"] = round(polygon_area(poly), 2)

                payload["instances"].append(inst)

        feat_header = {
            "cam_id": cam,
            "frame_id": frame_id,
            "stamp_ns": stamp_ns,
            "t_capture_ns": stamp_ns,
            "t_seg_ns": t_seg_ns,
            "frame_w": int(frame_w),
            "frame_h": int(frame_h),
            "type": "seg_track",
        }

        if self.pub is not None:
            try:
                self.pub.send_multipart(
                    [
                        self.seg_topic_b,
                        json.dumps(feat_header, separators=(",", ":")).encode("utf-8"),
                        json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                    ],
                    flags=zmq.NOBLOCK,
                )
            except zmq.Again:
                self.pub_drop_count += 1
                if self.debug and (self.pub_drop_count % 50 == 0):
                    print(f"[seg_track][{self.cam_id}] PUB backpressure drops={self.pub_drop_count}")

        self.n += 1
        if self.debug and (self.n % 50 == 0):
            print(f"[seg_track][{self.cam_id}] published={self.n} last_frame_id={frame_id}")

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
    ap.add_argument("--decode_timeout_s", type=float, default=1.0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cam_id = str(args.cam_id)

    video_cfg = get_zmq_endpoint(cfg, cam_id, "video")
    rtsp_cfg = cfg["cams"][cam_id]["rtsp"]

    video_connect = local_connect_addr(video_cfg["bind"])
    video_topic = video_cfg["topic"]
    rcvhwm = int(get_runtime(cfg, "rcvhwm", 3))
    codec = str(rtsp_cfg.get("codec", "h264")).lower()

    print(f"[seg_track] SUB video {video_connect} topic={video_topic} codec={codec}")

    ctx = zmq.Context.instance()

    while True:
        sub = None
        tracker = None
        decoder = None
        consecutive_errors = 0

        try:
            sub = make_sub_socket(ctx, video_connect, video_topic, rcvhwm)
            tracker = SegTracker(
                cfg_path=args.config,
                cam_id=cam_id,
                publish_external=True,
                debug=args.debug,
            )
            decoder = EncodedVideoDecoder(codec=codec, decoded_queue_size=2, debug=args.debug)
            decoder.start()

            print(f"[seg_track] cam={cam_id} pipeline ready")

            while True:
                try:
                    parts = sub.recv_multipart()
                    if len(parts) != 3:
                        raise RuntimeError(f"[seg_track] cam={cam_id} expected 3-part multipart, got {len(parts)}")

                    _topic_b, header_b, enc_b = parts
                    header = json.loads(header_b.decode("utf-8"))

                    decoder.push_encoded(enc_b)
                    frame = decoder.read_frame(timeout=args.decode_timeout_s)
                    if frame is None:
                        raise RuntimeError("decoder timed out waiting for BGR frame")

                    tracker.process_frame(frame, header)
                    consecutive_errors = 0

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    consecutive_errors += 1
                    print(f"[seg_track] cam={cam_id} frame error ({consecutive_errors}/{args.max_consecutive_errors}): {e}")
                    if consecutive_errors <= 2:
                        traceback.print_exc()

                    if consecutive_errors >= args.max_consecutive_errors:
                        raise RuntimeError(
                            f"[seg_track] cam={cam_id} too many consecutive errors, rebuilding node resources"
                        ) from e

                    time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n[seg_track] stopping...")
            break
        except Exception as e:
            print(f"[seg_track] cam={cam_id} restartable failure: {e}")
            time.sleep(args.restart_delay_s)
        finally:
            try:
                if sub is not None:
                    sub.close(0)
            except Exception:
                pass
            try:
                if decoder is not None:
                    decoder.stop()
            except Exception:
                pass
            try:
                if tracker is not None:
                    tracker.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
