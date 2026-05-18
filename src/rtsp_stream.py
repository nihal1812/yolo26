#!/usr/bin/env python3
import time
import json
import queue
import threading
import argparse
from urllib.parse import urlsplit, urlunsplit, quote

import zmq
import cv2

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from config_utils import load_cfg, get_rtsp_cfg, get_zmq_endpoint

Gst.init(None)


def build_auth_rtsp_url(rtsp_url: str, username: str = None, password: str = None) -> str:
    if not username:
        return rtsp_url

    parts = urlsplit(rtsp_url)
    if "@" in parts.netloc:
        return rtsp_url

    user = quote(str(username), safe="")
    pw = quote(str(password or ""), safe="")
    auth = f"{user}:{pw}@" if password is not None else f"{user}@"
    netloc = auth + parts.netloc
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def build_pipeline(rtsp_url: str, codec: str, latency_ms: int, transport: str):
    codec = str(codec).lower()

    if codec == "h265":
        depay = "rtph265depay"
        parse_caps = "video/x-h265,stream-format=byte-stream,alignment=au"
        parse = f"h265parse config-interval=-1 ! {parse_caps}"
        dec = "avdec_h265"
        encoding_name = "H265"
        proto_mask = 4 if str(transport).lower() == "tcp" else 1
    else:
        depay = "rtph264depay"
        parse_caps = "video/x-h264,stream-format=byte-stream,alignment=au"
        parse = f"h264parse config-interval=-1 ! {parse_caps}"
        dec = "avdec_h264"
        encoding_name = "H264"
        proto_mask = 4 if str(transport).lower() == "tcp" else 1

    pipeline_str = f"""
        rtspsrc name=src latency={latency_ms} protocols={proto_mask} do-rtsp-keep-alive=true !
        application/x-rtp,media=video,encoding-name={encoding_name} !
        {depay} !
        {parse} !
        tee name=t

        t. ! queue leaky=downstream max-size-buffers=10 !
        appsink name=encsink emit-signals=true sync=false max-buffers=10 drop=true

        t. ! queue leaky=downstream max-size-buffers=10 !
        {dec} !
        videoconvert !
        video/x-raw,format=BGR !
        appsink name=framesink emit-signals=true sync=false max-buffers=2 drop=true
    """

    pipeline = Gst.parse_launch(pipeline_str)
    src = pipeline.get_by_name("src")
    if src is None:
        raise RuntimeError("rtspsrc not found in pipeline")

    src.set_property("location", rtsp_url)
    return pipeline


class RtspStreamRuntime:
    """
    Per-camera source runtime.

    Publishes:
      - encoded stream on config key: video
      - JPEG UI stream on config key: video_ui

    Internal:
      - decoded BGR frames via read_frame()
    """

    def __init__(
        self,
        cfg_path: str,
        cam_id: str,
        publish_external: bool = True,
        decoded_queue_size: int = 2,
        debug: bool = False,
    ):
        self.cfg_path = cfg_path
        self.cam_id = str(cam_id)
        self.publish_external = bool(publish_external)
        self.decoded_queue_size = int(decoded_queue_size)
        self.debug = bool(debug)

        cfg = load_cfg(cfg_path)
        rtsp_cfg = get_rtsp_cfg(cfg, self.cam_id)
        video_cfg = get_zmq_endpoint(cfg, self.cam_id, "video")
        video_ui_cfg = get_zmq_endpoint(cfg, self.cam_id, "video_ui")

        self.rtsp_url = rtsp_cfg["url"]
        self.username = rtsp_cfg.get("username", None)
        self.password = rtsp_cfg.get("password", None)
        self.codec = str(rtsp_cfg.get("codec", "h264")).lower()
        self.latency_ms = int(rtsp_cfg.get("latency_ms", 120))
        self.transport = rtsp_cfg.get("transport", "tcp")

        self.video_bind = video_cfg["bind"]
        self.video_topic = video_cfg["topic"]
        self.video_sndhwm = int(video_cfg.get("sndhwm", 3))

        self.video_ui_bind = video_ui_cfg["bind"]
        self.video_ui_topic = video_ui_cfg["topic"]
        self.video_ui_sndhwm = int(video_ui_cfg.get("sndhwm", 3))
        self.video_ui_jpeg_quality = int(video_ui_cfg.get("jpeg_quality", 85))

        self.ctx = zmq.Context.instance()

        self.pub_video = None
        self.pub_video_ui = None

        if self.publish_external:
            self.pub_video = self.ctx.socket(zmq.PUB)
            self.pub_video.setsockopt(zmq.SNDHWM, self.video_sndhwm)
            self.pub_video.setsockopt(zmq.LINGER, 0)
            self.pub_video.bind(self.video_bind)

            self.pub_video_ui = self.ctx.socket(zmq.PUB)
            self.pub_video_ui.setsockopt(zmq.SNDHWM, self.video_ui_sndhwm)
            self.pub_video_ui.setsockopt(zmq.LINGER, 0)
            self.pub_video_ui.bind(self.video_ui_bind)

        self.pipeline = None
        self.rtspsrc = None
        self.bus = None
        self.loop = None
        self.loop_thread = None
        self.frame_id = 0
        self.running = False
        self.stopping = False

        self.decoded_q = queue.Queue(maxsize=self.decoded_queue_size)
        self.err_q = queue.Queue(maxsize=8)

        self._saw_encoded = False
        self._saw_decoded = False

    def _push_latest_decoded(self, item):
        try:
            self.decoded_q.put_nowait(item)
        except queue.Full:
            try:
                _ = self.decoded_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.decoded_q.put_nowait(item)
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
        effective_url = build_auth_rtsp_url(
            self.rtsp_url,
            username=self.username,
            password=self.password,
        )

        self.pipeline = build_pipeline(
            effective_url,
            self.codec,
            self.latency_ms,
            self.transport,
        )

        self.rtspsrc = self.pipeline.get_by_name("src")
        if self.rtspsrc is None:
            raise RuntimeError("rtspsrc not found after pipeline build")

        encsink = self.pipeline.get_by_name("encsink")
        framesink = self.pipeline.get_by_name("framesink")
        if encsink is None or framesink is None:
            raise RuntimeError("encsink/framesink not found")

        self.loop = GLib.MainLoop()
        self.running = True
        self.stopping = False
        self._saw_encoded = False
        self._saw_decoded = False
        self.frame_id = 0

        latest_header = {
            "cam_id": self.cam_id,
            "frame_id": -1,
            "stamp_ns": 0,
            "codec": self.codec,
        }
        latest_lock = threading.Lock()

        def on_new_encoded(sink):
            if self.stopping:
                return Gst.FlowReturn.OK

            sample = sink.emit("pull-sample")
            if sample is None:
                return Gst.FlowReturn.OK

            buf = sample.get_buffer()
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                return Gst.FlowReturn.OK

            try:
                self._saw_encoded = True

                stamp_ns = time.time_ns()
                header = {
                    "cam_id": self.cam_id,
                    "frame_id": self.frame_id,
                    "stamp_ns": stamp_ns,
                    "codec": self.codec,
                }
                enc = bytes(mapinfo.data)

                if self.pub_video is not None:
                    self.pub_video.send_multipart(
                        [
                            self.video_topic.encode("utf-8"),
                            json.dumps(header, separators=(",", ":")).encode("utf-8"),
                            enc,
                        ]
                    )

                with latest_lock:
                    latest_header.clear()
                    latest_header.update(header)

                if self.debug and (self.frame_id % 50 == 0):
                    print(
                        f"[rtsp_stream][{self.cam_id}] encoded frame_id={header['frame_id']} "
                        f"bytes={len(enc)}"
                    )

                self.frame_id += 1
            finally:
                buf.unmap(mapinfo)

            return Gst.FlowReturn.OK

        def on_new_frame(sink):
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
                import numpy as np

                self._saw_decoded = True
                bgr = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3)).copy()

                with latest_lock:
                    hdr = dict(latest_header)

                self._push_latest_decoded((hdr, bgr))

                if self.pub_video_ui is not None:
                    ok_jpg, jpg = cv2.imencode(
                        ".jpg",
                        bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.video_ui_jpeg_quality],
                    )
                    if ok_jpg:
                        ui_header = {
                            "cam_id": self.cam_id,
                            "frame_id": int(hdr.get("frame_id", -1)),
                            "stamp_ns": int(hdr.get("stamp_ns", time.time_ns())),
                            "codec": "jpeg",
                            "width": int(w),
                            "height": int(h),
                        }
                        try:
                            self.pub_video_ui.send_multipart(
                                [
                                    self.video_ui_topic.encode("utf-8"),
                                    json.dumps(ui_header, separators=(",", ":")).encode("utf-8"),
                                    jpg.tobytes(),
                                ],
                                flags=zmq.NOBLOCK,
                            )
                        except zmq.Again:
                            pass

                if self.debug and hdr.get("frame_id", 0) % 50 == 0:
                    print(
                        f"[rtsp_stream][{self.cam_id}] decoded frame_id={hdr.get('frame_id')} "
                        f"shape={bgr.shape}"
                    )
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
                self._push_error("gst_eos", RuntimeError("GStreamer EOS"))
                if self.loop is not None and self.loop.is_running():
                    self.loop.quit()

            elif mtype == Gst.MessageType.WARNING and self.debug:
                err, dbg = message.parse_warning()
                print(f"[rtsp_stream][{self.cam_id}] GST warning: {err} debug={dbg}")

            elif mtype == Gst.MessageType.STATE_CHANGED and self.debug:
                if message.src == self.pipeline:
                    old_state, new_state, pending = message.parse_state_changed()
                    print(
                        f"[rtsp_stream][{self.cam_id}] pipeline state "
                        f"{old_state.value_nick} -> {new_state.value_nick}"
                    )

            return True

        encsink.connect("new-sample", on_new_encoded)
        framesink.connect("new-sample", on_new_frame)

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", on_bus_message)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("rtsp_stream failed to set pipeline to PLAYING")

        self.loop_thread = threading.Thread(target=self._loop_runner, daemon=True)
        self.loop_thread.start()

        print(f"[rtsp_stream] cam_id={self.cam_id}")
        if self.publish_external:
            print(f"[rtsp_stream] PUB video    @ {self.video_bind} topic={self.video_topic} codec={self.codec}")
            print(f"[rtsp_stream] PUB video_ui @ {self.video_ui_bind} topic={self.video_ui_topic} codec=jpeg")
        else:
            print(f"[rtsp_stream] external video publish disabled for {self.cam_id}")
        print(f"[rtsp_stream] cam={self.cam_id} pipeline running")

    def read_frame(self, timeout=0.5):
        if not self.running:
            return None

        if not self.err_q.empty():
            src, err = self.err_q.get_nowait()
            raise RuntimeError(f"[rtsp_stream][{self.cam_id}] {src}: {err}") from err

        try:
            return self.decoded_q.get(timeout=timeout)
        except queue.Empty:
            if self.debug:
                print(
                    f"[rtsp_stream][{self.cam_id}] timeout waiting frame "
                    f"(encoded_seen={self._saw_encoded}, decoded_seen={self._saw_decoded})"
                )
            return None

    def stop(self):
        self.stopping = True
        self.running = False

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

        try:
            if self.pub_video is not None:
                self.pub_video.close(0)
        except Exception:
            pass
        self.pub_video = None

        try:
            if self.pub_video_ui is not None:
                self.pub_video_ui.close(0)
        except Exception:
            pass
        self.pub_video_ui = None

        self.pipeline = None
        self.rtspsrc = None
        self.bus = None
        self.loop = None
        self.loop_thread = None

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cam_id", required=True)
    ap.add_argument("--restart_delay_s", type=float, default=2.0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    rtsp = None
    try:
        while True:
            try:
                rtsp = RtspStreamRuntime(
                    cfg_path=args.config,
                    cam_id=args.cam_id,
                    publish_external=True,
                    decoded_queue_size=2,
                    debug=args.debug,
                )
                rtsp.start()

                while True:
                    _ = rtsp.read_frame(timeout=1.0)

            except KeyboardInterrupt:
                print("\n[rtsp_stream] stopping...")
                break
            except Exception as e:
                print(f"[rtsp_stream] cam={args.cam_id} restartable failure: {e}")
                time.sleep(args.restart_delay_s)
            finally:
                if rtsp is not None:
                    rtsp.stop()
                    rtsp = None
    finally:
        if rtsp is not None:
            rtsp.stop()


if __name__ == "__main__":
    main()
