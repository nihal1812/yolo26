#!/usr/bin/env python3
import time
import json
import argparse
import zmq

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from config_utils import load_cfg, get_rtsp_cfg, get_zmq_endpoint

Gst.init(None)


def build_pipeline(rtsp_url: str, codec: str, latency_ms: int, transport: str):
    """
    Outputs encoded access units via appsink (NOT decoded frames).
    """
    codec = str(codec).lower()
    if codec == "h265":
        depay = "rtph265depay"
        parse = "h265parse config-interval=-1"
    else:
        depay = "rtph264depay"
        parse = "h264parse config-interval=-1"

    proto = "tcp" if str(transport).lower() == "tcp" else "udp"

    pipeline_str = f"""
        rtspsrc location={rtsp_url} latency={latency_ms} protocols={proto} !
        {depay} !
        {parse} !
        appsink name=encsink emit-signals=true sync=false max-buffers=1 drop=true
    """
    return Gst.parse_launch(pipeline_str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cam_id", required=True)
    ap.add_argument("--restart_delay_s", type=float, default=2.0)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cam_id = str(args.cam_id)

    rtsp_cfg = get_rtsp_cfg(cfg, cam_id)
    video_cfg = get_zmq_endpoint(cfg, cam_id, "video")

    rtsp_url = rtsp_cfg["url"]
    codec = rtsp_cfg.get("codec", "h264")
    latency_ms = int(rtsp_cfg.get("latency_ms", 120))
    transport = rtsp_cfg.get("transport", "tcp")

    video_bind = video_cfg["bind"]
    topic = video_cfg["topic"]
    sndhwm = int(video_cfg.get("sndhwm", 3))

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, sndhwm)
    pub.bind(video_bind)

    frame_id = 0

    print(f"[rtsp_stream] cam_id={cam_id}")
    print(f"[rtsp_stream] PUB video @ {video_bind} topic={topic} codec={codec}")

    while True:
        pipeline = None
        loop = None
        bus = None

        try:
            pipeline = build_pipeline(rtsp_url, codec, latency_ms, transport)
            appsink = pipeline.get_by_name("encsink")
            if appsink is None:
                raise RuntimeError("appsink 'encsink' not found")

            loop = GLib.MainLoop()

            def on_new_sample(sink):
                nonlocal frame_id
                sample = sink.emit("pull-sample")
                if sample is None:
                    return Gst.FlowReturn.OK

                buf = sample.get_buffer()
                ok, mapinfo = buf.map(Gst.MapFlags.READ)
                if not ok:
                    return Gst.FlowReturn.OK

                try:
                    stamp_ns = time.time_ns()
                    header = {
                        "cam_id": cam_id,
                        "frame_id": frame_id,
                        "stamp_ns": stamp_ns,
                        "codec": codec,
                    }
                    pub.send_multipart([
                        topic.encode("utf-8"),
                        json.dumps(header).encode("utf-8"),
                        bytes(mapinfo.data),
                    ])
                    frame_id += 1
                finally:
                    buf.unmap(mapinfo)

                return Gst.FlowReturn.OK

            def on_bus_message(_bus, message):
                mtype = message.type
                if mtype == Gst.MessageType.ERROR:
                    err, dbg = message.parse_error()
                    print(f"[rtsp_stream] cam={cam_id} GST ERROR: {err} debug={dbg}")
                    if loop is not None and loop.is_running():
                        loop.quit()
                elif mtype == Gst.MessageType.EOS:
                    print(f"[rtsp_stream] cam={cam_id} GST EOS")
                    if loop is not None and loop.is_running():
                        loop.quit()
                return True

            appsink.connect("new-sample", on_new_sample)

            bus = pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", on_bus_message)

            pipeline.set_state(Gst.State.PLAYING)
            print(f"[rtsp_stream] cam={cam_id} pipeline running")

            loop.run()

            raise RuntimeError("GStreamer loop exited; rebuilding pipeline")

        except KeyboardInterrupt:
            print("\n[rtsp_stream] stopping...")
            break
        except Exception as e:
            print(f"[rtsp_stream] cam={cam_id} restartable failure: {e}")
            time.sleep(args.restart_delay_s)
        finally:
            try:
                if bus is not None:
                    bus.remove_signal_watch()
            except Exception:
                pass
            try:
                if pipeline is not None:
                    pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass

    pub.close(0)


if __name__ == "__main__":
    main()