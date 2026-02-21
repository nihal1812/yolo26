#!/usr/bin/env python3
import time
import json
import argparse
import yaml
import zmq

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init(None)

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_pipeline(rtsp_url: str, codec: str, latency_ms: int, transport: str):
    """
    Outputs *encoded* access units via appsink (NOT decoded frames).
    """
    codec = codec.lower()
    if codec == "h265":
        depay = "rtph265depay"
        parse = "h265parse config-interval=-1"
    else:
        depay = "rtph264depay"
        parse = "h264parse config-interval=-1"

    proto = "tcp" if transport.lower() == "tcp" else "udp"
    # protocols property: tcp=4 udp=2 (often). Using 'protocols=tcp' string is common but varies.
    # We'll use 'protocols=tcp' or 'protocols=udp' (works on most setups).
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
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    cam_id = cfg["system"]["cam_id"]
    rtsp_url = cfg["rtsp"]["url"]
    codec = cfg["rtsp"].get("codec", "h264")
    latency_ms = int(cfg["rtsp"].get("latency_ms", 120))
    transport = cfg["rtsp"].get("transport", "tcp")

    video_bind = cfg["zmq"]["video"]["bind"]
    topic = cfg["zmq"]["video"]["topic"]
    sndhwm = int(cfg["zmq"]["video"].get("sndhwm", 3))

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, sndhwm)
    pub.bind(video_bind)

    pipeline = build_pipeline(rtsp_url, codec, latency_ms, transport)
    appsink = pipeline.get_by_name("encsink")

    frame_id = 0

    def on_new_sample(sink):
        nonlocal frame_id
        sample = sink.emit("pull-sample")
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
                topic.encode(),
                json.dumps(header).encode(),
                mapinfo.data
            ])
            frame_id += 1
        finally:
            buf.unmap(mapinfo)

        return Gst.FlowReturn.OK

    appsink.connect("new-sample", on_new_sample)

    pipeline.set_state(Gst.State.PLAYING)
    loop = GLib.MainLoop()

    print(f"[rtsp_stream] PUB video @ {video_bind} topic={topic} codec={codec} cam_id={cam_id}")
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[rtsp_stream] stopping...")
    finally:
        pipeline.set_state(Gst.State.NULL)
        pub.close()

if __name__ == "__main__":
    main()
