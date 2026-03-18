#!/usr/bin/env python3
import argparse
import json
import time
from collections import deque

import redis
import zmq
import numpy as np
import cv2
import requests

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from config_utils import load_cfg, get_stream, get_zmq_endpoint, local_connect_addr

Gst.init(None)


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


def resolve_decisions_stream(cfg, cam_id: str):
    try:
        return get_stream(cfg, "decisions_enriched", cam_id)
    except Exception:
        return get_stream(cfg, "decisions", cam_id)


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


class GstDecoder:
    def __init__(self, codec="h264"):
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
        self._latest = None
        self.appsink.connect("new-sample", self._on_sample)
        self.pipeline.set_state(Gst.State.PLAYING)

    def _on_sample(self, sink):
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
    ap.add_argument("--cam_id", required=True)
    ap.add_argument("--frame_buffer", type=int, default=60)
    ap.add_argument("--pose_buffer", type=int, default=120)
    ap.add_argument("--decisions_block_ms", type=int, default=100)
    ap.add_argument("--decisions_count", type=int, default=100)

    ap.add_argument("--ema_alpha", type=float, default=0.30)
    ap.add_argument("--state_ttl_s", type=float, default=4.0)
    ap.add_argument("--decision_freshness_s", type=float, default=3.0)
    ap.add_argument("--track_memory_ttl_s", type=float, default=4.0)
    ap.add_argument("--iou_match_thr", type=float, default=0.20)
    ap.add_argument("--iou_loose_thr", type=float, default=0.08)

    ap.add_argument("--send_every_n_frames", type=int, default=1)
    ap.add_argument("--decisions_stream", default=None)
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
    )
    rdb.ping()

    decisions_stream = args.decisions_stream or resolve_decisions_stream(cfg, cam_id)

    p3 = cfg.get("pipeline3", {})
    wh = p3.get("webhooks", {})
    overlay_url = wh.get("overlay_url")
    timeout_s = float(wh.get("timeout_s", 3.0))
    if not overlay_url:
        raise RuntimeError("pipeline3.webhooks.overlay_url missing in config")

    video_cfg = get_zmq_endpoint(cfg, cam_id, "video")
    pose_cfg = get_zmq_endpoint(cfg, cam_id, "pose_features")

    ctx = zmq.Context.instance()

    video_connect = local_connect_addr(video_cfg["bind"])
    video_topic = video_cfg["topic"]

    pose_connect = local_connect_addr(pose_cfg["bind"])
    pose_topic = pose_cfg["topic"]

    sub_v = ctx.socket(zmq.SUB)
    sub_v.connect(video_connect)
    sub_v.setsockopt(zmq.SUBSCRIBE, video_topic.encode("utf-8"))

    sub_p = ctx.socket(zmq.SUB)
    sub_p.connect(pose_connect)
    sub_p.setsockopt(zmq.SUBSCRIBE, pose_topic.encode("utf-8"))

    poller = zmq.Poller()
    poller.register(sub_v, zmq.POLLIN)
    poller.register(sub_p, zmq.POLLIN)

    decoder = GstDecoder(codec=codec)

    frame_buf = {}
    pose_buf = {}
    frame_fifo = deque(maxlen=args.frame_buffer)
    pose_fifo = deque(maxlen=args.pose_buffer)

    # render state keyed by stable state_id
    # state_id = gid:<gid> when possible, else pid:<pid>
    render_states = {}

    # local tracker memory: pid -> state_id
    local_track_memory = {}

    last_decision_id = "0-0"
    sent_frame_count = 0

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

    def state_id_from_decision(dec):
        gid = dec.get("global_person_id", None)
        pid = int(dec.get("person_track_id", -1))
        if gid is not None:
            try:
                return f"gid:{int(gid)}"
            except Exception:
                pass
        return f"pid:{pid}"

    def prune_state():
        now = time.time()

        dead_states = []
        for sid, st in render_states.items():
            if (now - st.get("updated_at", 0.0)) > args.state_ttl_s:
                dead_states.append(sid)
        for sid in dead_states:
            render_states.pop(sid, None)

        dead_tracks = []
        for pid, tm in local_track_memory.items():
            if (now - tm.get("updated_at", 0.0)) > args.track_memory_ttl_s:
                dead_tracks.append(pid)
        for pid in dead_tracks:
            local_track_memory.pop(pid, None)

    def update_render_state_from_decision(dec):
        pid = int(dec.get("person_track_id", -1))
        if pid < 0:
            return

        raw_score = dec.get("score", None)
        S = dec.get("S", None)
        gid = dec.get("global_person_id", None)
        suspicion = S if S is not None else raw_score
        suspicion = clamp01(0.0 if suspicion is None else suspicion)

        sid = state_id_from_decision(dec)
        prev = render_states.get(sid, None)

        if prev is None:
            display_s = suspicion
            last_bbox = None
            last_matched_pid = pid
        else:
            old = float(prev.get("display_s", suspicion))
            display_s = (1.0 - args.ema_alpha) * old + args.ema_alpha * suspicion
            last_bbox = prev.get("last_bbox", None)
            last_matched_pid = prev.get("last_matched_pid", pid)

        now = time.time()
        render_states[sid] = {
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
            "identity_enriched": bool(dec.get("identity_enriched", False)),
            "updated_at": now,
            "last_bbox": last_bbox,
            "last_matched_pid": last_matched_pid,
            "match_source": "decision_update",
        }

        local_track_memory[pid] = {
            "state_id": sid,
            "updated_at": now,
        }

    def is_state_fresh(st):
        now = time.time()
        if (now - st.get("updated_at", 0.0)) > args.decision_freshness_s:
            return False
        return True

    def score_state_match(person_bbox, pid, st):
        score = -1e9
        reasons = []

        if not is_state_fresh(st):
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
        if iou >= args.iou_match_thr:
            reasons.append(f"iou_strong:{iou:.3f}")
        elif iou >= args.iou_loose_thr:
            reasons.append(f"iou_loose:{iou:.3f}")

        age = max(0.0, time.time() - float(st.get("updated_at", time.time())))
        freshness_bonus = max(0.0, 2.0 - age)
        score += freshness_bonus
        reasons.append(f"fresh:{freshness_bonus:.2f}")

        return score, reasons

    def match_person_to_state(person):
        pid = int(person.get("track_id", -1))
        pb = person.get("bbox_xyxy", None)
        if pid < 0 or pb is None:
            return None, "none"

        # 1) direct local-track memory
        mem = local_track_memory.get(pid, None)
        if mem is not None:
            sid = mem.get("state_id")
            st = render_states.get(sid)
            if st is not None and is_state_fresh(st):
                iou = iou_xyxy(pb, st.get("last_bbox", None))
                if st.get("pid_hint") == pid or st.get("last_matched_pid") == pid or iou >= args.iou_loose_thr:
                    return sid, "track_memory"

        # 2) exact fresh pid_hint search
        for sid, st in render_states.items():
            if not is_state_fresh(st):
                continue
            if st.get("pid_hint", None) == pid:
                return sid, "pid_hint"

        # 3) IoU / freshness fallback across all fresh states
        best_sid = None
        best_score = -1e9
        best_iou = 0.0

        for sid, st in render_states.items():
            sc, _ = score_state_match(pb, pid, st)
            iou = iou_xyxy(pb, st.get("last_bbox", None))
            if sc > best_score:
                best_score = sc
                best_sid = sid
                best_iou = iou

        if best_sid is not None and best_iou >= args.iou_match_thr:
            return best_sid, "iou_fallback"

        return None, "unmatched"

    print(f"[bbox_overlay] cam_id={cam_id}")
    print(f"[bbox_overlay] SUB video {video_connect} topic={video_topic}")
    print(f"[bbox_overlay] SUB pose  {pose_connect} topic={pose_topic}")
    print(f"[bbox_overlay] decisions={decisions_stream}")
    print(f"[bbox_overlay] POST {overlay_url}")

    try:
        while True:
            events = dict(poller.poll(timeout=10))

            if sub_v in events:
                _, header_b, enc = sub_v.recv_multipart()
                try:
                    header = json.loads(header_b.decode("utf-8"))
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
                    pose = json.loads(payload_b.decode("utf-8"))
                    fid = int(pose.get("frame_id", 0))
                    if fid > 0:
                        put_buf(pose_buf, pose_fifo, fid, pose)
                except Exception:
                    pass

            dec_streams = rdb.xread(
                {decisions_stream: last_decision_id},
                block=args.decisions_block_ms,
                count=args.decisions_count,
            )
            if dec_streams:
                for mid, fields in parse_xread(dec_streams):
                    last_decision_id = mid
                    js = fields.get(b"json", b"{}")
                    try:
                        dec = json.loads(b2s(js))
                        update_render_state_from_decision(dec)
                    except Exception:
                        pass

            prune_state()

            if not frame_buf or not pose_buf:
                continue

            latest_fid = frame_fifo[-1] if frame_fifo else None
            if latest_fid is None:
                continue

            frame_fid, frame_item = find_nearest(frame_buf, latest_fid, max_delta=0)
            pose_fid, pose = find_nearest(pose_buf, latest_fid, max_delta=3)

            if frame_item is None or pose is None:
                continue

            _, frame = frame_item
            out = frame.copy()

            people = pose.get("people", [])
            active_state_ids = set()

            for person in people:
                pid = int(person.get("track_id", -1))
                if pid < 0:
                    continue

                pb = person.get("bbox_xyxy", None)
                kp = person.get("keypoints_xy", None)
                if not pb:
                    continue

                sid, match_source = match_person_to_state(person)
                st = render_states.get(sid) if sid is not None else None

                display_s = float(st.get("display_s", 0.0)) if st else 0.0
                raw_score = st.get("raw_score", None) if st else None
                S = st.get("S", None) if st else None
                gid = st.get("global_person_id", None) if st else None

                color = suspicion_to_color(display_s)

                x1, y1, x2, y2 = [int(v) for v in pb]
                x1 = clamp(x1, 0, out.shape[1] - 1)
                x2 = clamp(x2, 0, out.shape[1] - 1)
                y1 = clamp(y1, 0, out.shape[0] - 1)
                y2 = clamp(y2, 0, out.shape[0] - 1)

                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

                if st is not None:
                    st["last_bbox"] = [float(v) for v in pb]
                    st["last_matched_pid"] = int(pid)
                    st["updated_at"] = time.time()
                    st["match_source"] = match_source
                    active_state_ids.add(sid)

                    local_track_memory[pid] = {
                        "state_id": sid,
                        "updated_at": time.time(),
                    }

                score_txt = "na" if raw_score is None else f"{float(raw_score):.2f}"
                s_txt = "na" if S is None else f"{float(S):.2f}"
                gid_txt = "na" if gid is None else str(int(gid))
                src_txt = match_source
                disp_txt = f"pid={pid} gid={gid_txt} score={score_txt} S={s_txt} disp={display_s:.2f} src={src_txt}"
                cv2.putText(
                    out,
                    disp_txt,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    color,
                    2,
                )

                if isinstance(kp, list):
                    for pt in kp:
                        if not pt or len(pt) < 2:
                            continue
                        px, py = int(pt[0]), int(pt[1])
                        if 0 <= px < out.shape[1] and 0 <= py < out.shape[0]:
                            cv2.circle(out, (px, py), 2, color, -1)

            sent_frame_count += 1
            if args.send_every_n_frames > 1 and (sent_frame_count % args.send_every_n_frames != 0):
                continue

            ok, jpg = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                continue

            meta = {
                "cam_id": cam_id,
                "frame_id": int(frame_fid),
                "pose_frame_id": int(pose_fid) if pose_fid is not None else None,
                "active_tracks": sorted([int(p.get("track_id", -1)) for p in people if int(p.get("track_id", -1)) >= 0]),
                "active_global_ids": sorted(
                    [int(render_states[s]["global_person_id"]) for s in active_state_ids
                     if render_states.get(s) is not None and render_states[s].get("global_person_id") is not None]
                ),
                "active_state_ids": sorted(list(active_state_ids)),
                "mode": "continuous_human_pose_overlay_robust_assoc",
            }

            ok_post, status_code, err = post_overlay_with_retry(
                overlay_url, meta, jpg.tobytes(), timeout_s, retries=2
            )
            if ok_post:
                print(f"[bbox_overlay] cam={cam_id} sent overlay frame={frame_fid} status={status_code}")
            else:
                print(f"[bbox_overlay] cam={cam_id} overlay error frame={frame_fid}: {err}")

    except KeyboardInterrupt:
        print("\n[bbox_overlay] stopping...")
    finally:
        decoder.close()
        sub_v.close()
        sub_p.close()


if __name__ == "__main__":
    main()