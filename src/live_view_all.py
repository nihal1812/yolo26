#!/usr/bin/env python3
import argparse
import json
import math
import time

import cv2
import zmq
import numpy as np

try:
    from src.config_utils import load_cfg
except ImportError:
    from config_utils import load_cfg


def b2s(x):
    return x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, bytearray)) else str(x)


def parse_cams_arg(cams_arg, cfg):
    if cams_arg:
        return [c.strip() for c in cams_arg.split(",") if c.strip()]
    return list(cfg.get("system", {}).get("active_cams", []))


def get_live_view_cfg(cfg):
    return cfg.get("live_view", {})


def get_cam_live_endpoint(cfg, cam_id):
    lv = get_live_view_cfg(cfg)
    zmq_cfg = lv.get("zmq", {})
    cam_cfg = zmq_cfg.get(cam_id, {})
    connect = cam_cfg.get("connect")
    topic = cam_cfg.get("topic")
    if not connect or not topic:
        raise KeyError(f"Missing live_view.zmq.{cam_id}.connect/topic")
    return connect, topic


def decode_message(parts):
    if len(parts) < 2:
        return None, None, None

    topic = b2s(parts[0])
    meta = {}

    if len(parts) >= 3:
        try:
            meta = json.loads(b2s(parts[1]))
        except Exception:
            meta = {}
        jpg_b = parts[2]
    else:
        jpg_b = parts[1]

    arr = np.frombuffer(jpg_b, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return topic, meta, frame


def fit_with_letterbox(img, dst_w, dst_h):
    if img is None:
        return np.zeros((dst_h, dst_w, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((dst_h, dst_w, 3), dtype=np.uint8)

    scale = min(dst_w / w, dst_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)

    x0 = (dst_w - nw) // 2
    y0 = (dst_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def draw_tile_header(tile, title, age_s=None):
    h, w = tile.shape[:2]
    bar_h = 28
    cv2.rectangle(tile, (0, 0), (w - 1, bar_h), (30, 30, 30), -1)

    txt = title
    if age_s is not None:
        txt += f" | age={age_s:.1f}s"

    cv2.putText(
        tile,
        txt,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def make_placeholder(cam_id, tile_w, tile_h, msg="no signal"):
    img = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    img[:] = 15
    cv2.putText(img, cam_id, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, msg, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2, cv2.LINE_AA)
    return img


def calc_grid(n):
    if n <= 0:
        return 1, 1
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cams", default=None)
    ap.add_argument("--window", default="Live View - All Cameras")
    ap.add_argument("--tile_w", type=int, default=640)
    ap.add_argument("--tile_h", type=int, default=360)
    ap.add_argument("--stale_s", type=float, default=3.0)
    ap.add_argument("--poll_ms", type=int, default=30)
    ap.add_argument("--max_fps", type=float, default=20.0)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cams = parse_cams_arg(args.cams, cfg)
    if not cams:
        raise RuntimeError("No cameras found")

    ctx = zmq.Context.instance()
    poller = zmq.Poller()
    sockets = {}
    state = {}

    for cam_id in cams:
        connect, topic = get_cam_live_endpoint(cfg, cam_id)

        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.LINGER, 0)
        sub.setsockopt(zmq.RCVHWM, 2)
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.connect(connect)
        sub.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))

        sockets[sub] = {"cam_id": cam_id, "topic": topic, "connect": connect}
        poller.register(sub, zmq.POLLIN)

        state[cam_id] = {
            "frame": None,
            "meta": {},
            "last_rx": 0.0,
            "topic": topic,
            "connect": connect,
        }

        print(f"[live_view_all] SUB {cam_id} {connect} topic={topic}")

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window, args.tile_w * 2, args.tile_h * 2)

    rows, cols = calc_grid(len(cams))
    min_frame_period = 1.0 / max(1e-6, args.max_fps)
    last_show = 0.0

    try:
        while True:
            events = dict(poller.poll(timeout=args.poll_ms))

            for sock, ev in events.items():
                if ev != zmq.POLLIN:
                    continue
                try:
                    parts = sock.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue
                except Exception as e:
                    print(f"[live_view_all] recv error: {e}")
                    continue

                topic, meta, frame = decode_message(parts)
                if frame is None:
                    continue

                cam_id = sockets[sock]["cam_id"]
                state[cam_id]["frame"] = frame
                state[cam_id]["meta"] = meta or {}
                state[cam_id]["last_rx"] = time.time()

            now = time.time()
            if (now - last_show) < min_frame_period:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            tiles = []
            for cam_id in cams:
                st = state[cam_id]
                age_s = None if st["last_rx"] <= 0 else (now - st["last_rx"])

                if st["frame"] is None:
                    tile = make_placeholder(cam_id, args.tile_w, args.tile_h, msg="waiting for overlay")
                elif age_s is not None and age_s > args.stale_s:
                    tile = fit_with_letterbox(st["frame"], args.tile_w, args.tile_h)
                    cv2.putText(
                        tile,
                        "STALE",
                        (20, args.tile_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    tile = fit_with_letterbox(st["frame"], args.tile_w, args.tile_h)

                fid = st["meta"].get("frame_id", None)
                label = cam_id if fid is None else f"{cam_id} | frame={fid}"
                tile = draw_tile_header(tile, label, age_s=age_s)
                tiles.append(tile)

            while len(tiles) < rows * cols:
                tiles.append(np.zeros((args.tile_h, args.tile_w, 3), dtype=np.uint8))

            row_imgs = []
            idx = 0
            for _ in range(rows):
                row_imgs.append(np.hstack(tiles[idx:idx + cols]))
                idx += cols
            dashboard = np.vstack(row_imgs)

            cv2.imshow(args.window, dashboard)
            last_show = now

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    except KeyboardInterrupt:
        print("\n[live_view_all] stopping...")
    finally:
        for sock in sockets:
            try:
                poller.unregister(sock)
            except Exception:
                pass
            try:
                sock.close(0)
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
