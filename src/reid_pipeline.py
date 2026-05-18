#!/usr/bin/env python3
"""
reid_pipeline.py

Unified in-process ReID pipeline runtime.
"""

import argparse
import json
import signal
import time
from typing import Dict, Tuple

import zmq

from config_utils import load_cfg, get_active_cams, get_zmq_endpoint, local_connect_addr
from reid_node import ReIDNode
from identity_stitcher import IdentityStitcher
from identity_enricher import IdentityEnricher
from incident_builder import IncidentBuilder


def safe_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return default


def safe_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def compact_json(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def make_sub_socket(ctx, connect_addr, topic, rcvhwm=1000, latest_only=False):
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.RCVHWM, int(rcvhwm))
    if latest_only:
        sub.setsockopt(zmq.CONFLATE, 1)
    sub.connect(connect_addr)
    sub.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
    return sub


class ZmqPublisher:
    def __init__(self, ctx, cfg):
        self.ctx = ctx
        self.cfg = cfg
        self.pubs: Dict[Tuple[str, str], Tuple[object, str]] = {}

    def _open_pub(self, cam_id: str, key: str):
        cache_key = (str(cam_id), str(key))
        if cache_key in self.pubs:
            return self.pubs[cache_key]

        try:
            ep = get_zmq_endpoint(self.cfg, cam_id, key)
        except Exception:
            return None

        bind_addr = ep["bind"]
        topic = ep["topic"]
        sndhwm = int(ep.get("sndhwm", 1000))

        pub = self.ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.LINGER, 0)
        pub.setsockopt(zmq.SNDHWM, sndhwm)
        pub.bind(bind_addr)

        self.pubs[cache_key] = (pub, topic)
        print(f"[reid_pipeline][pub] cam={cam_id} key={key} bind={bind_addr} topic={topic}")
        return self.pubs[cache_key]

    def _send_obj(self, cam_id: str, key: str, header: dict, payload: dict):
        opened = self._open_pub(cam_id, key)
        if opened is None:
            return

        pub, topic = opened
        pub.send_multipart([
            topic.encode("utf-8"),
            compact_json(header),
            compact_json(payload),
        ])

    def publish_reid_embedding(self, cam_id: str, obj: dict, maxlen=0):
        header = {
            "type": "reid_embedding",
            "event_id": obj.get("event_id"),
            "cam_id": obj.get("cam_id"),
            "person_track_id": obj.get("person_track_id"),
            "frame_id": obj.get("frame_id"),
            "stamp_ns": obj.get("stamp_ns"),
        }
        self._send_obj(cam_id, "reid_embeddings", header, obj)

    def publish_global_track(self, obj: dict, maxlen=0):
        cam_id = str(obj.get("cam_id", ""))
        header = {
            "type": "global_track",
            "event_id": obj.get("event_id"),
            "cam_id": obj.get("cam_id"),
            "person_track_id": obj.get("person_track_id"),
            "global_person_id": obj.get("global_person_id"),
            "frame_id": obj.get("frame_id"),
            "stamp_ns": obj.get("stamp_ns"),
        }
        self._send_obj(cam_id, "global_tracks", header, obj)

    def publish_enriched(self, internal_name: str, obj: dict, maxlen=0):
        cam_id = str(obj.get("cam_id", ""))

        key_map = {
            "scores": "scores_enriched",
            "decisions": "decisions_enriched",
            "alerts": "alerts_enriched",
            "clip_refs": "clip_refs_enriched",
        }
        key = key_map.get(internal_name)
        if key is None:
            return

        frame_id = safe_int(obj.get("frame_id_end", None), None)
        if frame_id is None:
            frame_id = safe_int(obj.get("frame_id", -1), -1)

        stamp_ns = safe_int(obj.get("stamp_ns_end", None), None)
        if stamp_ns is None:
            stamp_ns = safe_int(obj.get("stamp_ns", 0), 0)

        header = {
            "type": f"{internal_name}_enriched",
            "event_id": obj.get("event_id"),
            "cam_id": cam_id,
            "person_track_id": obj.get("person_track_id"),
            "global_person_id": obj.get("global_person_id"),
            "frame_id": frame_id,
            "stamp_ns": stamp_ns,
        }
        self._send_obj(cam_id, key, header, obj)

    def publish_incident(self, obj: dict, maxlen=0):
        latest = obj.get("latest_signal", {})
        cam_id = str(latest.get("cam_id", ""))

        header = {
            "type": "incident",
            "incident_id": obj.get("incident_id"),
            "event_id": latest.get("event_id"),
            "cam_id": latest.get("cam_id"),
            "person_track_id": latest.get("person_track_id"),
            "global_person_id": obj.get("global_person_id"),
            "frame_id": latest.get("frame_id"),
            "stamp_ns": latest.get("stamp_ns"),
        }
        self._send_obj(cam_id, "incidents", header, obj)

    def publish_incident_alert(self, obj: dict, maxlen=0):
        latest = obj.get("latest_signal", {})
        cam_id = str(latest.get("cam_id", ""))

        header = {
            "type": "incident_alert",
            "incident_id": obj.get("incident_id"),
            "event_id": latest.get("event_id"),
            "cam_id": latest.get("cam_id"),
            "person_track_id": latest.get("person_track_id"),
            "global_person_id": obj.get("global_person_id"),
            "frame_id": latest.get("frame_id"),
            "stamp_ns": latest.get("stamp_ns"),
        }
        self._send_obj(cam_id, "incident_alerts", header, obj)

    def close(self):
        for (pub, _topic) in self.pubs.values():
            try:
                pub.close(0)
            except Exception:
                pass
        self.pubs.clear()


class UnifiedReIDPipeline:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.running = True

        self.active_cams = get_active_cams(cfg)

        ccfg = cfg.get("identity_stitcher", {})
        self.stitcher = IdentityStitcher(
            active_cams=self.active_cams,
            max_idle_s=float(ccfg.get("max_idle_s", 90.0)),
            match_threshold=float(ccfg.get("match_threshold", 0.72)),
            same_cam_match_threshold=float(ccfg.get("same_cam_match_threshold", 0.88)),
            same_cam_local_reuse_threshold=float(ccfg.get("same_cam_local_reuse_threshold", 0.75)),
            min_transition_s_default=float(ccfg.get("min_transition_s_default", 0.0)),
            max_transition_s_default=float(ccfg.get("max_transition_s_default", 30.0)),
            topology=ccfg.get("topology", {}) or {},
            prototype_momentum=float(ccfg.get("prototype_momentum", 0.08)),
            same_cam_reuse_max_dt_s=float(ccfg.get("same_cam_reuse_max_dt_s", 8.0)),
            min_cross_cam_dt_s=float(ccfg.get("min_cross_cam_dt_s", 0.10)),
            debug_log=safe_bool(ccfg.get("debug_log", False), False),
            health_log_every_s=float(ccfg.get("health_log_every_s", 10.0)),
        )

        self.enricher = IdentityEnricher(
            mapping_ttl_s=float(args.mapping_ttl_s),
            pending_ttl_s=float(args.pending_ttl_s),
            max_pending=int(args.max_pending),
            max_pending_per_key=int(args.max_pending_per_key),
            defer_unmapped=bool(args.defer_unmapped),
            publisher=None,
        )

        self.incident_builder = IncidentBuilder(
            incident_ttl_s=float(args.incident_ttl_s),
            evidence_window_s=float(args.evidence_window_s),
            min_cams_for_cross_camera=int(args.min_cams_for_cross_camera),
            incident_open_thr=float(args.incident_open_thr),
            incident_alert_thr=float(args.incident_alert_thr),
            incident_alert_on_any_alert=bool(args.incident_alert_on_any_alert),
            max_history=int(args.max_history),
            dedupe_ttl_s=float(args.incident_dedupe_ttl_s),
            publisher=None,
        )

        self.ctx = zmq.Context.instance()
        self.publisher = ZmqPublisher(self.ctx, cfg)
        self.poller = zmq.Poller()

        self.reid_nodes: Dict[str, ReIDNode] = {}
        self.sock_to_cam: Dict[object, str] = {}
        self.local_event_meta: Dict[object, Tuple[str, str]] = {}

        for cam_id in self.active_cams:
            node = ReIDNode(
                cfg=cfg,
                cam_id=cam_id,
                publisher=self.publisher,
                device=args.device,
                emb_dim=args.emb_dim,
                latest_only=args.latest_only,
                max_frame_delta=args.max_frame_delta,
                use_seg_mask=args.use_seg_mask,
                min_bbox_h=args.min_bbox_h,
                min_bbox_w=args.min_bbox_w,
                pad_frac=args.pad_frac,
            )
            self.reid_nodes[cam_id] = node

            for sock in node.sockets():
                self.poller.register(sock, zmq.POLLIN)
                self.sock_to_cam[sock] = cam_id

        self._setup_local_event_subscribers()

    def _setup_local_event_subscribers(self):
        for cam_id in self.active_cams:
            for internal_name, zmq_key in [
                ("scores", "scores"),
                ("decisions", "decisions"),
                ("alerts", "alerts"),
            ]:
                try:
                    ep = get_zmq_endpoint(self.cfg, cam_id, zmq_key)
                except Exception:
                    continue

                connect_addr = local_connect_addr(ep["bind"])
                topic = ep["topic"]
                rcvhwm = int(ep.get("rcvhwm", self.args.local_rcvhwm))
                latest_only = bool(ep.get("latest_only", self.args.local_latest_only))

                sub = make_sub_socket(
                    self.ctx,
                    connect_addr=connect_addr,
                    topic=topic,
                    rcvhwm=rcvhwm,
                    latest_only=latest_only,
                )
                self.poller.register(sub, zmq.POLLIN)
                self.local_event_meta[sub] = (internal_name, cam_id)

                print(f"[reid_pipeline][sub] cam={cam_id} {internal_name} {connect_addr} topic={topic}")

            if self.args.include_clip_refs:
                try:
                    ep = get_zmq_endpoint(self.cfg, cam_id, "clip_refs")
                    connect_addr = local_connect_addr(ep["bind"])
                    topic = ep["topic"]
                    rcvhwm = int(ep.get("rcvhwm", self.args.local_rcvhwm))
                    latest_only = bool(ep.get("latest_only", self.args.local_latest_only))

                    sub = make_sub_socket(
                        self.ctx,
                        connect_addr=connect_addr,
                        topic=topic,
                        rcvhwm=rcvhwm,
                        latest_only=latest_only,
                    )
                    self.poller.register(sub, zmq.POLLIN)
                    self.local_event_meta[sub] = ("clip_refs", cam_id)

                    print(f"[reid_pipeline][sub] cam={cam_id} clip_refs {connect_addr} topic={topic}")
                except Exception:
                    pass

    def stop(self):
        self.running = False

    def process_embedding(self, emb_obj: dict):
        global_track_obj = self.stitcher.process_embedding(emb_obj)
        self.publisher.publish_global_track(global_track_obj)

        flushed = self.enricher.handle_global_track(global_track_obj)
        for internal_name, enriched_obj in flushed:
            self._publish_and_route_enriched(internal_name, enriched_obj)

    def process_local_event(self, internal_name: str, obj: dict):
        action, enriched_obj = self.enricher.handle_local_event(internal_name, obj)
        if action == "publish" and enriched_obj is not None:
            self._publish_and_route_enriched(internal_name, enriched_obj)

    def _publish_and_route_enriched(self, internal_name: str, enriched_obj: dict):
        self.publisher.publish_enriched(internal_name, enriched_obj)

        if internal_name == "decisions":
            incident_obj, incident_alert = self.incident_builder.ingest(enriched_obj, "decision")
            if incident_obj is not None:
                self.publisher.publish_incident(incident_obj)
            if incident_alert is not None:
                self.publisher.publish_incident_alert(incident_alert)

        elif internal_name == "alerts":
            incident_obj, incident_alert = self.incident_builder.ingest(enriched_obj, "alert")
            if incident_obj is not None:
                self.publisher.publish_incident(incident_obj)
            if incident_alert is not None:
                self.publisher.publish_incident_alert(incident_alert)

    def _handle_local_subscriber(self, sock):
        internal_name, _cam_id = self.local_event_meta[sock]
        parts = sock.recv_multipart()
        if len(parts) != 3:
            return

        _topic_b, _header_b, payload_b = parts
        try:
            obj = json.loads(payload_b.decode("utf-8"))
        except Exception:
            return

        self.process_local_event(internal_name, obj)

    def run(self):
        print(f"[reid_pipeline] active_cams={self.active_cams}")
        print("[reid_pipeline] unified in-process pipeline started")
        print("[reid_pipeline] hot path uses ZMQ subscribers + in-process routing")
        print("[reid_pipeline] Redis is not used")

        while self.running:
            try:
                events = dict(self.poller.poll(timeout=self.args.zmq_poll_ms))

                for sock in events.keys():
                    if sock in self.sock_to_cam:
                        cam_id = self.sock_to_cam.get(sock)
                        if cam_id is None:
                            continue

                        node = self.reid_nodes[cam_id]
                        produced = node.handle_socket_event(sock)
                        for emb_obj in produced:
                            self.process_embedding(emb_obj)

                    elif sock in self.local_event_meta:
                        self._handle_local_subscriber(sock)

                self.enricher.prune()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[reid_pipeline] error: {e}")
                time.sleep(0.2)

        self.close()

    def close(self):
        for node in self.reid_nodes.values():
            try:
                node.close()
            except Exception:
                pass

        for sock in list(self.local_event_meta.keys()):
            try:
                self.poller.unregister(sock)
            except Exception:
                pass
            try:
                sock.close(0)
            except Exception:
                pass

        self.publisher.close()
        print("[reid_pipeline] stopped")


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--emb_dim", type=int, default=512)
    ap.add_argument("--latest_only", action="store_true")
    ap.add_argument("--max_frame_delta", type=int, default=3)
    ap.add_argument("--use_seg_mask", action="store_true")
    ap.add_argument("--min_bbox_h", type=int, default=80)
    ap.add_argument("--min_bbox_w", type=int, default=30)
    ap.add_argument("--pad_frac", type=float, default=0.05)

    ap.add_argument("--zmq_poll_ms", type=int, default=20)
    ap.add_argument("--local_rcvhwm", type=int, default=1000)
    ap.add_argument("--local_latest_only", action="store_true")

    ap.add_argument("--mapping_ttl_s", type=float, default=120.0)
    ap.add_argument("--pending_ttl_s", type=float, default=3.0)
    ap.add_argument("--max_pending", type=int, default=50000)
    ap.add_argument("--max_pending_per_key", type=int, default=32)
    ap.add_argument("--defer_unmapped", action="store_true")
    ap.add_argument("--include_clip_refs", action="store_true")

    ap.add_argument("--incident_ttl_s", type=float, default=120.0)
    ap.add_argument("--evidence_window_s", type=float, default=20.0)
    ap.add_argument("--min_cams_for_cross_camera", type=int, default=2)
    ap.add_argument("--incident_open_thr", type=float, default=0.70)
    ap.add_argument("--incident_alert_thr", type=float, default=0.90)
    ap.add_argument("--incident_alert_on_any_alert", action="store_true")
    ap.add_argument("--max_history", type=int, default=100)
    ap.add_argument("--incident_dedupe_ttl_s", type=float, default=60.0)

    return ap


def main():
    args = build_arg_parser().parse_args()
    cfg = load_cfg(args.config)

    pipeline = UnifiedReIDPipeline(cfg, args)

    def _sig_handler(sig, frame):
        pipeline.stop()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    pipeline.run()


if __name__ == "__main__":
    main()
