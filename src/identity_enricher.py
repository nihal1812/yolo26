#!/usr/bin/env python3
"""
identity_enricher.py

Shared in-process enrichment stage.

Responsibilities:
- cache (cam_id, person_track_id) -> global_person_id
- enrich scores / decisions / alerts / clip_refs
- optionally defer events briefly if mapping is not yet available

Scalable design:
- in-process only
- bounded caches
- bounded deferred queue per key
- transport-agnostic optional publisher hooks
"""

import time
from collections import deque
from typing import Dict, Any, Tuple, Optional, List


def safe_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return default


class TTLCache:
    def __init__(self, ttl_s: float, max_items: int = 50000):
        self.ttl_s = float(ttl_s)
        self.max_items = int(max_items)
        self.store: Dict[Any, Tuple[float, Any]] = {}
        self.fifo = deque(maxlen=max_items)

    def put(self, key, value):
        now = time.time()
        self.store[key] = (now, value)
        self.fifo.append((now, key))
        self.prune()

    def get(self, key):
        item = self.store.get(key, None)
        if item is None:
            return None
        ts, val = item
        if (time.time() - ts) > self.ttl_s:
            self.store.pop(key, None)
            return None
        return val

    def pop(self, key, default=None):
        item = self.store.pop(key, None)
        if item is None:
            return default
        _ts, val = item
        return val

    def prune(self):
        now = time.time()
        while self.fifo:
            ts, key = self.fifo[0]
            if (now - ts) <= self.ttl_s and len(self.store) <= self.max_items:
                break
            self.fifo.popleft()
            self.store.pop(key, None)


def make_local_key(cam_id: str, person_track_id: int):
    return (str(cam_id), int(person_track_id))


def enrich_payload(obj: dict, global_person_id: Optional[int], mapping_debug: dict) -> dict:
    out = dict(obj)
    out["global_person_id"] = int(global_person_id) if global_person_id is not None else None
    out["identity_enriched"] = bool(global_person_id is not None)
    out["identity_debug"] = mapping_debug
    return out


class NullEnrichedPublisher:
    def publish_enriched(self, internal_name: str, payload: dict):
        return


class IdentityEnricher:
    def __init__(
        self,
        mapping_ttl_s: float = 120.0,
        pending_ttl_s: float = 3.0,
        max_pending: int = 50000,
        max_pending_per_key: int = 32,
        defer_unmapped: bool = False,
        publisher=None,
    ):
        self.local_to_global = TTLCache(ttl_s=mapping_ttl_s, max_items=max_pending)
        self.pending_events = TTLCache(ttl_s=pending_ttl_s, max_items=max_pending)
        self.defer_unmapped = bool(defer_unmapped)
        self.max_pending_per_key = int(max_pending_per_key)
        self.publisher = publisher if publisher is not None else NullEnrichedPublisher()

    def _publish_many(self, items: List[Tuple[str, dict]]):
        for internal_name, payload in items:
            try:
                self.publisher.publish_enriched(internal_name, payload)
            except Exception:
                pass

    def handle_global_track(self, obj: dict):
        cam_id = obj.get("cam_id", None)
        pid = safe_int(obj.get("person_track_id", None), None)
        gid = safe_int(obj.get("global_person_id", None), None)
        if cam_id is None or pid is None or gid is None:
            return []

        key = make_local_key(str(cam_id), int(pid))
        self.local_to_global.put(key, gid)

        flushed = []
        pend = self.pending_events.pop(key, default=None)
        if pend:
            for internal_name, pending_obj in pend:
                enriched = enrich_payload(
                    pending_obj,
                    gid,
                    {
                        "source": "pending_flush_after_global_track",
                        "local_key": [str(cam_id), int(pid)],
                    },
                )
                flushed.append((internal_name, enriched))

        self._publish_many(flushed)
        return flushed

    def handle_local_event(self, internal_name: str, obj: dict):
        cam_id = obj.get("cam_id", None)
        pid = safe_int(obj.get("person_track_id", None), None)
        if cam_id is None or pid is None:
            return "drop", None

        key = make_local_key(str(cam_id), int(pid))
        gid = self.local_to_global.get(key)

        if gid is not None:
            enriched = enrich_payload(
                obj,
                gid,
                {
                    "source": "cache_hit",
                    "local_key": [str(cam_id), int(pid)],
                },
            )
            try:
                self.publisher.publish_enriched(internal_name, enriched)
            except Exception:
                pass
            return "publish", enriched

        if self.defer_unmapped:
            current = self.pending_events.get(key)
            if current is None:
                current = []

            current.append((internal_name, obj))

            # bound pending list per key, keep newest items
            if len(current) > self.max_pending_per_key:
                current = current[-self.max_pending_per_key:]

            self.pending_events.put(key, current)
            return "deferred", None

        enriched = enrich_payload(
            obj,
            None,
            {
                "source": "cache_miss_passthrough",
                "local_key": [str(cam_id), int(pid)],
            },
        )
        try:
            self.publisher.publish_enriched(internal_name, enriched)
        except Exception:
            pass
        return "publish", enriched

    def prune(self):
        self.local_to_global.prune()
        self.pending_events.prune()
