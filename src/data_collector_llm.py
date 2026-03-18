#!/usr/bin/env python3
"""
data_collector_llm.py

Collects training/eval data without blocking realtime.

Consumes:
  - scores_enriched:{cam} or scores:{cam}
  - decisions_enriched:{cam} or decisions:{cam}
  - alerts_enriched:{cam} or alerts:{cam}
  - scalars_clip:{cam}
  - feedback:{cam}
  - optionally clips:{cam}
  - clip_refs_enriched:{cam} or clip_refs:{cam}

Produces:
  - train_events:{cam}
"""

import argparse
import json
import os
import time
from collections import deque
from typing import Optional, Tuple, List

import redis

from config_utils import load_cfg, get_stream


def b2s(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def safe_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return default


def safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def now_ns():
    return time.time_ns()


def parse_xread(streams):
    out = []
    for sname, msgs in streams:
        s = b2s(sname)
        for mid, fields in msgs:
            out.append((s, b2s(mid), fields))
    return out


def resolve_scores_stream(cfg, cam_id: str):
    try:
        return get_stream(cfg, "scores_enriched", cam_id)
    except Exception:
        return get_stream(cfg, "scores", cam_id)


def resolve_decisions_stream(cfg, cam_id: str):
    try:
        return get_stream(cfg, "decisions_enriched", cam_id)
    except Exception:
        return get_stream(cfg, "decisions", cam_id)


def resolve_alerts_stream(cfg, cam_id: str):
    try:
        return get_stream(cfg, "alerts_enriched", cam_id)
    except Exception:
        return get_stream(cfg, "alerts", cam_id)


def resolve_clip_refs_stream(cfg, cam_id: str):
    try:
        return get_stream(cfg, "clip_refs_enriched", cam_id)
    except Exception:
        try:
            return get_stream(cfg, "clip_refs", cam_id)
        except Exception:
            return f"clip_refs:{cam_id}"


def extract_feedback_text(obj: dict) -> str:
    if not isinstance(obj, dict):
        return ""

    candidates = [
        obj.get("comment"),
        obj.get("description"),
        obj.get("feedback_text"),
        obj.get("text"),
        obj.get("notes"),
        obj.get("reason"),
        obj.get("explanation"),
        obj.get("summary"),
    ]

    parts = []
    for item in candidates:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            parts.append(s)

    return "\n".join(parts).strip()


def summarize_reasoning_steps(text: str) -> List[str]:
    if not text:
        return []

    parts = []
    for block in text.splitlines():
        block = block.strip(" \t-")
        if not block:
            continue
        sub = [x.strip() for x in block.replace(";", ".").split(".")]
        for s in sub:
            if len(s) >= 6:
                parts.append(" ".join(s.split()))

    return parts[:8]


def validate_feedback_schema(x: dict, allowed_categories: set) -> Optional[dict]:
    if not isinstance(x, dict):
        return None

    label = safe_int(x.get("label", None), None)
    if label not in (0, 1):
        return None

    category = x.get("category", None)
    if category is not None:
        category = str(category).strip().lower()
        if allowed_categories and category not in allowed_categories:
            category = "needs_review" if "needs_review" in allowed_categories else None

    comment = x.get("comment", None)
    if comment is not None:
        comment = str(comment).strip()

    confidence = safe_float(x.get("confidence", 0.0), 0.0)
    confidence = max(0.0, min(1.0, confidence))

    needs_review = bool(x.get("needs_review", False))

    reasoning_summary = x.get("reasoning_summary", [])
    if not isinstance(reasoning_summary, list):
        reasoning_summary = []
    reasoning_summary = [str(v).strip() for v in reasoning_summary if str(v).strip()]

    if not reasoning_summary and comment:
        reasoning_summary = summarize_reasoning_steps(comment)

    return {
        "label": int(label),
        "category": category,
        "comment": comment,
        "confidence": confidence,
        "needs_review": needs_review,
        "reasoning_summary": reasoning_summary,
    }


def heuristic_feedback_parse(feedback_text: str, allowed_categories: set) -> Optional[dict]:
    text = (feedback_text or "").strip()
    if not text:
        return None

    low = text.lower()

    theft_cues = [
        "theft", "steal", "stole", "shoplift", "shoplifting",
        "conceal", "concealed", "hid", "hidden",
        "without paying", "did not pay", "didn't pay",
        "crossed billing", "passed checkout", "left with item",
    ]
    benign_cues = [
        "not theft", "false alarm", "normal", "benign", "no theft",
        "returned the item", "put it back", "paid for it",
        "already owned", "own bag", "customer checked",
    ]
    uncertain_cues = [
        "not sure", "unclear", "maybe", "possibly", "might have",
        "uncertain", "ambiguous",
    ]

    theft_hits = sum(1 for k in theft_cues if k in low)
    benign_hits = sum(1 for k in benign_cues if k in low)
    uncertain_hits = sum(1 for k in uncertain_cues if k in low)
    reasoning_summary = summarize_reasoning_steps(text)

    if theft_hits > benign_hits:
        return validate_feedback_schema({
            "label": 1,
            "category": "theft" if "theft" in allowed_categories else None,
            "comment": "Operator text indicates theft-like sequence of events",
            "confidence": 0.70 if uncertain_hits else 0.84,
            "needs_review": bool(uncertain_hits),
            "reasoning_summary": reasoning_summary,
        }, allowed_categories)

    if benign_hits > theft_hits:
        category = "false_alarm" if "false_alarm" in allowed_categories else (
            "benign" if "benign" in allowed_categories else None
        )
        return validate_feedback_schema({
            "label": 0,
            "category": category,
            "comment": "Operator text indicates benign / false-alarm behavior",
            "confidence": 0.70 if uncertain_hits else 0.84,
            "needs_review": bool(uncertain_hits),
            "reasoning_summary": reasoning_summary,
        }, allowed_categories)

    return validate_feedback_schema({
        "label": 0,
        "category": "needs_review" if "needs_review" in allowed_categories else None,
        "comment": "Ambiguous operator feedback; manual review recommended",
        "confidence": 0.30,
        "needs_review": True,
        "reasoning_summary": reasoning_summary,
    }, allowed_categories)


def build_feedback_json_schema(allowed_categories: List[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "integer", "enum": [0, 1]},
            "category": {"type": "string", "enum": allowed_categories},
            "comment": {"type": "string"},
            "reasoning_summary": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 8,
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "needs_review": {"type": "boolean"},
        },
        "required": [
            "label",
            "category",
            "comment",
            "reasoning_summary",
            "confidence",
            "needs_review",
        ],
    }


def get_openai_client(api_key: Optional[str], base_url: Optional[str], timeout_s: float):
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "OpenAI Python SDK is not installed. Install it with: pip install openai"
        ) from exc

    kwargs = {"timeout": timeout_s}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def extract_response_text(resp) -> str:
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    try:
        output = getattr(resp, "output", None)
        if not output:
            return ""
        chunks = []
        for item in output:
            content = getattr(item, "content", None) or []
            for c in content:
                c_text = getattr(c, "text", None)
                if isinstance(c_text, str) and c_text.strip():
                    chunks.append(c_text.strip())
        return "\n".join(chunks).strip()
    except Exception:
        return ""


def call_openai_feedback_parser(
    feedback_text: str,
    allowed_categories: set,
    model: str,
    timeout_s: float,
    api_key: Optional[str],
    base_url: Optional[str],
) -> Optional[dict]:
    text = (feedback_text or "").strip()
    if not text:
        return None

    ordered_categories = sorted(list(allowed_categories))
    client = get_openai_client(api_key=api_key, base_url=base_url, timeout_s=timeout_s)

    system_prompt = (
        "You normalize human operator feedback for a retail theft-detection training pipeline. "
        "The input may be a sentence or a long paragraph describing a sequence of events. "
        "Return only the structured judgment requested by the JSON schema. "
        "Rules: label=1 means theft/confirmed suspicious event. "
        "label=0 means not theft/false alarm/benign. "
        "If the text is mixed or uncertain, still choose the best label but set needs_review=true "
        "and lower confidence. reasoning_summary should be a short ordered list of event steps."
    )

    user_prompt = (
        f"Allowed categories: {ordered_categories}\n\n"
        f"Human operator feedback:\n{text}"
    )

    schema = build_feedback_json_schema(ordered_categories)

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        text={
            "format": {
                "type": "json_schema",
                "name": "feedback_normalization",
                "schema": schema,
                "strict": True,
            }
        },
    )

    raw = extract_response_text(resp)
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
    except Exception:
        return None

    return validate_feedback_schema(parsed, allowed_categories)


def parse_feedback_with_llm(
    feedback_text: str,
    allowed_categories: set,
    model: str,
    timeout_s: float,
    api_key: Optional[str],
    base_url: Optional[str],
    enable_fallback: bool = True,
) -> Optional[dict]:
    try:
        parsed = call_openai_feedback_parser(
            feedback_text=feedback_text,
            allowed_categories=allowed_categories,
            model=model,
            timeout_s=timeout_s,
            api_key=api_key,
            base_url=base_url,
        )
        if parsed is not None:
            return parsed
    except Exception as exc:
        print(f"[data_collector] LLM feedback parse failed: {exc}")

    if enable_fallback:
        return heuristic_feedback_parse(feedback_text, allowed_categories)
    return None


class TTLCache:
    def __init__(self, ttl_s: float, max_items: int = 20000):
        self.ttl_s = float(ttl_s)
        self.max_items = int(max_items)
        self.store = {}
        self.fifo = deque(maxlen=self.max_items)

    def put(self, key, value):
        t_s = time.time()
        self.store[key] = (t_s, value)
        self.fifo.append((t_s, key))
        self.prune()

    def get(self, key):
        item = self.store.get(key, None)
        if not item:
            return None
        t_s, val = item
        if (time.time() - t_s) > self.ttl_s:
            self.store.pop(key, None)
            return None
        return val

    def prune(self):
        now = time.time()
        while self.fifo:
            t_s, k = self.fifo[0]
            if (now - t_s) <= self.ttl_s and len(self.store) <= self.max_items:
                break
            self.fifo.popleft()
            self.store.pop(k, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cam_id", required=True)

    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--count", type=int, default=200)

    ap.add_argument("--scores_stream", default=None)
    ap.add_argument("--decisions_stream", default=None)
    ap.add_argument("--alerts_stream", default=None)
    ap.add_argument("--scalars_clip_stream", default=None)
    ap.add_argument("--feedback_stream", default=None)
    ap.add_argument("--clips_stream", default=None)
    ap.add_argument("--clip_refs_stream", default=None)
    ap.add_argument("--read_clips_stream", action="store_true")

    ap.add_argument("--train_events_stream", default=None)
    ap.add_argument("--redis_maxlen", type=int, default=50000)

    ap.add_argument("--cache_ttl_s", type=float, default=30.0)
    ap.add_argument("--pending_ttl_s", type=float, default=3600.0)

    ap.add_argument("--publish_partial", action="store_true")

    ap.add_argument("--gold_filter_unlabeled_ready", action="store_true")
    ap.add_argument("--gold_min_votes", type=int, default=2)
    ap.add_argument("--gold_min_S", type=float, default=0.85)
    ap.add_argument("--gold_require_gate_ok", action="store_true")
    ap.add_argument("--gold_max_missing_pose_ratio", type=float, default=0.55)
    ap.add_argument("--gold_max_missing_obj_ratio", type=float, default=0.75)

    ap.add_argument("--enable_llm_feedback_parse", action="store_true")
    ap.add_argument("--llm_model", type=str, default="gpt-5.4")
    ap.add_argument("--llm_timeout_s", type=float, default=8.0)
    ap.add_argument("--feedback_min_confidence", type=float, default=0.80)
    ap.add_argument(
        "--llm_allowed_categories",
        type=str,
        default="theft,false_alarm,benign,uncertain,needs_review",
    )
    ap.add_argument("--llm_api_key", type=str, default=None)
    ap.add_argument("--llm_base_url", type=str, default=None)
    ap.add_argument("--llm_disable_fallback", action="store_true")

    args = ap.parse_args()
    cfg = load_cfg(args.config)
    cam_id = str(args.cam_id)

    r_cfg = cfg.get("redis", {})
    rdb = redis.Redis(
        host=r_cfg.get("host", "127.0.0.1"),
        port=int(r_cfg.get("port", 6379)),
        db=int(r_cfg.get("db", 0)),
        password=r_cfg.get("password", None),
        decode_responses=False,
    )
    rdb.ping()

    scores_stream = args.scores_stream or resolve_scores_stream(cfg, cam_id)
    decisions_stream = args.decisions_stream or resolve_decisions_stream(cfg, cam_id)
    alerts_stream = args.alerts_stream or resolve_alerts_stream(cfg, cam_id)
    scalars_clip_stream = args.scalars_clip_stream or get_stream(cfg, "scalars_clip", cam_id)
    feedback_stream = args.feedback_stream or get_stream(cfg, "feedback", cam_id)
    clips_stream = args.clips_stream or get_stream(cfg, "clips", cam_id)
    clip_refs_stream = args.clip_refs_stream or resolve_clip_refs_stream(cfg, cam_id)
    train_events_stream = args.train_events_stream or get_stream(cfg, "train_events", cam_id)

    llm_api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY")
    pending_zset_prefix = r_cfg.get("pending_zset_prefix", "train:pending")

    def pending_key(cam, pid):
        return f"{pending_zset_prefix}:{cam}:pid:{pid}"

    print(f"[data_collector] cam_id={cam_id}")
    print("[data_collector] consuming:")
    print(f"  scores        = {scores_stream}")
    print(f"  decisions     = {decisions_stream}")
    print(f"  alerts        = {alerts_stream}")
    print(f"  scalars_clip  = {scalars_clip_stream}")
    print(f"  feedback      = {feedback_stream}")
    print(f"  clip_refs     = {clip_refs_stream}")
    if args.read_clips_stream:
        print(f"  clips         = {clips_stream} (READING ENABLED)")
    else:
        print(f"  clips         = {clips_stream} (not reading clip bytes; pointers only)")

    print(f"[data_collector] producing train_events = {train_events_stream}")

    last_ids = {
        scores_stream: "0-0",
        decisions_stream: "0-0",
        alerts_stream: "0-0",
        scalars_clip_stream: "0-0",
        feedback_stream: "0-0",
        clip_refs_stream: "0-0",
    }
    if args.read_clips_stream:
        last_ids[clips_stream] = "0-0"

    cache = TTLCache(ttl_s=args.cache_ttl_s, max_items=50000)

    def fuse_key(cam, pid, fid_end):
        return (str(cam), int(pid), int(fid_end))

    def stage_from_event(ev: dict) -> str:
        if ev.get("feedback") is not None:
            return "labeled"

        payload = ev.get("payload", {})
        has_score = isinstance(payload.get("score", None), dict)
        has_scal = isinstance(payload.get("scalars_clip", None), dict)

        if has_score and has_scal:
            return "ready"
        return "partial"

    def goldish_ok_for_unlabeled(ev: dict) -> Tuple[bool, List[str]]:
        reasons = []
        payload = ev.get("payload", {})
        dec = payload.get("decision", None)
        score_obj = payload.get("score", None)
        scal = payload.get("scalars_clip", None)

        if not isinstance(score_obj, dict):
            reasons.append("no_score")
        if not isinstance(scal, dict):
            reasons.append("no_scalars_clip")
        if reasons:
            return False, reasons

        miss_pose = None
        miss_obj = None
        if isinstance(dec, dict):
            miss_pose = safe_float(dec.get("missing_pose_ratio", None), None)
            miss_obj = safe_float(dec.get("missing_obj_ratio", None), None)
        if miss_pose is None:
            miss_pose = safe_float(score_obj.get("missing_pose_ratio", None), None)
        if miss_obj is None:
            miss_obj = safe_float(score_obj.get("missing_obj_ratio", None), None)

        if miss_pose is not None and miss_pose > float(args.gold_max_missing_pose_ratio):
            reasons.append("pose_missing_high")
        if miss_obj is not None and miss_obj > float(args.gold_max_missing_obj_ratio):
            reasons.append("obj_missing_high")

        votes = None
        S = None
        gate_ok = None

        if isinstance(dec, dict):
            votes = safe_int(dec.get("votes", None), None)
            S = safe_float(dec.get("S", None), None)
            gate_ok = dec.get("gate_ok", None)

        if args.gold_require_gate_ok and gate_ok is not True:
            reasons.append("gate_not_ok")

        ok_by_votes = (votes is not None and votes >= int(args.gold_min_votes))
        ok_by_S = (S is not None and S >= float(args.gold_min_S))

        if not (ok_by_votes or ok_by_S):
            reasons.append("not_strong_enough_votes_or_S")

        return (len(reasons) == 0), reasons

    def publish_train_event(ev: dict):
        ev["stage"] = stage_from_event(ev)

        if (not args.publish_partial) and ev["stage"] == "partial":
            return

        if "train_ok" not in ev:
            ev["train_ok"] = True
        if "train_ok_reasons" not in ev:
            ev["train_ok_reasons"] = []

        if args.gold_filter_unlabeled_ready and ev["stage"] == "ready" and ev.get("feedback") is None:
            ok, reasons = goldish_ok_for_unlabeled(ev)
            ev["train_ok"] = bool(ok)
            ev["train_ok_reasons"] = reasons
            if not ok:
                return

        fields = {
            "event_id": ev.get("event_id", ""),
            "cam_id": ev.get("cam_id", cam_id),
            "person_track_id": str(ev.get("person_track_id", -1)),
            "frame_id_end": str(ev.get("frame_id_end", -1)),
            "stamp_ns_end": str(ev.get("stamp_ns_end", 0)),
            "stage": ev["stage"],
            "train_ok": "1" if ev.get("train_ok", True) else "0",
            "json": json.dumps(ev),
        }

        gid = ev.get("global_person_id", None)
        if gid is not None:
            fields["global_person_id"] = str(gid)

        rdb.xadd(
            train_events_stream,
            fields,
            maxlen=args.redis_maxlen,
            approximate=True,
        )

    def cache_merge(key, patch: dict) -> dict:
        base = cache.get(key) or {
            "type": "train_event",
            "event_id": None,
            "cam_id": key[0],
            "person_track_id": key[1],
            "frame_id_end": key[2],
            "stamp_ns_end": 0,
            "global_person_id": None,
            "refs": {},
            "payload": {},
            "feedback": None,
            "created_ns": now_ns(),
            "updated_ns": now_ns(),
        }

        base["updated_ns"] = now_ns()

        if patch.get("event_id"):
            base["event_id"] = patch["event_id"]

        if "refs" in patch and isinstance(patch["refs"], dict):
            base["refs"].update(patch["refs"])
        if "payload" in patch and isinstance(patch["payload"], dict):
            base["payload"].update(patch["payload"])

        if patch.get("stamp_ns_end"):
            base["stamp_ns_end"] = int(patch["stamp_ns_end"])

        if patch.get("feedback") is not None:
            base["feedback"] = patch["feedback"]

        if "global_person_id" in patch and patch["global_person_id"] is not None:
            base["global_person_id"] = patch["global_person_id"]

        if "train_ok" in patch:
            base["train_ok"] = patch["train_ok"]
        if "train_ok_reasons" in patch:
            base["train_ok_reasons"] = patch["train_ok_reasons"]

        cache.put(key, base)
        return base

    def pending_prune(cam, pid):
        k = pending_key(cam, pid)
        cutoff = time.time() - float(args.pending_ttl_s)
        rdb.zremrangebyscore(k, 0, cutoff)

    def add_pending(cam, pid, ref_obj: dict):
        pending_prune(cam, pid)
        k = pending_key(cam, pid)
        t = time.time()
        rdb.zadd(k, {json.dumps(ref_obj): t})
        rdb.zremrangebyrank(k, 0, -51)

    def match_pending_latest(cam, pid) -> Optional[dict]:
        pending_prune(cam, pid)
        k = pending_key(cam, pid)
        items = rdb.zrevrange(k, 0, 0)
        if not items:
            return None
        try:
            return json.loads(b2s(items[0]))
        except Exception:
            return None

    def match_event_key(obj: dict):
        cam = obj.get("cam_id", cam_id)
        pid = safe_int(obj.get("person_track_id", -1), -1)
        fid_end = safe_int(obj.get("frame_id_end", -1), -1)
        stamp_ns_end = safe_int(obj.get("stamp_ns_end", 0), 0)
        event_id = obj.get("event_id") or (f"{cam}:{pid}:{fid_end}:{stamp_ns_end}" if pid >= 0 and fid_end >= 0 else None)

        if pid < 0 or fid_end < 0:
            return None, event_id, cam, pid, fid_end, stamp_ns_end
        return fuse_key(cam, pid, fid_end), event_id, cam, pid, fid_end, stamp_ns_end

    try:
        while True:
            streams = rdb.xread(last_ids, block=args.block_ms, count=args.count)
            if not streams:
                cache.prune()
                continue

            for sname, mid, fields in parse_xread(streams):
                last_ids[sname] = mid

                obj = None
                if b"json" in fields:
                    try:
                        obj = json.loads(b2s(fields[b"json"]))
                    except Exception:
                        obj = None

                if sname == scores_stream and obj:
                    key, event_id, cam, pid, fid_end, stamp_ns_end = match_event_key(obj)
                    if key is None:
                        continue

                    ev = cache_merge(key, {
                        "event_id": event_id,
                        "stamp_ns_end": stamp_ns_end,
                        "global_person_id": obj.get("global_person_id", None),
                        "refs": {"score_msg_id": mid},
                        "payload": {"score": obj},
                    })
                    publish_train_event(ev)

                elif sname == decisions_stream and obj:
                    key, event_id, cam, pid, fid_end, stamp_ns_end = match_event_key(obj)
                    if key is None:
                        continue

                    ev = cache_merge(key, {
                        "event_id": event_id,
                        "stamp_ns_end": stamp_ns_end,
                        "global_person_id": obj.get("global_person_id", None),
                        "refs": {"decision_msg_id": mid},
                        "payload": {"decision": obj},
                    })

                    if bool(obj.get("will_alert", False)):
                        add_pending(cam, pid, {
                            "source": "decision",
                            "event_id": event_id,
                            "decision_msg_id": mid,
                            "cam_id": cam,
                            "person_track_id": pid,
                            "global_person_id": obj.get("global_person_id", None),
                            "frame_id_end": fid_end,
                            "stamp_ns_end": stamp_ns_end,
                        })

                    publish_train_event(ev)

                elif sname == alerts_stream and obj:
                    key, event_id, cam, pid, fid_end, stamp_ns_end = match_event_key(obj)
                    if key is None:
                        continue

                    ev = cache_merge(key, {
                        "event_id": event_id,
                        "stamp_ns_end": stamp_ns_end,
                        "global_person_id": obj.get("global_person_id", None),
                        "refs": {"alert_msg_id": mid},
                        "payload": {"alert": obj},
                    })

                    add_pending(cam, pid, {
                        "source": "alert",
                        "event_id": event_id,
                        "alert_msg_id": mid,
                        "cam_id": cam,
                        "person_track_id": pid,
                        "global_person_id": obj.get("global_person_id", None),
                        "frame_id_end": fid_end,
                        "stamp_ns_end": stamp_ns_end,
                    })

                    publish_train_event(ev)

                elif sname == scalars_clip_stream and obj:
                    meta = obj.get("meta", {})
                    feats = obj.get("features", {})
                    policy_feats = obj.get("policy_features", {})

                    cam = meta.get("cam_id", cam_id)
                    pid = safe_int(meta.get("person_track_id", -1), -1)
                    fid_end = safe_int(meta.get("frame_id_end", -1), -1)
                    stamp_ns_end = safe_int(meta.get("stamp_ns_end", 0), 0)
                    event_id = meta.get("event_id") or f"{cam}:{pid}:{fid_end}:{stamp_ns_end}"

                    if pid < 0 or fid_end < 0:
                        continue

                    key = fuse_key(cam, pid, fid_end)
                    ev = cache_merge(key, {
                        "event_id": event_id,
                        "stamp_ns_end": stamp_ns_end,
                        "refs": {"scalars_clip_msg_id": mid},
                        "payload": {
                            "scalars_clip": {
                                "meta": meta,
                                "features": feats,
                                "policy_features": policy_feats,
                            }
                        },
                    })
                    publish_train_event(ev)

                elif sname == clip_refs_stream and obj:
                    key, event_id, cam, pid, fid_end, stamp_ns_end = match_event_key(obj)
                    if key is None:
                        continue

                    ev = cache_merge(key, {
                        "event_id": event_id,
                        "stamp_ns_end": stamp_ns_end,
                        "global_person_id": obj.get("global_person_id", None),
                        "refs": {"clip_ref_msg_id": mid},
                        "payload": {"clip_ref": obj},
                    })
                    publish_train_event(ev)

                elif args.read_clips_stream and (sname == clips_stream):
                    meta_js = fields.get(b"meta", b"{}")
                    try:
                        meta = json.loads(b2s(meta_js))
                    except Exception:
                        continue

                    cam = meta.get("cam_id", cam_id)
                    pid = safe_int(meta.get("person_track_id", -1), -1)
                    fid_end = safe_int(meta.get("frame_id_end", -1), -1)
                    stamp_ns_end = safe_int(meta.get("stamp_ns_end", 0), 0)
                    event_id = meta.get("event_id") or f"{cam}:{pid}:{fid_end}:{stamp_ns_end}"

                    if pid < 0 or fid_end < 0:
                        continue

                    key = fuse_key(cam, pid, fid_end)
                    ev = cache_merge(key, {
                        "event_id": event_id,
                        "stamp_ns_end": stamp_ns_end,
                        "global_person_id": meta.get("global_person_id", None),
                        "refs": {"clip_msg_id": mid},
                        "payload": {"clip_meta": meta},
                    })
                    publish_train_event(ev)

                elif sname == feedback_stream and obj:
                    cam = obj.get("cam_id", cam_id)
                    pid = safe_int(obj.get("person_track_id", -1), -1)
                    if pid < 0:
                        continue

                    allowed_categories = set(
                        x.strip().lower() for x in str(args.llm_allowed_categories).split(",") if x.strip()
                    )

                    fid_end = safe_int(obj.get("frame_id_end", -1), -1)
                    stamp_ns_end = safe_int(obj.get("stamp_ns_end", 0), 0)
                    feedback_event_id = obj.get("event_id", None)
                    alert_id = obj.get("alert_id", None)

                    matched_key = None
                    matched_event_id = feedback_event_id

                    if feedback_event_id:
                        parts = str(feedback_event_id).split(":")
                        if len(parts) >= 4:
                            try:
                                f_cam = parts[0]
                                f_pid = int(parts[1])
                                f_fid = int(parts[2])
                                matched_key = fuse_key(f_cam, f_pid, f_fid)
                            except Exception:
                                matched_key = None

                    if matched_key is None and fid_end is not None and fid_end >= 0:
                        matched_key = fuse_key(cam, pid, fid_end)
                        matched_event_id = matched_event_id or f"{cam}:{pid}:{fid_end}:{stamp_ns_end}"

                    if matched_key is None:
                        pend = match_pending_latest(cam, pid)
                        if pend:
                            matched_key = fuse_key(
                                pend.get("cam_id", cam),
                                pend.get("person_track_id", pid),
                                pend.get("frame_id_end", -1),
                            )
                            matched_event_id = matched_event_id or pend.get("event_id")

                    parsed_feedback = None

                    label_i = safe_int(obj.get("label", None), None)
                    if label_i in (0, 1):
                        parsed_feedback = validate_feedback_schema({
                            "label": label_i,
                            "category": obj.get("category", None),
                            "comment": obj.get("comment", None) or extract_feedback_text(obj),
                            "confidence": safe_float(obj.get("confidence", 1.0), 1.0),
                            "needs_review": bool(obj.get("needs_review", False)),
                            "reasoning_summary": obj.get("reasoning_summary", []),
                        }, allowed_categories)

                    if parsed_feedback is None and args.enable_llm_feedback_parse:
                        free_text = extract_feedback_text(obj)
                        if free_text:
                            parsed_feedback = parse_feedback_with_llm(
                                feedback_text=free_text,
                                allowed_categories=allowed_categories,
                                model=args.llm_model,
                                timeout_s=args.llm_timeout_s,
                                api_key=llm_api_key,
                                base_url=args.llm_base_url,
                                enable_fallback=(not args.llm_disable_fallback),
                            )

                    if parsed_feedback is None:
                        continue

                    free_text = extract_feedback_text(obj)
                    feedback_obj = {
                        "msg_id": mid,
                        "event_id": matched_event_id,
                        "label": int(parsed_feedback["label"]),
                        "category": parsed_feedback.get("category"),
                        "comment": parsed_feedback.get("comment"),
                        "reasoning_summary": parsed_feedback.get("reasoning_summary", []),
                        "raw_feedback_text": free_text,
                        "user_id": obj.get("user_id", None),
                        "alert_id": alert_id,
                        "received_ns": now_ns(),
                        "llm_confidence": parsed_feedback.get("confidence", 1.0),
                        "needs_review": parsed_feedback.get("needs_review", False),
                        "raw_feedback": obj,
                    }

                    feedback_train_ok = (
                        (not feedback_obj["needs_review"]) and
                        (float(feedback_obj.get("llm_confidence", 0.0)) >= float(args.feedback_min_confidence))
                    )

                    gid_from_feedback = obj.get("global_person_id", None)

                    if matched_key is not None and matched_key[2] >= 0:
                        ev = cache_merge(matched_key, {
                            "event_id": matched_event_id,
                            "global_person_id": gid_from_feedback,
                            "refs": {"feedback_msg_id": mid},
                            "feedback": feedback_obj,
                            "train_ok": feedback_train_ok,
                            "train_ok_reasons": [] if feedback_train_ok else [
                                "feedback_needs_review_or_low_confidence"
                            ],
                        })
                        publish_train_event(ev)
                    else:
                        orphan = {
                            "type": "train_event",
                            "event_id": matched_event_id,
                            "cam_id": cam,
                            "person_track_id": pid,
                            "global_person_id": gid_from_feedback,
                            "frame_id_end": -1,
                            "stamp_ns_end": 0,
                            "refs": {"feedback_msg_id": mid},
                            "payload": {"feedback_only": obj},
                            "feedback": feedback_obj,
                            "created_ns": now_ns(),
                            "updated_ns": now_ns(),
                            "stage": "labeled",
                            "train_ok": feedback_train_ok,
                            "train_ok_reasons": [] if feedback_train_ok else [
                                "feedback_needs_review_or_low_confidence"
                            ],
                            "note": "unmatched_feedback",
                        }
                        publish_train_event(orphan)

            cache.prune()

    except KeyboardInterrupt:
        print("\n[data_collector] stopping...")


if __name__ == "__main__":
    main()