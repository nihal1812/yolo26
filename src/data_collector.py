#!/usr/bin/env python3
"""
data_collector_llm.py

Collects training/eval data without blocking realtime.

Consumes:
  - scores:<cam>
  - decisions:<cam>
  - alerts:<cam>
  - scalars_clip:<cam>
  - feedback:<cam>

Produces:
  - train_events:<cam>    (unified fused event snapshots)

Features added here:
  - Paragraph / multi-line operator feedback support
  - Real LLM normalization using the OpenAI Responses API
  - Structured Outputs JSON schema validation
  - Confidence + review gating for training safety
"""

import argparse
import json
import os
import time
from collections import deque
from typing import Optional, Tuple, List

import yaml
import redis


# ----------------------------
# Utils
# ----------------------------
def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def extract_feedback_text(obj: dict) -> str:
    """
    Build one free-text string from common operator fields.
    Supports one-line notes, multi-line comments, and paragraph explanations.
    """
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
    """
    Simple fallback sentence splitter. Used only if the LLM fails.
    """
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
    """
    Conservative fallback if the LLM call fails.
    This lets the collector continue running instead of dropping everything.
    """
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
            "label": {
                "type": "integer",
                "enum": [0, 1],
                "description": "0 = not theft / false alarm / benign. 1 = theft / confirmed suspicious event."
            },
            "category": {
                "type": "string",
                "enum": allowed_categories,
                "description": "Normalized category for training and review workflows."
            },
            "comment": {
                "type": "string",
                "description": "Short normalized explanation, one concise sentence."
            },
            "reasoning_summary": {
                "type": "array",
                "description": "Short ordered summary of the operator-described sequence of events.",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 8,
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Model confidence in the normalization result."
            },
            "needs_review": {
                "type": "boolean",
                "description": "True when the operator text is ambiguous, contradictory, or weak."
            },
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
    """
    The OpenAI SDK exposes output_text for convenience.
    Keep a defensive fallback for SDK variations.
    """
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


def call_openai_feedback_parser(feedback_text: str,
                                allowed_categories: set,
                                model: str,
                                timeout_s: float,
                                api_key: Optional[str],
                                base_url: Optional[str]) -> Optional[dict]:
    """
    Real OpenAI Responses API call using Structured Outputs.
    The model is forced to emit JSON matching the provided schema.
    """
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


def parse_feedback_with_llm(feedback_text: str,
                            allowed_categories: set,
                            model: str,
                            timeout_s: float,
                            api_key: Optional[str],
                            base_url: Optional[str],
                            enable_fallback: bool = True) -> Optional[dict]:
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


# ----------------------------
# TTL cache (in-memory fuse)
# ----------------------------
class TTLCache:
    """
    key -> (t_s, value)
    Prunes by TTL and max_items
    """
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


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--count", type=int, default=200)

    # Stream names overrides
    ap.add_argument("--scores_stream", default=None)
    ap.add_argument("--decisions_stream", default=None)
    ap.add_argument("--alerts_stream", default=None)
    ap.add_argument("--scalars_clip_stream", default=None)
    ap.add_argument("--feedback_stream", default=None)

    ap.add_argument("--clips_stream", default=None)
    ap.add_argument("--read_clips_stream", action="store_true")

    ap.add_argument("--train_events_stream", default=None)
    ap.add_argument("--redis_maxlen", type=int, default=50000)

    # Correlation
    ap.add_argument("--cache_ttl_s", type=float, default=30.0)

    # Pending feedback matching (persisted in Redis)
    ap.add_argument("--pending_ttl_s", type=float, default=3600.0)

    # Publishing policy
    ap.add_argument("--publish_partial", action="store_true",
                    help="If set, publish every partial update (debug mode). Default: only ready/labeled.")

    # -------------------------
    # Gold-ish gating for unlabeled ready events
    # -------------------------
    ap.add_argument("--gold_filter_unlabeled_ready", action="store_true",
                    help="If set, only publish unlabeled 'ready' train_events when evidence looks clean.")
    ap.add_argument("--gold_min_votes", type=int, default=2,
                    help="Minimum decision.votes to count as gold-ish (if present).")
    ap.add_argument("--gold_min_S", type=float, default=0.85,
                    help="Minimum decision.S to count as gold-ish (if present).")
    ap.add_argument("--gold_require_gate_ok", action="store_true",
                    help="Require decision.gate_ok==True for unlabeled 'ready' events.")
    ap.add_argument("--gold_max_missing_pose_ratio", type=float, default=0.55)
    ap.add_argument("--gold_max_missing_obj_ratio", type=float, default=0.75)

    # -------------------------
    # LLM feedback normalization
    # -------------------------
    ap.add_argument("--enable_llm_feedback_parse", action="store_true",
                    help="Normalize descriptive/paragraph operator feedback into structured training labels.")
    ap.add_argument("--llm_model", type=str, default="gpt-5.4",
                    help="OpenAI model name used for feedback normalization.")
    ap.add_argument("--llm_timeout_s", type=float, default=8.0,
                    help="Timeout for the OpenAI API request.")
    ap.add_argument("--feedback_min_confidence", type=float, default=0.80,
                    help="Minimum confidence for LLM-normalized feedback to be marked train_ok.")
    ap.add_argument("--llm_allowed_categories", type=str,
                    default="theft,false_alarm,benign,uncertain,needs_review",
                    help="Comma-separated category enum allowed in normalized feedback.")
    ap.add_argument("--llm_api_key", type=str, default=None,
                    help="Optional API key override. If omitted, OPENAI_API_KEY environment variable is used.")
    ap.add_argument("--llm_base_url", type=str, default=None,
                    help="Optional alternate base URL. Leave empty for the official OpenAI API.")
    ap.add_argument("--llm_disable_fallback", action="store_true",
                    help="If set, do not use heuristic fallback when the LLM call fails.")

    args = ap.parse_args()
    cfg = load_cfg(args.config)
    cam_id = cfg["system"]["cam_id"]

    r_cfg = cfg.get("redis", {})
    rdb = redis.Redis(
        host=r_cfg.get("host", "127.0.0.1"),
        port=int(r_cfg.get("port", 6379)),
        db=int(r_cfg.get("db", 0)),
        password=r_cfg.get("password", None),
        decode_responses=False,
    )
    rdb.ping()

    scores_stream = args.scores_stream or r_cfg.get("scores_stream", f"scores:{cam_id}")
    decisions_stream = args.decisions_stream or r_cfg.get("decisions_stream", f"decisions:{cam_id}")
    alerts_stream = args.alerts_stream or r_cfg.get("alerts_stream", f"alerts:{cam_id}")
    scalars_clip_stream = args.scalars_clip_stream or r_cfg.get("scalars_clip_stream", f"scalars_clip:{cam_id}")
    feedback_stream = args.feedback_stream or r_cfg.get("feedback_stream", f"feedback:{cam_id}")

    clips_stream = args.clips_stream or r_cfg.get("clips_stream", f"clips:{cam_id}")
    train_events_stream = args.train_events_stream or r_cfg.get("train_events_stream", f"train_events:{cam_id}")

    llm_api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY")

    # Redis keys for pending alert refs (persisted)
    pending_zset_prefix = r_cfg.get("pending_zset_prefix", "train:pending")

    def pending_key(cam, pid):
        return f"{pending_zset_prefix}:{cam}:pid:{pid}"

    print("[data_collector] consuming:")
    print(f"  scores        = {scores_stream}")
    print(f"  decisions     = {decisions_stream}")
    print(f"  alerts        = {alerts_stream}")
    print(f"  scalars_clip  = {scalars_clip_stream}")
    print(f"  feedback      = {feedback_stream}")
    if args.read_clips_stream:
        print(f"  clips         = {clips_stream} (READING ENABLED)")
    else:
        print(f"  clips         = {clips_stream} (not reading clip bytes; pointers only)")

    print(f"[data_collector] producing train_events = {train_events_stream}")
    print(f"[data_collector] publish_partial={args.publish_partial}")
    print(f"[data_collector] pending_ttl_s={args.pending_ttl_s}")
    print(f"[data_collector] gold_filter_unlabeled_ready={args.gold_filter_unlabeled_ready}")
    print(f"[data_collector] enable_llm_feedback_parse={args.enable_llm_feedback_parse}")
    if args.enable_llm_feedback_parse:
        print(f"[data_collector] llm_model={args.llm_model}")
        print(f"[data_collector] llm_base_url={args.llm_base_url or 'official OpenAI API'}")
        print(f"[data_collector] llm_api_key_present={'yes' if llm_api_key else 'no'}")

    # XREAD cursors
    last_ids = {
        scores_stream: "0-0",
        decisions_stream: "0-0",
        alerts_stream: "0-0",
        scalars_clip_stream: "0-0",
        feedback_stream: "0-0",
    }
    if args.read_clips_stream:
        last_ids[clips_stream] = "0-0"

    # Correlation cache keyed by (cam, pid, fid_end)
    cache = TTLCache(ttl_s=args.cache_ttl_s, max_items=50000)

    def fuse_key(cam, pid, fid_end) -> Tuple[str, int, int]:
        return (str(cam), int(pid), int(fid_end))

    def event_stage(ev: dict) -> str:
        """
        partial: missing required things
        ready: has score + scalars (good for training even without feedback)
        labeled: has feedback attached
        """
        if ev.get("feedback") is not None:
            return "labeled"
        payload = ev.get("payload", {})
        has_score = isinstance(payload.get("score", None), dict)
        has_scal = isinstance(payload.get("scalars_clip", None), dict)
        if has_score and has_scal:
            return "ready"
        return "partial"

    def goldish_ok_for_unlabeled(ev: dict) -> Tuple[bool, List[str]]:
        """
        Only used for unlabeled READY events (optional filter).
        This reduces drift if you later do self-training / pseudo-labeling.
        """
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

        # Missing ratios can be from decision or score object
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

        # If decision exists, use votes/S/gate_ok
        votes = None
        S = None
        gate_ok = None
        have_feats = None

        if isinstance(dec, dict):
            votes = safe_int(dec.get("votes", None), None)
            S = safe_float(dec.get("S", None), None)
            gate_ok = dec.get("gate_ok", None)
            have_feats = dec.get("have_scalars_clip", None)

        if have_feats is False:
            reasons.append("decision_no_scalars_clip")

        if args.gold_require_gate_ok:
            if gate_ok is not True:
                reasons.append("gate_not_ok")

        ok_by_votes = (votes is not None and votes >= int(args.gold_min_votes))
        ok_by_S = (S is not None and S >= float(args.gold_min_S))

        if not (ok_by_votes or ok_by_S):
            reasons.append("not_strong_enough_votes_or_S")

        return (len(reasons) == 0), reasons

    def publish_train_event(ev: dict):
        ev["stage"] = event_stage(ev)

        if (not args.publish_partial) and ev["stage"] == "partial":
            return

        # Respect train_ok if already set by feedback normalization.
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

        rdb.xadd(
            train_events_stream,
            {
                "cam_id": ev.get("cam_id", cam_id),
                "person_track_id": str(ev.get("person_track_id", -1)),
                "frame_id_end": str(ev.get("frame_id_end", -1)),
                "stamp_ns_end": str(ev.get("stamp_ns_end", 0)),
                "stage": ev["stage"],
                "train_ok": "1" if ev.get("train_ok", True) else "0",
                "json": json.dumps(ev),
            },
            maxlen=args.redis_maxlen,
            approximate=True,
        )

    def cache_merge(key, patch: dict) -> dict:
        base = cache.get(key) or {
            "type": "train_event",
            "cam_id": key[0],
            "person_track_id": key[1],
            "frame_id_end": key[2],
            "stamp_ns_end": 0,
            "refs": {},
            "payload": {},
            "feedback": None,
            "created_ns": now_ns(),
            "updated_ns": now_ns(),
        }

        base["updated_ns"] = now_ns()

        if "refs" in patch and isinstance(patch["refs"], dict):
            base["refs"].update(patch["refs"])
        if "payload" in patch and isinstance(patch["payload"], dict):
            base["payload"].update(patch["payload"])

        if patch.get("stamp_ns_end"):
            base["stamp_ns_end"] = int(patch["stamp_ns_end"])

        if patch.get("feedback") is not None:
            base["feedback"] = patch["feedback"]

        if "train_ok" in patch:
            base["train_ok"] = patch["train_ok"]
        if "train_ok_reasons" in patch:
            base["train_ok_reasons"] = patch["train_ok_reasons"]

        cache.put(key, base)
        return base

    # ----------------------------
    # Pending alert/decision helpers (Redis ZSET)
    # ----------------------------
    def pending_prune(cam, pid):
        k = pending_key(cam, pid)
        cutoff = time.time() - float(args.pending_ttl_s)
        rdb.zremrangebyscore(k, 0, cutoff)

    def add_pending(cam, pid, ref_obj: dict):
        pending_prune(cam, pid)
        k = pending_key(cam, pid)
        t = time.time()
        rdb.zadd(k, {json.dumps(ref_obj): t})
        rdb.zremrangebyrank(k, 0, -51)  # keep ~50 newest

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

    # ----------------------------
    # Main loop
    # ----------------------------
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

                # --- scores
                if sname == scores_stream and obj:
                    cam = obj.get("cam_id", cam_id)
                    pid = safe_int(obj.get("person_track_id", -1), -1)
                    fid_end = safe_int(obj.get("frame_id_end", -1), -1)
                    if pid < 0 or fid_end < 0:
                        continue

                    key = fuse_key(cam, pid, fid_end)
                    ev = cache_merge(key, {
                        "stamp_ns_end": safe_int(obj.get("stamp_ns_end", 0), 0),
                        "refs": {"score_msg_id": mid},
                        "payload": {"score": obj},
                    })
                    publish_train_event(ev)

                # --- decisions
                elif sname == decisions_stream and obj:
                    cam = obj.get("cam_id", cam_id)
                    pid = safe_int(obj.get("person_track_id", -1), -1)
                    fid_end = safe_int(obj.get("frame_id_end", -1), -1)
                    if pid < 0 or fid_end < 0:
                        continue

                    key = fuse_key(cam, pid, fid_end)
                    ev = cache_merge(key, {
                        "stamp_ns_end": safe_int(obj.get("stamp_ns_end", 0), 0),
                        "refs": {"decision_msg_id": mid},
                        "payload": {"decision": obj},
                    })

                    if bool(obj.get("will_alert", False)):
                        add_pending(cam, pid, {
                            "source": "decision",
                            "decision_msg_id": mid,
                            "cam_id": cam,
                            "person_track_id": pid,
                            "frame_id_end": fid_end,
                        })

                    publish_train_event(ev)

                # --- alerts
                elif sname == alerts_stream and obj:
                    cam = obj.get("cam_id", cam_id)
                    pid = safe_int(obj.get("person_track_id", -1), -1)
                    fid_end = safe_int(obj.get("frame_id_end", -1), -1)
                    if pid < 0 or fid_end < 0:
                        continue

                    key = fuse_key(cam, pid, fid_end)
                    ev = cache_merge(key, {
                        "stamp_ns_end": safe_int(obj.get("stamp_ns_end", 0), 0),
                        "refs": {"alert_msg_id": mid},
                        "payload": {"alert": obj},
                    })

                    add_pending(cam, pid, {
                        "source": "alert",
                        "alert_msg_id": mid,
                        "cam_id": cam,
                        "person_track_id": pid,
                        "frame_id_end": fid_end,
                    })

                    publish_train_event(ev)

                # --- scalars_clip
                elif sname == scalars_clip_stream and obj:
                    meta = obj.get("meta", {})
                    feats = obj.get("features", {})
                    cam = meta.get("cam_id", cam_id)
                    pid = safe_int(meta.get("person_track_id", -1), -1)
                    fid_end = safe_int(meta.get("frame_id_end", -1), -1)
                    if pid < 0 or fid_end < 0:
                        continue

                    key = fuse_key(cam, pid, fid_end)
                    ev = cache_merge(key, {
                        "stamp_ns_end": safe_int(meta.get("stamp_ns_end", 0), 0),
                        "refs": {"scalars_clip_msg_id": mid},
                        "payload": {"scalars_clip": {"meta": meta, "features": feats}},
                    })
                    publish_train_event(ev)

                # --- optional: clips stream meta only
                elif args.read_clips_stream and (sname == clips_stream):
                    meta_js = fields.get(b"meta", b"{}")
                    try:
                        meta = json.loads(b2s(meta_js))
                    except Exception:
                        continue

                    cam = meta.get("cam_id", cam_id)
                    pid = safe_int(meta.get("person_track_id", -1), -1)
                    fid_end = safe_int(meta.get("frame_id_end", -1), -1)
                    if pid < 0 or fid_end < 0:
                        continue

                    key = fuse_key(cam, pid, fid_end)
                    ev = cache_merge(key, {
                        "stamp_ns_end": safe_int(meta.get("stamp_ns_end", 0), 0),
                        "refs": {"clip_msg_id": mid},
                        "payload": {"clip_meta": meta},
                    })
                    publish_train_event(ev)

                # --- feedback
                elif sname == feedback_stream and obj:
                    cam = obj.get("cam_id", cam_id)
                    pid = safe_int(obj.get("person_track_id", -1), -1)
                    if pid < 0:
                        continue

                    allowed_categories = set(
                        x.strip().lower() for x in str(args.llm_allowed_categories).split(",") if x.strip()
                    )

                    fid_end = safe_int(obj.get("frame_id_end", -1), -1)
                    alert_id = obj.get("alert_id", None)

                    matched_key = None
                    if fid_end is not None and fid_end >= 0:
                        matched_key = fuse_key(cam, pid, fid_end)
                    else:
                        pend = match_pending_latest(cam, pid)
                        if pend:
                            matched_key = fuse_key(
                                pend.get("cam_id", cam),
                                pend.get("person_track_id", pid),
                                pend.get("frame_id_end", -1),
                            )

                    parsed_feedback = None

                    # Path 1: already structured feedback
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

                    # Path 2: descriptive feedback -> LLM normalization
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

                    if matched_key is not None and matched_key[2] >= 0:
                        ev = cache_merge(matched_key, {
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
                            "cam_id": cam,
                            "person_track_id": pid,
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
