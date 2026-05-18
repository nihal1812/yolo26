#!/usr/bin/env python3
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_active_cams(cfg: dict):
    cams = cfg.get("system", {}).get("active_cams", [])
    if not isinstance(cams, list) or not cams:
        raise KeyError("config.system.active_cams missing or empty")
    return [str(c) for c in cams]


def get_cam_cfg(cfg: dict, cam_id: str):
    cams = cfg.get("cams", {})
    if cam_id not in cams:
        raise KeyError(f"config.cams.{cam_id} missing")
    cam_cfg = cams[cam_id]
    if not isinstance(cam_cfg, dict):
        raise TypeError(f"config.cams.{cam_id} must be a mapping")
    return cam_cfg


def get_rtsp_cfg(cfg: dict, cam_id: str):
    cam_cfg = get_cam_cfg(cfg, cam_id)
    rtsp = cam_cfg.get("rtsp", {})
    if not isinstance(rtsp, dict):
        raise TypeError(f"config.cams.{cam_id}.rtsp must be a mapping")
    return rtsp


def get_zmq_cfg(cfg: dict, cam_id: str):
    cam_cfg = get_cam_cfg(cfg, cam_id)
    zmq_cfg = cam_cfg.get("zmq", {})
    if not isinstance(zmq_cfg, dict):
        raise TypeError(f"config.cams.{cam_id}.zmq must be a mapping")
    return zmq_cfg


def get_global_zmq_cfg(cfg: dict):
    zmq_cfg = cfg.get("zmq", {})
    if zmq_cfg is None:
        return {}
    if not isinstance(zmq_cfg, dict):
        raise TypeError("config.zmq must be a mapping")
    return zmq_cfg


def _validate_endpoint_dict(node: Any, path_hint: str):
    if not isinstance(node, dict):
        raise TypeError(f"{path_hint} must be a mapping")
    return node


def get_zmq_endpoint(cfg: dict, cam_id: str, key: str):
    """
    Backward-compatible camera-scoped endpoint lookup.

    Looks only under:
      config.cams.{cam_id}.zmq.{key}
    """
    zmq_cfg = get_zmq_cfg(cfg, cam_id)
    if key not in zmq_cfg:
        raise KeyError(f"config.cams.{cam_id}.zmq.{key} missing")
    return _validate_endpoint_dict(zmq_cfg[key], f"config.cams.{cam_id}.zmq.{key}")


def get_any_zmq_endpoint(cfg: dict, key: str, cam_id: Optional[str] = None):
    """
    Flexible lookup for scalable architecture.

    Lookup order:
      1) config.cams.{cam_id}.zmq.{key}      if cam_id provided
      2) config.zmq.{key}                    global/shared endpoints
    """
    if cam_id is not None:
        try:
            return get_zmq_endpoint(cfg, cam_id, key)
        except Exception:
            pass

    global_zmq = get_global_zmq_cfg(cfg)
    if key in global_zmq:
        return _validate_endpoint_dict(global_zmq[key], f"config.zmq.{key}")

    if cam_id is not None:
        raise KeyError(
            f"ZMQ endpoint '{key}' not found in either "
            f"config.cams.{cam_id}.zmq.{key} or config.zmq.{key}"
        )

    raise KeyError(f"config.zmq.{key} missing")


def get_optional_zmq_endpoint(cfg: dict, key: str, cam_id: Optional[str] = None, default=None):
    """
    Same lookup behavior as get_any_zmq_endpoint(), but returns default instead of raising.
    """
    try:
        return get_any_zmq_endpoint(cfg, key=key, cam_id=cam_id)
    except Exception:
        return default


def has_zmq_endpoint(cfg: dict, key: str, cam_id: Optional[str] = None) -> bool:
    try:
        get_any_zmq_endpoint(cfg, key=key, cam_id=cam_id)
        return True
    except Exception:
        return False


def get_stream(cfg: dict, key: str, cam_id: str = None):
    streams = cfg.get("redis", {}).get("streams", {})
    if key not in streams:
        raise KeyError(f"config.redis.streams.{key} missing")
    tmpl = streams[key]
    if cam_id is None:
        return str(tmpl)
    return str(tmpl).format(cam=cam_id)


def has_stream(cfg: dict, key: str) -> bool:
    streams = cfg.get("redis", {}).get("streams", {})
    return key in streams


def get_runtime(cfg: dict, key: str, default=None):
    return cfg.get("runtime", {}).get(key, default)


def get_model_cfg(cfg: dict):
    models = cfg.get("models", {})
    if not isinstance(models, dict):
        raise TypeError("config.models must be a mapping")
    return models


def get_brain_args(cfg: dict, section: str):
    brain = cfg.get("brain", {})
    if not isinstance(brain, dict):
        return {}
    out = brain.get(section, {})
    return out if isinstance(out, dict) else {}


def get_section(cfg: dict, section: str, default=None):
    val = cfg.get(section, default if default is not None else {})
    if val is None:
        return default if default is not None else {}
    return val


def format_cam_value(value, cam_id: str):
    if isinstance(value, str):
        return value.format(cam=cam_id)
    return value


def format_cam_dict(d: dict, cam_id: str):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out[k] = format_cam_dict(v, cam_id)
        elif isinstance(v, list):
            out[k] = [format_cam_value(x, cam_id) for x in v]
        else:
            out[k] = format_cam_value(v, cam_id)
    return out


def local_connect_addr(bind_addr: str):
    """
    Convert bind address like tcp://*:5555 -> tcp://127.0.0.1:5555
    """
    return str(bind_addr).replace("*", "127.0.0.1")


def local_bind_addr(addr: str):
    """
    Normalize bind address; kept for symmetry/readability.
    """
    return str(addr)


def endpoint_bind_addr(ep: Dict[str, Any]) -> str:
    if not isinstance(ep, dict):
        raise TypeError("endpoint must be a mapping")
    bind = ep.get("bind", None)
    if not bind:
        raise KeyError("endpoint.bind missing")
    return str(bind)


def endpoint_connect_addr(ep: Dict[str, Any]) -> str:
    if not isinstance(ep, dict):
        raise TypeError("endpoint must be a mapping")
    if ep.get("connect"):
        return str(ep["connect"])
    bind = ep.get("bind", None)
    if not bind:
        raise KeyError("endpoint.bind missing")
    return local_connect_addr(bind)


def endpoint_topic(ep: Dict[str, Any], default: Optional[str] = None) -> Optional[str]:
    if not isinstance(ep, dict):
        raise TypeError("endpoint must be a mapping")
    topic = ep.get("topic", default)
    if topic is None:
        return None
    return str(topic)


def endpoint_sndhwm(ep: Dict[str, Any], default: int = 1000) -> int:
    if not isinstance(ep, dict):
        raise TypeError("endpoint must be a mapping")
    try:
        return int(ep.get("sndhwm", default))
    except Exception:
        return int(default)


def endpoint_rcvhwm(ep: Dict[str, Any], default: int = 1000) -> int:
    if not isinstance(ep, dict):
        raise TypeError("endpoint must be a mapping")
    try:
        return int(ep.get("rcvhwm", default))
    except Exception:
        return int(default)


def endpoint_latest_only(ep: Dict[str, Any], default: bool = False) -> bool:
    if not isinstance(ep, dict):
        raise TypeError("endpoint must be a mapping")
    v = ep.get("latest_only", default)
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")
