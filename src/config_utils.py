#!/usr/bin/env python3
from pathlib import Path
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


def get_zmq_endpoint(cfg: dict, cam_id: str, key: str):
    zmq_cfg = get_zmq_cfg(cfg, cam_id)
    if key not in zmq_cfg:
        raise KeyError(f"config.cams.{cam_id}.zmq.{key} missing")
    node = zmq_cfg[key]
    if not isinstance(node, dict):
        raise TypeError(f"config.cams.{cam_id}.zmq.{key} must be a mapping")
    return node


def get_stream(cfg: dict, key: str, cam_id: str = None):
    streams = cfg.get("redis", {}).get("streams", {})
    if key not in streams:
        raise KeyError(f"config.redis.streams.{key} missing")
    tmpl = streams[key]
    if cam_id is None:
        return str(tmpl)
    return str(tmpl).format(cam=cam_id)


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
    return str(bind_addr).replace("*", "127.0.0.1")