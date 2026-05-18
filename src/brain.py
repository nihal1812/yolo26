#!/usr/bin/env python3
"""
brain.py

Pipeline orchestrator for the theft-detection system.

This file is the actual orchestration runner. A separate main.py may simply
start this script/process, but this file owns the realtime and learning loops.

Production layout
-----------------
Realtime pipeline:
    ModelWorker(cam) -> PolicyWorker(cam)

Learning pipeline:
    DataCollectorWorker(cam01)
    DataCollectorWorker(cam02)
    ...
    DataCollectorWorker(camN)
        -> all publish training events to one shared ZMQ topic

    TrainerWorker(fleet/global scope)
        -> subscribes once to that shared topic
        -> downloads clips temporarily from S3 using event.clip_ref.s3_uri
        -> trains one shared fleet/global theft model
        -> uploads model / registry / dataset reports to S3
        -> publishes model_update / policy_update for the shared target

Important design choice
-----------------------
The trainer is not camera-specific and is not store-specific by default.
cam_id and site_id/store_id remain useful event metadata for debugging,
per-camera/per-store metrics, and future local threshold overrides, but they are
not the model identity.

Recommended trainer identity:
    target_id: global_retail
    model_scope: fleet

The data collector nodes are still per camera because each camera produces its
own detections, decisions, feedback, and clip references. They should all publish
training events to the same topic, for example:
    train_events.theft

The trainer subscribes once to that topic. You do not need to run a long command
for the trainer; settings should come from config.yaml under trainer_node.
"""

import time
import argparse
import multiprocessing as mp
import traceback

try:
    from .config_utils import load_cfg, get_active_cams, get_brain_args, format_cam_dict
    from .model_node import ModelWorker
    from .policy_node import PolicyWorker
    from .data_collector_llm import DataCollectorWorker

    # Fleet/global ZMQ/S3 trainer. Preferred entrypoint is the config factory.
    try:
        from .trainer_node import create_trainer_worker_from_config, TrainerWorker
    except Exception:
        create_trainer_worker_from_config = None
        from .trainer_node import TrainerWorker
except Exception:
    from config_utils import load_cfg, get_active_cams, get_brain_args, format_cam_dict
    from model_node import ModelWorker
    from policy_node import PolicyWorker
    from data_collector_llm import DataCollectorWorker

    try:
        from trainer_node import create_trainer_worker_from_config, TrainerWorker
    except Exception:
        create_trainer_worker_from_config = None
        from trainer_node import TrainerWorker


def ns_from_dict(d: dict):
    ns = argparse.Namespace()
    for k, v in (d or {}).items():
        setattr(ns, k, v)
    return ns


def ensure_defaults(ns, defaults: dict):
    for k, v in defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


def build_model_args(cfg: dict, cfg_path: str, cam_id: str):
    raw = format_cam_dict(get_brain_args(cfg, "model_node_args"), cam_id)
    ns = ns_from_dict(raw)

    defaults = {
        "config": cfg_path,
        "cam_id": cam_id,
        "device": "cuda:0",
        "fp16": False,
        "disable_redis_updates": False,
        "scores_stream": None,
        "updates_stream": None,
        "updates_target": None,
        "check_updates_every_loops": 1,
        "global_tracks_stream": None,
        "clip_connect": None,
        "clip_topic": None,
        "scalars_connect": None,
        "scalars_topic": None,
        "global_tracks_connect": None,
        "global_tracks_topic": None,
        "scores_bind": None,
        "scores_topic": None,
        "zmq_rcvhwm": 256,
        "block_ms": 1000,
        "max_drain_per_step": 64,
        "scalar_cache_size": 5000,
        "scalar_cache_ttl_s": 5.0,
        "weights": None,
        "model_version": "bootstrap_v0",
        "prefer_champion_registry": False,
        "champion_registry_dir": "models",
        "allow_runtime_model_selection": False,
        "global_cache_ttl_s": 30.0,
        "identity_recent_window_s": 30.0,
        "identity_state_idle_s": 60.0,
        "enable_identity_memory_fusion": False,
        "identity_memory_weight": 0.20,
    }
    return ensure_defaults(ns, defaults)


def build_policy_args(cfg: dict, cfg_path: str, cam_id: str):
    raw = format_cam_dict(get_brain_args(cfg, "policy_node_args"), cam_id)
    ns = ns_from_dict(raw)

    defaults = {
        "config": cfg_path,
        "cam_id": cam_id,
        "scores_stream": None,
        "alerts_stream": None,
        "decisions_stream": None,
        "policy_updates_stream": None,
        "policy_target": None,
        "block_ms": 1000,
        "count": 50,
        "threshold": 0.75,
        "M": 8,
        "K": 4,
        "cooldown_s": 10.0,
        "max_missing_pose_ratio": 0.55,
        "max_missing_obj_ratio": 0.75,
        "drop_if_scalar_missing": False,
        "use_suspicion_policy": False,
        "suspicion_tau_s": 6.0,
        "suspicion_alert_thr": 0.85,
        "suspicion_persist_clips": 2,
        "suspicion_gain": 0.55,
        "suspicion_model_weight": 0.55,
        "suspicion_votes_weight": 0.45,
        "min_votes_to_alert": 1,
        "vote_use_heuristic_score": False,
        "heuristic_vote_thr": 0.80,
        "vote_contact_ratio_thr": 0.35,
        "vote_visibility_drop_thr": 0.25,
        "vote_carry_score_thr": 0.70,
        "vote_disappeared_after_contact": False,
        "state_ttl_s": 120.0,
        "redis_maxlen": 20000,
        "check_updates_every_loops": 1,
        "use_global_identity_state": False,
        "score_field": "score",
        "prefer_fused_if_present": False,
    }
    return ensure_defaults(ns, defaults)


def build_collector_args(cfg: dict, cfg_path: str, cam_id: str):
    raw = format_cam_dict(get_brain_args(cfg, "data_collector_args"), cam_id)
    ns = ns_from_dict(raw)

    defaults = {
        "config": cfg_path,
        "cam_id": cam_id,
        "block_ms": 1000,
        "count": 200,
        "scores_stream": None,
        "decisions_stream": None,
        "alerts_stream": None,
        "feedback_stream": None,
        "clip_refs_stream": None,
        "train_events_stream": None,
        "redis_maxlen": 50000,
        "cache_ttl_s": 30.0,
        "pending_ttl_s": 3600.0,
        "publish_partial": False,
        "gold_filter_unlabeled_ready": False,
        "gold_min_votes": 2,
        "gold_min_S": 0.85,
        "gold_require_gate_ok": False,
        "gold_max_missing_pose_ratio": 0.55,
        "gold_max_missing_obj_ratio": 0.75,
        "enable_llm_feedback_parse": False,
        "llm_model": "gpt-5.4",
        "llm_timeout_s": 8.0,
        "feedback_min_confidence": 0.80,
        "llm_allowed_categories": "theft,false_alarm,benign,uncertain,needs_review",
        "llm_api_key": None,
        "llm_base_url": None,
        "llm_disable_fallback": False,
        "use_scalars_clip_zmq": True,
        "scalars_connect": None,
        "scalars_topic": None,
        "zmq_rcvhwm": 512,

        # Production learning design:
        # Every per-camera collector should publish to the same shared training
        # topic. The fleet/global trainer subscribes once to that topic.
        # These names are examples; your data_collector_llm.py must support them
        # or map them from its existing config fields.
        "train_events_zmq_enabled": True,
        "train_events_zmq_mode": "pub",
        "train_events_zmq_bind": None,
        "train_events_zmq_connect": None,
        "train_events_topic": "train_events.theft",
    }
    return ensure_defaults(ns, defaults)


def build_trainer(cfg_path: str):
    """
    Build one fleet/global trainer.

    The trainer reads all runtime settings from config.yaml, especially:

    trainer_node:
      target_id: global_retail
      model_scope: fleet
      require_site_id: false
      zmq:
        input_mode: sub
        input_connect: tcp://127.0.0.1:5690
        input_topic: train_events.theft
        output_mode: pub
        output_bind: tcp://*:5691
      storage:
        require_s3_clips: true

    Clip locations are not hardcoded here. Each training event must contain:

      payload.clip_ref.s3_uri

    or:

      clip_ref.s3_uri

    The trainer downloads each clip temporarily from S3, loads it, and removes
    the local temp copy. Clips do not need to remain on the device.
    """
    if create_trainer_worker_from_config is not None:
        return create_trainer_worker_from_config(cfg_path)

    # Fallback for slightly different trainer implementation.
    if hasattr(TrainerWorker, "from_config"):
        return TrainerWorker.from_config(cfg_path)

    # Last-resort fallback: pass a small namespace. Prefer the factory above.
    cfg = load_cfg(cfg_path)
    trainer_cfg = cfg.get("trainer_node", {}) or {}
    storage_cfg = trainer_cfg.get("storage", {}) or {}
    zmq_cfg = trainer_cfg.get("zmq", {}) or {}

    ns = argparse.Namespace(
        config=cfg_path,
        target_id=trainer_cfg.get("target_id", trainer_cfg.get("fleet_id", "global_retail")),
        fleet_id=trainer_cfg.get("fleet_id", trainer_cfg.get("target_id", "global_retail")),
        site_id=trainer_cfg.get("site_id"),
        model_scope=trainer_cfg.get("model_scope", "fleet"),
        input_mode=zmq_cfg.get("input_mode", "sub"),
        input_connect=zmq_cfg.get("input_connect"),
        input_bind=zmq_cfg.get("input_bind"),
        input_topic=zmq_cfg.get("input_topic", "train_events.theft"),
        output_mode=zmq_cfg.get("output_mode", "pub"),
        output_bind=zmq_cfg.get("output_bind"),
        output_connect=zmq_cfg.get("output_connect"),
        require_s3_clips=storage_cfg.get("require_s3_clips", True),
    )
    return TrainerWorker(ns)


def run_realtime_pipeline(cfg_path: str):
    try:
        cfg = load_cfg(cfg_path)
        cams = get_active_cams(cfg)

        workers = []
        for cam in cams:
            model_args = build_model_args(cfg, cfg_path, cam)
            policy_args = build_policy_args(cfg, cfg_path, cam)

            model = ModelWorker(model_args)
            policy = PolicyWorker(policy_args)

            workers.append((cam, model, policy))

        print("[brain] realtime started")

        while True:
            for cam, model, policy in workers:
                model.step()
                policy.step()

    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise


def run_learning_pipeline(cfg_path: str):
    """
    Runs all per-camera collectors and exactly one fleet/global trainer.

    Old behavior:
        for each camera:
            collector(cam)
            trainer(cam)

    New production behavior:
        for each camera:
            collector(cam)
        trainer(global/fleet)

    This prevents the training dataset from being split into weak camera-specific
    models. All camera events contribute to one shared fleet model while keeping
    cam_id in event metadata for debugging and metrics.
    """
    try:
        cfg = load_cfg(cfg_path)
        cams = get_active_cams(cfg)

        try:
            from data_collector_llm import setup_logging as setup_collector_logging
            setup_collector_logging("INFO")
        except Exception:
            pass

        collectors = []
        for cam in cams:
            collector_args = build_collector_args(cfg, cfg_path, cam)
            print(
                "[brain] collector_zmq "
                f"cam={cam} "
                f"train_events_zmq_mode={getattr(collector_args, 'train_events_zmq_mode', None)} "
                f"train_events_zmq_bind={getattr(collector_args, 'train_events_zmq_bind', None)} "
                f"train_events_zmq_connect={getattr(collector_args, 'train_events_zmq_connect', None)} "
                f"train_events_topic={getattr(collector_args, 'train_events_topic', None)}"
            )
            collector = DataCollectorWorker(collector_args)
            collectors.append((cam, collector))

        trainer = build_trainer(cfg_path)

        print("[brain] learning started")
        print("[brain] collectors:", cams)
        print("[brain] trainer: one fleet/global trainer from trainer_node config")
        print(
            "[brain] training_mode=batch_based: collectors accumulate/store S3 feedback "
            "samples and publish train_batch messages only when batch thresholds are met"
        )

        while True:
            for cam, collector in collectors:
                collector.step()

            # The trainer subscribes to the shared train_events topic and drains
            # any available messages. Its own implementation should keep this
            # step lightweight and run training in a background thread.
            trainer.step()

    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    cams = get_active_cams(cfg)

    print("[brain] cams:", cams)

    realtime_proc = mp.Process(
        target=run_realtime_pipeline,
        args=(args.config,),
        name="realtime_pipeline",
    )

    learning_proc = mp.Process(
        target=run_learning_pipeline,
        args=(args.config,),
        name="learning_pipeline",
    )

    realtime_proc.start()
    learning_proc.start()

    print("[brain] running")

    restart_counts = {
        "realtime": 0,
        "learning": 0,
    }
    base_restart_delay_s = 2.0
    restart_backoff_step_s = 1.0
    restart_backoff_cap_s = 30.0

    def restart_delay(name: str) -> float:
        count = int(restart_counts.get(name, 0))
        return min(
            base_restart_delay_s + max(0, count - 1) * restart_backoff_step_s,
            restart_backoff_cap_s,
        )

    def wait_for_restart_delay(delay_s: float):
        t0 = time.time()
        while time.time() - t0 < delay_s:
            time.sleep(0.1)

    def restart_child(name: str, old_proc: mp.Process, target):
        restart_counts[name] = int(restart_counts.get(name, 0)) + 1
        count = restart_counts[name]
        exitcode = getattr(old_proc, "exitcode", None)

        try:
            old_proc.join(timeout=1.0)
        except Exception:
            pass

        delay = restart_delay(name)
        print(
            f"[brain] {name} crashed -> restarting "
            f"exitcode={exitcode} restart_count={count} backoff_s={delay:.1f}"
        )
        wait_for_restart_delay(delay)

        proc = mp.Process(
            target=target,
            args=(args.config,),
            name=f"{name}_pipeline",
        )
        proc.start()
        return proc

    try:
        while True:
            if not realtime_proc.is_alive():
                realtime_proc = restart_child("realtime", realtime_proc, run_realtime_pipeline)

            if not learning_proc.is_alive():
                learning_proc = restart_child("learning", learning_proc, run_learning_pipeline)

            time.sleep(1)
    except KeyboardInterrupt:
        pass

    realtime_proc.terminate()
    learning_proc.terminate()

    realtime_proc.join()
    learning_proc.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
