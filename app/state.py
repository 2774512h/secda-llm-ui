import json
from pathlib import Path

import streamlit as st

CONFIG_PATH = Path("app_config.json")

DEFAULT_CONFIG = {
    "model": None,
    "finetune": None,
    "eval": None,
}


def _normalize_config(cfg):
    if cfg is None:
        cfg = {}

    if "model" not in cfg:
        cfg["model"] = None
    if "finetune" not in cfg:
        cfg["finetune"] = None
    if "eval" not in cfg:
        cfg["eval"] = None

    return cfg


def _load_config_from_disk():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return _normalize_config(json.load(f))
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def _save_config_to_disk(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(_normalize_config(cfg), f, indent=2)


def get_config():
    if "config" not in st.session_state:
        st.session_state["config"] = _load_config_from_disk()
    else:
        st.session_state["config"] = _normalize_config(st.session_state["config"])
    return st.session_state["config"]


def set_section(section: str, value):
    cfg = get_config()
    cfg[section] = value
    cfg = _normalize_config(cfg)
    st.session_state["config"] = cfg
    _save_config_to_disk(cfg)


def validate_config(config):
    config = _normalize_config(config)

    model_cfg = config.get("model") or {}
    finetune_cfg = config.get("finetune") or {}
    eval_cfg = config.get("eval") or {}

    if not model_cfg.get("name"):
        return False, "Choose an inference model in the Model page."
    if not finetune_cfg.get("method"):
        return False, "Choose a fine-tuning method and dataset in the Fine-tune page."
    if not eval_cfg.get("suite"):
        return False, "Choose an evaluation suite in the Evaluate page."

    return True, "Configuration complete and ready to run."


def set_last_run_id(run_id: str):
    st.session_state["last_run_id"] = run_id


def get_last_run_id():
    return st.session_state.get("last_run_id") or None