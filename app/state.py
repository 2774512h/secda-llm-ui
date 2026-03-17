import streamlit as st

DEFAULT_CONFIG = {
    "model": None,
    "finetune": None,
    "eval": None,
}

def _normalize_config(cfg):
    if cfg is None:
        cfg = {}

    if "eval" not in cfg and "evaluate" in cfg:
        cfg["eval"] = cfg.get("evaluate")

    if "model" not in cfg:
        cfg["model"] = None
    if "finetune" not in cfg:
        cfg["finetune"] = None
    if "eval" not in cfg:
        cfg["eval"] = None

    if "evaluate" in cfg:
        del cfg["evaluate"]

    return cfg

def get_config():
    if "config" not in st.session_state:
        st.session_state["config"] = DEFAULT_CONFIG.copy()
    else:
        st.session_state["config"] = _normalize_config(st.session_state["config"])
    return st.session_state["config"]

def set_section(section: str, value):
    cfg = get_config()
    cfg[section] = value
    if section == "eval" and "evaluate" in cfg:
        del cfg["evaluate"]
    st.session_state["config"] = _normalize_config(cfg)

def validate_config(config):
    config = _normalize_config(config)
    if config.get("model") is None:
        return (False, "Pick a model")
    if config.get("finetune") is None:
        return (False, "Pick a finetuning method")
    if config.get("eval") is None:
        return (False, "Pick an evaluation method")
    return (True, "Config settings saved")
    
def set_last_run_id(run_id: str):
    st.session_state["last_run_id"] = run_id

def get_last_run_id():
    return st.session_state.get("last_run_id") or None