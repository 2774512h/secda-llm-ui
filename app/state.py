import streamlit as st

DEFAULT_CONFIG = {
    "model": None,
    "finetune": None,
    "evaluate": None,
}

def get_config():
    if "config" not in st.session_state:
        st.session_state["config"] = DEFAULT_CONFIG.copy()
    return st.session_state["config"]

def set_section(section: str, value):
    cfg = get_config()
    cfg[section] = value
    st.session_state["config"] = cfg

def validate_config(config):
    if config.get("model") is None:
        return (False, "Pick a model")
    if config.get("finetune") is None:
        return (False, "Pick a finetuning method")
    if config.get("evaluate") is None:
        return (False, "Pick an evaluation method")
    return (True, "Config settings saved")
    
def set_last_run_id(run_id: str):
    st.session_state["last_run_id"] = run_id

def get_last_run_id():
    return st.session_state.get("last_run_id") or None