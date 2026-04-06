import streamlit as st

from app.state import get_config, set_section
from engine.models.ollama import get_status, list_models

st.title("1) Choose a Model")

ok, msg = get_status()
st.info(msg)

st.subheader("Installed Ollama models on this machine")
models = list_models()
if not models:
    st.warning("No Ollama models found on this machine.")
else:
    for m in models:
        st.write(m.id)

cfg = get_config() or {}
current_model = cfg.get("model") or {}

training_base_model = st.text_input(
    "Training base model for fine-tuning (HF id or local path)",
    value=current_model.get("training_base_model", current_model.get("base_model", "")),
    help="Required for fine-tuning and evaluation.",
)

if st.button("Save model settings"):
    if not training_base_model.strip():
        st.error("Training base model is required.")
    else:
        model_section = {
            "provider": "ollama",
            "training_base_model": training_base_model.strip(),
        }
        set_section("model", model_section)
        st.success("Model settings saved.")
