import streamlit as st
from state import get_config, set_section
from engine.models.ollama import get_status, list_models

st.title("1) Choose a Model")

ok, msg = get_status()
st.info(msg)

models = list_models()
if not models:
    st.warning("No ollama models found on this machine.")
    models = [
        type("M", (), {"id": "llama3:8b", "display": "llama3:8b (placeholder)"}),
        type("M", (), {"id": "mistral:7b", "display": "mistral:7b (placeholder)"}),
    ]

cfg = get_config() or {}
current = cfg.get("model") or {}

labels = [m.display for m in models]

default_index = 0
current_name = current.get("name")
if isinstance(current_name, str):
    for i, m in enumerate(models):
        if m.id == current_name:
            default_index = i
            break

choice = st.selectbox("Models", labels, index=default_index)
label_to_id = {m.display: m.id for m in models}
chosen_id = label_to_id.get(choice)

base_model = st.text_input(
    "Training base model (HF id or local path)",
    value=current.get("base_model", ""),
    placeholder="e.g. meta-llama/Meta-Llama-3-8B or ./models/llama3_8b_hf"
)

if st.button("Save model"):
    if not chosen_id:
        st.error("No model selected.")
    else:
        model_section = {"provider": "ollama", "name": chosen_id,}
        
        if base_model.strip():
            model_section["base_model"] = base_model.strip()
        else:
            st.warning("Saved Ollama model for inference. Fine-tuning will require a training base model.")

        set_section("model", model_section)
        st.success("Model saved!")

st.subheader("Current config")
st.json(get_config())
