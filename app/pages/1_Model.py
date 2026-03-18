import streamlit as st

from app.state import get_config, set_section
from engine.models.ollama import get_status, list_models

st.title("1) Choose a Model")

ok, msg = get_status()
st.info(msg)

models = list_models()
if not models:
    st.warning("No Ollama models found on this machine.")
    st.stop()

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

choice = st.selectbox("Inference model(s) in Ollama installed on this machine", labels, index=default_index)

label_to_id = {m.display: m.id for m in models}
chosen_id = label_to_id.get(choice)

base_model = st.text_input(
    "Training base model for fine-tuning (HF id or local path)",
    value=current.get("base_model", ""),
    placeholder="e.g. meta-llama/Meta-Llama-3-8B or ./models/llama3_8b_hf",
    help="Required for fine-tuning. The Ollama model above is used for inference/export.",
)

if st.button("Save model settings"):
    if not chosen_id:
        st.error("No inference model selected.")
    else:
        model_section = {
            "provider": "ollama",
            "name": chosen_id,
        }

        if base_model.strip():
            model_section["base_model"] = base_model.strip()
        else:
            st.warning(
                "Saved Ollama model for inference only. Fine-tuning will require a training base model."
            )

        set_section("model", model_section)
        st.success("Model settings saved.")

st.subheader("Current config")
st.json(get_config())
