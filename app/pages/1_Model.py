import streamlit as st
from state import get_config, set_section
from engine.models.ollama import get_status, list_models, pull_model, generate 

# Display header
st.title("1) Choose a Model")

# Display info box
ok, msg = get_status()
st.info(msg)

models = list_models()

if not models:
    st.warning("No ollama models found on this machine.")
    models = [
        type("M", (), {"id": "llama3:8b", "display": "llama3:8b (placeholder)"}),
        type("M", (), {"id": "mistral:7b", "display": "mistral:7b (placeholder)"}),
    ]

cfg = get_config()
current = cfg["model"]
labels = [m.display for m in models]

default_index = 0
if current and isinstance(current, dict) and current.get("name"):
    for i, m in enumerate(models):
        if m.id in current["name"]:
            default_index = i 
            break

choice = st.selectbox("Models", labels, index=default_index)
label_to_id = {m.display: m.id for m in models}
chosen_id = label_to_id.get(choice)

if st.button("Save model"):
    if not chosen_id:
        st.error("No model selected.")
    else:
        set_section("model", {"provider": "ollama", "name": chosen_id})
        st.success("Model saved!")

st.subheader("Current config")
st.json(get_config())
