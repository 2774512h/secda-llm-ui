import streamlit as st
from state import get_config, set_section

st.title("2) Fine-tune")

cfg = get_config()
if not cfg["model"]:
    st.warning("Pick a model first.")
    st.stop()

method = st.selectbox("Method", ["lora", "qlora", "full", "prompt-tuning"])
dataset = st.text_input("Dataset name/id", value="dataset-v1")
epochs = st.number_input("Epochs", min_value=1, max_value=50, value=3)

if st.button("Save fine-tune settings"):
    set_section("finetune", {"method": method, "dataset": dataset, "epochs": int(epochs)})
    st.success("Fine-tune settings saved!")

st.subheader("Current config")
st.json(get_config())
