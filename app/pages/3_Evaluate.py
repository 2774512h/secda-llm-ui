import streamlit as st
from state import get_config, set_section

st.title("3) Evaluate")

cfg = get_config()
if not cfg["model"] or not cfg["finetune"]:
    st.warning("Pick a model and fine-tune settings first.")
    st.stop()

suite = st.selectbox("Evaluation suite", ["smoke", "full"])
metrics = st.multiselect("Metrics", ["accuracy", "latency", "cost"], default=["accuracy"])

if st.button("Save evaluation settings"):
    set_section("evaluate", {"suite": suite, "metrics": metrics})
    st.success("Evaluation settings saved!")

st.subheader("Current config")
st.json(get_config())
