import streamlit as st
import json
from pathlib import Path

from app.state import get_config, validate_config, set_last_run_id, get_last_run_id
from engine.pipeline import run_pipeline

st.set_page_config(page_title="LLM DSE UI", layout="wide")

st.title("LLM Design Space Exploration UI")
st.write("Workflow: **Model → Fine-tune → Evaluate**")

cfg = get_config()
(ok,msg) = validate_config(cfg)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Run control")
    st.info(msg)

    if st.button("Run", disabled=not ok):
        with st.spinner("Running pipeline..."):
            run = run_pipeline(cfg)

        set_last_run_id(run.run_id)
        st.success(f"Run completed (or failed): {run.run_id}")
        st.code(str(run.root))
with col2:
    st.subheader("Current config")
    st.json(cfg)

    last_run_id = get_last_run_id()
    st.subheader("Latest run")
    if not last_run_id:
        st.caption("No runs yet.")
    else:
        st.write(f"Run ID: `{last_run_id}`")

        run_dir = Path("runs") / last_run_id
        status_path = run_dir / "status.json"
        metrics_path = run_dir / "metrics.json"

        if status_path.exists():
            st.write("Status:")
            st.json(json.loads(status_path.read_text(encoding="utf-8")))
        else:
            st.warning("status.json not found.")

        if metrics_path.exists():
            st.write("Metrics:")
            st.json(json.loads(metrics_path.read_text(encoding="utf-8")))
        else:
            st.info("metrics.json not found (run may still be running or failed).")
