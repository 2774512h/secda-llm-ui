import streamlit as st
import json
from pathlib import Path

from app.state import get_config, validate_config, set_last_run_id, get_last_run_id
from engine.pipeline import run_pipeline
from engine.export_ollama import export_to_ollama

st.set_page_config(page_title="SECDA LLM UI", layout="wide")
st.title("SECDA LLM Design Space Exploration UI")
st.write("Workflow: **Model → Fine-tune → Evaluate → Export standalone model to Ollama**")

cfg = get_config()
(ok, msg) = validate_config(cfg)

col1, col2 = st.columns([2, 1])

# ---------------------------------------------------
# LEFT COLUMN (Run control + Export)
# ---------------------------------------------------

with col1:
    st.subheader("Run pipeline")
    st.info(msg)

    if st.button("Run", disabled=not ok):
        try:
            with st.spinner("Running pipeline..."):
                run = run_pipeline(cfg)

            set_last_run_id(run.run_id)
            st.success(f"Run completed (or failed): {run.run_id}")
            st.code(str(run.root))

        except Exception as e:
            st.error("Pipeline crashed")
            st.exception(e)

    # ----------------------------
    # EXPORT SECTION
    # ----------------------------

    last_run_id = get_last_run_id()

    if last_run_id:
        run_dir = Path("runs") / last_run_id
        status_path = run_dir / "status.json"

        if status_path.exists():
            status_obj = json.loads(status_path.read_text(encoding="utf-8"))
            state = status_obj.get("state") or status_obj.get("status")

            if state == "done":
                st.divider()
                st.subheader("Export to Ollama")

                default_name = f"dse-{last_run_id}"

                model_name = st.text_input(
                    "New Ollama model name",
                    value=default_name,
                    key="export_model_name",
                )

                register = st.checkbox(
                    "Register in Ollama (requires Ollama CLI)",
                    value=True,
                    key="export_register",
                )

                st.caption("Default export creates a standalone model first, then registers it in Ollama.")

                if st.button("Export to Ollama", key="export_button"):
                    try:
                        with st.spinner("Exporting..."):
                            summary = export_to_ollama(
                                run_dir=run_dir,
                                run_id=last_run_id,
                                ollama_new_model_name=model_name,
                                register=register,
                            )

                        attempted = summary.get("attempted_register", False)
                        create_error = summary.get("ollama_create_error")

                        if attempted and create_error:
                            st.error("Export finished, but Ollama registration failed")
                        else:
                            st.success("Export complete")

                        st.json(summary)

                    except Exception as e:
                        st.error("Export failed")
                        st.exception(e)

# ---------------------------------------------------
# RIGHT COLUMN (Config + Latest Run)
# ---------------------------------------------------

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