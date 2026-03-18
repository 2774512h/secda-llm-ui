import streamlit as st

from app.state import get_config, set_section

st.title("3) Evaluate")

cfg = get_config()
if not cfg["model"] or not cfg["finetune"]:
    st.warning("Choose model and fine-tune settings first.")
    st.stop()

current = cfg.get("eval") or {}
finetune_cfg = cfg.get("finetune") or {}

supported_suites = ["LoRA", "QLoRA", "Full", "Partial"]

current_suite = current.get("suite")
if current_suite not in supported_suites:
    finetune_method = finetune_cfg.get("method", "LoRA")
    current_suite = finetune_method if finetune_method in supported_suites else "LoRA"

inherited_dataset = (
    finetune_cfg.get("eval_dataset_path")
    or finetune_cfg.get("dataset_path")
    or finetune_cfg.get("dataset")
    or ""
)
inherited_max_seq_len = int(finetune_cfg.get("max_seq_len", 1024))

use_dataset_override_default = bool(current.get("dataset_path"))
use_max_seq_len_override_default = "max_seq_len" in current

default_dataset_override = current.get("dataset_path", "")
default_limit = int(current.get("limit", 0))
default_batch_size = int(current.get("batch_size", 4))
default_max_seq_len_override = int(current.get("max_seq_len", inherited_max_seq_len))
default_max_new_tokens = int(current.get("max_new_tokens", 256))

suite = st.selectbox(
    "Evaluation suite",
    supported_suites,
    index=supported_suites.index(current_suite),
)

if suite == "Partial":
    st.info("Partial fine-tuning currently uses the same evaluator implementation as Full in the pipeline.")

st.subheader("Inherited from fine-tuning")
st.text_input("Dataset path / name / id", value=inherited_dataset, disabled=True)
st.number_input(
    "Max sequence length",
    min_value=64,
    max_value=32768,
    value=inherited_max_seq_len,
    step=64,
    disabled=True,
)

st.subheader("Evaluation settings")
limit = st.number_input(
    "Maximum examples to evaluate (0 = use all)",
    min_value=0,
    max_value=1_000_000,
    value=default_limit,
)
batch_size = st.number_input("Evaluation batch size", min_value=1, max_value=1024, value=default_batch_size)
max_new_tokens = st.number_input("Max new tokens", min_value=1, max_value=4096, value=default_max_new_tokens)

st.subheader("Optional overrides")
use_dataset_override = st.checkbox("Use a different evaluation dataset", value=use_dataset_override_default)
if use_dataset_override:
    dataset_path_override = st.text_input("Evaluation dataset override", value=default_dataset_override)

use_max_seq_len_override = st.checkbox(
    "Override max sequence length for evaluation",
    value=use_max_seq_len_override_default,
)
if use_max_seq_len_override:
    max_seq_len_override = st.number_input(
        "Evaluation max sequence length",
        min_value=64,
        max_value=32768,
        value=default_max_seq_len_override,
        step=64,
    )

st.subheader("What this evaluation currently computes")
st.markdown(
    """
- Teacher-forced evaluation loss
- Generation exact match against `target_json`
- Raw predictions saved to artifact files
"""
)

if st.button("Save evaluation settings"):
    eval_cfg = {
        "suite": suite,
        "batch_size": int(batch_size),
        "max_new_tokens": int(max_new_tokens),
    }

    if int(limit) > 0:
        eval_cfg["limit"] = int(limit)

    if use_dataset_override and dataset_path_override.strip():
        eval_cfg["dataset_path"] = dataset_path_override.strip()

    if use_max_seq_len_override:
        eval_cfg["max_seq_len"] = int(max_seq_len_override)

    set_section("eval", eval_cfg)
    st.success("Evaluation settings saved.")

st.subheader("Current config")
st.json(get_config())