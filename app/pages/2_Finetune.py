import streamlit as st
from state import get_config, set_section

st.title("2) Fine-tune")

cfg = get_config()
if not cfg["model"]:
    st.warning("Pick a model first.")
    st.stop()

current = cfg.get("finetune") or {}
supported_methods = ["LoRA", "QLoRA", "Full", "Partial"]

current_method = current.get("method", "LoRA")
current_dataset = current.get("dataset_path") or current.get("dataset") or ""
current_epochs = int(current.get("epochs", 3))
current_lr = float(current.get("lr", 2e-4))
current_batch_size = int(current.get("batch_size", 1))
current_max_seq_len = int(current.get("max_seq_len", 1024))
current_grad_accum = int(current.get("grad_accum", 1))

current_lora = current.get("LoRA") or {}
current_qlora = current.get("QLoRA") or {}
current_partial = current.get("Partial") or {}

default_lora_target_modules = current_lora.get("target_modules")
if isinstance(default_lora_target_modules, list):
    default_lora_target_modules = ", ".join(default_lora_target_modules)
elif not default_lora_target_modules:
    default_lora_target_modules = ""

default_qlora_target_modules = current_qlora.get("target_modules")
if isinstance(default_qlora_target_modules, list):
    default_qlora_target_modules = ", ".join(default_qlora_target_modules)
elif not default_qlora_target_modules:
    default_qlora_target_modules = ""

method = st.selectbox(
    "Method",
    supported_methods,
    index=supported_methods.index(current_method) if current_method in supported_methods else 0,
)

dataset = st.text_input("Dataset path/name/id", value=current_dataset)

st.subheader("Common training settings")
epochs = st.number_input("Epochs", min_value=1, max_value=50, value=current_epochs)
lr = st.number_input("Learning rate", min_value=0.0, value=current_lr, format="%.8f")
batch_size = st.number_input("Batch size", min_value=1, max_value=128, value=current_batch_size)
max_seq_len = st.number_input("Max sequence length", min_value=64, max_value=32768, value=current_max_seq_len, step=64)

if method in ["Full", "Partial"]:
    grad_accum = st.number_input("Gradient accumulation", min_value=1, max_value=1024, value=current_grad_accum)

if method == "LoRA":
    st.subheader("LoRA settings")
    lora_r = st.number_input("LoRA rank (r)", min_value=1, max_value=256, value=int(current_lora.get("r", 8)))
    lora_alpha = st.number_input("LoRA alpha", min_value=1, max_value=1024, value=int(current_lora.get("alpha", 16)))
    lora_dropout = st.number_input(
        "LoRA dropout",
        min_value=0.0,
        max_value=1.0,
        value=float(current_lora.get("dropout", 0.05)),
        format="%.4f",
    )
    lora_target_modules_raw = st.text_input(
        "LoRA target modules (comma-separated, leave blank to auto-detect)",
        value=default_lora_target_modules,
    )

if method == "QLoRA":
    st.subheader("QLoRA settings")
    qlora_r = st.number_input("QLoRA rank (r)", min_value=1, max_value=256, value=int(current_qlora.get("r", 8)))
    qlora_alpha = st.number_input("QLoRA alpha", min_value=1, max_value=1024, value=int(current_qlora.get("alpha", 16)))
    qlora_dropout = st.number_input(
        "QLoRA dropout",
        min_value=0.0,
        max_value=1.0,
        value=float(current_qlora.get("dropout", 0.05)),
        format="%.4f",
    )
    qlora_target_modules_raw = st.text_input(
        "QLoRA target modules (comma-separated, leave blank to auto-detect)",
        value=default_qlora_target_modules,
    )

    st.subheader("QLoRA quantization settings")
    use_4bit = st.checkbox("Use 4-bit quantization", value=bool(current_qlora.get("use_4bit", True)))
    bnb_4bit_quant_type = st.selectbox(
        "4-bit quantization type",
        ["nf4", "fp4"],
        index=["nf4", "fp4"].index(current_qlora.get("bnb_4bit_quant_type", "nf4"))
        if current_qlora.get("bnb_4bit_quant_type", "nf4") in ["nf4", "fp4"]
        else 0,
    )
    bnb_4bit_use_double_quant = st.checkbox(
        "Use double quantization",
        value=bool(current_qlora.get("bnb_4bit_use_double_quant", True)),
    )
    bnb_4bit_compute_dtype = st.selectbox(
        "4-bit compute dtype",
        ["bf16", "fp16"],
        index=["bf16", "fp16"].index(current_qlora.get("bnb_4bit_compute_dtype", "bf16"))
        if current_qlora.get("bnb_4bit_compute_dtype", "bf16") in ["bf16", "fp16"]
        else 0,
    )

if method == "Full":
    st.subheader("Full fine-tuning settings")
    st.info("Full fine-tuning updates the whole model and generally needs more memory and compute than LoRA or QLoRA.")

if method == "Partial":
    st.subheader("Partial fine-tuning settings")
    unfreeze_last_n = st.number_input(
        "Unfreeze last N transformer blocks",
        min_value=1,
        max_value=1024,
        value=int(current_partial.get("unfreeze_last_n", 4)),
    )
    unfreeze_lm_head = st.checkbox(
        "Unfreeze LM head",
        value=bool(current_partial.get("unfreeze_lm_head", True)),
    )
    unfreeze_layer_norms = st.checkbox(
        "Unfreeze layer norms",
        value=bool(current_partial.get("unfreeze_layer_norms", False)),
    )

if st.button("Save fine-tune settings"):
    finetune_cfg = {
        "method": method,
        "dataset": dataset,
        "dataset_path": dataset,
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "max_seq_len": int(max_seq_len),
    }

    if method in ["Full", "Partial"]:
        finetune_cfg["grad_accum"] = int(grad_accum)

    if method == "LoRA":
        lora_target_modules = [x.strip() for x in lora_target_modules_raw.split(",") if x.strip()]
        finetune_cfg["LoRA"] = {
            "r": int(lora_r),
            "alpha": int(lora_alpha),
            "dropout": float(lora_dropout),
        }
        if lora_target_modules:
            finetune_cfg["LoRA"]["target_modules"] = lora_target_modules

    if method == "QLoRA":
        qlora_target_modules = [x.strip() for x in qlora_target_modules_raw.split(",") if x.strip()]
        finetune_cfg["QLoRA"] = {
            "r": int(qlora_r),
            "alpha": int(qlora_alpha),
            "dropout": float(qlora_dropout),
            "use_4bit": bool(use_4bit),
            "bnb_4bit_quant_type": bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": bool(bnb_4bit_use_double_quant),
            "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype,
        }
        if qlora_target_modules:
            finetune_cfg["QLoRA"]["target_modules"] = qlora_target_modules

    if method == "Partial":
        finetune_cfg["Partial"] = {
            "unfreeze_last_n": int(unfreeze_last_n),
            "unfreeze_lm_head": bool(unfreeze_lm_head),
            "unfreeze_layer_norms": bool(unfreeze_layer_norms),
        }

    set_section("finetune", finetune_cfg)
    st.success("Fine-tune settings saved!")

st.subheader("Current config")
st.json(get_config())