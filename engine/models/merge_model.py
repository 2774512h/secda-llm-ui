from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def _choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _choose_torch_dtype(device: str):
    if device == "cuda":
        return torch.float16
    return torch.float32


def _safe_remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _validate_adapter_dir(adapter_dir: Path) -> None:
    if not adapter_dir.exists():
        raise RuntimeError(f"Adapter dir not found: {adapter_dir}")
    if not adapter_dir.is_dir():
        raise RuntimeError(f"Adapter path is not a directory: {adapter_dir}")

    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        raise RuntimeError(
            f"Adapter dir does not look like a PEFT adapter directory: missing {adapter_config}"
        )


def _load_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_base_model(base_model: str, device: str):
    torch_dtype = _choose_torch_dtype(device)

    if device == "cuda":
        return AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
        )

    return AutoModelForCausalLM.from_pretrained(base_model)


def merge_adapter_into_model(
    *,
    base_model: str,
    adapter_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Merge a PEFT adapter into its base Hugging Face causal LM and save
    a standalone full model directory.

    Intended default path:
    - LoRA adapter -> merged full model
    - QLoRA adapter -> may work depending on how the adapter/base were trained and saved

    Returns a small metadata dict describing the merge result.
    """
    t0 = time.time()

    adapter_path = Path(adapter_dir)
    output_path = Path(output_dir)

    _validate_adapter_dir(adapter_path)

    device = _choose_device()
    torch_dtype = _choose_torch_dtype(device)

    _safe_remove_dir(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(base_model)
    base = _load_base_model(base_model, device=device)

    peft_model = PeftModel.from_pretrained(base, str(adapter_path))

    try:
        merged_model = peft_model.merge_and_unload()
    except Exception as e:
        raise RuntimeError(
            "Failed to merge adapter into base model. "
            "This can happen if the adapter/base pair is incompatible or if the "
            "adapter type is not mergeable in the current setup. "
            f"Underlying error: {type(e).__name__}: {e}"
        ) from e

    merged_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    merge_summary = {
        "base_model": base_model,
        "adapter_dir": str(adapter_path),
        "output_dir": str(output_path),
        "device": device,
        "torch_dtype": str(torch_dtype),
        "runtime_s": time.time() - t0,
    }

    (output_path / "merge_summary.json").write_text(
        json.dumps(merge_summary, indent=2),
        encoding="utf-8",
    )

    config_path = output_path / "config.json"
    if not config_path.exists():
        raise RuntimeError(
            f"Merged model save completed, but config.json was not found in {output_path}"
        )

    return merge_summary