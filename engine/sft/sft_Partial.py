from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import time

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

from engine.runs.run_manager import RunPaths, append_log


@dataclass
class CanonicalExample:
    device: str
    knobs: Dict[str, List[int]]
    task: str
    target_json: Dict[str, Any]


class PromptFormatter:
    def format(self, ex: CanonicalExample) -> tuple[str, str]:
        """Returns prompt_text and completion_text"""
        knobs_text = json.dumps(ex.knobs, indent=2, sort_keys=True)
        target_text = json.dumps(ex.target_json, indent=2, sort_keys=True)
        prompt = (
            "DEVICE:\n" + (ex.device or "") + "\n\n"
            "KNOBS:\n" + (knobs_text or "") + "\n\n"
            "TASK:\n" + (ex.task or "") + "\n\n"
            "OUTPUT JSON (values only):\n"
        )
        completion = target_text + "\n"
        return prompt, completion


def load_jsonl_dataset(path: str) -> List[CanonicalExample]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset_path not found: {path}")

    examples: List[CanonicalExample] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            knobs = obj.get("knobs", {})
            target_json = obj.get("target_json", {})

            if not isinstance(knobs, dict):
                raise ValueError(f"Line {line_no}: 'knobs' must be a dict, got {type(knobs).__name__}")
            if not isinstance(target_json, dict):
                raise ValueError(f"Line {line_no}: 'target_json' must be a dict, got {type(target_json).__name__}")

            examples.append(
                CanonicalExample(
                    device=str(obj.get("device", "")),
                    knobs=knobs,
                    task=str(obj.get("task", "")),
                    target_json=target_json,
                )
            )

    if not examples:
        raise ValueError("dataset is empty (no JSONL records)")
    return examples


class SFTDataset(Dataset):
    def __init__(self, examples: List[CanonicalExample], tokenizer, max_seq_len: int, formatter: PromptFormatter):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.formatter = formatter

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        prompt, completion = self.formatter.format(ex)

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        completion_ids = self.tokenizer(completion, add_special_tokens=False).input_ids

        input_ids = (prompt_ids + completion_ids)[: self.max_seq_len]
        labels = ([-100] * len(prompt_ids) + completion_ids)[: self.max_seq_len]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def collate_batch(features: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].shape[0] for x in features)

    def pad_1d(x: torch.Tensor, pad_value: int) -> torch.Tensor:
        pad_len = max_len - x.shape[0]
        if pad_len <= 0:
            return x
        return torch.cat([x, torch.full((pad_len,), pad_value, dtype=x.dtype)], dim=0)

    input_ids = torch.stack([pad_1d(x["input_ids"], pad_token_id) for x in features])
    attention_mask = torch.stack([pad_1d(x["attention_mask"], 0) for x in features])
    labels = torch.stack([pad_1d(x["labels"], -100) for x in features])

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def find_transformer_blocks(model):
    """
    Return the list-like container of transformer blocks for common decoder-only architectures.
    Raises ValueError if not found.
    """
    # LLaMA / Mistral / Qwen2 style: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    # GPT-2 / Falcon style: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h

    # OPT style: model.model.decoder.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers

    raise ValueError(
        "Could not locate transformer blocks. "
        "Tried: model.model.layers, model.transformer.h, model.model.decoder.layers"
    )


def set_requires_grad(module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def count_params(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = (100.0 * trainable / total) if total else 0.0
    return trainable, total, pct


def apply_partial_unfreeze(model, unfreeze_last_n: int, unfreeze_lm_head: bool, unfreeze_layer_norms: bool) -> None:
    # Freeze everything first
    set_requires_grad(model, False)

    blocks = find_transformer_blocks(model)
    n_layers = len(blocks)
    if unfreeze_last_n <= 0:
        raise ValueError("Partial SFT requires unfreeze_last_n >= 1")
    if unfreeze_last_n > n_layers:
        raise ValueError(f"unfreeze_last_n={unfreeze_last_n} exceeds number of layers={n_layers}")

    # Unfreeze last N blocks
    for layer in blocks[-unfreeze_last_n:]:
        set_requires_grad(layer, True)

    # Optional: always unfreeze lm_head
    if unfreeze_lm_head and hasattr(model, "lm_head") and model.lm_head is not None:
        set_requires_grad(model.lm_head, True)

    # Optional: unfreeze all LayerNorm/RMSNorm modules
    if unfreeze_layer_norms:
        for m in model.modules():
            name = m.__class__.__name__.lower()
            if "layernorm" in name or name in ("rmsnorm",):
                set_requires_grad(m, True)


class PartialSFT:
    def run(self, config: Dict[str, Any], run: RunPaths) -> Dict[str, Any]:
        model_cfg = config.get("model") or {}
        base_model = model_cfg.get("training_base_model")
        if not base_model:
            raise ValueError("Partial SFT requires config.model.base_model (string)")

        finetune = config.get("finetune") or {}
        partial_cfg = finetune.get("Partial") or {}

        dataset_path = finetune.get("dataset_path") or finetune.get("dataset")
        if not dataset_path:
            raise ValueError("Partial SFT requires config.finetune.dataset_path (or finetune.dataset for smoke test)")

        # Partial knobs
        unfreeze_last_n = int(partial_cfg.get("unfreeze_last_n", 4))
        unfreeze_lm_head = bool(partial_cfg.get("unfreeze_lm_head", True))
        unfreeze_layer_norms = bool(partial_cfg.get("unfreeze_layer_norms", False))

        # Training knobs
        epochs = int(finetune.get("epochs", 1))
        lr = float(finetune.get("lr", 5e-5))
        batch_size = int(finetune.get("batch_size", 1))
        max_seq_len = int(finetune.get("max_seq_len", 1024))
        grad_accum = int(finetune.get("grad_accum", 1))

        # Output dirs
        model_dir = run.root / "artifacts" / "model"
        partial_model_dir = model_dir / "partial_model"
        partial_model_dir.mkdir(parents=True, exist_ok=True)

        # Device / dtype
        use_cuda = torch.cuda.is_available()
        use_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
        append_log(run, f"[SFT][Partial] torch.cuda.is_available()={use_cuda}, bf16_supported={use_bf16}")

        device = torch.device("cuda" if use_cuda else "cpu")
        torch_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else None)

        # Tokenizer
        append_log(run, f"[SFT][Partial] Loading tokenizer: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Model
        append_log(run, f"[SFT][Partial] Loading model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype)
        model.to(device)

        # Apply partial unfreeze policy
        apply_partial_unfreeze(model, unfreeze_last_n, unfreeze_lm_head, unfreeze_layer_norms)
        trainable, total, pct = count_params(model)
        append_log(run, f"[SFT][Partial] Unfrozen last_n={unfreeze_last_n}, lm_head={unfreeze_lm_head}, lns={unfreeze_layer_norms}")
        append_log(run, f"[SFT][Partial] Trainable params: {trainable}/{total} ({pct:.2f}%)")

        # Dataset
        examples = load_jsonl_dataset(dataset_path)
        formatter = PromptFormatter()
        train_ds = SFTDataset(examples, tokenizer, max_seq_len=max_seq_len, formatter=formatter)

        # Training args
        args = TrainingArguments(
            output_dir=str(model_dir / "trainer_out"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            logging_steps=5,
            save_strategy="no",
            report_to=[],
            fp16=bool(use_cuda and not use_bf16),
            bf16=bool(use_bf16),
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            data_collator=lambda feats: collate_batch(feats, pad_token_id=tokenizer.pad_token_id),
        )

        append_log(run, f"[SFT][Partial] Starting training: epochs={epochs}, bs={batch_size}, accum={grad_accum}, lr={lr}")
        t0 = time.time()
        train_result = trainer.train()
        dt = time.time() - t0
        append_log(run, f"[SFT][Partial] Training done in {dt:.2f}s")

        # Save full model + tokenizer (partial still modifies base weights)
        append_log(run, f"[SFT][Partial] Saving model to {partial_model_dir}")
        model.save_pretrained(str(partial_model_dir))
        tokenizer.save_pretrained(str(partial_model_dir))

        summary = {
            "method": "Partial",
            "base_model": base_model,
            "dataset_path": dataset_path,
            "partial": {
                "unfreeze_last_n": unfreeze_last_n,
                "unfreeze_lm_head": unfreeze_lm_head,
                "unfreeze_layer_norms": unfreeze_layer_norms,
                "trainable_params": trainable,
                "total_params": total,
                "trainable_pct": pct,
            },
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "max_seq_len": max_seq_len,
            "cuda_used": use_cuda,
            "bf16_used": bool(use_bf16),
            "train_runtime_s": dt,
            "train_metrics": dict(train_result.metrics) if hasattr(train_result, "metrics") else {},
        }
        (model_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return {
            "type": "full_model",
            "method": "Partial",
            "base_model": base_model,
            "model_dir": str(partial_model_dir),
            "summary_path": str(model_dir / "training_summary.json"),
        }