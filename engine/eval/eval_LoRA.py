from __future__ import annotations

from typing import Any, Dict
import json
import time
import re

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from engine.runs.run_manager import RunPaths, append_log

from engine.sft.sft_LoRA import load_jsonl_dataset, SFTDataset, PromptFormatter, collate_batch


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_first_json(text: str):
    """Extract the first JSON object from a string and parse it as a dict (or return None)."""
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


@torch.no_grad()
def eval_teacher_forced_loss(
    model,
    tokenizer,
    examples,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Teacher-forced eval: run prompt+true completion through the model and compute loss
    only on completion tokens (prompt tokens are masked in labels by SFTDataset).
    """
    formatter = PromptFormatter()
    ds = SFTDataset(examples, tokenizer, max_seq_len=max_seq_len, formatter=formatter)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda feats: collate_batch(feats, pad_token_id=tokenizer.pad_token_id),
    )

    losses = []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(float(out.loss.detach().cpu().item()))

    mean_loss = sum(losses) / max(1, len(losses))
    return {"eval_loss": mean_loss}


@torch.no_grad()
def eval_generation(
    model,
    tokenizer,
    examples,
    device: torch.device,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """
    Generation eval: prompt-only -> generate -> parse JSON -> exact match vs target_json.
    """
    formatter = PromptFormatter()

    exact = 0
    total = 0
    rows = []

    for ex in examples:
        prompt, _completion = formatter.format(ex)

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic baseline for eval comparability
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        pred = extract_first_json(text)

        is_exact = 1 if (pred == ex.target_json) else 0
        exact += is_exact
        total += 1

        rows.append(
            {
                "prompt": prompt,
                "generated_text": text,
                "parsed_json": pred,
                "target_json": ex.target_json,
                "exact_match": is_exact,
            }
        )

    return {
        "gen_exact_match": (exact / total) if total else 0.0,
        "predictions": rows,
    }


class LoRAEval:
    def run(self, config, run: RunPaths, sft_artifact: Dict[str, Any]) -> Dict[str, Any]:
        base_model = sft_artifact["base_model"]
        adapter_dir = sft_artifact["adapter_dir"]
        t0 = time.time()

        out_dir = run.root / "artifacts" / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)

        eval_cfg = config.get("eval") or {}
        finetune_cfg = config.get("finetune") or {}

        dataset_path = (
            eval_cfg.get("dataset_path")
            or finetune_cfg.get("eval_dataset_path")
            or finetune_cfg.get("dataset_path")
            or finetune_cfg.get("dataset")
        )
        if not dataset_path:
            raise ValueError("Eval needs a dataset path (eval.dataset_path or finetune.dataset_path).")

        examples = load_jsonl_dataset(dataset_path)

        limit = eval_cfg.get("limit")
        if limit is not None:
            examples = examples[: int(limit)]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        append_log(run, f"[EVAL][LoRA] device={device.type}")

        append_log(run, f"[EVAL][LoRA] Loading tokenizer: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        append_log(run, f"[EVAL][LoRA] Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(base_model)

        append_log(run, f"[EVAL][LoRA] Loading adapter: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)

        model.to(device)
        model.eval()

        max_seq_len = int(eval_cfg.get("max_seq_len", finetune_cfg.get("max_seq_len", 1024)))
        batch_size = int(eval_cfg.get("batch_size", 4))
        max_new_tokens = int(eval_cfg.get("max_new_tokens", 256))

        append_log(run, f"[EVAL][LoRA] Dataset: {dataset_path} (n={len(examples)})")

        loss_metrics = eval_teacher_forced_loss(model, tokenizer, examples, max_seq_len, batch_size, device)
        append_log(run, f"[EVAL][LoRA] eval_loss={loss_metrics['eval_loss']:.4f}")

        gen_metrics = eval_generation(model, tokenizer, examples, device, max_new_tokens)
        append_log(run, f"[EVAL][LoRA] gen_exact_match={gen_metrics['gen_exact_match']:.3f}")

        metrics = {
            "base_model": base_model,
            "adapter_dir": adapter_dir,
            "dataset_path": str(dataset_path),
            "n_examples": len(examples),
            **loss_metrics,
            "gen_exact_match": gen_metrics["gen_exact_match"],
            "runtime_s": time.time() - t0,
        }

        (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        pred_path = out_dir / "predictions.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for row in gen_metrics["predictions"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return {
            "eval_loss": float(loss_metrics["eval_loss"]),
            "gen_exact_match": float(gen_metrics["gen_exact_match"]),
            "eval_metrics_path": str(out_dir / "eval_metrics.json"),
            "predictions_path": str(pred_path),
    }



