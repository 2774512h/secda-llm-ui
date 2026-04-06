from __future__ import annotations

import shutil
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from engine.models.ollama import get_status, create_model
from engine.models.merge_model import merge_adapter_into_model

def normalize_model_for_export(
    *,
    sft_artifact: Dict[str, Any],
    export_dir: Path,
) -> Tuple[Path, str, Optional[Dict[str, Any]]]:
    """
    Normalize a training artifact into a standalone full-model directory for export.

    Returns:
      (model_dir, normalized_from, normalization_details)

    normalized_from:
      - "merged_adapter"  -> adapter artifact was merged into a full model
      - "full_model"      -> artifact already points at a full model directory
    """
    artifact_type = str(sft_artifact.get("artifact_type") or sft_artifact.get("type") or "").strip().lower()
    method = str(sft_artifact.get("method") or "").strip().lower()

    if artifact_type == "adapter":
        base_model = sft_artifact.get("base_model")
        adapter_dir_str = sft_artifact.get("adapter_dir")
        can_merge = bool(sft_artifact.get("can_merge", False))
        merge_supported = bool(sft_artifact.get("merge_supported", can_merge))

        if not base_model:
            raise RuntimeError("Adapter export normalization requires 'base_model' in sft_artifact.json.")
        if not adapter_dir_str:
            raise RuntimeError("Adapter export normalization requires 'adapter_dir' in sft_artifact.json.")
        if not can_merge or not merge_supported:
            raise RuntimeError(
                f"Artifact method '{method}' is an adapter, but this run is not marked mergeable."
            )

        adapter_dir = Path(adapter_dir_str)
        if not adapter_dir.exists():
            raise RuntimeError(f"Adapter dir not found: {adapter_dir}")

        merged_model_dir = export_dir / "merged_model"

        merge_result = merge_adapter_into_model(
            base_model=base_model,
            adapter_dir=str(adapter_dir),
            output_dir=str(merged_model_dir),
        )

        if not merged_model_dir.exists():
            raise RuntimeError(
                f"Merge completed but merged model dir was not created: {merged_model_dir}"
            )

        return merged_model_dir, "merged_adapter", {
            "base_model": base_model,
            "adapter_dir": str(adapter_dir),
            "merged_model_dir": str(merged_model_dir),
            "merge_result": merge_result,
            "method": method,
        }

    if artifact_type in {"full_model", "full_model_hf", "partial_full_model"}:
        candidate_paths = [
            sft_artifact.get("model_dir"),
            sft_artifact.get("full_model_dir"),
            sft_artifact.get("merged_model_dir"),
        ]
        model_dir_str = next((p for p in candidate_paths if p), None)
        if not model_dir_str:
            raise RuntimeError(
                "Full-model export requires one of: model_dir, full_model_dir, or merged_model_dir "
                "in sft_artifact.json."
            )

        model_dir = Path(model_dir_str)
        if not model_dir.exists():
            raise RuntimeError(f"Full model dir not found: {model_dir}")

        return model_dir, "full_model", {
            "model_dir": str(model_dir),
            "method": method,
        }

    raise RuntimeError(
        f"Unsupported export artifact type: {artifact_type!r}. "
        "Expected 'adapter', 'full_model', 'full_model_hf', or 'partial_full_model'."
    )

def write_full_model_modelfile(
    modelfile_path: Path,
    model_path: Path,
) -> None:
    modelfile_path.write_text(
        "\n".join(
            [
                f"FROM {model_path.resolve()}",
                "",
                "# Optional: keep deterministic-ish defaults for evaluation / RAG",
                "PARAMETER temperature 0.2",
                "PARAMETER top_p 0.9",
                "",
            ]
        ),
        encoding="utf-8",
    )

def export_to_ollama(
    *,
    run_dir: Path,
    run_id: str,
    ollama_new_model_name: str,
    register: bool = True,
) -> Dict[str, Any]:
    """
    Export a fine-tuned run to Ollama.

    Default export behavior is standalone-model-first:
    - Adapter artifacts are normalized by merging them into their base model first
    - Full-model artifacts are exported directly

    Supported paths:
    - LoRA adapter -> merge -> full model -> Ollama
    - QLoRA adapter -> merge if supported -> full model -> Ollama
    - Full fine-tune -> export directly
    - Partial fine-tune -> export directly if artifact already points to a full model
    """
    t0 = time.time()
    run_dir = Path(run_dir)

    sft_artifact_path = run_dir / "artifacts" / "model" / "sft_artifact.json"
    if not sft_artifact_path.exists():
        raise RuntimeError(f"Missing SFT artifact: {sft_artifact_path}")

    sft_artifact = json.loads(sft_artifact_path.read_text(encoding="utf-8"))
    artifact_type = str(sft_artifact.get("artifact_type") or sft_artifact.get("type") or "").strip().lower()
    method = str(sft_artifact.get("method") or "").strip().lower()

    export_dir = run_dir / "artifacts" / "export_ollama"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    modelfile_path = export_dir / "Modelfile"

    model_path_for_modelfile, normalized_from, normalization_details = normalize_model_for_export(
        sft_artifact=sft_artifact,
        export_dir=export_dir,
    )

    write_full_model_modelfile(
        modelfile_path=modelfile_path,
        model_path=model_path_for_modelfile,
    )

    export_mode = "standalone_model"

    ok, status_msg = get_status()

    create_output: Optional[str] = None
    create_error: Optional[str] = None

    if register:
        if not ok:
            create_error = f"Ollama not available: {status_msg}"
        else:
            try:
                create_output = create_model(ollama_new_model_name, str(modelfile_path))
            except Exception as e:
                create_error = f"{type(e).__name__}: {e}"

    summary = {
        "run_id": run_id,
        "export_mode": export_mode,
        "artifact_type": artifact_type,
        "finetune_method": method,
        "ollama_new_model_name": ollama_new_model_name,
        "ollama_status": {"ok": ok, "message": status_msg},
        "normalized_from": normalized_from,
        "normalization_details": normalization_details,
        "sft_artifact": sft_artifact,
        "export_dir": str(export_dir),
        "modelfile_path": str(modelfile_path),
        "model_path_used": str(model_path_for_modelfile) if model_path_for_modelfile else None,
        "attempted_register": bool(register),
        "ollama_create_output": create_output,
        "ollama_create_error": create_error,
        "runtime_s": time.time() - t0,
        "notes": [
            "Default Ollama export is standalone-model-first.",
            "Adapter artifacts are normalized by merging into the recorded base model before export.",
            "Full-model artifacts are exported directly using Modelfile FROM <model_dir>.",
            "QLoRA export depends on whether the artifact is marked merge-supported.",
        ]
    }

    (export_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary