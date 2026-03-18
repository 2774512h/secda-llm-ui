from __future__ import annotations

import shutil
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from engine.models.ollama import get_status, create_model

def _maybe_convert_adapter_to_gguf(
    adapter_dir: Path,
    export_dir: Path,
    llama_cpp_convert_script: Optional[str],
) -> Tuple[Path, Optional[str]]:
    """
    Best-effort conversion of a PEFT adapter directory to a GGUF adapter using llama.cpp's convert_lora_to_gguf.py.

    Returns:
      (adapter_path_to_use_in_modelfile, warning_message)
    """
    # If user didn't provide a converter, just use adapter_dir as-is.
    if not llama_cpp_convert_script:
        return adapter_dir, (
            "No llama.cpp convert script provided; using adapter directory directly in Modelfile. "
            "If Ollama rejects it, provide llama_cpp_convert_script to convert adapter to GGUF."
        )

    script = Path(llama_cpp_convert_script)
    if not script.exists():
        return adapter_dir, f"llama_cpp_convert_script not found at {script}; using adapter directory directly."

    # Heuristic: if adapter already contains a .gguf file, prefer it
    gguf_candidates = list(adapter_dir.glob("*.gguf"))
    if gguf_candidates:
        return gguf_candidates[0], None

    # Otherwise, attempt conversion into export_dir/adapter.gguf
    out_path = export_dir / "adapter.gguf"

    # llama.cpp script usage can vary; this is a best-effort call.
    # Many setups accept: python convert_lora_to_gguf.py <adapter_dir> --outfile <out>
    cmd = ["python", str(script), str(adapter_dir), "--outfile", str(out_path)]

    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = (cp.stdout or "") + (cp.stderr or "")
        if cp.returncode != 0 or not out_path.exists():
            return adapter_dir, (
                "Attempted adapter conversion to GGUF but it failed; using adapter directory directly.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Output:\n{out.strip()}"
            )
        return out_path, None
    except Exception as e:
        return adapter_dir, f"Adapter conversion threw {type(e).__name__}: {e}. Using adapter directory directly."

def export_to_ollama(
    *,
    run_dir: Path,
    run_id: str,
    ollama_new_model_name: str,
    ollama_base_model: str,
    register: bool = True,
    llama_cpp_convert_script: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Real Ollama export for LoRA runs via Modelfile + ADAPTER.

    - Reads runs/<run_id>/artifacts/model/sft_artifact.json (must include base_model + adapter_dir)
    - Writes runs/<run_id>/artifacts/export_ollama/Modelfile + export_summary.json
    - Modelfile uses:
        FROM <ollama_base_model>
        ADAPTER <adapter path>
      (ADAPTER path can be relative to Modelfile) :contentReference[oaicite:2]{index=2}
    - Optionally registers with `ollama create` if Ollama is available.
    """
    t0 = time.time()
    run_dir = Path(run_dir)

    sft_artifact_path = run_dir / "artifacts" / "model" / "sft_artifact.json"
    if not sft_artifact_path.exists():
        raise RuntimeError(f"Missing SFT artifact: {sft_artifact_path}")

    sft_artifact = json.loads(sft_artifact_path.read_text(encoding="utf-8"))
    adapter_dir = Path(sft_artifact.get("adapter_dir", ""))
    if not adapter_dir.exists():
        raise RuntimeError(f"Adapter dir not found: {adapter_dir}")

    export_dir = run_dir / "artifacts" / "export_ollama"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Convert adapter if needed (best-effort)
    adapter_path_for_modelfile, convert_warning = _maybe_convert_adapter_to_gguf(
        adapter_dir=adapter_dir,
        export_dir=export_dir,
        llama_cpp_convert_script=llama_cpp_convert_script,
    )

    # Stage adapter beside the Modelfile for Ollama.
    # On Windows, pointing ADAPTER at an external directory can be flaky.
    staged_adapter_path = None

    if adapter_path_for_modelfile.is_dir():
        required = ["adapter_config.json", "adapter_model.safetensors"]
        missing = [name for name in required if not (adapter_path_for_modelfile / name).exists()]
        if missing:
            raise RuntimeError(
                f"Adapter directory is missing required files: {missing} in {adapter_path_for_modelfile}"
            )

        shutil.copy2(adapter_path_for_modelfile / "adapter_config.json", export_dir / "adapter_config.json")
        shutil.copy2(adapter_path_for_modelfile / "adapter_model.safetensors", export_dir / "adapter_model.safetensors")
        staged_adapter_path = "."
    else:
        # GGUF adapter file case
        staged_name = adapter_path_for_modelfile.name
        shutil.copy2(adapter_path_for_modelfile, export_dir / staged_name)
        staged_adapter_path = f"./{staged_name}"

    modelfile_path = export_dir / "Modelfile"
    modelfile_path.write_text(
        "\n".join(
            [
                f"FROM {ollama_base_model}",
                f"ADAPTER {staged_adapter_path}",
                "",
                "# Optional: keep deterministic-ish defaults for evaluation / RAG",
                "PARAMETER temperature 0.2",
                "PARAMETER top_p 0.9",
                "",
            ]
        ),
        encoding="utf-8",
    )

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
        "ollama_base_model": ollama_base_model,
        "ollama_new_model_name": ollama_new_model_name,
        "ollama_status": {"ok": ok, "message": status_msg},
        "sft_artifact": sft_artifact,
        "export_dir": str(export_dir),
        "modelfile_path": str(modelfile_path),
        "adapter_path_used": str(adapter_path_for_modelfile),
        "adapter_path_in_modelfile": str(staged_adapter_path),
        "adapter_convert_warning": convert_warning,
        "attempted_register": bool(register),
        "ollama_create_output": create_output,
        "ollama_create_error": create_error,
        "runtime_s": time.time() - t0,
        "notes": [
            "This export uses Ollama Modelfile ADAPTER support (base model must match adapter’s training base).",
            "If registration fails, check base model exists in Ollama and adapter format compatibility.",
        ],
    }

    (export_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary