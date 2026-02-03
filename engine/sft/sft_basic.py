from __future__ import annotations
from typing import Any, Dict

from engine.runs.run_manager import RunPaths, append_log

class BasicSFT:


    def run(self, config: Dict[str, Any], run: RunPaths) -> Dict[str,Any]:
        
        finetune = config.get("finetune") or {}

        method = finetune.get("method", "basic")
        dataset_id = finetune.get("dataset", "")
        epochs = finetune.get("epochs", 1)

        if not dataset_id:
            append_log(run, "[SFT] WARNING: No dataset provided")
        
        append_log(run, f"[SFT] Selected method={method}")
        

        artifact = {
            "type": "base",
            "note": "BasicSFT performs no training; evaluation uses the base model.",
            "method": method,
            "dataset": dataset_id,
            "epochs": epochs,
        }

        append_log(run, f"[SFT] Artifact descriptor: {artifact}")

        return artifact 