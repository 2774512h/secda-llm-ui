from __future__ import annotations 

from typing import Any, Dict, Tuple

from engine.runs.run_manager import (
    RunPaths,
    append_log,
    create_run,
    set_status,
    write_metrics,
)

from engine.sft.sft_basic import BasicSFT
from engine.sft.sft_LoRA import LoRASFT
from engine.eval.eval_basic import BasicEval
from engine.eval.eval_LoRA import LoRAEval

SFT_METHODS = {
    "basic" : BasicSFT,
    "LoRA" : LoRASFT, 
    "QLoRA" : BasicSFT,
    "full" : BasicSFT,
    "half" : BasicSFT,
}

EVAL_METHODS = {
    "full" : BasicEval,
    "LoRA" : LoRAEval,
}

def select_sft(config: Dict[str, Any]) -> Tuple[str,Any]:
    finetune = config.get("finetune") or {}
    method = finetune.get("method","basic")
    cls = SFT_METHODS.get(method, BasicSFT)
    return method, cls()

def select_eval(config: Dict[str,Any]) -> Tuple[str,Any]:
    eval = config.get("evaluate") or {}
    suite = eval.get("suite","full")
    cls = EVAL_METHODS.get(suite, BasicEval)
    return suite, cls()

def run_pipeline(config: Dict[str, Any]) -> RunPaths:

    run = create_run(config)

    try:
        set_status(run, "running", "Pipeline started")
        append_log(run, "[PIPELINE] Pipeline started")

        sft_name, sft = select_sft(config)
        append_log(run, f"[PIPELINE] Selected SFT method: {sft_name}")
        set_status(run, "running", f"Running SFT: {sft_name}")

        sft_artifact = sft.run(config, run)
        append_log(run, f"[PIPELINE] SFT artifact: {sft_artifact}")

        eval_name, evaluator = select_eval(config)
        append_log(run, f"[PIPELINE] Selected eval suite: {eval_name}")
        set_status(run, "running", f"Running eval: {eval_name}")

        metrics = evaluator.run(config, run, sft_artifact)

        write_metrics(run, metrics)
        append_log(run, f"[PIPELINE] Metrics written: {metrics}")
        set_status(run, "done", "Run completed successfully")
        return run
    
    except Exception as e:
        append_log(run, f"[PIPELINE] FAILED: {repr(e)}")
        set_status(run, "failed", f"{type(e).__name__}: {e}")
        import traceback
        append_log(run, traceback.format_exc())
        return run