from __future__ import annotations 
from typing import List, Any, Dict
import time

from engine.models.ollama import generate, get_status
from engine.runs.run_manager import RunPaths, append_log, append_output

DEFAULT_SMOKE_PROMPTS: List[str] = [
    "Return 'OK' only.",
    "What is 2+2? Answer with a single number.",
    "Say hello in one short sentence.",
]

class BasicEval:

    def run(self, config: Dict[str, Any], run: RunPaths, sft_artifact: Dict[str, Any]) -> Dict[str, Any]:

        model_cfg = config.get("model") or {}
        model_id = model_cfg.get("name", "")

        evaluate_cfg = config.get("evaluate") or {}
        suite = evaluate_cfg.get("suite", "smoke")

        ok, msg = get_status()
        append_log(run, f"[EVAL] Ollama status: {msg}")
        append_log(run, f"[EVAL] suite={suite}, model={model_id}")
        append_log(run, f"[EVAL] sft_artifact={sft_artifact}")

        # For now we just use a fixed smoke set.
        # Later: load from data registry or eval dataset file.
        prompts = DEFAULT_SMOKE_PROMPTS

        successes = 0
        latencies: List[float] = []

        for i, prompt in enumerate(prompts):
            t0 = time.time()
            success = False
            try:
                response = generate(model_id, prompt)
                success = True
                successes += 1
            except Exception as e:
                response = f"[ERROR] {type(e).__name__}: {e}"

            dt = time.time() - t0
            latencies.append(dt)

            # outputs.jsonl record: one line per test case
            append_output(
                run,
                {
                    "case_id": i,
                    "suite": suite,
                    "model": model_id,
                    "prompt": prompt,
                    "response": response,
                    "latency_s": dt,
                    "success": success,
                },
            )

        avg_latency = sum(latencies) / max(len(latencies), 1)
        success_rate = successes / max(len(prompts), 1)

        metrics = {
            "suite": suite,
            "model": model_id,
            "num_cases": len(prompts),
            "successes": successes,
            "success_rate": success_rate,
            "avg_latency_s": avg_latency,
        }

        append_log(run, f"[EVAL] metrics={metrics}")
        return metrics
        
