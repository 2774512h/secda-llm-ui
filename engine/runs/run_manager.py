from __future__ import annotations
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

RUNS_DIR = Path("runs")

@dataclass(frozen=True)
class RunPaths:
    run_id: str
    root: Path
    config: Path
    status: Path
    metrics: Path
    outputs: Path
    logs: Path

def write_json(path: Path, obj: Dict[str, Any]) -> None:
    """"Helper to write JSON consistently"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def create_run(config: Dict[str,Any]) -> RunPaths:
    """
    Creates a new run directory and saves the config. 
    Run_id is currently the timestamp and a random suffix
    """
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    root = RUNS_DIR / run_id
    root.mkdir(parents=True, exist_ok=True)

    paths = RunPaths(
        run_id=run_id,
        root=root,
        config=root / "config.json",
        status=root / "status.json",
        metrics=root / "metrics.json",
        outputs=root / "outputs.jsonl",
        logs=root / "logs.txt",
    )

    write_json(paths.config, config)
    set_status(paths, state="created", detail="Run folder created")
    return paths 

def set_status(paths: RunPaths, state: str, detail: str = "", extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {"state": state, "detail": detail, "timestamp": time.time()}
    if extra:
        payload.update(extra)
    write_json(paths.status, payload)

def write_metrics(paths: RunPaths, metrics: Dict[str, Any]) -> None:
    write_json(paths.metrics, metrics)
    

def append_log(paths: RunPaths, line: str) -> None:
    paths.logs.parent.mkdir(parents=True, exist_ok=True)
    with paths.logs.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")

def append_output(paths: RunPaths, record: Dict[str, Any]) -> None:
    paths.outputs.parent.mkdir(parents=True, exist_ok=True)
    with paths.outputs.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")