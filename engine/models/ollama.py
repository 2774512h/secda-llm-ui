from __future__ import annotations 

import os 
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

import httpx

"""
Check Ollama exists, what models are installed, optionally download any, and ask it to generate text

"""

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

@dataclass(frozen=True)
class OllamaModel:
    id: str
    display: str


def _cli_available() -> bool:
    return shutil.which("ollama") is not None


def _http_available(host: str = DEFAULT_OLLAMA_HOST, timeout_s: float = 0.8) -> bool:
    try:
        r = httpx.get(f"{host}/api/tags", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False

def get_status() -> Tuple[bool, str]:
    """
    Returns (ok, message) about Ollama availability.
    """
    if _http_available():
        return True, f"Ollama reachable via HTTP at {DEFAULT_OLLAMA_HOST}"
    if _cli_available():
        return True, "Ollama CLI found on PATH (HTTP not detected)"
    return False, "Ollama not detected (no HTTP service and no CLI on PATH)"

def list_models() -> List[OllamaModel]:
    """
    Prefer HTTP if available; fall back to CLI; else return empty list.
    """
    # HTTP mode (preferred)
    if _http_available():
        r = httpx.get(f"{DEFAULT_OLLAMA_HOST}/api/tags", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        models = []
        for m in data.get("models", []):
            name = m.get("name")
            if name:
                models.append(OllamaModel(id=name, display=name))
        return models

    # CLI fallback
    if _cli_available():
        cp = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
        if cp.returncode != 0:
            return []
        lines = [ln.strip() for ln in cp.stdout.splitlines() if ln.strip()]
        if len(lines) <= 1:
            return []
        out: List[OllamaModel] = []
        for ln in lines[1:]:
            parts = ln.split()
            if parts:
                name = parts[0]
                out.append(OllamaModel(id=name, display=name))
        return out

    return []

def pull_model(model_id: str) -> str:
    """
    Pull via CLI if available. (HTTP pull exists but CLI is simplest.)
    """
    if not _cli_available():
        raise RuntimeError("Cannot pull model: Ollama CLI not found on PATH.")
    cp = subprocess.run(["ollama", "pull", model_id], capture_output=True, text=True, check=False)
    out = (cp.stdout or "") + (cp.stderr or "")
    if cp.returncode != 0:
        raise RuntimeError(out.strip() or "Pull failed")
    return out.strip()


def generate(model_id: str, prompt: str) -> str:
    """
    Generate text from an Ollama model.
    - Prefer HTTP
    - Fall back to CLI
    """
    if not model_id:
        raise RuntimeError("No model selected (model_id is empty).")

    # HTTP (preferred)
    if _http_available():
        payload = {"model": model_id, "prompt": prompt, "stream": False}
        r = httpx.post(f"{DEFAULT_OLLAMA_HOST}/api/generate", json=payload, timeout=120.0)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()

    # CLI fallback
    if _cli_available():
        cp = subprocess.run(["ollama", "run", model_id, prompt], capture_output=True, text=True, check=False)
        if cp.returncode != 0:
            raise RuntimeError((cp.stderr or cp.stdout).strip())
        return (cp.stdout or "").strip()

    raise RuntimeError("Ollama not available. Install/run Ollama on the machine you're using.")

