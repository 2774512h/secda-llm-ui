# secda-llm-ui

A local Streamlit application for LLM design space exploration, fine-tuning, evaluation, and export.

The project provides a UI-driven workflow for:
- selecting a base model
- configuring a fine-tuning strategy
- running a training/evaluation pipeline
- exporting a standalone model for Ollama

The current app entrypoint describes the workflow as:

**Model → Fine-tune → Evaluate → Export standalone model to Ollama**

## Repository structure

```text
secda-llm-ui/
├── app/            # Streamlit UI pages and app state
├── data/           # Local datasets used for training/evaluation
├── engine/         # Training, export, model, and pipeline logic
├── requirements.txt
└── README.md
```

### Main directories

- `app/`  
  Contains the Streamlit application, home page, configuration state, and UI pages for model and fine-tuning setup.

- `data/`  
  Contains local dataset files used by the training pipeline. A dataset must be present here before running a fine-tuning job.

- `engine/`  
  Contains the core implementation for model handling, supervised fine-tuning methods, pipeline execution, and export to Ollama.

## Requirements

This project is intended to be run locally.

### Software
- Python 3.10+ recommended
- `pip`
- Windows PowerShell was used for the run command below
- Python packages listed in `requirements.txt`

Key dependencies include:
- `streamlit`
- `transformers`
- `peft`
- `accelerate`
- `torch`
- `datasets`
- `bitsandbytes`
- `safetensors`
- `sentencepiece`

### Hardware
- A **GPU is preferred**
- **Sufficient memory is required to load the selected base model and any fine-tuning adapters**
- Larger models will require substantially more VRAM and/or system RAM
- A small model such as `TinyLlama/TinyLlama_v1.1` is recommended for a quick smoke test

## Dataset requirements

A dataset must exist under:

```text
secda-llm-ui/data
```

For a minimal smoke test, the project expects:

```text
data/smoke_train.jsonl
```

This file should be provided before running training.

## Installation

From the repository root:

```powershell
pip install -r requirements.txt
```

## Running the application

From the project root, start the Streamlit app with:

```powershell
python -m streamlit run app\home.py
```

Then open the local Streamlit URL shown in the terminal.

## Basic workflow

The UI is designed around the following sequence:

1. Select a base model
2. Configure a fine-tuning method
3. Run the pipeline
4. Review the latest run output
5. Optionally export the finished model to Ollama

## Smoke test / minimal verification

There is currently **no automated test suite** in this repository.  
To verify the project works locally, perform a small fine-tuning run through the UI.

### Recommended smoke-test setup

Use the following settings to keep the run lightweight:

- **Base model:** `TinyLlama/TinyLlama_v1.1`
- **Fine-tuning method:** **LoRA**
- **Dataset:** `data/smoke_train.jsonl`
- Prefer a **small batch size**
- Prefer **1 epoch**
- Use a **short max sequence length**
- Keep adapter rank and related LoRA settings modest
- Avoid unnecessarily large evaluation or export settings during the first run

### Suggested goal of the smoke test

A successful smoke test should:
- launch the Streamlit UI
- accept the selected model and dataset
- start and complete a short LoRA training run
- write run artifacts under `runs/`
- show status and metrics in the app

## Export

After a successful run, the UI can export a standalone model and optionally register it with Ollama.

If Ollama registration is enabled, this requires a working local Ollama installation/CLI.

## Notes

- This project is currently documented for **local use only**
- No deployment instructions are required beyond local installation and execution
- No `.env` setup is required for the documented local workflow
