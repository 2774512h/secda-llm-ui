# User Manual

## Overview

`secda-llm-ui` is a local Streamlit application for experimenting with language model fine-tuning workflows.

It is intended to help you:
- choose a base model
- configure fine-tuning
- run a training/evaluation pipeline
- inspect outputs from the latest run
- export a trained model for use with Ollama

The main workflow is:

**Model → Fine-tune → Evaluate → Export standalone model to Ollama**

## Before you start

Make sure you have:

- Python installed
- all dependencies installed from `requirements.txt`
- enough available memory to load your chosen model
- a dataset placed in the `data/` directory

A GPU is strongly preferred. Small models are much easier to run locally, especially for first-time validation.

## Starting the application

Open PowerShell in the repository root and run:

```powershell
python -m streamlit run app\home.py
```

Streamlit will print a local URL in the terminal. Open that address in your browser.

## Preparing data

Before running fine-tuning, place your dataset file inside:

```text
secda-llm-ui/data
```

For a minimal test run, use:

```text
data/smoke_train.jsonl
```

This dataset is intended for a quick smoke test to confirm the project is functioning.

## Using the UI

### 1. Choose a model

Go to the model configuration page and select the base model you want to use.

For the first run, use:

```text
TinyLlama/TinyLlama_v1.1
```

This keeps the hardware requirement lower than using a larger instruction model.

Save the settings.

### 2. Configure fine-tuning

Open the fine-tuning section and choose a training strategy.

For the first run, use:

- **LoRA** fine-tuning

This is the recommended smoke-test option because it is much lighter than full fine-tuning.

To keep the run as small as possible:
- use a small batch size
- use a single epoch
- keep sequence length short
- use modest LoRA adapter settings
- avoid aggressive evaluation settings on the first pass

Save the settings.

### 3. Configure evaluation 

Make sure you also select the LoRA Eval suite. 

Save the settings.

### 4. Run the pipeline

Return to the main page and click **Run**.

The application will:
- validate the current configuration
- start the pipeline
- show the pipeline is running
- save the latest run ID
- display status and metrics upon completion

### 5. Review outputs

After the run completes, the home page shows details for the latest run.

You should expect to see:
- a run ID
- status information
- metrics if they were produced

Run outputs are written under the local `runs/` directory.

### 6. Export to Ollama

If the run completes successfully, the application can export the resulting model.

In the export section:
- enter a new Ollama model name, or use the default generated name
- choose whether to register it in Ollama
- run the export

If registration is enabled, a working local Ollama installation is required.

## Recommended first-run procedure

Use this exact approach to verify the project works:

1. Install dependencies
2. Place `smoke_train.jsonl` inside `data/`
3. Start the app
4. Select `TinyLlama/TinyLlama_v1.1`
5. Choose **LoRA**
6. Keep settings minimal to reduce memory and runtime
7. Click **Run**
8. Confirm that a run directory is created and status is shown in the UI

## Troubleshooting

### The model will not load
Possible causes:
- not enough VRAM
- not enough system RAM
- model too large for the machine

Try:
- switching to `TinyLlama/TinyLlama_v1.1`
- closing other memory-heavy applications
- using a GPU-enabled environment if available

### Training and exportation stage is very slow

This is very common, even on small models the pipeline and exportation process take some time. 

### The app opens but training fails
Check:
- that `data/smoke_train.jsonl` exists
- that the selected settings are valid
- that all dependencies installed successfully

### Ollama export fails
Check:
- that the run completed successfully first
- that Ollama is installed locally if registration is enabled
- that the base model is part of the Ollama family

## Intended usage

This project is intended to be run **locally**.  
No deployment process is required for normal use.

## Test status

There is currently **no automated test suite** documented for this project.

The recommended proof that the software works is a minimal LoRA smoke test using:
- base model: `TinyLlama/TinyLlama_v1.1`
- dataset: `data/smoke_train.jsonl`
