# My LLM

This repository is a modular extraction of `My LLM.ipynb`. The notebook is an educational GPT-2-compatible implementation with attention demonstrations, language-model pretraining utilities, GPT-2 124M weight loading, and SMS spam classification fine-tuning.

## Current specification

The final fine-tuning path uses GPT-2 small: vocabulary 50,257, context length 1,024, embedding size 768, 12 layers, 12 heads, dropout 0.0, and QKV bias enabled. The classification experiment replaces the language-model head with a two-class head and trains the head plus final normalization using AdamW (`5e-5`, weight decay `0.1`) for 5 epochs. Dataset sequences are padded/truncated to 120 GPT-2 tokens.

## Installation and workflows

Use Python 3.10+ and install `requirements.txt`. Run commands from the repository root with `PYTHONPATH=src`.

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python scripts/prepare_data.py
PYTHONPATH=src python scripts/train.py
PYTHONPATH=src python scripts/evaluate.py
PYTHONPATH=src python scripts/generate.py "The capital of France is" --tokens 10
```

The UCI SMS Spam Collection is downloaded only by `prepare_data.py`. GPT-2 checkpoint conversion is optional and requires TensorFlow; pass `--gpt2-model-dir gpt2` to `train.py` when those converted parameters are available. The original helper is preserved as `gpt_download3.py`, and the local GPT-2 files are not assumed to be a trained PyTorch checkpoint.

## Reproducibility

The notebook uses seed 123, a 70/10/20 balanced SMS split, batch size 8, and no scheduler or gradient accumulation. See [RUNBOOK.md](RUNBOOK.md) for the exact sequence and [PROJECT_AUDIT.md](PROJECT_AUDIT.md) for the full audit. No notebook result values were invented; the notebook did not contain a reliable saved result report.

## Limitations

The notebook mixes multiple intermediate definitions and contains a separate 256-token pretraining configuration. The modular code uses the final 1024-token, QKV-biased configuration used by the fine-tuning master setup. Loading the original GPT-2 TensorFlow checkpoint still requires TensorFlow and network access. Full tests require PyTorch and the other dependencies.
