# Reproduction Runbook

1. Create a Python environment and install `requirements.txt`.
2. From the project root, set `PYTHONPATH=src`.
3. Run `python scripts/prepare_data.py`. This downloads the UCI SMS Spam Collection, balances ham against spam using seed 123, maps ham/spam to 0/1, and writes `data/train.csv`, `data/validation.csv`, and `data/test.csv`.
4. Obtain GPT-2 124M parameters if pretrained initialization is required. The notebook's `gpt_download3.py` conversion path needs TensorFlow; do not treat the TensorFlow files under `gpt2/` as a PyTorch state dict.
5. Run `python scripts/train.py`. This performs the notebook's five-epoch classifier experiment and writes `checkpoints/spam_classifier.pt`. To initialize from GPT-2 weights, add `--gpt2-model-dir gpt2` after installing TensorFlow; otherwise the CLI starts from the model's random initialization and is only a pipeline smoke run.
6. Run `python scripts/evaluate.py --checkpoint checkpoints/spam_classifier.pt`.
7. Run `python scripts/generate.py "The capital of France is" --tokens 10` for a generation smoke check. Do not pass `spam_classifier.pt`: its two-class head is not compatible with language-model generation.

The original notebook remains available at `notebooks/original_colab.ipynb`.
