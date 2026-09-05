# Project Audit

Source audited: `/home/ujwal-mahajan/Desktop/My LLM.ipynb` (208 cells). The source notebook was preserved unchanged and copied to `notebooks/original_colab.ipynb`.

## 1. Model architecture

The notebook builds self-attention, causal attention, a per-head multi-head wrapper, and a fused multi-head implementation. The final GPT path is a pre-layer-normalized decoder-only transformer: token embedding plus learned positional embedding, dropout, 12 transformer blocks, final custom LayerNorm, and a linear vocabulary head. Each block has causal multi-head attention, an output projection, a two-layer feed-forward network with approximate GELU, residual connections, and dropout.

## 2. Configuration and parameters

The final fine-tuning configuration is GPT-2 small: `vocab_size=50257`, `context_length=1024`, `emb_dim=768`, `n_heads=12`, `n_layers=12`, `drop_rate=0.0`, and `qkv_bias=True`. Earlier cells use context length 256, dropout 0.1, and no QKV bias. The repository preserves the final configuration in `configs/model.yaml`.

The GPT-2 architecture conventionally has about 124M parameters because the token embedding and output head are tied. The notebook instantiates an independent `out_head`, so a fresh PyTorch object allocates about 163.0M parameter entries; the GPT-2 loading cell assigns the same embedding values to the head but does not explicitly tie the module parameters. This distinction is documented rather than silently changed.

## 3. Tokenizer

The notebook uses `tiktoken.get_encoding("gpt2")`, GPT-2 BPE vocabulary, and token ID 50256 as padding/EOS in the classifier path. Tokenization is not trained or modified. The extracted wrapper is in `src/custom_llm/tokenizer/gpt2.py`.

## 4. Datasets and preprocessing

The language-modeling demonstration downloads `the-verdict.txt` from the rasbt/LLMs-from-scratch GitHub repository, splits it 80/20 by character offset, creates next-token input/target windows, and uses context length and stride 1024 in the later setup. The supervised path downloads the UCI SMS Spam Collection, downsamples ham to the spam count, maps ham/spam to 0/1, shuffles with seed 123, and splits 70/10/20. Classifier examples are truncated and padded to 120 GPT-2 tokens.

## 5. Data loading

Language modeling uses `GPT_DatasetV1` and `DataLoader` with batch size 2, `drop_last=True` for training and false for validation, zero workers, and context-sized stride. Classification uses `SpamDataset`, batch size 8, shuffled training data, and zero/default workers.

## 6. Training

The notebook contains a language-model training helper using next-token cross-entropy, AdamW, evaluation intervals, and sample generation. The later supervised helper trains the last-token classification logits, evaluates every 50 steps over up to 5 batches, and runs for 5 epochs. There is no gradient accumulation, scheduler, warmup, or mixed-precision implementation.

## 7. Optimizer and scheduler

The supervised experiment uses `torch.optim.AdamW(lr=5e-5, weight_decay=0.1)`. No scheduler appears in the notebook. The language-model optimizer cell uses AdamW but its final hyperparameters are not recorded as a stable experiment configuration.

## 8. Loss and evaluation

Language modeling flattens `[batch, tokens, vocab]` logits and targets for cross-entropy. Classification takes the final sequence position and applies cross-entropy to two logits. Accuracy is computed over selected loader batches. Perplexity is calculated in the earlier language-model section. No authoritative result table is present.

## 9. Checkpointing

The notebook saves `model.pth` and `model_and_optimizer.pth` in the current working directory, but does not define a durable experiment naming scheme. The extracted training CLI writes `checkpoints/spam_classifier.pt` with model config and history. GPT-2 source checkpoint files are downloaded by `gpt_download3.py` and require TensorFlow to convert.

## 10. Inference and generation

There is greedy generation (`generate_text_simple`) and temperature/top-k generation (`generate`), both truncating the prompt to the supported context length. The notebook tests generic text prompts and a spam prompt. The classifier has `classify_review`, which returns spam/not spam from the final position.

## 11. Experiments already present

The notebook demonstrates attention progressively, creates a randomly initialized GPT model, evaluates language-model loss/perplexity, attempts small-corpus training, loads GPT-2 124M weights, verifies selected weights, generates text, and fine-tunes a two-class SMS spam head. The notebook's displayed outputs are not a reproducible results record and are not copied into claims in this repository.

## 12. Dependencies

Runtime dependencies are PyTorch, tiktoken, pandas, NumPy, matplotlib, requests, and tqdm. TensorFlow below 2.17 is only used by the GPT-2 TensorFlow checkpoint conversion path. The notebook also contains a Colab pip command for protobuf/TensorFlow compatibility.

## 13. Colab-specific behavior and external state

The notebook uses `!pip install`, notebook execution order, relative working-directory files, and implicit Colab runtime state. It references GitHub raw text, the UCI dataset URL, OpenAI's GPT-2 blob storage through `gpt_download3.py`, and a suggested NVIDIA RTX 3050 6GB. No secrets or environment variables were found.

## 14. Duplicated, obsolete, or risky code

There are repeated definitions of `LayerNorm`, `FeedForward`, `MultiHeadAttention`, `GPTModel`, loss helpers, and training helpers. Earlier cells include a malformed `FeedForward.forward` indentation, inconsistent `norm1`/`ln1` and `attn`/`att` names, a likely typo in `allowed_special`, and repeated reinitialization that changes configuration. Later master cells supersede those variants. Dataset downloading disables SSL verification. Several cells assume `train.csv`, `validation.csv`, `test.csv`, `params`, `assign`, or previously executed definitions already exist. The extracted implementation removes only these verified duplicates from the runnable path; the original cells remain available for historical comparison.

## 15. Migration decisions

The modular code uses the final coherent GPT-2-compatible architecture and supervised experiment values. Paths are configurable through CLI arguments, data downloads are explicit, seeds and device selection are retained, and no full training run is started automatically. Because this environment has no PyTorch/pandas/tiktoken installation, runtime validation is limited to static compilation until dependencies are installed.
