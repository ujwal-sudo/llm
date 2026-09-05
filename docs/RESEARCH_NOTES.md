# Research Notes

The notebook starts with trainable self-attention, causal attention, and two multi-head implementations, then builds a pre-layer-normalized GPT-style transformer. The canonical model has token and positional embeddings, 12 transformer blocks, custom LayerNorm, GELU feed-forward layers, causal attention, and a tied output embedding when GPT-2 weights are loaded.

The language-modeling experiment uses the small `the-verdict.txt` sample, GPT-2 BPE tokenization, 80/20 character split, context/stride 1024 in its later setup, cross-entropy, and AdamW training helpers. The notebook does not provide a complete, trustworthy training-results record.

The supervised experiment downloads the UCI SMS Spam Collection, balances classes, splits 70/10/20 with seed 123, pads to 120 tokens, replaces the output head with two logits, and freezes the transformer while allowing the head and final normalization to train. It uses AdamW at `5e-5`, weight decay `0.1`, for 5 epochs.

Open questions include whether the intended research target is the language-modeling path or the SMS classifier, whether pretrained GPT-2 loading should be made a hard prerequisite, and which of the notebook's conflicting 256/1024 context configurations is intended for future experiments.
