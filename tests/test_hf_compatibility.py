import os

import pytest
import torch

from custom_llm.convert_hf_gpt2 import load_hf_gpt2_into_custom_model
from custom_llm.model import GPTModel


def test_hf_gpt2_logits_match_custom_model():
    transformers = pytest.importorskip("transformers")
    checkpoint_dir = os.environ.get("HF_GPT2_LOCAL_DIR")
    if not checkpoint_dir:
        pytest.skip("Set HF_GPT2_LOCAL_DIR to a local Hugging Face GPT-2 124M checkpoint")

    reference = transformers.GPT2LMHeadModel.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
    ).eval()
    config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    custom = GPTModel(config).eval()
    load_hf_gpt2_into_custom_model(custom, reference.state_dict())

    input_ids = torch.tensor([[15496, 995, 428, 318, 257, 1332]], dtype=torch.long)
    with torch.no_grad():
        reference_logits = reference(input_ids).logits
        custom_logits = custom(input_ids)

    difference = (reference_logits - custom_logits).abs()
    max_difference = difference.max().item()
    mean_difference = difference.mean().item()
    tolerance = {"rtol": 1e-4, "atol": 1e-5}
    print(f"max absolute logit difference: {max_difference:.8e}")
    print(f"mean absolute logit difference: {mean_difference:.8e}")
    print(f"tolerance: rtol={tolerance['rtol']}, atol={tolerance['atol']}")
    torch.testing.assert_close(custom_logits, reference_logits, **tolerance)
