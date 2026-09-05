import torch

from custom_llm.config import ModelConfig
from custom_llm.evaluation import count_parameters
from custom_llm.model import GPTModel


def test_tiny_forward_pass():
    cfg = ModelConfig(vocab_size=100, context_length=16, emb_dim=32, n_heads=4, n_layers=2).as_dict()
    model = GPTModel(cfg)
    output = model(torch.randint(0, cfg["vocab_size"], (2, 8)))
    assert output.shape == (2, 8, cfg["vocab_size"])
    assert count_parameters(model) > 0
