from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.0
    qkv_bias: bool = True

    def as_dict(self):
        return asdict(self)


GPT2_124M = ModelConfig()
SEED = 123
DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
