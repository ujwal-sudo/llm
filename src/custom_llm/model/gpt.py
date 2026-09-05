import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device))
            * (x + 0.044715 * x.pow(3))))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        if d_out % num_heads:
            raise ValueError("d_out must be divisible by num_heads")
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        batch, tokens, _ = x.shape
        keys = self.W_key(x).view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        scores = queries @ keys.transpose(2, 3)
        scores = scores.masked_fill(self.mask.bool()[:tokens, :tokens], -torch.inf)
        weights = torch.softmax(scores / self.head_dim**0.5, dim=-1)
        context = (self.dropout(weights) @ values).transpose(1, 2).contiguous()
        return self.out_proj(context.view(batch, tokens, self.d_out))


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"], cfg["drop_rate"], cfg["n_heads"], cfg["qkv_bias"])
        self.ffn = FeedForward(cfg)
        self.ln1 = LayerNorm(cfg["emb_dim"])
        self.ln2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.drop_shortcut(self.att(self.ln1(x)))
        return x + self.drop_shortcut(self.ffn(self.ln2(x)))


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        _, seq_len = in_idx.shape
        if seq_len > self.pos_emb.num_embeddings:
            raise ValueError(f"Sequence length {seq_len} exceeds context length {self.pos_emb.num_embeddings}")
        positions = torch.arange(seq_len, device=in_idx.device)
        x = self.drop_emb(self.tok_emb(in_idx) + self.pos_emb(positions))
        return self.out_head(self.final_norm(self.trf_blocks(x)))
