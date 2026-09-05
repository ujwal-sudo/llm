import numpy as np
import torch


def assign(left, right):
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"Shape mismatch: {left.shape}, {right.shape}")
    value = torch.from_numpy(right).to(device=left.device, dtype=left.dtype) if isinstance(right, np.ndarray) else right.to(left.device)
    left.data.copy_(value)
    return left


def load_model_weights_into_gpt(model, params):
    model.tok_emb.weight = assign(model.tok_emb.weight, params["wte"])
    model.pos_emb.weight = assign(model.pos_emb.weight, params["wpe"])
    for block, source in zip(model.trf_blocks, params["blocks"]):
        q_w, k_w, v_w = np.split(source["attn"]["c_attn"]["w"], 3, axis=-1)
        for layer, weights in ((block.att.W_query, q_w.T), (block.att.W_key, k_w.T), (block.att.W_value, v_w.T)):
            layer.weight = assign(layer.weight, weights)
        q_b, k_b, v_b = np.split(source["attn"]["c_attn"]["b"], 3, axis=-1)
        for layer, bias in ((block.att.W_query, q_b), (block.att.W_key, k_b), (block.att.W_value, v_b)):
            layer.bias = assign(layer.bias, bias)
        block.att.out_proj.weight = assign(block.att.out_proj.weight, source["attn"]["c_proj"]["w"].T)
        block.att.out_proj.bias = assign(block.att.out_proj.bias, source["attn"]["c_proj"]["b"])
        block.ffn.layers[0].weight = assign(block.ffn.layers[0].weight, source["mlp"]["c_fc"]["w"].T)
        block.ffn.layers[0].bias = assign(block.ffn.layers[0].bias, source["mlp"]["c_fc"]["b"])
        block.ffn.layers[2].weight = assign(block.ffn.layers[2].weight, source["mlp"]["c_proj"]["w"].T)
        block.ffn.layers[2].bias = assign(block.ffn.layers[2].bias, source["mlp"]["c_proj"]["b"])
        for norm, source_norm in ((block.ln1, source["ln_1"]), (block.ln2, source["ln_2"])):
            norm.scale = assign(norm.scale, source_norm["g"])
            norm.shift = assign(norm.shift, source_norm["b"])
    model.final_norm.scale = assign(model.final_norm.scale, params["g"])
    model.final_norm.shift = assign(model.final_norm.shift, params["b"])
    model.out_head.weight = assign(model.out_head.weight, params["wte"])
    return model
