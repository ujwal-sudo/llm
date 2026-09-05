"""Convert a standard Hugging Face GPT-2 state dict for the custom GPTModel.

This module deliberately does not modify the custom model or the existing
TensorFlow loader. It only translates names and Conv1D/Linear tensor layouts.
"""

from collections.abc import Mapping

import torch


VOCAB_SIZE = 50257
CONTEXT_LENGTH = 1024
EMB_DIM = 768
NUM_LAYERS = 12
QKV_DIM = 3 * EMB_DIM


def _required_hf_keys():
    keys = {
        "transformer.wte.weight",
        "transformer.wpe.weight",
        "transformer.ln_f.weight",
        "transformer.ln_f.bias",
    }
    for index in range(NUM_LAYERS):
        prefix = f"transformer.h.{index}"
        keys.update(
            {
                f"{prefix}.attn.c_attn.weight",
                f"{prefix}.attn.c_attn.bias",
                f"{prefix}.attn.c_proj.weight",
                f"{prefix}.attn.c_proj.bias",
                f"{prefix}.ln_1.weight",
                f"{prefix}.ln_1.bias",
                f"{prefix}.mlp.c_fc.weight",
                f"{prefix}.mlp.c_fc.bias",
                f"{prefix}.mlp.c_proj.weight",
                f"{prefix}.mlp.c_proj.bias",
                f"{prefix}.ln_2.weight",
                f"{prefix}.ln_2.bias",
            }
        )
    return keys


def _allowed_non_parameter_keys():
    return {
        f"transformer.h.{index}.attn.bias"
        for index in range(NUM_LAYERS)
    } | {
        f"transformer.h.{index}.attn.masked_bias"
        for index in range(NUM_LAYERS)
    }


def _require_shape(state_dict, key, shape):
    if key not in state_dict:
        raise KeyError(f"Missing Hugging Face parameter: {key}")
    actual = tuple(state_dict[key].shape)
    if actual != tuple(shape):
        raise ValueError(f"Shape mismatch for {key}: expected {shape}, got {actual}")


def validate_hf_gpt2_state_dict(state_dict):
    """Validate that ``state_dict`` is a complete GPT-2 small state dict."""
    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping of parameter names to tensors")

    required = _required_hf_keys()
    allowed = required | _allowed_non_parameter_keys() | {"lm_head.weight"}
    missing = sorted(required - set(state_dict))
    unexpected = sorted(set(state_dict) - allowed)
    if missing:
        raise KeyError(f"Missing Hugging Face parameters: {missing}")
    if unexpected:
        raise KeyError(f"Unexpected Hugging Face parameters: {unexpected}")

    _require_shape(state_dict, "transformer.wte.weight", (VOCAB_SIZE, EMB_DIM))
    _require_shape(state_dict, "transformer.wpe.weight", (CONTEXT_LENGTH, EMB_DIM))
    _require_shape(state_dict, "transformer.ln_f.weight", (EMB_DIM,))
    _require_shape(state_dict, "transformer.ln_f.bias", (EMB_DIM,))

    for index in range(NUM_LAYERS):
        prefix = f"transformer.h.{index}"
        _require_shape(state_dict, f"{prefix}.attn.c_attn.weight", (EMB_DIM, QKV_DIM))
        _require_shape(state_dict, f"{prefix}.attn.c_attn.bias", (QKV_DIM,))
        _require_shape(state_dict, f"{prefix}.attn.c_proj.weight", (EMB_DIM, EMB_DIM))
        _require_shape(state_dict, f"{prefix}.attn.c_proj.bias", (EMB_DIM,))
        _require_shape(state_dict, f"{prefix}.ln_1.weight", (EMB_DIM,))
        _require_shape(state_dict, f"{prefix}.ln_1.bias", (EMB_DIM,))
        _require_shape(state_dict, f"{prefix}.mlp.c_fc.weight", (EMB_DIM, 4 * EMB_DIM))
        _require_shape(state_dict, f"{prefix}.mlp.c_fc.bias", (4 * EMB_DIM,))
        _require_shape(state_dict, f"{prefix}.mlp.c_proj.weight", (4 * EMB_DIM, EMB_DIM))
        _require_shape(state_dict, f"{prefix}.mlp.c_proj.bias", (EMB_DIM,))
        _require_shape(state_dict, f"{prefix}.ln_2.weight", (EMB_DIM,))
        _require_shape(state_dict, f"{prefix}.ln_2.bias", (EMB_DIM,))

    if "lm_head.weight" in state_dict:
        _require_shape(state_dict, "lm_head.weight", (VOCAB_SIZE, EMB_DIM))
        if not torch.equal(state_dict["lm_head.weight"], state_dict["transformer.wte.weight"]):
            raise ValueError("GPT-2 lm_head.weight is not tied to transformer.wte.weight")


def _copy(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("All checkpoint values must be torch tensors")
    return tensor.detach().clone()


def convert_hf_gpt2_state_dict(state_dict, target_model=None):
    """Convert a Hugging Face GPT-2 124M state dict to custom-model names.

    If ``target_model`` is supplied, its causal-mask buffers are included so
    the returned dictionary can be loaded with ``strict=True``.
    """
    validate_hf_gpt2_state_dict(state_dict)
    converted = {
        "tok_emb.weight": _copy(state_dict["transformer.wte.weight"]),
        "pos_emb.weight": _copy(state_dict["transformer.wpe.weight"]),
        "final_norm.scale": _copy(state_dict["transformer.ln_f.weight"]),
        "final_norm.shift": _copy(state_dict["transformer.ln_f.bias"]),
        "out_head.weight": _copy(state_dict["transformer.wte.weight"]),
    }

    for index in range(NUM_LAYERS):
        source = f"transformer.h.{index}"
        target = f"trf_blocks.{index}"
        q_weight, k_weight, v_weight = state_dict[f"{source}.attn.c_attn.weight"].split(EMB_DIM, dim=-1)
        q_bias, k_bias, v_bias = state_dict[f"{source}.attn.c_attn.bias"].split(EMB_DIM, dim=-1)
        converted.update(
            {
                f"{target}.att.W_query.weight": _copy(q_weight.transpose(0, 1)),
                f"{target}.att.W_query.bias": _copy(q_bias),
                f"{target}.att.W_key.weight": _copy(k_weight.transpose(0, 1)),
                f"{target}.att.W_key.bias": _copy(k_bias),
                f"{target}.att.W_value.weight": _copy(v_weight.transpose(0, 1)),
                f"{target}.att.W_value.bias": _copy(v_bias),
                f"{target}.att.out_proj.weight": _copy(state_dict[f"{source}.attn.c_proj.weight"].transpose(0, 1)),
                f"{target}.att.out_proj.bias": _copy(state_dict[f"{source}.attn.c_proj.bias"]),
                f"{target}.ln1.scale": _copy(state_dict[f"{source}.ln_1.weight"]),
                f"{target}.ln1.shift": _copy(state_dict[f"{source}.ln_1.bias"]),
                f"{target}.ffn.layers.0.weight": _copy(state_dict[f"{source}.mlp.c_fc.weight"].transpose(0, 1)),
                f"{target}.ffn.layers.0.bias": _copy(state_dict[f"{source}.mlp.c_fc.bias"]),
                f"{target}.ffn.layers.2.weight": _copy(state_dict[f"{source}.mlp.c_proj.weight"].transpose(0, 1)),
                f"{target}.ffn.layers.2.bias": _copy(state_dict[f"{source}.mlp.c_proj.bias"]),
                f"{target}.ln2.scale": _copy(state_dict[f"{source}.ln_2.weight"]),
                f"{target}.ln2.shift": _copy(state_dict[f"{source}.ln_2.bias"]),
            }
        )

    if target_model is not None:
        for name, tensor in target_model.state_dict().items():
            if name.endswith(".att.mask"):
                converted[name] = _copy(tensor)

    return converted


def load_hf_gpt2_into_custom_model(model, state_dict):
    """Convert and strictly load a Hugging Face GPT-2 state dict."""
    converted = convert_hf_gpt2_state_dict(state_dict, target_model=model)
    expected = set(model.state_dict())
    actual = set(converted)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise KeyError(f"Converted state dict mismatch: missing={missing}, unexpected={unexpected}")
    model.load_state_dict(converted, strict=True)
    return model
