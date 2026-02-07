import math

import torch
import torch.nn.functional as F

def silly_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 2.
    # scale_factor = query.size(-1) ** -0.5
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    q = query
    k = key.transpose(-2, -1) * scale_factor

    mask = attn_bias < 0

    num1 = (q @ k) / 2.
    num2 = (q.abs() @ k.abs()) / 2.
    num3 = (q.square() @ k.square()) / 4.
    num4 = ((q*q.abs()) @ (k*k.abs())) / 4.
    num5 = ((q*q.square()) @ (k*k.square())) / 12.
    num6 = ((q.abs().pow(3)) @ (k.abs().pow(3))) / 12.
    attn_weight_num = num1+num2+num3+num4+num5+num6+1
    attn_weight_num = attn_weight_num.masked_fill(mask, 0.)

    # attn_weight_num = q.relu() @ k.relu()
    # # attn_weight_num = F.softplus(attn_weight_num)
    # attn_weight_num = attn_weight_num.masked_fill(mask, 0.)
    
    attn_weight_den = torch.sum(attn_weight_num, dim=-1, keepdim=True).clamp(1e-6)
    attn_weight = attn_weight_num / attn_weight_den
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value