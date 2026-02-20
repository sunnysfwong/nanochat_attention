import math

import torch
import torch.nn.functional as F

def silly_attention(
        query, key, value, attn_mask=None, 
        # dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    # scale_factor = 1.
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
    k = key.transpose(-2, -1) 

    mask = attn_bias < 0

    q_abs = q.abs()
    k_abs = k.abs()
    q_sq = q.square()
    k_sq = k.square()

    ord1 = (q @ k + q_abs @ k_abs) / 2.
    ord2 = (q_sq @ k_sq + (q_abs*q)@(k_abs*k)) / 4.
    ord3 = ((q_sq*q)@(k_sq*k) + (q_abs*q_sq)@(k_abs*k_sq))/ 12.
    attn_weight_num = 1+ord1+ord2+ord3
    attn_weight_num = attn_weight_num.masked_fill(mask, 0.)

    tmp = torch.cumsum(torch.logical_not(mask).float(), dim=-1)
    exponent = tmp - tmp[...,-1:]
    factor = torch.tensor(1.005, device=exponent.device).pow(exponent)
    attn_weight_num = attn_weight_num * factor

    # attn_weight_num = q.relu() @ k.relu()
    # # attn_weight_num = F.softplus(attn_weight_num)
    # attn_weight_num = attn_weight_num.masked_fill(mask, 0.)
    
    attn_weight_den = torch.sum(attn_weight_num, dim=-1, keepdim=True).clamp(1e-6)
    attn_weight = attn_weight_num / attn_weight_den
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value