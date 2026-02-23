import torch

def mesa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs):
    b, h, l, d = q.shape
    s = k.size(2)
    alpha = 10
    if l == s:
    # Per-step outer products — shape (B, H, S, D, D)
        k_outer  = torch.einsum('bhsd,bhse->bhsde', k, k)  # k_i^T k_i at each step
        kv_outer = torch.einsum('bhsd,bhse->bhsde', k, v)  # k_i^T v_i at each step

        H = torch.eye(d, device=k.device, dtype=k.dtype) * alpha
        H = H + torch.cumsum(k_outer, dim=2)   # (B, H, S, D, D)
        G = torch.cumsum(kv_outer, dim=2)      # (B, H, S, D, D)

        tmp = torch.linalg.solve(H, q.unsqueeze(-1).to(H.dtype))
        out = tmp.transpose(-2, -1) @ G
        return out.squeeze(-2)

    k1, k2 = torch.tensor_split(k, [s-l], 2)
    v1, v2 = torch.tensor_split(v, [s-l], 2)
    k1_T = k1.transpose(-2,-1)
    k_outer  = torch.einsum('bhsd,bhse->bhsde', k2, k2)
    kv_outer = torch.einsum('bhsd,bhse->bhsde', k2, v2)

    H = torch.eye(d, device=k.device, dtype=k.dtype) * alpha
    H = H + torch.unsqueeze(k1_T @ k1, 2) # (B, H, S, D, D)
    H = H + torch.cumsum(k_outer, dim=2)   # (B, H, S, D, D)

    G = torch.unsqueeze(k1_T @ v1, 2)
    G = G + torch.cumsum(kv_outer, dim=2)      # (B, H, S, D, D)

    tmp = torch.linalg.solve(H, q.unsqueeze(-1).to(H.dtype))
    out = tmp.transpose(-2, -1) @ G
    return out.squeeze(-2)