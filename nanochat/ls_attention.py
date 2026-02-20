import torch

def ls_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs):
    B,H,S,D = q.shape
    state_0 = 0.01*torch.eye(D).to(k.device)
    state_1 = torch.zeros(B,H,D,D).to(k.device)
    out = []
    for i in range(S):
        q_i = q[:,:,i:i+1,:]
        k_i = k[:,:,i:i+1,:]
        v_i = v[:,:,i:i+1,:]
        k_i_T = k_i.transpose(-2,-1)
        tmp = k_i @ state_0
        state_0 = state_0 - tmp.transpose(-2,-1) @ tmp / (1+tmp @ k_i_T)
        state_1 = state_1 + k_i_T @ v_i
        z = q_i @ state_0 @ state_1
        out.append(z)
    return torch.cat(out, dim=2)