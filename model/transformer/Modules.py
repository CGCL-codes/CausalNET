import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(x, mask, gnn_coef, dim=-1):
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x-maxes)
    x_exp_mask = x_exp * (1-mask)  # mask
    x_exp_decay = x_exp_mask * gnn_coef  # decay
    x_exp_decay_sum = torch.sum(x_exp_decay, dim, keepdim=True)
    probs = x_exp_decay / (x_exp_decay_sum + 1e-9)
    return probs 


class ScaledDotProductAttention(nn.Module):
    """ 
    Scaled Dot-Product Attention 
    """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
    
    def forward(self, q, k, v, mask_pad, mask_sub=None, mask_causal=None, decay_coef=None, event_num=None):
        """ Attention Score Func V1: for complex real-world datasets """
        mask = 1 - (1 - mask_pad) * (1 - mask_sub) * (1 - mask_causal)  # 1=mask
        
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(F.softmax(attn, dim=-1))
        
        # Mask (trainable)
        if  mask_causal is not None:
            attn  = attn * (1 - mask)

        # Decay (trainable)
        if decay_coef is not None:
            attn = attn * decay_coef
        
        output = torch.matmul(attn, v)

        return output, attn
    
    # def forward(self, q, k, v, mask_pad, mask_sub=None, mask_causal=None, decay_coef=None, event_num=None):
    #     """ Attention Score Func V2: for simple synthetic datasets """
    #     attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
    #     mask = 1 - (1 - mask_pad) * (1 - mask_sub) * (1 - mask_causal)  # 1=mask

    #     attn = self.dropout(masked_softmax(attn, mask, decay_coef, dim=-1))
        
    #     # # Decay (trainable)
    #     # if gnn_coef is not None:
    #     #     attn = attn * gnn_coef
        
    #     output = torch.matmul(attn, v)

    #     return output, attn