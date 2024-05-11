import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer
from copy import deepcopy

def get_non_pad_mask(seq):
    """ 
    Get the non-padding positions. 
    """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ 
    For masking out the padding part of key sequence. 
    :Return: bool triu matrix, True=mask
    """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ 
    For masking out the subsequent info, i.e., masked self-attention. 
    :Return: 01 triu matrix with diag=0, 1=mask
    """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)  # triu matrix with diag=0
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls (copy b times)
    return subsequent_mask


def get_attn_causal_mask(event_num, event_type, causal_graph):
    """ 
    Only attention to the (causal) parents of the predicted event is allowed.
    """
    n_node = causal_graph.shape[1]
    sz_b, len_s = event_type.size()
    len_q, len_k = len_s, len_s
    device = event_type.device.type
    
    # records which event types are the (causal) parents of the predicted event in each batch
    event_type = event_type - 1  # -1 because 0 means PAD
    target_types = []
    for batch_id in range(sz_b):         
        idx = event_num[batch_id] - 1
        cur_type = event_type[batch_id][idx].item()
        target_types.append(cur_type)
    parent_types = causal_graph[range(sz_b), :, target_types]  # batch*n_node
    
    # to be improved: a little time-consuming
    causal_mask = torch.zeros((sz_b, len_k), device=device)  # bs*len_k
    parent_type_encoding = parent_types.unsqueeze(1)  # -> (sz_b, 1, n_node)
    seq_type = deepcopy(event_type).to(torch.int64)
    seq_type_encoding = torch.eye(n_node, dtype=torch.float32, device=device)[seq_type].transpose(1, 2)  # (sz_b, n_node, len_k)
    causal_mask = torch.matmul(parent_type_encoding, seq_type_encoding).squeeze(1)  # (sz_b, len_k)
       
    causal_mask = 1 - causal_mask  # let '1' means mask
    causal_mask = causal_mask.unsqueeze(1).expand(-1, len_q, -1)  # bs*len_q*len_k
    
    # specially, allow a event to attention to itself, though it may not be a (causal) parent of the predited event
    causal_mask = causal_mask * (1-torch.eye(len_q, len_k)).to(device)
    
    return causal_mask


def get_causal_decay_coef(event_num, event_type, event_topo, theta, A, causal_graph):
    """ 
    Causal decay among topological nodes
    """
    n_node = causal_graph.shape[1]
    n_topo = A[0].shape[0]
    sz_b, len_s = event_topo.size()
    len_q, len_k = len_s, len_s
    device = event_topo.device.type
    
    # identify the 'topological node' of the each predicted event
    event_topo = event_topo - 1  # -1 because 0 means PAD
    target_topos = []
    for batch_id in range(sz_b): 
        idx = event_num[batch_id] - 1
        cur_topo = event_topo[batch_id][idx].item()
        target_topos.append(cur_topo)
        
    # identify the 'event type' of the each predicted event
    event_type = event_type - 1  # -1 because 0 means PAD
    target_types = []
    for batch_id in range(sz_b):
        idx = event_num[batch_id] - 1
        cur_type = event_type[batch_id][idx].item()
        target_types.append(cur_type)
    
    # identify the causal decay coef between the predicted event and each historical event
    parents = torch.zeros(sz_b, n_topo, n_node).to(device) 
    max_hop = len(theta) - 1
    for k in range(0, max_hop+1):        
        theta_k = theta[k].permute(2, 3, 0, 1)  # (n_topo,n_topo,n_node,n_node) -> (n_node,n_node,n_topo,n_topo)
        theta_k_adj = theta_k * A[k]
        sig_theta_k = torch.sigmoid(theta_k_adj)
        sig_theta_k = sig_theta_k * A[k].ne(0).float()  # mask topo nodes out of max_hop
        sig_theta_k = sig_theta_k.permute(2, 3, 0, 1)  # (n_node,n_node,n_topo,n_topo) -> (n_topo,n_topo,n_node,n_node)
        parents += sig_theta_k[:, target_topos, :, target_types]  # sz_b*n_topo*n_node
    causal_decay_coef = torch.zeros((sz_b, len_k), device=device)  # bs*len_k

    # record decay coefs
    causal_decay_coef = parents[torch.arange(sz_b)[:,None].expand(-1,len_k), event_topo.to(torch.int64), event_type.to(torch.int64)]  # (sz_b, len_k)
    causal_decay_coef = causal_decay_coef.unsqueeze(1).expand(-1, len_q, -1)  # (sz_b, len_k) -> (sz_b, len_q, len_k)
    
    return causal_decay_coef


class Decoder(nn.Module):
    """ 
    A decoder model with self attention mechanism. 
    """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])


    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask


    def forward(self, event_num, event_type, event_time, event_topo, non_pad_mask, causal_graph, theta, A):
        """ 
        Encode event sequences via Topology-informed Causal Attention. 
        Mask: 1 = mask
        """
        # mask future
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        # mask padding
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)  # to 0,1 matrix
        # mask non-causal
        slf_attn_mask_causal = get_attn_causal_mask(event_num, event_type, causal_graph)
        # causal decay coef
        causal_decay_coef = get_causal_decay_coef(event_num, event_type, event_topo, theta, A, causal_graph)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        dec_output = self.event_emb(event_type)

        for dec_layer in self.layer_stack:
            dec_output += tem_enc
            dec_output, _ = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                mask_pad=slf_attn_mask_keypad,
                mask_sub=slf_attn_mask_subseq,
                mask_causal=slf_attn_mask_causal,
                decay_coef=causal_decay_coef,
                event_num=event_num)
        return dec_output


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout,
            num_types=None, num_topos=None):
        super().__init__()

        self.decoder = Decoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.num_types = num_types

        # (1) predict the likelihood of the next event
        #     specifically, used in the softplus func
        ## convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)
        ## parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        ## parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # (2) predict the attributes of the next event
        ## timestamp
        self.time_predictor = Predictor(d_model, 1)

        ## event type
        self.type_predictor = Predictor(d_model, num_types)
        
        ## event type
        # self.topo_predictor = Predictor(d_model, num_topos)


    def forward(self, event_num, event_type, event_time, event_topo, causal_graph, theta, A):
        """
        Generate the hidden vectors and predictions.
        """

        non_pad_mask = get_non_pad_mask(event_type)

        dec_output = self.decoder(event_num, event_type, event_time, event_topo, non_pad_mask, causal_graph, theta, A)

        time_prediction = self.time_predictor(dec_output, non_pad_mask)

        type_prediction = self.type_predictor(dec_output, non_pad_mask)

        return dec_output, (type_prediction, time_prediction)
