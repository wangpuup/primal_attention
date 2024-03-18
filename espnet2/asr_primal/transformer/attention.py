#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import math
import random

import torch
from torch import nn

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, n_attn, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        assert n_attn % n_head == 0
        self.d_k = n_attn // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_attn)
        self.linear_k = nn.Linear(n_feat, n_attn)
        self.linear_v = nn.Linear(n_feat, n_attn)
        self.linear_out = nn.Linear(n_attn, n_feat)
        
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

class PrimalAttention(nn.Module):

    def __init__(self, n_head, n_feat, low_rank, n_attn, dropout_rate):

        """Construct an MultiHeadedAttention object."""
        #n_feat dim after pre-encoder (e.g. conv2d)
        #n_attn total dim of each attention layer 
        super(PrimalAttention, self).__init__()
        assert n_attn % n_head == 0
        self.d_k = n_attn // n_head
        self.h = n_head

        # primal parameters
        self.We = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.h, self.d_k, low_rank)))
        self.Wr = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.h, self.d_k, low_rank)))
        self.Lambda = nn.Parameter(nn.init.uniform_(torch.Tensor(self.h, low_rank)))
        self.concate_weight = nn.Linear(2 * low_rank, self.d_k)
        
        self.linear_q = nn.Linear(n_feat, n_attn)
        self.linear_k = nn.Linear(n_feat, n_attn)
        self.linear_v = nn.Linear(n_feat, n_attn)
        self.linear_out = nn.Linear(n_attn, n_feat)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def feature_map(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1)
    
    def forward(self, query, key, value, p_layer, mask):

        n_batch, seq_len, _ = query.size()
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k) #B L H D
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)

        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)

        if p_layer == True:
            # generate weights for Primal attention
            normal = (((torch.arange(seq_len)).float() + 1.0)).to(q.device)
            
            # we conduct cumsum before the non-linear map (Causal map)
            q = q.cumsum(dim=2) / normal[None, None, :, None]
            k = k.cumsum(dim=2) / normal[None, None, :, None]
            # feature_map
            q = self.feature_map(q)
            k = self.feature_map(k)
            
            escore = torch.einsum('...nd,...de->...ne', q, self.We.unsqueeze(0).repeat(q.size(0),1,1,1)) #B H L s
            rscore = torch.einsum('...nd,...de->...ne', k, self.Wr.unsqueeze(0).repeat(k.size(0),1,1,1))
            scores = torch.cat((escore, rscore), dim=-1) #B H L 2s
            
            self.attn = self.concate_weight(scores).transpose(1,2).contiguous().reshape(n_batch, seq_len, self.h * self.d_k)
            x = self.attn
        else:
            v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
            v = v.transpose(1, 2)  # (batch, head, time2, d_k)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
                min_value = torch.finfo(scores.dtype).min
                scores = scores.masked_fill(mask, min_value)
                self.attn = torch.softmax(scores, dim=-1).masked_fill(
                    mask, 0.0
                    )  # (batch, head, time1, time2)
            else:
                self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
                
            p_attn = self.dropout(self.attn)
            x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
            x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
            )  # (batch, time1, d_model)
            escore = 0
            rscore = 0

        return self.linear_out(x), [escore, rscore, self.We, self.Wr, self.Lambda]


