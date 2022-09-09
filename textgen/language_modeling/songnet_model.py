# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SongNet model

refer: ACL2020 paper: Rigid Formats Controlled Text Generation, Piji Li
url: https://www.aclweb.org/anthology/2020.acl-main.68
"""
import os
import sys
import math
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm, trange
import torch
from torch import nn
import torch.nn.functional as F
from transformers.optimization import AdamW, Adafactor
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from textgen.config.model_args import SongNetArgs
from textgen.language_modeling.songnet_utils import (
    ZHCharTokenizer, s2t, s2xy, s2xy_polish,
    SongNetDataLoader,
    BOS, EOS,
)

has_cuda = torch.cuda.is_available()
os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, 1.)
        nn.init.constant_(self.bias, 0.)

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__
    INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, '_guyu_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._guyu_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._guyu_instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def gelu(x):
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return cdf * x


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, device, size, padding_idx, label_smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = padding_idx
        self.size = size
        self.device = device

        self.smoothing_value = label_smoothing / (size - 2)
        self.one_hot = torch.full((1, size), self.smoothing_value).to(device)
        self.one_hot[0, self.padding_idx] = 0

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        real_size = output.size(1)
        if real_size > self.size:
            real_size -= self.size
        else:
            real_size = 0

        model_prob = self.one_hot.repeat(target.size(0), 1)
        if real_size > 0:
            ext_zeros = torch.full((model_prob.size(0), real_size), self.smoothing_value).to(self.device)
            model_prob = torch.cat((model_prob, ext_zeros), -1)
        model_prob.scatter_(1, target, self.confidence)
        model_prob.masked_fill_((target == self.padding_idx), 0.)

        return F.kl_div(output, model_prob, reduction='sum')


class TransformerLayer(nn.Module):

    def __init__(self, embed_dim, ff_embed_dim, num_heads, dropout, with_external=False, weights_dropout=True):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout, weights_dropout)
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim)
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim)
        self.attn_layer_norm = LayerNorm(embed_dim)
        self.ff_layer_norm = LayerNorm(embed_dim)
        self.with_external = with_external
        self.dropout = dropout
        if self.with_external:
            self.external_attn = MultiheadAttention(embed_dim, num_heads, dropout, weights_dropout)
            self.external_layer_norm = LayerNorm(embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, kv=None,
                self_padding_mask=None, self_attn_mask=None,
                external_memories=None, external_padding_mask=None,
                need_weights=False):
        # x: seq_len x bsz x embed_dim
        residual = x
        if kv is None:
            x, self_attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_padding_mask,
                                          attn_mask=self_attn_mask, need_weights=need_weights)
        else:
            x, self_attn = self.self_attn(query=x, key=kv, value=kv, key_padding_mask=self_padding_mask,
                                          attn_mask=self_attn_mask, need_weights=need_weights)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.attn_layer_norm(residual + x)

        if self.with_external:
            residual = x
            x, external_attn = self.external_attn(query=x, key=external_memories, value=external_memories,
                                                  key_padding_mask=external_padding_mask, need_weights=need_weights)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.external_layer_norm(residual + x)
        else:
            external_attn = None

        residual = x
        x = gelu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ff_layer_norm(residual + x)

        return x, self_attn, external_attn

    def work_incremental(self, x, self_padding_mask=None, self_attn_mask=None,
                         external_memories=None, external_padding_mask=None, incremental_state=None):
        # x: seq_len x bsz x embed_dim
        residual = x
        x, self_attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_padding_mask,
                                      attn_mask=self_attn_mask, incremental_state=incremental_state)
        x = self.attn_layer_norm(residual + x)

        if self.with_external:
            residual = x
            x, external_attn = self.external_attn(query=x, key=external_memories, value=external_memories,
                                                  key_padding_mask=external_padding_mask)
            x = self.external_layer_norm(residual + x)
        else:
            external_attn = None
        residual = x
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = self.ff_layer_norm(residual + x)

        return x, self_attn, external_attn


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., weights_dropout=True):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.weights_dropout = weights_dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=False,
                incremental_state=None):
        """ Input shape: Time x Batch x Channel
            key_padding_mask: Time x batch
            attn_mask:  tgt_len x src_len
        """
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            bidx = self._get_bidx(incremental_state)
        else:
            saved_state = None
            bidx = None

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key']
                if bidx is not None:
                    prev_key = prev_key[bidx]
                prev_key = prev_key.contiguous().view(bsz * self.num_heads, -1, self.head_dim)
                k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value']
                if bidx is not None:
                    prev_value = prev_value[bidx]
                prev_value = prev_value.contiguous().view(bsz * self.num_heads, -1, self.head_dim)
                v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)
            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)
        # k,v: bsz*heads x src_len x dim
        # q: bsz*heads x tgt_len x dim

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_weights.masked_fill_(
                attn_mask.unsqueeze(0),
                float('-inf')
            )

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.transpose(0, 1).unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.weights_dropout:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        if not self.weights_dropout:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        if need_weights:
            # maximum attention weight over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

            attn_weights, _ = attn_weights.max(dim=1)
            attn_weights = attn_weights.transpose(0, 1)
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def _get_input_buffer(self, incremental_state):
        return get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer, )

    def _get_bidx(self, incremental_state):
        if "bidx" in incremental_state:
            return incremental_state["bidx"]
        else:
            return None


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, std=0.02)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


class SelfAttentionMask(nn.Module):
    def __init__(self, init_size=100, device=0):
        super(SelfAttentionMask, self).__init__()
        self.weights = SelfAttentionMask.get_mask(init_size)
        self.device = device

    @staticmethod
    def get_mask(size):
        weights = torch.triu(torch.ones((size, size), dtype=torch.bool), 1)
        return weights

    def forward(self, size):
        if self.weights is None or size > self.weights.size(0):
            self.weights = SelfAttentionMask.get_mask(size)
        res = self.weights[:size, :size].cuda(self.device).detach()
        return res


class LearnedPositionalEmbedding(nn.Module):
    """This module produces LearnedPositionalEmbedding.
    """

    def __init__(self, embedding_dim, init_size=1024, device=0):
        super(LearnedPositionalEmbedding, self).__init__()
        self.weights = nn.Embedding(init_size, embedding_dim)
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weights.weight, std=0.02)

    def forward(self, input, offset=0):
        """Input is expected to be of size [seq_len x bsz]."""
        seq_len, bsz = input.size()
        positions = (offset + torch.arange(seq_len)).cuda(self.device)
        res = self.weights(positions).unsqueeze(1).expand(-1, bsz, -1)
        return res


class SongNet(nn.Module):
    """SongNet model network"""

    def __init__(self, tokenizer, device=0, embed_dim=768, ff_embed_dim=768 * 4, num_heads=12, dropout=0.2,
                 num_layers=12, smoothing_factor=0.1):
        """
        Init model network
        """
        super(SongNet, self).__init__()
        self.tokenizer = tokenizer
        self.embed_dim = embed_dim

        self.tok_embed = Embedding(self.tokenizer.size, embed_dim, self.tokenizer.padding_idx)
        self.pos_embed = LearnedPositionalEmbedding(embed_dim, device=device)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout, with_external=True))
        self.emb_layer_norm = LayerNorm(embed_dim)
        self.one_more = nn.Linear(embed_dim, embed_dim)
        self.one_more_layer_norm = LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, self.tokenizer.size)

        self.attn_mask = SelfAttentionMask(device=device)
        self.smoothing = LabelSmoothing(device, self.tokenizer.size, self.tokenizer.padding_idx, smoothing_factor)

        self.dropout = dropout
        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.one_more.bias, 0.)
        nn.init.normal_(self.one_more.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def label_smotthing_loss(self, y_pred, y, y_mask, avg=True):
        seq_len, bsz = y.size()
        y_pred = torch.log(y_pred.clamp(min=1e-8))
        loss = self.smoothing(y_pred.view(seq_len * bsz, -1), y.view(seq_len * bsz, -1))
        if avg:
            return loss / torch.sum(y_mask)
        else:
            return loss / bsz

    def nll_loss(self, y_pred, y, y_mask, avg=True):
        cost = -torch.log(torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1)))
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = torch.sum(cost * y_mask, 0) / torch.sum(y_mask, 0)
        else:
            cost = torch.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        ppl = 2 ** cost
        return cost.sum().item(), ppl.sum().item()

    def work_incremental(self, enc, src_padding_mask, ys_inp, ys_tpl, ys_seg, ys_pos, incremental_state=None):
        seq_len, bsz = ys_inp.size()
        x = self.tok_embed(ys_inp) + self.pos_embed(ys_inp) + self.tok_embed(ys_tpl) + self.tok_embed(
            ys_seg) + self.tok_embed(ys_pos)
        x = self.emb_layer_norm(x)
        padding_mask = torch.eq(ys_inp, self.tokenizer.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        if incremental_state is None:
            self_attn_mask = self.attn_mask(seq_len)
            incremental_state = {}
        else:
            x = x[-1, :, :].unsqueeze(0)
            self_attn_mask = None

        for layer in self.layers:
            x, _, _ = layer.work_incremental(
                x,
                self_padding_mask=padding_mask,
                self_attn_mask=self_attn_mask,
                external_memories=enc,
                external_padding_mask=src_padding_mask,
                incremental_state=incremental_state
            )
        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        probs = torch.softmax(self.out_proj(x), -1)
        _, pred_y = probs.max(-1)
        return probs, pred_y, incremental_state

    def work(self, enc, src_padding_mask, ys_inp, ys_tpl, ys_seg, ys_pos):
        seq_len, bsz = ys_inp.size()
        self_attn_mask = self.attn_mask(seq_len)
        x = self.tok_embed(ys_inp) + self.pos_embed(ys_inp) + self.tok_embed(ys_tpl) + self.tok_embed(
            ys_seg) + self.tok_embed(ys_pos)
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        padding_mask = torch.eq(ys_inp, self.tokenizer.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        for layer in self.layers:
            x, _, _ = layer(
                x,
                self_padding_mask=padding_mask,
                self_attn_mask=self_attn_mask,
                external_memories=enc,
                external_padding_mask=src_padding_mask
            )
        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        probs = torch.softmax(self.out_proj(x), -1)
        _, pred_y = probs.max(-1)
        return probs, pred_y

    def encode(self, xs_tpl, xs_seg, xs_pos):
        padding_mask = torch.eq(xs_tpl, self.tokenizer.padding_idx)
        x = self.tok_embed(xs_tpl) + self.tok_embed(xs_seg) + self.tok_embed(xs_pos)
        x = self.emb_layer_norm(x)
        return x, padding_mask

    def ppl(self, xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk):
        enc, src_padding_mask = self.encode(xs_tpl, xs_seg, xs_pos)
        seq_len, bsz = ys_inp.size()
        self_attn_mask = self.attn_mask(seq_len)
        x = self.tok_embed(ys_inp) + self.pos_embed(ys_inp) + self.tok_embed(ys_tpl) + self.tok_embed(
            ys_seg) + self.tok_embed(ys_pos)
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        padding_mask = torch.eq(ys_truth, self.tokenizer.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        for layer in self.layers:
            x, _, _ = layer(
                x,
                self_padding_mask=padding_mask,
                self_attn_mask=self_attn_mask,
                external_memories=enc,
                external_padding_mask=src_padding_mask
            )
        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        pred = torch.softmax(self.out_proj(x), -1)
        nll, ppl = self.nll_loss(pred, ys_truth, msk)
        return nll, ppl, bsz

    def forward(self, xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk):
        enc, src_padding_mask = self.encode(xs_tpl, xs_seg, xs_pos)
        seq_len, bsz = ys_inp.size()
        self_attn_mask = self.attn_mask(seq_len)
        x = self.tok_embed(ys_inp) + self.pos_embed(ys_inp) + self.tok_embed(ys_tpl) + self.tok_embed(
            ys_seg) + self.tok_embed(ys_pos)
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        padding_mask = torch.eq(ys_truth, self.tokenizer.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        for layer in self.layers:
            x, _, _ = layer(
                x,
                self_padding_mask=padding_mask,
                self_attn_mask=self_attn_mask,
                external_memories=enc,
                external_padding_mask=src_padding_mask
            )
        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        pred = torch.softmax(self.out_proj(x), -1)

        loss = self.label_smotthing_loss(pred, ys_truth, msk)

        _, pred_y = pred.max(-1)
        n_tokens = msk.float().sum().item()
        acc = (torch.eq(pred_y, ys_truth).float() * msk).sum().item()

        nll, ppl = self.nll_loss(pred, ys_truth, msk)
        return (pred_y, ys_truth), loss, acc, nll, ppl, n_tokens, bsz


class SongNetModel:
    def __init__(
            self,
            model_type='songnet',
            model_name='shibing624/songnet-base-chinese-couplet',
            args=None,
            use_cuda=has_cuda,
            cuda_device=-1,
            **kwargs,
    ):
        """
        Initializes a SongNetModel model.

        Args:
            model_type: The type of model (songnet)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, SongNetArgs):
            self.args = args

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `use_cuda=False`."
                )
        else:
            self.device = "cpu"
        logger.debug(f"Device: {self.device}")

        self.results = {}

        if model_name:
            self.tokenizer = ZHCharTokenizer.from_pretrained(model_name, **kwargs)
            self.model = SongNet(
                self.tokenizer,
                device=0,
                embed_dim=self.args.embed_dim,
                ff_embed_dim=self.args.ff_embed_dim,
                num_heads=self.args.num_heads,
                dropout=self.args.dropout,
                num_layers=self.args.num_layers,
                smoothing_factor=self.args.smoothing_factor,
            )
            self.model.load_state_dict(torch.load(os.path.join(model_name, 'pytorch_model.bin')))

        self.args.model_type = model_type
        if model_name is None:
            self.args.model_name = "SongNet_from_scratch"
        else:
            self.args.model_name = model_name

    def train_model(
            self,
            train_file,
            output_dir=None,
            show_running_loss=True,
            args=None,
            eval_file=None,
            verbose=True,
            **kwargs,
    ):
        """
        Trains the model using 'train_file'

        Args:
            train_file: Path to text file containing the text to train the language model on.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_data (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down training significantly as the predicted sequences need to be generated.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)
        if self.args.evaluate_during_training and eval_file is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_data is not specified."
                " Pass eval_data to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if (
                os.path.exists(output_dir)
                and os.listdir(output_dir)
                and not self.args.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set args.overwrite_output_dir = True to overcome.".format(output_dir)
            )

        self._move_model_to_device()
        train_dataloader = self.load_and_cache_examples(train_file, verbose=verbose)
        os.makedirs(output_dir, exist_ok=True)
        global_step, training_details = self.train(
            train_dataloader,
            output_dir,
            show_running_loss=show_running_loss,
            eval_file=eval_file,
            verbose=verbose,
            **kwargs,
        )
        self.save_model(model=self.model)

        if verbose:
            logger.info(
                " Training of {} model complete. Saved to {}.".format(
                    self.args.model_name, output_dir
                )
            )

        return global_step, training_details

    def train(
            self,
            train_dataloader,
            output_dir,
            show_running_loss=True,
            eval_file=None,
            verbose=True,
            **kwargs,
    ):
        """
        Trains the model on train_dataloader.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args
        device = self.device

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = (
                    args.max_steps
                    // (len(train_dataloader) // args.gradient_accumulation_steps)
                    + 1
            )
        else:
            t_total = (
                    len(train_dataloader)
                    // args.gradient_accumulation_steps
                    * args.num_train_epochs
            )

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [
                p for n, p in model.named_parameters() if n in params
            ]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                               and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                               and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = (
            warmup_steps if args.warmup_steps == 0 else args.warmup_steps
        )

        if args.optimizer == "AdamW":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon,
            )
        elif args.optimizer == "Adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adafactor_eps,
                clip_threshold=args.adafactor_clip_threshold,
                decay_rate=args.adafactor_decay_rate,
                beta1=args.adafactor_beta1,
                weight_decay=args.weight_decay,
                scale_parameter=args.adafactor_scale_parameter,
                relative_step=args.adafactor_relative_step,
                warmup_init=args.adafactor_warmup_init,
            )

        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                    args.optimizer
                )
            )

        if args.scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)

        elif args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps
            )

        elif args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
            )

        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "polynomial_decay_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                lr_end=args.polynomial_decay_schedule_lr_end,
                power=args.polynomial_decay_schedule_power,
            )

        else:
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        if (
                args.model_name
                and os.path.isfile(os.path.join(args.model_name, "optimizer.pt"))
                and os.path.isfile(os.path.join(args.model_name, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(args.model_name, "optimizer.pt"))
            )
            scheduler.load_state_dict(
                torch.load(os.path.join(args.model_name, "scheduler.pt"))
            )

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        logger.info(" Training started")

        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0
        )
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (
                        len(train_dataloader) // args.gradient_accumulation_steps
                )
                steps_trained_in_current_epoch = global_step % (
                        len(train_dataloader) // args.gradient_accumulation_steps
                )

                logger.info(
                    "   Continuing training from checkpoint, will skip to saved global_step"
                )
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info(
                    "   Will skip the first %d steps in the current epoch",
                    steps_trained_in_current_epoch,
                )
            except ValueError:
                logger.info("   Starting fine-tuning.")
        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(**kwargs)

        for current_epoch in train_iterator:
            model.train()
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(
                f"Epoch {epoch_number + 1} of {args.num_train_epochs}"
            )
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                
                inputs = self._get_inputs_dict(batch)
                res, loss, acc, nll, ppl, n_tokens, bsz = model(**inputs)

                if args.n_gpu > 1:
                    loss = (loss.mean())
                current_loss = loss.item()
                current_ppl = ppl / bsz

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. "
                        f"Running Loss: {current_loss:9.4f} PPL: {current_ppl:9.4f}"
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )

                        self.save_model(
                            output_dir_current, optimizer, scheduler, model=model
                        )

                    if args.evaluate_during_training and (
                            args.evaluate_during_training_steps > 0
                            and global_step % args.evaluate_during_training_steps == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = self.eval_model(
                            eval_file,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent,
                            **kwargs,
                        )

                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )

                        if args.save_eval_checkpoints:
                            self.save_model(
                                output_dir_current,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        training_progress_scores["train_ppl"].append(current_ppl)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(
                                args.output_dir, "training_progress_scores.csv"
                            ),
                            index=False,
                        )

                        if not best_eval_metric:
                            best_eval_metric = results[args.early_stopping_metric]
                            self.save_model(
                                args.best_model_dir,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )
                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if (
                                    results[args.early_stopping_metric] - best_eval_metric
                                    < args.early_stopping_delta
                            ):
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if (
                                            early_stopping_counter
                                            < args.early_stopping_patience
                                    ):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(
                                                f" No improvement in {args.early_stopping_metric}"
                                            )
                                            logger.info(
                                                f" Current step: {early_stopping_counter}"
                                            )
                                            logger.info(
                                                f" Early stopping patience: {args.early_stopping_patience}"
                                            )
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args.early_stopping_patience} steps reached"
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        else:
                            if (
                                    results[args.early_stopping_metric] - best_eval_metric
                                    > args.early_stopping_delta
                            ):
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if (
                                            early_stopping_counter
                                            < args.early_stopping_patience
                                    ):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(
                                                f" No improvement in {args.early_stopping_metric}"
                                            )
                                            logger.info(
                                                f" Current step: {early_stopping_counter}"
                                            )
                                            logger.info(
                                                f" Early stopping patience: {args.early_stopping_patience}"
                                            )
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args.early_stopping_patience} steps reached"
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        model.train()

            epoch_number += 1
            output_dir_current = os.path.join(
                output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number)
            )

            if args.save_model_every_epoch:
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:
                results = self.eval_model(
                    eval_file,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    **kwargs,
                )

                if args.save_eval_checkpoints:
                    self.save_model(
                        output_dir_current, optimizer, scheduler, results=results
                    )

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                training_progress_scores["train_ppl"].append(current_ppl)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(
                    os.path.join(args.output_dir, "training_progress_scores.csv"),
                    index=False,
                )

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    self.save_model(
                        args.best_model_dir,
                        optimizer,
                        scheduler,
                        model=model,
                        results=results,
                    )
                if best_eval_metric and args.early_stopping_metric_minimize:
                    if (
                            results[args.early_stopping_metric] - best_eval_metric
                            < args.early_stopping_delta
                    ):
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )
                        early_stopping_counter = 0
                    else:
                        if (
                                args.use_early_stopping
                                and args.early_stopping_consider_epochs
                        ):
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(
                                        f" No improvement in {args.early_stopping_metric}"
                                    )
                                    logger.info(
                                        f" Current step: {early_stopping_counter}"
                                    )
                                    logger.info(
                                        f" Early stopping patience: {args.early_stopping_patience}"
                                    )
                            else:
                                if verbose:
                                    logger.info(
                                        f" Patience of {args.early_stopping_patience} steps reached"
                                    )
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )
                else:
                    if (
                            results[args.early_stopping_metric] - best_eval_metric
                            > args.early_stopping_delta
                    ):
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )
                        early_stopping_counter = 0
                    else:
                        if (
                                args.use_early_stopping
                                and args.early_stopping_consider_epochs
                        ):
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(
                                        f" No improvement in {args.early_stopping_metric}"
                                    )
                                    logger.info(
                                        f" Current step: {early_stopping_counter}"
                                    )
                                    logger.info(
                                        f" Early stopping patience: {args.early_stopping_patience}"
                                    )
                            else:
                                if verbose:
                                    logger.info(
                                        f" Patience of {args.early_stopping_patience} steps reached"
                                    )
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )

        return (
            global_step,
            tr_loss / global_step
            if not self.args.evaluate_during_training
            else training_progress_scores,
        )

    def eval_model(
            self, eval_file, output_dir=None, verbose=True, silent=False, **kwargs
    ):
        """
        Evaluates the model on eval_data. Saves results to output_dir.

        Args:
            eval_file: evaluate file
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down evaluation significantly as the predicted sequences need to be generated.
        Returns:
            results: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"

        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        eval_dataloader = self.load_and_cache_examples(
            eval_file, evaluate=True, verbose=verbose, silent=silent
        )
        os.makedirs(output_dir, exist_ok=True)

        result = self.evaluate(
            eval_dataloader, output_dir, verbose=verbose, silent=silent, **kwargs
        )
        self.results.update(result)
        if verbose:
            logger.info(self.results)

        return self.results

    def evaluate(self, eval_dataloader, output_dir, verbose=True, silent=False, **kwargs):
        """
        Evaluates the model on eval_dataloader.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args
        eval_output_dir = output_dir
        device = self.device

        results = {}

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        nb_eval_steps = 0
        model.eval()

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        avg_ppl = 0.0
        count = 0
        for batch in tqdm(
                eval_dataloader, disable=args.silent or silent, desc="Running Evaluation"
        ):
            inputs = self._get_inputs_dict(batch)
            with torch.no_grad():
                nll, ppl, bsz = model.ppl(**inputs)
                avg_ppl += ppl
                count += bsz
            nb_eval_steps += 1
        avg_ppl = avg_ppl / count
        results["eval_ppl"] = avg_ppl
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

        return results

    def _top_k(self, enc, src_padding_mask, inp_ys_tpl, inp_ys_seg, inp_ys_pos, s):
        inp_y, m = s2t(s, self.tokenizer)
        inp_y = inp_y.to(self.device)
        res = []
        s_ = []
        for l in range(inp_ys_tpl.size(0)):
            probs, pred = self.model.work(
                enc,
                src_padding_mask,
                inp_y,
                inp_ys_tpl[0:l + 1, :],
                inp_ys_seg[0:l + 1, :],
                inp_ys_pos[0:l + 1, :]
            )
            next_tk = []
            for i in range(len(s)):
                ctk = self.tokenizer.idx2token(inp_ys_tpl[l, i].item())
                if ctk not in ["<c0>", "<c1>", "<c2>"]:
                    next_tk.append(ctk)
                    continue
                logits = probs[len(s[i]) - 1, i]
                ps, idx = torch.topk(logits, k=self.args.k)
                ps = ps / torch.sum(ps)
                sampled = torch.multinomial(ps, num_samples=1)
                sampled_idx = idx[sampled]
                next_tk.append(self.tokenizer.idx2token(sampled_idx.item()))
            s_ = []
            for sent, t in zip(s, next_tk):
                if t == EOS:
                    res.append(sent)
                else:
                    s_.append(sent + [t])
            if not s_:
                break
            s = s_  # set back to s
            inp_y, m = s2t(s, self.tokenizer)
            inp_y = inp_y.to(self.device)
        res += s_
        return res[-1]

    def _top_k_inc(self, enc, src_padding_mask, inp_ys_tpl, inp_ys_seg, inp_ys_pos, s):
        incremental_state = None
        inp_y, m = s2t(s, self.tokenizer)
        inp_y = inp_y.to(self.device)
        res = []
        s_ = []
        for l in range(inp_ys_tpl.size(0)):
            probs, pred, incremental_state = self.model.work_incremental(
                enc,
                src_padding_mask,
                inp_y,
                inp_ys_tpl[0:l + 1, :],
                inp_ys_seg[0:l + 1, :],
                inp_ys_pos[0:l + 1, :],
                incremental_state
            )
            next_tk = []
            for i in range(len(s)):
                ctk = self.tokenizer.idx2token(inp_ys_tpl[l, i].item())
                if ctk not in ["<c0>", "<c1>", "<c2>"]:
                    next_tk.append(ctk)
                    continue
                if l == 0:
                    logits = probs[len(s[i]) - 1, i]
                else:
                    logits = probs[0, i]
                ps, idx = torch.topk(logits, k=self.args.k)
                ps = ps / torch.sum(ps)
                sampled = torch.multinomial(ps, num_samples=1)
                sampled_idx = idx[sampled]
                next_tk.append(self.tokenizer.idx2token(sampled_idx.item()))
            s_ = []
            bidx = [1] * len(s)
            for idx, (sent, t) in enumerate(zip(s, next_tk)):
                if t == EOS:
                    res.append(sent)
                    bidx[idx] = 0
                else:
                    s_.append(sent + [t])
            if not s_:
                break
            s = s_  # set back to s
            inp_y, m = s2t(s, self.tokenizer)
            inp_y = inp_y.to(self.device)
            bidx = torch.BoolTensor(bidx).to(self.device)
            incremental_state["bidx"] = bidx
        res += s_
        return res[-1]

    def generate(self, sentences, split_on_space=False):
        """
        Performs predictions on a list of text.

        Args:
            sentences: A python list of text (str) to be sent to the model for prediction. Note that the prefix should be prepended to the text.
            split_on_space (optional): If True, input is english string, if False, input is chinese string.

        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        self._move_model_to_device()
        self.model.eval()

        all_outputs = []
        # Batching
        for sent in tqdm(
                sentences,
                desc="Generating outputs",
                disable=self.args.silent,
        ):
            batch = [sent]
            xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk = \
                s2xy(batch, self.tokenizer, self.args.max_length, min_len=2)
            xs_tpl = xs_tpl.to(self.device)
            xs_seg = xs_seg.to(self.device)
            xs_pos = xs_pos.to(self.device)
            ys_tpl = ys_tpl.to(self.device)
            ys_seg = ys_seg.to(self.device)
            ys_pos = ys_pos.to(self.device)
            with torch.no_grad():
                enc, src_padding_mask = self.model.encode(xs_tpl, xs_seg, xs_pos)
            s = [[BOS]]
            outputs = self._top_k(enc, src_padding_mask, ys_tpl, ys_seg, ys_pos, s)
            if self.args.skip_special_tokens:
                outputs = [s for s in outputs if s not in self.tokenizer.special_tokens]
            if split_on_space:
                outputs = ' '.join(outputs)
            else:
                outputs = ''.join(outputs)
            all_outputs.append(outputs)

        return all_outputs

    def fill_mask(self, sentences, split_on_space=False):
        """
        Performs mask predictions on a list of text.

        Args:
            sentences: A python list of text (str) to be sent to the model for prediction. 
            split_on_space (optional): If True, input is english string, if False, input is chinese string.

        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        self._move_model_to_device()
        self.model.eval()
        all_outputs = []
        # Batching
        for sent in tqdm(
                sentences,
                desc="Generating outputs",
                disable=self.args.silent,
        ):
            batch = [sent]
            xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk = \
                s2xy_polish(batch, self.tokenizer, self.args.max_length)
            xs_tpl = xs_tpl.to(self.device)
            xs_seg = xs_seg.to(self.device)
            xs_pos = xs_pos.to(self.device)
            ys_tpl = ys_tpl.to(self.device)
            ys_seg = ys_seg.to(self.device)
            ys_pos = ys_pos.to(self.device)
            with torch.no_grad():
                enc, src_padding_mask = self.model.encode(xs_tpl, xs_seg, xs_pos)
            s = [[BOS]]
            outputs = self._top_k(enc, src_padding_mask, ys_tpl, ys_seg, ys_pos, s)
            if self.args.skip_special_tokens:
                outputs = [s for s in outputs if s not in self.tokenizer.special_tokens]
            if split_on_space:
                outputs = ' '.join(outputs)
            else:
                outputs = ''.join(outputs)
            all_outputs.append(outputs)
        return all_outputs

    def load_and_cache_examples(
            self, file_path, evaluate=False, no_cache=False, verbose=True, silent=False
    ):
        """
        Creates a DataLoader from file_path.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        args = self.args
        if not no_cache:
            no_cache = args.no_cache
        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)
        mode = "dev" if evaluate else "train"
        return SongNetDataLoader(
                tokenizer,
                args,
                file_path,
                mode,
            )

    def compute_metrics(self, labels, preds, **kwargs):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            labels: List of target sequences
            preds: List of model generated outputs
            **kwargs: Custom metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down evaluation significantly as the predicted sequences need to be generated.

        Returns:
            result: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"
        assert len(labels) == len(preds)

        results = {}
        for metric, func in kwargs.items():
            results[metric] = func(labels, preds)

        return results

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_inputs_dict(self, batch):
        """
        Get input dict, to device
        format: input_ids, attention_mask, labels, decoder_attention_mask, decoder_input_ids
        """
        batch = tuple(t.to(self.device) for t in batch)
        # xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk
        inputs = {
            "xs_tpl": batch[0],
            "xs_seg": batch[1],
            "xs_pos": batch[2],
            "ys_truth": batch[3],
            "ys_inp": batch[4],
            "ys_tpl": batch[5],
            "ys_seg": batch[6],
            "ys_pos": batch[7],
            "msk": batch[8],
        }
        return inputs

    def _create_training_progress_scores(self, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "train_loss": [],
            "train_ppl": [],
            "eval_ppl": [],
            **extra_metrics,
        }

        return training_progress_scores

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def save_model(
            self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None
    ):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Save model and tokenizer
            self.tokenizer.save_pretrained(output_dir)
            torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(
                    optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                )
                torch.save(
                    scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                )
            self.save_model_args(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = SongNetArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]
