# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SongNet model

refer: ACL2020 paper: Rigid Formats Controlled Text Generation
paper author: Piji Li
url: https://www.aclweb.org/anthology/2020.acl-main.68
"""
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
import math
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm, trange

from textgen.config.model_args import SongNetArgs
from textgen.songnet.songnet_utils import ZHCharTokenizer, batchify, s2t, s2xy, s2xy_polish, DataLoader

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
    def __init__(self, tokenizer, device=0, embed_dim=768, ff_embed_dim=768 * 4, num_heads=12, dropout=0.2,
                 layers=12, smoothing_factor=0.1, approx=None):
        super(SongNet, self).__init__()
        self.tokenizer = tokenizer
        self.embed_dim = embed_dim

        self.tok_embed = Embedding(self.tokenizer.size, embed_dim, self.tokenizer.padding_idx)
        self.pos_embed = LearnedPositionalEmbedding(embed_dim, device=device)

        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout, with_external=True))
        self.emb_layer_norm = LayerNorm(embed_dim)
        self.one_more = nn.Linear(embed_dim, embed_dim)
        self.one_more_layer_norm = LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, self.tokenizer.size)

        self.attn_mask = SelfAttentionMask(device=device)
        self.smoothing = LabelSmoothing(device, self.tokenizer.size, self.tokenizer.padding_idx, smoothing_factor)

        self.dropout = dropout
        self.device = device

        self.approx = approx
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
            x, _, _ = layer.work_incremental(x, self_padding_mask=padding_mask,
                                             self_attn_mask=self_attn_mask,
                                             external_memories=enc,
                                             external_padding_mask=src_padding_mask,
                                             incremental_state=incremental_state)

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
            x, _, _ = layer(x, self_padding_mask=padding_mask,
                            self_attn_mask=self_attn_mask,
                            external_memories=enc,
                            external_padding_mask=src_padding_mask, )

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
            x, _, _ = layer(x, self_padding_mask=padding_mask,
                            self_attn_mask=self_attn_mask,
                            external_memories=enc,
                            external_padding_mask=src_padding_mask)

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
            x, _, _ = layer(x, self_padding_mask=padding_mask,
                            self_attn_mask=self_attn_mask,
                            external_memories=enc,
                            external_padding_mask=src_padding_mask)

        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        pred = torch.softmax(self.out_proj(x), -1)

        loss = self.label_smotthing_loss(pred, ys_truth, msk)

        _, pred_y = pred.max(-1)
        tot_tokens = msk.float().sum().item()
        acc = (torch.eq(pred_y, ys_truth).float() * msk).sum().item()

        nll, ppl = self.nll_loss(pred, ys_truth, msk)
        return (pred_y, ys_truth), loss, acc, nll, ppl, tot_tokens, bsz


class SongNetModel:
    def __init__(
            self,
            model_path,
            vocab_path,
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
        if model_path and vocab_path:
            ckpt = torch.load(model_path, map_location='cpu')
            model_args = ckpt['args']
            logger.debug(f'model args: {model_args}')
            tokenizer = ZHCharTokenizer(vocab_path, min_occur_cnt=model_args.min_occur_cnt, **kwargs)
            device = 0 if self.device != 'cpu' else 0
            model = SongNet(
                tokenizer,
                device,
                embed_dim=model_args.embed_dim,
                ff_embed_dim=model_args.ff_embed_dim,
                num_heads=model_args.num_heads,
                dropout=model_args.dropout,
                layers=model_args.layers,
                smoothing_factor=model_args.smoothing,
                approx=None
            )
            model.load_state_dict(ckpt['model'])
            model = model.to(self.device)
            self.model = model
            self.tokenizer = tokenizer
            self.model_args = model_args

        self.args.model_type = model_type
        if model_name is None:
            self.args.model_name = "SongNet_from_scratch"
        else:
            self.args.model_name = model_name

    def train_model(self, train_df, eval_df=None):
        pass

    def eval_model(self, eval_df):
        pass

    def _top_k_inc(self, enc, src_padding_mask, inp_ys_tpl, inp_ys_seg, inp_ys_pos, s, k=32):
        incremental_state = None
        inp_y, m = s2t(s, self.tokenizer)
        inp_y = inp_y.to(self.device)
        res = []
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
                if ctk != "<c1>" and ctk != "<c2>" and ctk != "<c0>":
                    next_tk.append(ctk)
                    continue
                if l == 0:
                    logits = probs[len(s[i]) - 1, i]
                else:
                    logits = probs[0, i]
                ps, idx = torch.topk(logits, k=k)
                ps = ps / torch.sum(ps)
                sampled = torch.multinomial(ps, num_samples=1)
                sampled_idx = idx[sampled]
                next_tk.append(self.tokenizer.idx2token(sampled_idx.item()))

            sents = []
            bidx = [1] * len(s)
            for idx, (sent, t) in enumerate(zip(s, next_tk)):
                if t == "<eos>":
                    sents.append(sent)
                    bidx[idx] = 0
                else:
                    sents.append(sent + [t])
            if not sents:
                break
            s = sents
            inp_y, m = s2t(s, self.tokenizer)
            inp_y = inp_y.to(self.device)
            bidx = torch.BoolTensor(bidx).to(self.device)
            incremental_state["bidx"] = bidx
            res += sents
        return res

    def predict(self, sentences, skip_special_tokens=True):
        """
        Performs predictions on a list of text.

        Args:
            sentences: A python list of text (str) to be sent to the model for prediction. 

        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        self._move_model_to_device()
        self.model.eval()
        all_outputs = []
        # Batching
        for sent in sentences:
            batch = [sent]
            xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk = \
                s2xy(batch, self.tokenizer, self.model_args.max_len, min_len=2)

            xs_tpl = xs_tpl.to(self.device)
            xs_seg = xs_seg.to(self.device)
            xs_pos = xs_pos.to(self.device)
            ys_tpl = ys_tpl.to(self.device)
            ys_seg = ys_seg.to(self.device)
            ys_pos = ys_pos.to(self.device)
            enc, src_padding_mask = self.model.encode(xs_tpl, xs_seg, xs_pos)
            s = [['<bos>']]
            res = self._top_k_inc(enc, src_padding_mask, ys_tpl, ys_seg, ys_pos, s)[-1]
            if skip_special_tokens:
                res = [s for s in res if s not in self.tokenizer.special_tokens]
            all_outputs.append(''.join(res))
        return all_outputs

    def predict_mask(self, sentences, skip_special_tokens=True):
        """
        Performs mask predictions on a list of text.

        Args:
            sentences: A python list of text (str) to be sent to the model for prediction. 

        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        self._move_model_to_device()
        self.model.eval()
        all_outputs = []
        # Batching
        for sent in sentences:
            batch = [sent]
            xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk = \
                s2xy_polish(batch, self.tokenizer, self.model_args.max_len, min_len=2)

            xs_tpl = xs_tpl.to(self.device)
            xs_seg = xs_seg.to(self.device)
            xs_pos = xs_pos.to(self.device)
            ys_tpl = ys_tpl.to(self.device)
            ys_seg = ys_seg.to(self.device)
            ys_pos = ys_pos.to(self.device)
            enc, src_padding_mask = self.model.encode(xs_tpl, xs_seg, xs_pos)
            s = [['<bos>']]
            res = self._top_k_inc(enc, src_padding_mask, ys_tpl, ys_seg, ys_pos, s)[-1]
            if skip_special_tokens:
                res = [s for s in res if s not in self.tokenizer.special_tokens]
            all_outputs.append(''.join(res))
        return all_outputs

    def _move_model_to_device(self):
        self.model.to(self.device)

    def save_model(
            self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None
    ):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
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
