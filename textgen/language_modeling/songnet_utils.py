# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import random
import torch
import numpy as np
import os
import pickle
from loguru import logger
from torch.utils.data import Dataset
from filelock import FileLock
from tqdm.auto import tqdm

PAD, UNK, BOS, EOS = '<pad>', '<unk>', '<bos>', '<eos>'
BOC, EOC = '<boc>', '<eoc>'
LS, RS, SP = '<s>', '</s>', ' '
CS = ['<c-1>'] + ['<c' + str(i) + '>' for i in range(32)]  # content
SS = ['<s-1>'] + ['<s' + str(i) + '>' for i in range(512)]  # segment
PS = ['<p-1>'] + ['<p' + str(i) + '>' for i in range(512)]  # position
TS = ['<t-1>'] + ['<t' + str(i) + '>' for i in range(32)]  # other types
PUNCS = {",", ".", "?", "!", ":", "，", "。", "？", "！", "："}


class ZHCharTokenizer(object):
    def __init__(self, vocab_file, specials=None):
        special_tokens = [PAD, UNK, BOS, EOS, BOC, EOC, LS, RS, SP] + CS + SS + PS + TS \
                    + (specials if specials is not None else [])
        vocabs = self.load_vocab(vocab_file)
        idx2token = special_tokens + [v for v in vocabs if v not in special_tokens]
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]
        self.special_tokens = special_tokens

    @staticmethod
    def load_vocab(vocab_file):
        vocabs = []
        with open(vocab_file, encoding='utf8') as f:
            for line in f:
                line = line.strip('\n')
                if line:
                    vocabs.append(line)
        return vocabs

    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def random_token(self):
        return self.idx2token(1 + np.random.randint(self.size - 1))

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)

    def __repr__(self):
        return f"ZHCharTokenizer<_token2idx size:{len(self._token2idx)}>"

    @classmethod
    def from_pretrained(cls, model_dir, *init_inputs, **kwargs):
        r"""
        Instantiate a predefined tokenizer.
        """
        vocab_file = os.path.join(model_dir, 'vocab.txt')
        try:
            tokenizer = cls(vocab_file, *init_inputs, **kwargs)
        except OSError:
            raise OSError(
                "Unable to load vocabulary from file. "
                "Please check that the provided vocabulary is accessible and not corrupted."
            )
        return tokenizer

    def save_pretrained(self, output_dir):
        r"""
        Save vocab.
        """
        vocab_file = os.path.join(output_dir, 'vocab.txt')
        with open(vocab_file, 'w', encoding='utf8') as f:
            for token, idx in self._token2idx.items():
                f.write(token + '\n')
            logger.info("Vocab saved in {}".format(vocab_file))


def lists2tensor(xs, tokenizer=None):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        if tokenizer is not None:
            y = tokenizer.token2idx(x) + [tokenizer.padding_idx] * (max_len - len(x))
        else:
            y = x + [0] * (max_len - len(x))
        ys.append(y)
    return ys


def batchify(data, tokenizer):
    xs_tpl, xs_seg, xs_pos, \
    ys_truth, ys_inp, \
    ys_tpl, ys_seg, ys_pos, msk = [], [], [], [], [], [], [], [], []
    # logger.debug(f"data:{data}, first: {data[0]}")
    for xs_tpl_i, xs_seg_i, xs_pos_i, ys_i, ys_tpl_i, ys_seg_i, ys_pos_i in data:
        # logger.debug(f"first is xs_tpl_i: {xs_tpl_i}")
        xs_tpl.append(xs_tpl_i)
        xs_seg.append(xs_seg_i)
        xs_pos.append(xs_pos_i)

        ys_truth.append(ys_i)
        ys_inp.append([BOS] + ys_i[:-1])
        ys_tpl.append(ys_tpl_i)
        ys_seg.append(ys_seg_i)
        ys_pos.append(ys_pos_i)

        msk.append([1 for i in range(len(ys_i))])

    xs_tpl = torch.LongTensor(lists2tensor(xs_tpl, tokenizer)).t_().contiguous()
    xs_seg = torch.LongTensor(lists2tensor(xs_seg, tokenizer)).t_().contiguous()
    xs_pos = torch.LongTensor(lists2tensor(xs_pos, tokenizer)).t_().contiguous()
    ys_truth = torch.LongTensor(lists2tensor(ys_truth, tokenizer)).t_().contiguous()
    ys_inp = torch.LongTensor(lists2tensor(ys_inp, tokenizer)).t_().contiguous()
    ys_tpl = torch.LongTensor(lists2tensor(ys_tpl, tokenizer)).t_().contiguous()
    ys_seg = torch.LongTensor(lists2tensor(ys_seg, tokenizer)).t_().contiguous()
    ys_pos = torch.LongTensor(lists2tensor(ys_pos, tokenizer)).t_().contiguous()
    msk = torch.FloatTensor(lists2tensor(msk)).t_().contiguous()
    return xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk


def s2t(strs, tokenizer):
    inp, msk = [], []
    for x in strs:
        inp.append(x)
        msk.append([1 for i in range(len(x))])

    inp = torch.LongTensor(lists2tensor(inp, tokenizer)).t_().contiguous()
    msk = torch.FloatTensor(lists2tensor(msk)).t_().contiguous()
    return inp, msk


def s2xy(lines, tokenizer, max_len, min_len):
    data = []
    for line in lines:
        res = parse_line(line, max_len, min_len)
        if not res:
            continue
        data.append(res)
    return batchify(data, tokenizer)


def parse_line(line, max_len, min_len=2):
    line = line.strip()
    if not line:
        return None
    fs = line.split("<s2>")
    author, cipai = fs[0].split("<s1>")
    sents = fs[1].strip()
    if len(sents) > max_len:
        sents = sents[:max_len]
    if len(sents) < min_len:
        return None
    sents = sents.split("</s>")

    ys = []
    xs_tpl = []
    xs_seg = []
    xs_pos = []

    ctx = cipai
    ws = [w for w in ctx]
    xs_tpl = ws + [EOC]
    xs_seg = [SS[0] for w in ws] + [EOC]
    xs_pos = [SS[i + 300] for i in range(len(ws))] + [EOC]

    ys_tpl = []
    ys_seg = []
    ys_pos = []
    for si, sent in enumerate(sents):
        ws = []
        sent = sent.strip()
        if not sent:
            continue
        for w in sent:
            ws.append(w)
            if w.strip() and w not in PUNCS:
                ys_tpl.append(CS[2])
            else:
                ys_tpl.append(CS[1])
        ys += ws + [RS]
        if ws[-1] in PUNCS:
            ys_tpl[-2] = CS[3]
        else:
            ys_tpl[-1] = CS[3]
        ys_tpl += [RS]
        ys_seg += [SS[si + 1] for w in ws] + [RS]
        ys_pos += [PS[len(ws) - i] for i in range(len(ws))] + [RS]

    ys += [EOS]
    ys_tpl += [EOS]
    ys_seg += [EOS]
    ys_pos += [EOS]

    xs_tpl += ys_tpl
    xs_seg += ys_seg
    xs_pos += ys_pos

    if len(ys) < min_len:
        return None
    return xs_tpl, xs_seg, xs_pos, ys, ys_tpl, ys_seg, ys_pos


def s2xy_polish(lines, tokenizer, max_len, min_len=2):
    data = []
    for line in lines:
        res = parse_line_polish(line, max_len, min_len)
        data.append(res)
    return batchify(data, tokenizer)


def parse_line_polish(line, max_len, min_len):
    line = line.strip()
    if not line:
        return None
    fs = line.split("<s2>")
    author, cipai = fs[0].split("<s1>")
    sents = fs[1].strip()
    if len(sents) > max_len:
        sents = sents[:max_len]
    if len(sents) < min_len:
        return None
    sents = sents.split("</s>")

    ys = []
    xs_tpl = []
    xs_seg = []
    xs_pos = []

    ctx = cipai
    ws = [w for w in ctx]
    xs_tpl = ws + [EOC]
    xs_seg = [SS[0] for w in ws] + [EOC]
    xs_pos = [SS[i + 300] for i in range(len(ws))] + [EOC]

    ys_tpl = []
    ys_seg = []
    ys_pos = []
    for si, sent in enumerate(sents):
        ws = []
        sent = sent.strip()
        if not sent:
            continue
        for w in sent:
            ws.append(w)
            if w == "_":
                ys_tpl.append(CS[2])
            else:
                ys_tpl.append(w)
        ys += ws + [RS]
        ys_tpl += [RS]
        ys_seg += [SS[si + 1] for w in ws] + [RS]
        ys_pos += [PS[len(ws) - i] for i in range(len(ws))] + [RS]

    ys += [EOS]
    ys_tpl += [EOS]
    ys_seg += [EOS]
    ys_pos += [EOS]

    xs_tpl += ys_tpl
    xs_seg += ys_seg
    xs_pos += ys_pos

    if len(ys) < min_len:
        return None

    return xs_tpl, xs_seg, xs_pos, ys, ys_tpl, ys_seg, ys_pos


class SongNetDataLoader(object):
    def __init__(self, tokenizer, args, file_path, mode):
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.examples = self.convert_text_to_ids(file_path, tokenizer, args)
        
    def convert_text_to_ids(self, file_path, tokenizer, args):
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        data = []
        for line in lines:
            res = preprocess_data(line, args)
            if not res:
                continue
            data.append(res)
        return data

    def __iter__(self):
        if self.mode == "train":
            random.shuffle(self.examples)
        idx = 0
        while idx < len(self.examples):
            yield batchify(self.examples[idx:idx + self.args.train_batch_size], self.tokenizer)
            idx += self.args.train_batch_size
    
    def __len__(self):
        return len(self.examples) // self.args.train_batch_size

def preprocess_data(line, args):
    """Convert line text to idx"""
    fs = line.split("<s2>", 1)
    author, cipai = fs[0].split("<s1>", 1)
    sents = fs[1].strip()
    if len(sents) > args.max_length:
        sents = sents[:args.max_length]
    if len(sents) < args.min_length:
        return None
    sents = sents.split("</s>")

    ys = []
    xs_tpl = []
    xs_seg = []
    xs_pos = []

    ctx = cipai
    ws = [w for w in ctx]
    xs_tpl = ws + [EOC]
    xs_seg = [SS[0] for w in ws] + [EOC]
    xs_pos = [SS[i + 300] for i in range(len(ws))] + [EOC]

    ys_tpl = []
    ys_seg = []
    ys_pos = []
    for si, sent in enumerate(sents):
        ws = []
        sent = sent.strip()
        if not sent:
            continue
        for w in sent:
            ws.append(w)
            if w.strip() and w not in PUNCS:
                ys_tpl.append(CS[2])
            else:
                ys_tpl.append(CS[1])
        ys += ws + [RS]
        if ws[-1] in PUNCS:
            ys_tpl[-2] = CS[3]
        else:
            ys_tpl[-1] = CS[3]
        ys_tpl += [RS]
        ys_seg += [SS[si + 1] for w in ws] + [RS]
        ys_pos += [PS[len(ws) - i] for i in range(len(ws))] + [RS]

    ys += [EOS]
    ys_tpl += [EOS]
    ys_seg += [EOS]
    ys_pos += [EOS]

    xs_tpl += ys_tpl
    xs_seg += ys_seg
    xs_pos += ys_pos

    if len(ys) < args.min_length:
        return None
    return xs_tpl, xs_seg, xs_pos, ys, ys_tpl, ys_seg, ys_pos
