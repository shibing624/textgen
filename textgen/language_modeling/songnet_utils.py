# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys
import random

import numpy as np
import torch
import shutil
import tarfile
import zipfile
import six
import requests

from tqdm.autonotebook import tqdm

PAD, UNK, BOS, EOS = '<pad>', '<unk>', '<bos>', '<eos>'
BOC, EOC = '<boc>', '<eoc>'
LS, RS, SP = '<s>', '</s>', ' '
CS = ['<c-1>'] + ['<c' + str(i) + '>' for i in range(32)]  # content
SS = ['<s-1>'] + ['<s' + str(i) + '>' for i in range(512)]  # segment
PS = ['<p-1>'] + ['<p' + str(i) + '>' for i in range(512)]  # position
TS = ['<t-1>'] + ['<t' + str(i) + '>' for i in range(32)]  # other types
PUNCS = {",", ".", "?", "!", ":", "，", "。", "？", "！", "："}


class SongNetTokenizer(object):
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
        return f"SongNetTokenizer<vocab size:{len(self._token2idx)}>"

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
        self.mode = mode
        self.batch_size = args.train_batch_size if mode == "train" else args.eval_batch_size
        self.examples = self.read_file(file_path, args.max_length, args.min_length)

    @staticmethod
    def read_file(file_path, max_length, min_length):
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        data = []
        for line in lines:
            res = preprocess_data(line, max_length, min_length)
            if not res:
                continue
            data.append(res)
        return data

    def __iter__(self):
        if self.mode == "train":
            random.shuffle(self.examples)
        idx = 0
        while idx < len(self.examples):
            yield batchify(self.examples[idx:idx + self.batch_size], self.tokenizer)
            idx += self.batch_size

    def __len__(self):
        from math import ceil
        length = len(self.examples)
        length = ceil(length / self.batch_size)
        return length


def preprocess_data(line, max_length, min_length):
    """Convert line text to idx"""
    fs = line.split("<s2>", 1)
    author, cipai = fs[0].split("<s1>", 1)
    sents = fs[1].strip()
    if len(sents) > max_length:
        sents = sents[:max_length]
    if len(sents) < min_length:
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

    if len(ys) < min_length:
        return None
    return xs_tpl, xs_seg, xs_pos, ys, ys_tpl, ys_seg, ys_pos


class Optim:
    """Optimizer wrapper that implements rate."""

    def __init__(self, emb_dim, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.emb_dim = emb_dim
        self.lr = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.lr = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lr` above"""
        if step is None:
            step = self._step
        return self.factor * (self.emb_dim ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, m):
        self.optimizer.load_state_dict(m)


def http_get(url, path, extract: bool = True):
    """
    Downloads a URL to a given path on disc
    """
    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
        req.raise_for_status()
        return

    download_filepath = path + "_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()

    if extract:
        data_dir = os.path.dirname(os.path.abspath(path))
        _extract_archive(path, data_dir, 'auto')


def _extract_archive(file_path, path='.', archive_format='auto'):
    """
    Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.
    :param file_path: path to the archive file
    :param path: path to extract the archive file
    :param archive_format: Archive format to try for extracting the file.
        Options are 'auto', 'tar', 'zip', and None.
        'tar' includes tar, tar.gz, and tar.bz files.
        The default 'auto' is ['tar', 'zip'].
        None or an empty list will return no matches found.
    :return: True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    if archive_format is None:
        return False
    if archive_format == 'auto':
        archive_format = ['tar', 'zip']
    if isinstance(archive_format, six.string_types):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError,
                        KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False


######################
# Download from HuggingFace models
######################

from typing import Dict, Optional, Union, List
from pathlib import Path
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub import HfApi, hf_hub_url, cached_download
import fnmatch

def snapshot_download(
    repo_id: str,
    revision: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    user_agent: Union[Dict, str, None] = None,
    ignore_files: Optional[List[str]] = None,
    use_auth_token: Union[bool, str, None] = None
) -> str:
    """
    Method derived from huggingface_hub.
    Adds a new parameters 'ignore_files', which allows to ignore certain files / file-patterns
    """
    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    _api = HfApi()
    model_info = _api.model_info(repo_id=repo_id, revision=revision)

    storage_folder = os.path.join(
        cache_dir, repo_id.replace("/", "_")
    )

    for model_file in model_info.siblings:
        if ignore_files is not None:
            skip_download  = False
            for pattern in ignore_files:
                if fnmatch.fnmatch(model_file.rfilename, pattern):
                    skip_download = True
                    break

            if skip_download:
                continue

        url = hf_hub_url(
            repo_id, filename=model_file.rfilename, revision=model_info.sha
        )
        relative_filepath = os.path.join(*model_file.rfilename.split("/"))

        # Create potential nested dir
        nested_dirname = os.path.dirname(
            os.path.join(storage_folder, relative_filepath)
        )
        os.makedirs(nested_dirname, exist_ok=True)

        path = cached_download(
            url,
            cache_dir=storage_folder,
            force_filename=relative_filepath,
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
            use_auth_token=use_auth_token,
        )

        if os.path.exists(path + ".lock"):
            os.remove(path + ".lock")

    return storage_folder
