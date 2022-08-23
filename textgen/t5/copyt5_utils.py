# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import pickle
from multiprocessing import Pool

import jieba
from datasets import Dataset as HFDataset
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from loguru import logger
import json
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
import collections
from transformers import AdamW, get_linear_schedule_with_warmup, T5ForConditionalGeneration
from transformers import T5ForConditionalGeneration, BertTokenizer, AutoTokenizer
from tqdm import tqdm
import copy

jieba.setLogLevel('ERROR')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_start_method('spawn')


class ZHTokenizer(BertTokenizer):

    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


def mask_select(inputs, mask):
    input_dim = inputs.ndim
    mask_dim = mask.ndim
    mask = mask.reshape(-1).bool()
    if input_dim > mask_dim:
        inputs = inputs.reshape((int(mask.size(-1)), -1))[mask]
    else:
        inputs = inputs.reshape(-1)[mask]
    return inputs


def copy_loss(inputs, targets, mask, eps=1e-6):
    mask = mask[:, 1:]
    inputs = inputs[:, :-1]
    targets = targets[:, 1:]
    inputs = mask_select(inputs, mask)
    targets = mask_select(targets, mask)
    log_preds = (inputs + eps).log()
    loss = F.nll_loss(log_preds, targets)
    return loss


def ce_loss(inputs, targets, mask):
    mask = mask[:, 1:]
    inputs = inputs[:, :-1]
    targets = targets[:, 1:]
    inputs = mask_select(inputs, mask)
    targets = mask_select(targets, mask)
    loss = F.cross_entropy(inputs, targets)
    return loss


def preprocess_batch_for_hf_dataset(dataset, tokenizer, args):
    if args.preprocess_inputs:
        return tokenizer.prepare_seq2seq_batch(
            src_texts=[
                prefix + ": " + input_text
                for prefix, input_text in zip(dataset["prefix"], dataset["input_text"])
            ],
            tgt_texts=dataset["target_text"],
            max_length=args.max_seq_length,
            max_target_length=args.max_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )
    else:
        return tokenizer.prepare_seq2seq_batch(
            src_texts=[
                prefix + input_text
                for prefix, input_text in zip(dataset["prefix"], dataset["input_text"])
            ],
            tgt_texts=dataset["target_text"],
            max_length=args.max_seq_length,
            max_target_length=args.max_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )


def load_hf_dataset(data, tokenizer, args):
    if isinstance(data, str):
        dataset = load_dataset(
            "csv",
            data_files=data,
            delimiter="\t",
            download_mode="force_redownload"
            if args.reprocess_input_data
            else "reuse_dataset_if_exists",
        )
    else:
        dataset = HFDataset.from_pandas(data)

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(x, tokenizer=tokenizer, args=args),
        batched=True,
    )

    dataset.set_format(type="pt", columns=["input_ids", "attention_mask"])

    if isinstance(data, str):
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        return dataset["train"]
    else:
        return dataset


def preprocess_data(data):
    prefix, input_text, target_text, tokenizer, args = data

    batch = tokenizer.prepare_seq2seq_batch(
        src_texts=[prefix + ": " + input_text],
        tgt_texts=[target_text],
        max_length=args.max_seq_length,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
    )
    input_ids = batch["input_ids"][0]
    attention_mask = batch["attention_mask"][0]
    labels = batch["labels"][0]
    return (input_ids, attention_mask, labels)


class CopyT5Dataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        cached_features_file = os.path.join(
            args.cache_dir,
            args.model_name.replace("/", "_")
            + "_cached_"
            + str(args.max_seq_length)
            + str(len(data)),
        )

        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not args.no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s" % cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s" % args.cache_dir)

            data = [
                (prefix, input_text, target_text, tokenizer, args)
                for prefix, input_text, target_text in zip(
                    data["prefix"], data["input_text"], data["target_text"]
                )
            ]

            if (mode == "train" and args.use_multiprocessing) or (
                    mode == "dev" and args.use_multiprocessing_for_evaluation
            ):
                if args.multiprocessing_chunksize == -1:
                    chunksize = max(len(data) // (args.process_count * 2), 500)
                else:
                    chunksize = args.multiprocessing_chunksize

                with Pool(args.process_count) as p:
                    self.examples = list(
                        tqdm(
                            p.imap(preprocess_data, data, chunksize=chunksize),
                            total=len(data),
                            disable=args.silent,
                        )
                    )
            else:
                self.examples = [preprocess_data(d) for d in tqdm(data, disable=args.silent)]
            if not args.no_cache:
                logger.info(" Saving features into cached file %s" % cached_features_file)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
