# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

import pandas as pd
from loguru import logger
from transformers import AutoTokenizer

sys.path.append('..')
from textgen.llama.llama_utils import LlamaPretrainingDataset
from textgen import GptArgs


def load_data(file_path):
    return [i for i in open(file_path, 'r', encoding='utf-8').read().split('\n\n') if i]


def test_data():
    tokenizer = AutoTokenizer.from_pretrained('shibing624/chinese-alpaca-plus-7b-hf')
    args = GptArgs()
    args.model_name = 'shibing624/chinese-alpaca-plus-7b-hf'
    args.no_cache = True
    print('args', args)
    logger.info(args)
    train_data = load_data('../examples/data/pt.txt')
    train_df = pd.DataFrame(train_data, columns=["text"])
    eval_df = train_df[:10]
    ds = LlamaPretrainingDataset(
        tokenizer,
        args,
        train_df,
        mode="train",
    )
    print(ds, len(ds))
    a = list(ds)
    b = [i for i in ds]
    assert a == b
    for i in list(ds)[:3]:
        print(type(i), i)
        for k, v in i.items():
            print(k, v)
    assert ds is not None
