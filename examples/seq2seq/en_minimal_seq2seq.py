# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys

import pandas as pd
import torch

sys.path.append('../..')
from textgen.seq2seq import Seq2SeqModel

use_cuda = torch.cuda.is_available()


def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            terms = line.strip().split()
            data.append([terms[0], terms[1]])
    return data


def load_qa_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            q = ''
            line = line.strip()
            if line.startswith('='):
                continue
            if line.startswith('Q: '):
                q = line[3:]
            if line.startswith('A: '):
                a = line[3:]
                data.append([q, a])
    return data


train_data = [
    ["one", "1"],
    ["two", "2"],
    ["three", "3"],
    ["four", "4"],
    ["five", "5"],
    ["six", "6"],
    ["seven", "7"],
    ["eight", "8"],
]

sub_train_data = load_qa_data('dialog_en.txt')
train_data += sub_train_data

train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

eval_data = [
    ["nine", "9"],
    ["zero", "0"],
]
sub_eval_data = load_qa_data('dialog_en.txt')[:10]
eval_data += sub_eval_data

eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 50,
    "train_batch_size": 8,
    "num_train_epochs": 25,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "silent": False,
    "evaluate_generated_text": True,
    "evaluate_during_training": False,
    "evaluate_during_training_verbose": False,
    "use_multiprocessing": False,
    "save_best_model": True,
    "max_length": 50,
    "output_dir": './en_output/',
}

# encoder_type=None, encoder_name=None, decoder_name=None, encoder_decoder_type=None, encoder_decoder_name=None,
model = Seq2SeqModel("bert", "bert-base-cased", "bert-base-cased", args=model_args, use_cuda=use_cuda, )


def count_matches(labels, preds):
    print(labels)
    print(preds)
    return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])


model.train_model(train_df, eval_data=eval_df, matches=count_matches)

print(model.eval_model(eval_df, matches=count_matches))

print(model.predict(["four", "five"]))
print(model.predict(
    ["that 's the kind of guy she likes ? Pretty ones ?", "Not the hacking and gagging and spitting part ."]))
