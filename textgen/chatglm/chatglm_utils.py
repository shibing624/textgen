# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import pickle
from multiprocessing import Pool

import datasets
import numpy as np
from datasets import Dataset as HFDataset
from datasets import load_dataset
from loguru import logger
from rouge import Rouge
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def preprocess_data(data):
    instruction, input_text, target_text, tokenizer, args = data

    prompt = f"问：{instruction}\n"
    if input_text:
        prompt += f"{input_text}\n"
    prompt += "答："

    prompt_ids = tokenizer.encode(prompt, max_length=args.max_seq_length, truncation=True)
    target_ids = tokenizer.encode(target_text, max_length=args.max_length, truncation=True,
                                  add_special_tokens=False)
    input_ids = prompt_ids + target_ids
    input_ids = input_ids[:(args.max_seq_length + args.max_length)] + [tokenizer.eos_token_id]

    return input_ids


def preprocess_batch_for_hf_dataset(dataset, tokenizer, args):
    data = (dataset["instruction"], dataset["input"], dataset["output"], tokenizer, args)
    dataset['input_ids'] = preprocess_data(data)
    return dataset


def load_hf_dataset(data, tokenizer, args, mode):
    if isinstance(data, str):
        if data.endswith('.json') or data.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=data)
        elif os.path.isdir(data):
            dataset = datasets.load_from_disk(data)
        else:
            dataset = load_dataset(
                data,
                download_mode="force_redownload"
                if args.reprocess_input_data
                else "reuse_dataset_if_exists",
            )
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        dataset = dataset['train']
        if mode == 'dev' and args.max_eval_samples is not None:
            max_eval_samples = min(len(dataset), args.max_eval_samples)
            dataset = dataset.select(range(max_eval_samples))
    else:
        dataset = HFDataset.from_pandas(data)

    dataset = dataset.shuffle().map(
        lambda x: preprocess_batch_for_hf_dataset(x, tokenizer=tokenizer, args=args),
        batched=False, remove_columns=dataset.column_names
    ).filter(lambda x: tokenizer.gmask_token_id in list(x['input_ids']))  # exclude samples without gmask

    dataset.set_format(type="np", columns=["input_ids"])

    return dataset["input_ids"]


class ChatGlmDataset(Dataset):
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
                (instruction, input_text, target_text, tokenizer, args)
                for instruction, input_text, target_text in zip(
                    data["instruction"], data["input"], data["output"]
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


def compute_bleu(label, pred, weights=None):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    weights = weights or (0.25, 0.25, 0.25, 0.25)
    return np.mean([sentence_bleu(references=[list(a)], hypothesis=list(b),
                                  smoothing_function=SmoothingFunction().method1, weights=weights)
                    for a, b in zip(label, pred)])


def compute_rouge(label, pred, weights=None, mode='weighted'):
    weights = weights or (0.2, 0.4, 0.4)
    if isinstance(label, str):
        label = [label]
    if isinstance(pred, str):
        pred = [pred]
    label = [' '.join(x) for x in label]
    pred = [' '.join(x) for x in pred]

    def _compute_rouge(label, pred):
        try:
            scores = Rouge().get_scores(hyps=label, refs=pred)[0]
            scores = [scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']]
        except ValueError:
            scores = [0, 0, 0]
        return scores

    scores = np.mean([_compute_rouge(*x) for x in zip(label, pred)], axis=0)
    if mode == 'weighted':
        return {'rouge': sum(s * w for s, w in zip(scores, weights))}
    elif mode == '1':
        return {'rouge-1': scores[0]}
    elif mode == '2':
        return {'rouge-2': scores[1]}
    elif mode == 'l':
        return {'rouge-l': scores[2]}
    elif mode == 'all':
        return {'rouge-1': scores[0], 'rouge-2': scores[1], 'rouge-l': scores[2]}
