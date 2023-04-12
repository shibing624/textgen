# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import pickle
from multiprocessing import Pool

import datasets
from datasets import Dataset as HFDataset
from datasets import load_dataset
from loguru import logger
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def preprocess_data(data):
    instruction, input_text, target_text, tokenizer, args = data

    prompt = f"问：{instruction}\n"
    if input_text:
        prompt += f"{input_text}\n"
    prompt += "答："
    if target_text:
        prompt += f"{target_text}"

    full_max_length = args.max_seq_length + args.max_length
    example = tokenizer(
        prompt,
        truncation=True,
        max_length=full_max_length,
        padding=False,
        return_tensors=None,
    )
    if (
            example["input_ids"][-1] != tokenizer.eos_token_id
            and len(example["input_ids"]) < full_max_length
    ):
        example["input_ids"].append(tokenizer.eos_token_id)
        example["attention_mask"].append(1)

    example["labels"] = example["input_ids"].copy()

    return example


def preprocess_batch_for_hf_dataset(dataset, tokenizer, args):
    data = (dataset["instruction"], dataset["input"], dataset["output"], tokenizer, args)
    dataset = preprocess_data(data)
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

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(x, tokenizer=tokenizer, args=args),
        batched=False,
    ).filter(lambda x: len(x['input_ids']) > 0)

    dataset.set_format(type="np", columns=["input_ids"])

    return dataset


class LlamaDataset(Dataset):
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
