# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import pickle
from dataclasses import dataclass
from itertools import chain
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Union

import datasets
from datasets import Dataset as HFDataset
from datasets import load_dataset
from loguru import logger
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n{input_text}\n\n### Response:\n\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n\n"
    ),
}


def preprocess_instruction_data(data):
    instruction, input_text, target_text, tokenizer, args = data

    if input_text:
        prompt = PROMPT_DICT["prompt_input"].format(instruction=instruction, input_text=input_text)
    else:
        prompt = PROMPT_DICT["prompt_no_input"].format(instruction=instruction)

    full_prompt = prompt + target_text + tokenizer.eos_token
    full_max_length = args.max_seq_length + args.max_length
    example = tokenizer(
        full_prompt,
        truncation=True,
        max_length=full_max_length,
        padding=False,
        add_special_tokens=False
    )
    example["labels"] = example["input_ids"].copy()
    if not args.is_train_on_prompt:
        user_example = tokenizer(
            prompt,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            add_special_tokens=False
        )
        user_prompt_len = len(user_example["input_ids"])
        # set labels to full max length to adjust for DataCollatorForSeq2Seq padding
        example["labels"] = [-100] * (full_max_length - len(example['labels']) + user_prompt_len) + \
                            example["labels"][user_prompt_len:]

    return example


def preprocess_batch_for_hf_instruction_dataset(example, tokenizer, args):
    data = (example["instruction"], example["input"], example["output"], tokenizer, args)
    example = preprocess_instruction_data(data)
    return example


def load_hf_instruction_dataset(data, tokenizer, args, mode):
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
        lambda x: preprocess_batch_for_hf_instruction_dataset(x, tokenizer=tokenizer, args=args),
        batched=False, remove_columns=dataset.column_names
    ).filter(lambda x: len(x['input_ids']) > 0)

    return dataset


class LlamaInstructionDataset(Dataset):
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
            logger.debug(" Creating features from dataset file at %s" % args.cache_dir)

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
                            p.imap(preprocess_instruction_data, data, chunksize=chunksize),
                            total=len(data),
                            disable=args.silent,
                        )
                    )
            else:
                self.examples = [preprocess_instruction_data(d) for d in tqdm(data, disable=args.silent)]
            if not args.no_cache:
                logger.info(" Saving features into cached file %s" % cached_features_file)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])


def group_texts(examples, block_size: int = 1024):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def load_hf_pretraining_dataset(data, tokenizer, args, mode):
    if isinstance(data, str):
        if data.endswith('.json') or data.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=data)
        elif data.endswith('.txt'):
            dataset = load_dataset("text", data_files=data)
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

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer=tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on dataset",
    )
    grouped_datasets = tokenized_dataset.map(
        lambda x: group_texts(x, block_size=args.block_size),
        batched=True,
        desc=f"Grouping texts in chunks of {args.block_size}",
    ).filter(lambda x: len(x['input_ids']) > 0)
    dataset = grouped_datasets

    return dataset


class LlamaPretrainingDataset(Dataset):
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
            logger.debug(" Creating features from dataset file at %s" % args.cache_dir)
            dataset = load_hf_pretraining_dataset(data, tokenizer, args, mode)
            self.examples = list(dataset)
            if not args.no_cache:
                logger.info(" Saving features into cached file %s" % cached_features_file)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def preprocess_reward_function(examples, tokenizer):
    """
    Turn the dataset into pairs of Question + Answer, where input_ids_chosen is the preferred question + answer 
        and text_rejected is the other.
    """
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_j": [],
        "input_ids_rejected": [],
        "attention_mask_k": [],
    }
    for question, chosen, rejected in zip(examples["question"], examples["response_chosen"],
                                          examples["response_rejected"]):
        tokenized_chosen = tokenizer("Question: " + question + "\n\nAnswer: " + chosen, truncation=True)
        tokenized_rejected = tokenizer("Question: " + question + "\n\nAnswer: " + rejected, truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples


def load_hf_reward_dataset(data, tokenizer, args, mode):
    if isinstance(data, str):
        if data.endswith('.json') or data.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=data)
        elif data.endswith('.txt'):
            dataset = load_dataset("text", data_files=data)
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

    tokenized_dataset = dataset.map(
        lambda x: preprocess_reward_function(x, tokenizer=tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on dataset",
    )
    dataset = tokenized_dataset.filter(
        lambda x: len(x['input_ids_chosen']) > 0 and len(x['input_ids_rejected']) > 0 and len(
            x['input_ids_chosen']) <= args.max_length and len(x['input_ids_rejected']) <= args.max_length)

    return dataset


class LlamaRewardDataset(Dataset):
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
            logger.debug(" Creating features from dataset file at %s" % args.cache_dir)
            dataset = load_hf_reward_dataset(data, tokenizer, args, mode)
            self.examples = list(dataset)
            if not args.no_cache:
                logger.info(" Saving features into cached file %s" % cached_features_file)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        for feature in features:
            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_chosen["input_ids"],
            "attention_mask_j": batch_chosen["attention_mask"],
            "input_ids_k": batch_rejected["input_ids"],
            "attention_mask_k": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        return batch


def preprocess_reinforcement_learning_function(examples, tokenizer):
    """
    Preprocesses a single example in reinforcement learning dataset
    """
    new_examples = {
        "query": [],
        "input_ids": [],
    }
    for question in examples["question"]:
        query = "Question: " + question + "\n\nAnswer: "
        tokenized_question = tokenizer(query, truncation=True)
        new_examples["query"].append(query)
        new_examples["input_ids"].append(tokenized_question["input_ids"])

    return new_examples


def load_hf_reinforcement_learning_dataset(data, tokenizer, args, mode):
    if isinstance(data, str):
        if data.endswith('.json') or data.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=data)
        elif data.endswith('.txt'):
            dataset = load_dataset("text", data_files=data)
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

    tokenized_dataset = dataset.map(
        lambda x: preprocess_reinforcement_learning_function(x, tokenizer=tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on dataset",
    )
    dataset = tokenized_dataset.filter(lambda x: 0 < len(x['input_ids']) <= args.max_seq_length)
    dataset.set_format(type="torch")

    return dataset


class LlamaRLDataset(Dataset):
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
            logger.debug(" Creating features from dataset file at %s" % args.cache_dir)
            dataset = load_hf_reward_dataset(data, tokenizer, args, mode)
            self.examples = list(dataset)
            if not args.no_cache:
                logger.info(" Saving features into cached file %s" % cached_features_file)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
