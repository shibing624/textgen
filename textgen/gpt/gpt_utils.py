# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import pickle
import re

import datasets
from datasets import Dataset as HFDataset
from datasets import load_dataset
from loguru import logger
from torch.utils.data import Dataset

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n{input_text}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
    "prompt_multi_round_no_input": (
        "Below is an multi-round dialogue between human and assistant. "
        "Write a response as an assistant that appropriately completes the human request in each round by incorporating previous context.\n\n"
        "{instruction}{output_text}"
    ),
}


def generate_prompt(instruction, input_text, output_text):
    """Generate prompt for instruction."""
    if 'Human:' in instruction and 'Assistant:' in instruction:
        instruction = instruction.replace('Human:', '### Human:')
        instruction = instruction.replace('Assistant:', '### Assistant:')
        prompt = PROMPT_DICT['prompt_multi_round_no_input'].format(instruction=instruction, output_text=output_text)
        return prompt, 'multi_round'
    else:
        if input_text:
            prompt = PROMPT_DICT["prompt_input"].format(instruction=instruction, input_text=input_text)
        else:
            prompt = PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
        return prompt, 'single_round'


def preprocess_data(data):
    instruction, input_text, target_text, tokenizer, args = data
    IGNORE_INDEX = -100
    EOS_TOKEN = tokenizer.eos_token
    full_max_length = args.max_seq_length + args.max_length

    prompt, round_type = generate_prompt(instruction, input_text, target_text)
    if round_type == 'multi_round':
        prompt = re.sub(r'(?<!\n)\n### ', f'\n{EOS_TOKEN}### ', prompt)
        prompt += EOS_TOKEN
        example = tokenizer(prompt, max_length=full_max_length, truncation=True, padding=False)
        labels = example['input_ids'].copy()

        if not args.is_train_on_prompt:
            source_len = len(tokenizer(
                PROMPT_DICT['prompt_multi_round_no_input'].split('\n\n')[0] + '\n\n')['input_ids'])
            labels[:source_len] = [IGNORE_INDEX] * source_len
            input_tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
            matches = re.finditer(r'### (?!Assistant:)(.*?)</s>', prompt, re.DOTALL)
            for match in matches:
                start_pos, end_pos = match.span()
                start_idx = None
                end_idx = None
                current_pos = 0
                current_idx = 0

                while current_pos < start_pos:
                    current_pos += len(input_tokens[current_idx]) + 1
                    current_idx += 1
                start_idx = current_idx

                while current_pos < end_pos:
                    current_pos += len(input_tokens[current_idx]) + 1
                    current_idx += 1
                end_idx = current_idx - 1

                if start_idx is not None and end_idx is not None:
                    for i in range(start_idx, end_idx - 1):
                        labels[i] = IGNORE_INDEX
        # Padding labels to full max length
        example['labels'] = [IGNORE_INDEX] * (full_max_length - len(labels)) + labels
    else:
        full_prompt = prompt + target_text + tokenizer.eos_token
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
            # Padding labels to full max length to equalize the length of input_ids after collator
            example["labels"] = [IGNORE_INDEX] * (full_max_length - len(example['labels']) + user_prompt_len) + \
                                example["labels"][user_prompt_len:]
    return {"input_ids": example['input_ids'], "labels": example["labels"]}


def preprocess_batch_for_hf_instruction_dataset(example, tokenizer, args):
    data = (example["instruction"], example["input"], example["output"], tokenizer, args)
    example = preprocess_data(data)
    return example


def load_hf_instruction_dataset(tokenizer, args, data, mode):
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


class InstructionDataset(Dataset):
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

            self.examples = list(load_hf_instruction_dataset(tokenizer, args, data, mode))
            if not args.no_cache:
                logger.info(" Saving features into cached file %s" % cached_features_file)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
