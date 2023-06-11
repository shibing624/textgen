# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import pickle
import re
from multiprocessing import Pool

import datasets
from datasets import Dataset as HFDataset
from datasets import load_dataset
from loguru import logger
from torch.utils.data import Dataset
from tqdm.auto import tqdm

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
        instruction = instruction.replace('Human:', '### Human: ')
        instruction = instruction.replace('Assistant:', '### Assistant: ')
        prompt = PROMPT_DICT['prompt_multi_round_no_input'].format(instruction=instruction, output_text=output_text)
        return prompt, 'multi_round'
    else:
        if input_text:
            prompt = PROMPT_DICT["prompt_input"].format(instruction=instruction, input_text=input_text)
        else:
            prompt = PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
        return prompt, 'single_round'


def preprocess_instruction_data(data):
    instruction, input_text, target_text, tokenizer, args = data
    IGNORE_INDEX = -100
    EOS_TOKEN = tokenizer.eos_token

    prompt, round_type = generate_prompt(instruction, input_text, target_text)
    if round_type == 'multi_round':
        prompt = re.sub(r'(?<!\n)\n### ', f'\n{EOS_TOKEN}### ', prompt)
        prompt += EOS_TOKEN
        example = tokenizer(prompt, return_offsets_mapping=True)
        labels = example['input_ids'].copy()
        source_len = len(tokenizer(PROMPT_DICT['prompt_multi_round_no_input'].split('\n\n')[0] + '\n\n')['input_ids'])
        labels[:source_len] = [IGNORE_INDEX] * source_len
        offsets = example["offset_mapping"]

        matches = re.finditer(r'### (?!Assistant:)(.*?)<\/s>', prompt, re.DOTALL)
        for match in matches:
            start_pos, end_pos = match.span()
            start_idx = None
            end_idx = None

            for i, (start, end) in enumerate(offsets):
                if start <= start_pos < end:
                    start_idx = i
                if start <= end_pos < end:
                    end_idx = i

            if start_idx is not None and end_idx is not None:
                for i in range(start_idx, end_idx - 1):
                    labels[i] = -100

        example['labels'] = labels
        return example
    else:
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
