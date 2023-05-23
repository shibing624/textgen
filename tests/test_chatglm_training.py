# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import pytest
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk

sys.path.append('..')
from textgen import ChatGlmModel


def preprocess_batch_for_hf_dataset(example, tokenizer, args):
    instruction, input_text, target_text = example["instruction"], example["input"], example["output"]
    prompt = f"{instruction}\n答："
    prompt_ids = tokenizer.encode(prompt, max_length=args.max_seq_length)
    target_ids = tokenizer.encode(target_text, max_length=args.max_length,
                                  add_special_tokens=False)
    input_ids = prompt_ids + target_ids
    input_ids = input_ids[:(args.max_seq_length + args.max_length)] + [tokenizer.eos_token_id]

    example['input_ids'] = input_ids
    return example


class MyDataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        if data.endswith('.json') or data.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=data)
        elif os.path.isdir(data):
            dataset = load_from_disk(data)
        else:
            dataset = load_dataset(data)
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        dataset = dataset["train"]
        dataset = dataset.map(
            lambda x: preprocess_batch_for_hf_dataset(x, tokenizer, args),
            batched=False, remove_columns=dataset.column_names
        )
        dataset.set_format(type="np", columns=["input_ids"])

        self.examples = dataset["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def test_train_name():
    model = ChatGlmModel(
        "chatglm", "THUDM/chatglm-6b",
        args={
            "dataset_class": MyDataset,
            "use_peft": True,
            "max_seq_length": 128,
            "max_length": 128,
            "per_device_train_batch_size": 8,
            "eval_batch_size": 8,
            "num_train_epochs": 20,
            "save_steps": 50,
            "output_dir": "tmp_outputs",
        }
    )
    model.train_model('instruction_name.json', eval_data='instruction_name.json')

    sents = ['我要开一家美妆店，帮我起一个店铺名\n答：']
    response = model.predict(sents)
    print(response)
    assert len(response) > 0


def test_second_predict():
    model = ChatGlmModel("chatglm", "THUDM/chatglm-6b",
                         args={"use_peft": True}, peft_name='tmp_outputs')
    # load model from peft_name is equal to load model from output_dir
    sents = ['我要开一家美妆店，帮我起一个店铺名\n答：']
    response = model.predict(sents)
    print(response)
    assert len(response) > 0


test_train_name()
test_second_predict()
