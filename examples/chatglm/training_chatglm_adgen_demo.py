# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import os
import argparse
from loguru import logger
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset

sys.path.append('../..')
from textgen import ChatGlmModel


def preprocess_batch_for_hf_dataset(example, tokenizer, args):
    input_text, target_text = example["content"], example["summary"]
    instruction = '改写为电商广告文案：'
    prompt = f"问：{instruction}\n{input_text}\n答："
    prompt_ids = tokenizer.encode(prompt, max_length=args.max_seq_length)
    target_ids = tokenizer.encode(target_text, max_length=args.max_length,
                                  add_special_tokens=False)
    input_ids = prompt_ids + target_ids
    input_ids = input_ids[:(args.max_seq_length + args.max_length)] + [tokenizer.eos_token_id]

    example['input_ids'] = input_ids
    return example


class AdgDataset(Dataset):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default="shibing624/AdvertiseGen", type=str,
                        help='Datasets name, eg:shibing624/AdvertiseGen')
    parser.add_argument('--model_type', default='chatglm', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='THUDM/chatglm-6b', type=str, help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs-adg/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=128, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=128, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=0.2, type=float, help='Number of training epochs')
    parser.add_argument('--batch_size', default=3, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)
    model = None
    # fine-tune chatGLM model
    if args.do_train:
        logger.info('Loading data...')
        model_args = {
            "dataset_class": AdgDataset,
            'use_lora': True,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "max_length": args.max_length,
            "per_device_train_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "output_dir": args.output_dir,
        }
        model = ChatGlmModel(args.model_type, args.model_name, args=model_args)

        model.train_model(args.train_file)
    if args.do_predict:
        if model is None:
            model = ChatGlmModel(
                args.model_type, args.model_name,
                args={'use_lora': True, 'eval_batch_size': args.batch_size,
                      'output_dir': args.output_dir, "max_length": args.max_length, }
            )
        sents = ['改写为电商广告文案：\n类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞\n答：', ]
        response = model.predict(sents)
        print(response)


if __name__ == '__main__':
    main()
