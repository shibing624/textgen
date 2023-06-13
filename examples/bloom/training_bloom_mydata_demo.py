# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from loguru import logger

sys.path.append('../..')
from textgen import BloomModel


def load_data(data_dir):
    path = Path(data_dir)
    files = [os.path.join(path, file.name) for file in path.glob("*.json")]
    logger.info(f"training files: {' '.join(files)}")
    all_datasets = []
    for file in files:
        raw_dataset = load_dataset("json", data_files=file)
        all_datasets.append(raw_dataset["train"])
    all_datasets = concatenate_datasets(all_datasets)
    logger.debug(f"all_datasets size:{all_datasets}, first line: {next(iter(all_datasets))}")
    return all_datasets.to_pandas()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', default='../data/json_files/', type=str, help='Train data file')
    parser.add_argument('--test_data_dir', default='../data/json_files/', type=str, help='Test data file')
    parser.add_argument('--model_type', default='bloom', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='bigscience/bloomz-560m', type=str, help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs-multi-round-v1/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=128, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=128, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=1.0, type=float, help='Number of training epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)
    model = None
    if args.do_train:
        logger.info('Loading data...')
        model_args = {
            'use_peft': True,
            "overwrite_output_dir": True,
            "reprocess_input_data": True,
            "max_seq_length": args.max_seq_length,
            "max_length": args.max_length,
            "per_device_train_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "output_dir": args.output_dir,
        }
        model = BloomModel(args.model_type, args.model_name, args=model_args)
        train_df = load_data(args.train_data_dir)
        logger.debug('train_data: {}'.format(train_df))
        eval_df = train_df[:10]
        train_df = train_df[10:]
        model.train_model(train_df, eval_data=eval_df)
    if args.do_predict:
        if model is None:
            model = BloomModel(
                args.model_type, args.model_name,
                peft_name=args.output_dir,
                args={'use_peft': True, 'eval_batch_size': args.batch_size, "max_length": args.max_length, }
            )

        def generate_prompt(instruction):
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: """

        response = model.predict([generate_prompt("给出三个保持健康的秘诀。")])
        print(response)
        response = model.predict(["介绍下北京"], add_system_prompt=True)
        print(response)

        # Chat with model
        response, history = model.chat('What is the sum of 1 and 2?', add_system_prompt=True)
        print(response)
        response, history = model.chat('what is the multiplication result of two num? please think step by step.',
                                       history=history, add_system_prompt=True)
        print(response)


if __name__ == '__main__':
    main()
