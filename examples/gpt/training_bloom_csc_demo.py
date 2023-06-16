# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import os
import sys

from datasets import load_dataset, load_from_disk
from loguru import logger

sys.path.append('../..')
from textgen import GptModel


def preprocess_function(example):
    original_text, wrong_ids, correct_text = example["original_text"], example["wrong_ids"], example["correct_text"]
    logger.info('original_text:{}, wrong_ids:{}, correct_text:{}'.format(original_text, wrong_ids, correct_text))
    example['instruction'] = '对下面中文拼写纠错：'
    example['input'] = original_text
    example['output'] = correct_text + '\n错误字：' + '，'.join([correct_text[i] for i in wrong_ids])
    return example


def load_data(data):
    if data.endswith('.json') or data.endswith('.jsonl'):
        dataset = load_dataset("json", data_files=data)
    elif os.path.isdir(data):
        dataset = load_from_disk(data)
    else:
        dataset = load_dataset(data)
    dataset = dataset["train"]
    dataset = dataset.map(preprocess_function, batched=False, remove_columns=dataset.column_names)
    return dataset.to_pandas()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default="shibing624/CSC", type=str, help='Train file')
    parser.add_argument('--model_type', default='bloom', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='bigscience/bloomz-560m', type=str, help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs-csc-v1/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=128, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=128, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=1.0, type=float, help='Number of training epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)
    model = None
    # fine-tune chatGLM model
    if args.do_train:
        logger.info('Loading data...')
        model_args = {
            'use_peft': True,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "max_length": args.max_length,
            "per_device_train_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "output_dir": args.output_dir,
        }
        model = GptModel(args.model_type, args.model_name, args=model_args)
        train_df = load_data(args.train_file)
        logger.debug('train_data: {}'.format(train_df))
        eval_df = train_df[:10]
        train_df = train_df[10:]
        model.train_model(train_df, eval_data=eval_df)
    if args.do_predict:
        if model is None:
            model = GptModel(
                args.model_type, args.model_name,
                peft_name=args.output_dir,
                args={'use_peft': True, 'eval_batch_size': args.batch_size, "max_length": args.max_length, }
            )

        def generate_prompt(q):
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n对下面中文拼写纠错：\n{q}\n### Response: """

        sents = ['少先队员因该为老人让坐。',
                 '下个星期，我跟我朋唷打算去法国玩儿。']
        response = model.predict([generate_prompt(q) for q in sents])
        print(response)


if __name__ == '__main__':
    main()
