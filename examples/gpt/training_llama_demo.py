# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys

from loguru import logger

sys.path.append('../..')
from textgen import GptModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='../data/sharegpt_zh_100_format.jsonl', type=str, help='Train file')
    parser.add_argument('--test_file', default='../data/sharegpt_zh_100_format.jsonl', type=str, help='Test file')
    parser.add_argument('--model_type', default='llama', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='shibing624/chinese-alpaca-plus-7b-hf', type=str,
                        help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--bf16', action='store_true', help='Whether to use bf16 mixed precision training.')
    parser.add_argument('--output_dir', default='./outputs-llama-demo/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=128, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=128, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=0.2, type=float, help='Number of training epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--eval_steps', default=50, type=int, help='Eval every X steps')
    parser.add_argument('--save_steps', default=50, type=int, help='Save checkpoint every X steps')
    args = parser.parse_args()
    logger.info(args)
    model = None
    # fine-tune Llama model
    if args.do_train:
        logger.info('Loading data...')
        model_args = {
            "use_peft": True,
            "overwrite_output_dir": True,
            "reprocess_input_data": True,
            "max_seq_length": args.max_seq_length,
            "max_length": args.max_length,
            "per_device_train_batch_size": args.batch_size,
            "eval_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "output_dir": args.output_dir,
            "resume_from_checkpoint": args.output_dir,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "bf16": args.bf16,
        }
        model = GptModel(args.model_type, args.model_name, args=model_args)
        model.train_model(args.train_file)
    if args.do_predict:
        if model is None:
            model = GptModel(
                args.model_type, args.model_name,
                peft_name=args.output_dir,
                args={'use_peft': True, 'eval_batch_size': args.batch_size, "max_length": args.max_length, }
            )

        response = model.predict(["介绍下北京"])
        print(response)

        # Chat model with multi turns conversation
        response, history = model.chat('What is the sum of 1 and 2?')
        print(response)
        response, history = model.chat('what is the multiplication result of two num? please think step by step.',
                                       history=history)
        print(response)


if __name__ == '__main__':
    main()
