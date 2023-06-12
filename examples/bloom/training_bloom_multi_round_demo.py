# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys

from loguru import logger

sys.path.append('../..')
from textgen import BloomModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default="../data/multi_round_alpaca.json", type=str, help='Train file')
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

        model.train_model(args.train_file, eval_data=args.train_file)
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
        response = model.predict([generate_prompt("介绍下北京")])
        print(response)

        response, history = model.chat('what is your name?', add_system_prompt=True)
        print(response)
        response, history = model.chat('how to spell it', history=history, add_system_prompt=True)
        print(response)


if __name__ == '__main__':
    main()
