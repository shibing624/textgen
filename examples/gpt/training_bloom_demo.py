# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys

import pandas as pd
from loguru import logger

sys.path.append('../..')
from textgen import GptModel

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            terms = line.split('\t')
            instruction = '对下面中文拼写纠错：'
            if len(terms) == 2:
                data.append([instruction, terms[0], terms[1]])
            else:
                logger.warning(f'line error: {line}')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='../data/zh_csc_train.tsv', type=str, help='Training data file')
    parser.add_argument('--test_file', default='../data/zh_csc_test.tsv', type=str, help='Test data file')
    parser.add_argument('--model_type', default='bloom', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='bigscience/bloomz-560m', type=str, help='Model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=128, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=128, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=1, type=float, help='Number of training epochs')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)
    model = None
    # fine-tune GptModel model
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
        }
        model = GptModel(args.model_type, args.model_name, args=model_args)
        train_data = load_data(args.train_file)
        logger.debug('train_data: {}'.format(train_data[:10]))
        train_df = pd.DataFrame(train_data, columns=["instruction", "input", "output"])
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
        test_data = load_data(args.test_file)[:10]
        test_df = pd.DataFrame(test_data, columns=["instruction", "input", "output"])
        logger.debug('test_df: {}'.format(test_df))

        def get_prompt(arr):
            if arr['input'].strip():
                return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{arr['instruction']}\n### Input:\n{arr['input']}\n\n### Response:"""
            else:
                return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{arr['instruction']}\n\n### Response:"""

        test_df['prompt'] = test_df.apply(get_prompt, axis=1)
        test_df['predict_after'] = model.predict(test_df['prompt'].tolist())
        logger.debug('test_df result: {}'.format(test_df[['output', 'predict_after']]))
        out_df = test_df[['instruction', 'input', 'output', 'predict_after']]
        out_df.to_json('test_result.json', force_ascii=False, orient='records', lines=True)

        def generate_prompt(instruction):
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: """

        response = model.predict([generate_prompt("给出三个保持健康的秘诀。")])
        print(response)
        response = model.predict([generate_prompt("介绍下北京")])
        print(response)


if __name__ == '__main__':
    main()
