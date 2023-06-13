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
from textgen import ChatGlmModel


def preprocess_function(example):
    example['instruction'] = '改写为电商广告文案：'
    example['input'] = example["content"]
    example['output'] = example["summary"]
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
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
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
            "eval_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "output_dir": args.output_dir,
            "resume_from_checkpoint": args.output_dir,
        }
        model = ChatGlmModel(args.model_type, args.model_name, args=model_args)
        train_df = load_data(args.train_file)
        logger.debug('train_data: {}'.format(train_df))
        eval_df = train_df[:10]
        train_df = train_df[10:]
        model.train_model(train_df, eval_data=eval_df)
    if args.do_predict:
        if model is None:
            model = ChatGlmModel(
                args.model_type, args.model_name,
                peft_name=args.output_dir,
                args={'use_peft': True, 'eval_batch_size': args.batch_size, "max_length": args.max_length}
            )
        sents = [
            '改写为电商广告文案：\n类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞\n答：', ]
        response = model.predict(sents)
        print(response)


if __name__ == '__main__':
    main()
