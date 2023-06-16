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
    parser.add_argument('--train_file', default='tatsu-lab/alpaca', type=str,
                        help='Dataset name (e.g. tatsu-lab/alpaca, shibing624/alpaca-zh, BelleGroup/train_1M_CN, '
                             'Chinese-Vicuna/guanaco_belle_merge_v1.0)')
    parser.add_argument('--model_type', default='llama', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='shibing624/chinese-alpaca-plus-7b-hf', type=str,
                        help='model name or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs-alpaca/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=256, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=256, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=20, type=float, help='Number of training epochs')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)
    model = None
    # fine-tune LLAMA model
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

        model.train_model(args.train_file)
    if args.do_predict:
        if model is None:
            model = GptModel(
                args.model_type, args.model_name,
                peft_name=args.output_dir,
                args={'use_peft': True, 'eval_batch_size': args.batch_size, "max_length": args.max_length, }
            )

        def generate_prompt(instruction):
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:{instruction}\n\n### Response: """

        sents = [
            '用一句话描述地球为什么是独一无二的。',
            '给定两个数字，计算它们的平均值。 数字: 25, 36\n',
            '基于以下提示填写以下句子的空格。 空格应填写一个形容词 句子: ______出去享受户外活动，包括在公园里散步，穿过树林或在海岸边散步。',
            '我能用lightning数据线给安卓手机充电吗？',
            '为什么天空是蓝色的？',
            '如何做披萨？',
            '为什么冥王星被踢出太阳系？',
            '列举太阳系的全部行星',
            '详细说明DNA和RNA的区别',
            '中国的“东北三省”指的是哪里？',
            '经常吃烫的东西会罹患什么病？',
            '盐酸莫西沙星能否用于治疗肺炎？',
            '机场代码KIX代表的是哪个机场？',
            '给出三个保持健康的秘诀',
            '给定一篇文章，纠正里面的语法错误。\n我去年很喜欢在公园里跑步，但因为最近天气太冷所以我不去了。',
        ]
        prompt_sents = [generate_prompt(sent) for sent in sents]
        response = model.predict(prompt_sents)
        print(response)


if __name__ == '__main__':
    main()
