# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import argparse
from loguru import logger

sys.path.append('../..')
from textgen import ChatGlmModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='tatsu-lab/alpaca', type=str,
                        help='Dataset name (e.g. tatsu-lab/alpaca, shibing624/alpaca-zh, BelleGroup/train_1M_CN, '
                             'Chinese-Vicuna/guanaco_belle_merge_v1.0)')
    parser.add_argument('--model_type', default='chatglm', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='THUDM/chatglm-6b', type=str, help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs-alpaca/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=256, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=256, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=1.0, type=float, help='Number of training epochs')
    parser.add_argument('--batch_size', default=3, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)
    model = None
    # fine-tune chatGLM model
    if args.do_train:
        logger.info('Loading data...')
        model_args = {
            'use_lora': True,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "max_length": args.max_length,
            "per_device_train_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "output_dir": args.output_dir,
            "use_hf_datasets": True,
        }
        model = ChatGlmModel(args.model_type, args.model_name,  args=model_args)

        model.train_model(args.train_file)
    if args.do_predict:
        if model is None:
            model = ChatGlmModel(
                args.model_type, args.model_name,
                args={'use_lora': True, 'eval_batch_size': args.batch_size,
                      'output_dir': args.output_dir, "max_length": args.max_length, }
            )
        sents = [
            '问：用一句话描述地球为什么是独一无二的。\n答：',
            '问：给定两个数字，计算它们的平均值。 数字: 25, 36\n答：',
            '问：基于以下提示填写以下句子的空格。 空格应填写一个形容词 句子: ______出去享受户外活动，包括在公园里散步，穿过树林或在海岸边散步。\n答：',
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
        ]
        response = model.predict(sents)
        print(response)
        response, history = model.chat("给出三个保持健康的秘诀。", history=[])
        print(response)
        response, history = model.chat(
            "给定一篇文章，纠正里面的语法错误。\n我去年很喜欢在公园里跑步，但因为最近天气太冷所以我不去了。\n",
            history=history)
        print(response)
        del model

        ref_model = ChatGlmModel(args.model_type, args.model_name,
                                 args={'use_lora': False, 'eval_batch_size': args.batch_size})
        response = ref_model.predict(sents)
        print(response)
        response, history = ref_model.chat("给出三个保持健康的秘诀。", history=[])
        print(response)
        response, history = ref_model.chat(
            "给定一篇文章，纠正里面的语法错误。\n我去年很喜欢在公园里跑步，但因为最近天气太冷所以我不去了。\n",
            history=history)
        print(response)


if __name__ == '__main__':
    main()
