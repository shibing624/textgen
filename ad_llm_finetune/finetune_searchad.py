# -*- coding: utf-8 -*-
"""
@description: 广告数据，指令微调
"""
import sys
import argparse
from loguru import logger
import pandas as pd
import os
import torch
pwd_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd_path, '..'))
sys.path.append('..')
from textgen import ChatGlmModel
import random


def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            terms = line.split('\t')
            if len(terms) == 3:
                instruction = terms[0].replace('\\n', '\n')
                input = terms[1].replace('\\n', '\n')
                output = terms[2].replace('\\n', '\n')
                data.append([instruction, input, output])
            else:
                logger.warning(f'line error: {line}')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='/apdcephfs_cq3/share_2973545/data/dataset/llm_sogou_ad/llm_sogou_ad_train.txt', type=str, help='Training data file')
    parser.add_argument('--test_file', default='/apdcephfs_cq3/share_2973545/data/dataset/llm_sogou_ad/llm_sogou_ad_test.txt', type=str, help='Test data file')
    parser.add_argument('--eval_file', default='/apdcephfs_cq3/share_2973545/data/dataset/llm_sogou_ad/llm_sogou_ad_eval.txt', type=str, help='Eval data file')
    parser.add_argument('--model_type', default='chatglm', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='/apdcephfs_cq3/share_2973545/data/models/THUDM-chatglm-6b', type=str, help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--do_debug', action='store_true', help='Whether to run debug.')
    parser.add_argument('--is_train_on_prompt', action='store_true', help='Whether to compute loss on prompt')
    parser.add_argument('--output_dir', default='./outputs/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=256, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=2048, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=1, type=float, help='Number of training epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--eval_steps', default=500, type=int, help='Eval every X steps')
    parser.add_argument('--save_steps', default=500, type=int, help='Save checkpoint every X steps')
    #parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    logger.info(args)
    #torch.cuda.set_device(args.local_rank)
    model = None
    # fine-tune ChatGlmModel
    if args.do_train:
        logger.info('Loading data...')
        model_args = {
            "use_peft": True,
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "max_length": args.max_length,
            "per_device_train_batch_size": args.batch_size,
            "eval_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "is_train_on_prompt": args.is_train_on_prompt,
            "output_dir": args.output_dir,
            "resume_from_checkpoint": args.output_dir,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
        }
        model = ChatGlmModel(args.model_type, args.model_name, args=model_args)
        train_data = load_data(args.train_file)
        logger.debug('train_data: {}'.format(train_data[:10]))
        train_df = pd.DataFrame(train_data, columns=["instruction", "input", "output"])
        eval_data = load_data(args.eval_file)
        logger.debug('eval_data: {}'.format(eval_data[:10]))
        eval_df = pd.DataFrame(eval_data, columns=["instruction", "input", "output"])
        model.train_model(train_df, eval_data=eval_df)
    if args.do_predict:
        if model is None:
            model = ChatGlmModel(
                args.model_type, args.model_name,
                args={'use_peft': True, 'eval_batch_size': args.batch_size,
                      'output_dir': args.output_dir, "max_length": args.max_length, }
            )
        #test_data = load_data(args.test_file)[:8]
        test_data = random.sample(load_data(args.test_file), 8)
        test_df = pd.DataFrame(test_data, columns=["instruction", "input", "output"])
        logger.debug('test_df: {}'.format(test_df))

        def get_prompt(arr):
            if arr['input'].strip():
                return f"{arr['instruction']}\n{arr['input']}\n"
            else:
                return f"{arr['instruction']}\n"

        test_df['prompt'] = test_df.apply(get_prompt, axis=1)
        test_df['predict_after'] = model.predict(test_df['prompt'].tolist())
        logger.debug('test_df result: {}'.format(test_df[['output', 'predict_after']]))
        out_df = test_df[['instruction', 'input', 'output', 'predict_after']]
        #out_df.to_json('test_result.json', force_ascii=False, orient='records', lines=True)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
        print(test_df)

        response, history = model.chat("给出三个保持健康的秘诀。", history=[])
        print(response)
        response, history = model.chat(
            "给定一篇文章，纠正里面的语法错误。\n我去年很喜欢在公园里跑步，但因为最近天气太冷所以我不去了。\n",
            history=history)
        print(response)
    if args.do_debug:
        if model is None:
            model = ChatGlmModel(
                args.model_type, args.model_name, args.peft_name,
                args={'use_peft': True, 'eval_batch_size': args.batch_size,
                      'output_dir': args.output_dir, "max_length": args.max_length, }
            )

        prompt = ["对于给定的搜索query，生成2条低点击率的广告\nquery：注册会计师",
                  "对于给定的搜索query，生成4条高点击率的广告\nquery：注册会计师",
                  "对于给定的搜索query，生成4条高点击率的广告\nquery：淘宝",
                  "对于给定的搜索query，生成1条低点击率的广告\nquery：淘宝",
                  "对于给定的搜索query，生成5条高点击率的广告\nquery：微信",
                  "对于给定的搜索query，生成3条低点击率的广告\nquery：微信",
                  "对于给定的搜索query，生成4条高点击率的广告\nquery：轮胎生产日期怎么看",
                  "对于给定的搜索query，生成3条低点击率的广告\nquery：轮胎生产日期怎么看"]
        r = model.predict(prompt)
        print(prompt)
        print(r)  
        print("\n")

        response, history = model.chat("给出三个保持健康的秘诀。", history=[])
        print(response)
        response, history = model.chat(
            "给定一篇文章，纠正里面的语法错误。\n我去年很喜欢在公园里跑步，但因为最近天气太冷所以我不去了。\n",
            history=history)
        print(response)


if __name__ == '__main__':
    main()
