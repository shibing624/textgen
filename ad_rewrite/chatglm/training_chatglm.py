# -*- coding: utf-8 -*-
import sys
import argparse
from loguru import logger
import os
import pandas as pd

sys.path.append('../..')
from textgen import ChatGlmModel


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

def load_predict_data(file_path,industry_name):
    '''
    加载预测数据，不同行业的输入文件格式不一样。
    '''
    data = []
    if industry_name == "线上服务":
        with open(file_path, 'r', encoding='utf-8') as f:
            index = 0
            for line in f:
                index+=1
                if index==1:
                    continue
                terms = line.strip('\n').split('\t')
                if len(terms)!=2:
                    continue
                app, app_desc = terms
                instruction = '根据以下提供的信息，生成搜索广告标题和描述：'
                input_data = "app："+app +"\n"+"app描述："+app_desc
                data.append([instruction, input_data])
    return data

def load_online_service_data(file_path):
    max_input_length = 0
    max_out_length = 0
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            terms = line.strip('\n').split('\t')
            if len(terms)!=4:
                continue
            app, app_desc, gen_title, gen_desc = terms
            instruction = '根据以下提供的信息，生成搜索广告标题和描述：'
            input_data = "app："+app +"\n"+"app描述："+app_desc
            output_data = "标题："+gen_title +"\n"+ "描述：" + gen_desc
            data.append([instruction, input_data, output_data])
            max_input_length = max(len(instruction+input_data),max_input_length)
            max_out_length = max(len(output_data),max_out_length)
    print("max_input_length:{}".format(max_input_length))
    print("max_out_length:{}".format(max_out_length))
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='../data/zh_csc_train.tsv', type=str, help='Training data file')
    parser.add_argument('--test_file', default='../data/zh_csc_test.tsv', type=str, help='Test data file')
    parser.add_argument('--predict_file', default='../data/zh_csc_test.tsv', type=str, help='Predict data file')
    parser.add_argument('--model_type', default='chatglm', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='THUDM/chatglm-6b', type=str, help='Transformers model or path')
    parser.add_argument('--lora_name', default=None, type=str, help='Peft lora model name or dir')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_test', action='store_true', help='Whether to run test.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=128, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=128, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=0.2, type=float, help='Number of training epochs')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)
    model = None
    # fine-tune chatGLM model
    if args.do_train:
        logger.info('Loading data...')
        model_args = {
            "use_lora": True,
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "max_length": args.max_length,
            "per_device_train_batch_size": args.batch_size,
            "eval_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "output_dir": args.output_dir,
            "resume_from_checkpoint": args.output_dir,
        }
        model = ChatGlmModel(args.model_type, args.model_name, lora_name=args.lora_name, args=model_args)
        train_data = load_online_service_data(args.train_file)
        logger.debug('train_data: {}'.format(train_data[:10]))
        train_df = pd.DataFrame(train_data, columns=["instruction", "input", "output"])
        model.train_model(train_df)
    if args.do_test:
        if model is None:
            model = ChatGlmModel(
                args.model_type, args.model_name,
                lora_name=args.lora_name,
                args={'use_lora': True, 'eval_batch_size': args.batch_size,
                      "max_length": args.max_length, }
            )
        test_data = load_online_service_data(args.test_file)
        test_df = pd.DataFrame(test_data, columns=["instruction", "input", "output"])
        logger.debug('test_df: {}'.format(test_df))

        def get_prompt(arr):
            if arr['input'].strip():
                return f"问：{arr['instruction']}\n{arr['input']}\n答："
            else:
                return f"问：{arr['instruction']}\n答："

        test_df['prompt'] = test_df.apply(get_prompt, axis=1)
        test_df['predict_after'] = model.predict(test_df['prompt'].tolist())
        logger.debug('test_df result: {}'.format(test_df[['output', 'predict_after']]))
        out_df = test_df[['instruction', 'input', 'output', 'predict_after']]
        out_df.to_json('test_result.json', force_ascii=False, orient='records', lines=True)
    
    if args.do_predict:
        if model is None:
            model = ChatGlmModel(
                args.model_type, args.model_name,
                lora_name=args.lora_name,
                args={'use_lora': True,  # 加载的训练好的lora模型，代码里优先加载args.output_dir+"/"+lora_bin_name(默认是adapter_model.bin）； 如果没有，则加载args.lora_name指定的lora模型
                     'eval_batch_size': args.batch_size,
                      "max_length": args.max_length, 
                      "do_sample":True, "temperature":0.9, "top_p":0.7, # do_sample=True: 使用随机采样; do_sample=False：使用贪心搜索/beam search
                      "num_beams":2, "num_return_sequences": 2  # num_beams=1:使用贪心搜索； num_beams>1：使用beam search  num_return_sequences必须<=num_beams
                      } 
            )
        predict_data = load_predict_data(args.predict_file, industry_name="线上服务")[:10]
        predict_df = pd.DataFrame(predict_data, columns=["instruction", "input"])
        logger.debug('predict_df: {}'.format(predict_df))

        def get_prompt(arr):
            if arr['input'].strip():
                return f"{arr['instruction']}\n{arr['input']}\n"
            else:
                return f"{arr['instruction']}\n"

        predict_df['prompt'] = predict_df.apply(get_prompt, axis=1)
        predict_df['predict_after'] = model.predict(predict_df['prompt'].tolist())
        logger.debug('test_df result: {}'.format(predict_df[['predict_after']]))
        out_df = predict_df[['instruction', 'input', 'predict_after']]
        out_df.to_json(args.output_dir+"/"+'predict_result.json', force_ascii=False, orient='records', lines=True)


        # response, history = model.chat("给出三个保持健康的秘诀。", history=[])
        # print(response)
        # response, history = model.chat(
        #     "给定一篇文章，纠正里面的语法错误。\n我去年很喜欢在公园里跑步，但因为最近天气太冷所以我不去了。\n",
        #     history=history)
        # print(response)


if __name__ == '__main__':
    main()
