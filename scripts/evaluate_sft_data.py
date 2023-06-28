# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import argparse
import json
import os
import time

import openai
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")
REQ_TIME_GAP = 3


def openai_reply(content, model_name, max_tokens, temperature):
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user", "content": content}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response = completion.choices[0].message["content"]
    return response


@retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(3))
def completion_with_backoff(**kwargs):
    # 重试间隔时间1到3秒，重试次数3
    return openai_reply(**kwargs)


def get_chatgpt_response(content, model_name, max_tokens, temperature):
    try:
        response = str(completion_with_backoff(content=content, model_name=model_name, max_tokens=max_tokens,
                                               temperature=temperature))
        logger.debug(f"Successfully get chatgpt response, content:{content}, res:{response}")
    except Exception as e:
        logger.error(e)
        response = ''
    time.sleep(REQ_TIME_GAP)
    return response


def save_jsonl(data_list, json_path):
    dir = os.path.dirname(os.path.abspath(json_path))
    if not os.path.exists(dir):
        print(dir)
        os.makedirs(dir)
    with open(json_path, 'w', encoding='utf-8') as f:
        for entry in data_list:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    print(f'save to {json_path}, size: {len(data_list)}')


def load_jsonl(json_path):
    json_list = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for json_str in f:
            json_list.append(json.loads(json_str))
    return json_list


def read_data(file_path):
    return [line for line in open(file_path, 'r', encoding='utf-8').readlines() if line]


def generate_prompt(data_list, prefix):
    prompts = []
    for data in data_list:
        prompt = prefix + f"{data['instruction']}\n{data['input']}\n模型回答：{data['output']}\n请针对模型回答给出得分，顺便给出理由："
        prompts.append(prompt)
    return prompts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='./seed_medical_sft_data.jsonl', type=str, help='Input data file')
    parser.add_argument('--output_file', default='./scores.jsonl', type=str, help='Output file')
    parser.add_argument('--model_name', default='gpt-3.5-turbo', type=str, help='gpt-3.5-turbo or gpt-4')
    parser.add_argument('--max_tokens', default=1024, type=int, help='Output max sequence length')
    parser.add_argument('--temperature', default=0.2, type=float, help='Number of training epochs')
    args = parser.parse_args()
    logger.info(args)
    data_list = load_jsonl(args.input_file)
    logger.info(f'data size: {len(data_list)}, first data: {data_list[0]}')
    eval_prompt = """你需要研究评价标准来对模型回答给出分数，满分为1分，最低分为0分。请按照"得分:"这样的形式输出分数。评价标准要求模型回答语句通顺，符合问题要求，同时是真实且没有恶意的。\n"""
    prompts = generate_prompt(data_list, eval_prompt)
    logger.debug(f"first prompt: {prompts[0]}")

    res = []
    try:
        for i, (data, c) in tqdm(enumerate(zip(data_list, prompts))):
            r = get_chatgpt_response(c, args.model_name, args.max_tokens, args.temperature)
            out_dict = {'instruction': data['instruction'], 'input': data['input'], 'output': data['output'],
                        'score': r}
            if r:
                res.append(out_dict)
    except KeyboardInterrupt:
        logger.warning('KeyboardInterrupt')
    save_jsonl(res, args.output_file)
