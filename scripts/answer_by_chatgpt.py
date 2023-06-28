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
MAX_API_RETRY = 2


def openai_reply(content, model_name):
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user", "content": content}
        ],
        temperature=1.0,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return completion.choices[0].message.content


@retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(3))
def completion_with_backoff(**kwargs):
    # 重试间隔时间1到3秒，重试次数3
    return openai_reply(**kwargs)


def get_chatgpt_response(content, model_name):
    try:
        logger.debug(f"input openai_reply, content:{content}")
        response = str(completion_with_backoff(content=content, model_name=model_name))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='./medical_question.txt', type=str, help='Input data file')
    parser.add_argument('--output_file', default='./medical_question_result.jsonl', type=str, help='Output file')
    parser.add_argument('--model_name', default='gpt-3.5-turbo', type=str, help='gpt-3.5-turbo or gpt-4')
    args = parser.parse_args()
    logger.info(args)
    data_list = read_data(args.input_file)
    logger.info(f'data size: {len(data_list)}, first data: {data_list[0]}')
    prompt = """你是一个专业的医生，请基于专业知识，准确并认真的回答以下问题：\n"""
    prompts = [prompt + q for q in data_list]
    print('first prompt:', prompts[0])

    res = []
    try:
        for c in tqdm(prompts):
            r = get_chatgpt_response(c, args.model_name)
            out_dict = {'instruction': c, 'input': '', 'output': r}
            if r:
                res.append(out_dict)
    except KeyboardInterrupt:
        logger.warning('KeyboardInterrupt')
    save_jsonl(res, args.output_file)
