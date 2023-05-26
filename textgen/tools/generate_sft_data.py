# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import argparse
import json
import os
import re
import time
from multiprocessing.dummy import Pool
from random import choices

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


def parse_response(responses):
    data_list = []
    response_list = [i for i in re.split("###", responses) if i]
    for response in response_list:
        intruction_pattern = re.compile(
            r"(?<=(?:" + '|'.join(['指令:', '指令：']) + "))[\s\S]*?(?=" + '|'.join(['输入:', '输入：']) + ")")
        input_pattern = re.compile(
            r"(?<=(?:" + '|'.join(['输入:', '输入：']) + "))[\s\S]*?(?=" + '|'.join(['输出:', '输出：']) + ")")
        output_pattern = re.compile(r"(?<=(?:" + '|'.join(['输出:', '输出：']) + "))[\s\S]*?(?=$)")
        intruction_match = intruction_pattern.search(response)
        input_match = input_pattern.search(response)
        output_match = output_pattern.search(response)
        if intruction_match and input_match and output_match:
            inst = re.sub(r'\d+\.$', '', intruction_match.group().strip()).strip('\n').rstrip()
            input = re.sub(r'\d+\.$', '', input_match.group().strip()).strip('\n').rstrip()
            input = "" if "无输入" in input else input
            output = output_match.group().strip().strip('\n')
            if '指令:' in output and '输入:' in output and '输出:' in output:  # 返回若没有以###号区分，取第一条数据
                output_pattern_new = re.compile(r"(?<=(?:" + "))[\s\S]*?(?=" + '|'.join(['指令:', '指令：']) + ")")
                output_match_new = output_pattern_new.search(output)
                if output_match_new:
                    output = re.sub(r'\d+\.$', '', output_match_new.group().strip()).strip('\n').rstrip()
            out = {"instruction": inst, "input": input, "output": output}
            data_list.append(out)
    return data_list


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


def openai_reply(messages, model_name="gpt-3.5-turbo"):
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=1.0,
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    response = completion.choices[0].message["content"]
    return response


@retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(3))
def completion_with_backoff(**kwargs):
    # 重试间隔时间1到3秒，重试次数3
    return openai_reply(**kwargs)


def get_chatgpt_response(i):
    """Get response from chatgpt model"""
    task_subset = choices(seed_task_list, k=3)
    prompt_ = prompt
    for idx, task_dict in enumerate(task_subset):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        input = "<无输入>" if input.lower() == "" else input
        prompt_ = prompt_ + f"###\n"
        prompt_ += f"{idx + 1}. 指令: {instruction}\n"
        prompt_ += f"{idx + 1}. 输入:\n{input}\n"
        prompt_ += f"{idx + 1}. 输出:\n{output}\n"
    prompt_ += f"###\n"
    messages = [{"role": "assistant", "content": prompt_}]
    logger.debug(f'prompt: {messages}')
    try:
        response = completion_with_backoff(messages=messages, model_name=model_name)
        logger.debug(f"Successfully get chatgpt response, content:{prompt_}, res:{response}")
    except Exception as e:
        logger.error(e)
        response = ''
    instruction_list = parse_response(response)
    time.sleep(REQ_TIME_GAP)
    pbar.update(1)
    if instruction_list:
        data_result.extend(instruction_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_file', default='./seed_medical_sft_data.jsonl', type=str, help='Seed file')
    parser.add_argument('--output_file', default='./medical_sft_result.jsonl', type=str, help='Output file')
    parser.add_argument('--model_name', default='gpt-3.5-turbo', type=str, help='gpt-3.5-turbo or gpt-4')
    parser.add_argument('--num_instructions_to_generate', default=2, type=int,
                        help='Number of instructions to generate')
    args = parser.parse_args()
    print(args)
    model_name = args.model_name
    seed_task_list = load_jsonl(args.seed_file)

    prompt = f"请给出10个多样化的任务指令。这些任务指令将被提供给GPT模型，我们将评估GPT模型完成指令的能力。\n \
    以下是你提供指令需要满足的要求：\n \
    1.指令用中文书写，指令应该是一个医疗任务。\n \
    2.指令类型应该是多样化的，包括各种类型的任务，类别种类例如：病情诊断，病因分析，病理诊断，治疗方案，就医建议，指标解读，药物剂量，用药建议，医疗建议，医学知识，疾病描述，后果表述，注意事项，功效作用，医疗费用，预防措施，预后评估，其他\n \
    3.你应该给指令生成适当的输入，输入字段应包含为指令提供的具体示例，它应该是一个医疗问题，含有有用的医学信息，例如病灶描述，体检指标数值，药物剂量等，不应包含简单的占位符。输入应提供充实的内容，使指令具有挑战性。\n \
    4.输出应该是针对指令和输入的恰当回答，如果输入的信息不足以进行判断需要进一步询问。\n \
    5.输入输出相关的疾病应该是多样化的，包含各种类型的疾病和药品信息。\n \
    下面是几个任务指令的示例：\n\n"

    data_result = []
    data_list = [[i] for i in range(args.num_instructions_to_generate)]
    with tqdm(total=len(data_list)) as pbar:
        pbar.set_description("Generating dialogue")
        try:
            pool = Pool(processes=2)
            res = pool.starmap(get_chatgpt_response, [[i] for i in data_list])
            pool.close()
            pool.join()
        except Exception as e:
            print(e)
        pbar.update(len(data_list))
    print('save all')
    save_jsonl(data_result, args.output_file)
