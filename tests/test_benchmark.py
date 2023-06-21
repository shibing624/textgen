# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import sys

import pandas as pd

sys.path.append('..')
from textgen import GptModel, ChatGlmModel

pwd_path = os.path.abspath(os.path.dirname(__file__))


def llama_generate_prompt(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:{instruction}\n\n### Response: """


def chatglm_generate_prompt(instruction):
    return f"""{instruction}\n答："""


sentences = [i.strip() for i in open(os.path.join(pwd_path, '../examples/data/llm_benchmark_test.txt')).readlines() if
             i.strip()]


def test_llama_13b_lora():
    m = GptModel('llama', "decapoda-research/llama-13b-hf", peft_name='shibing624/llama-13b-belle-zh-lora')

    predict_sentences = [llama_generate_prompt(s) for s in sentences]
    res = m.predict(predict_sentences)
    for s, i in zip(sentences, res):
        print('input:', s, '\noutput:', i)
        print()

    res_dict = {'input': sentences, 'output': res}
    df = pd.DataFrame.from_dict(res_dict)
    json_file = os.path.join(pwd_path, 'llama_13b_lora_llm_benchmark_test_result.json')
    df.to_json(json_file, force_ascii=False, orient='records', lines=True)
    df.to_excel(json_file + '.xlsx', index=False)


def test_llama_7b_alpaca_plus():
    m = GptModel('llama', "shibing624/chinese-alpaca-plus-7b-hf", args={'use_peft': False})
    predict_sentences = [llama_generate_prompt(s) for s in sentences]
    res = m.predict(predict_sentences)
    for s, i in zip(sentences, res):
        print('input:', s, '\noutput:', i)
        print()
    res_dict = {'input': sentences, 'output': res}
    df = pd.DataFrame.from_dict(res_dict)
    json_file = os.path.join(pwd_path, 'llama_7b_alpaca_plus_llm_benchmark_test_result.json')
    df.to_json(json_file, force_ascii=False, orient='records', lines=True)
    df.to_excel(json_file + '.xlsx', index=False)


def test_llama_13b_alpaca_plus():
    m = GptModel('llama', "shibing624/chinese-alpaca-plus-13b-hf", args={'use_peft': False})
    predict_sentences = [llama_generate_prompt(s) for s in sentences]
    res = m.predict(predict_sentences)
    for s, i in zip(sentences, res):
        print('input:', s, '\noutput:', i)
        print()
    res_dict = {'input': sentences, 'output': res}
    df = pd.DataFrame.from_dict(res_dict)
    json_file = os.path.join(pwd_path, 'llama_13b_alpaca_plus_llm_benchmark_test_result.json')
    df.to_json(json_file, force_ascii=False, orient='records', lines=True)
    df.to_excel(json_file + '.xlsx', index=False)


def test_chatglm_6b():
    m = ChatGlmModel('chatglm', "THUDM/chatglm-6b", peft_name=None, args={'use_peft': False})
    predict_sentences = [chatglm_generate_prompt(s) for s in sentences]
    res = m.predict(predict_sentences)
    for s, i in zip(sentences, res):
        print('input:', s, '\noutput:', i)
        print()

    res_dict = {'input': sentences, 'output': res}
    df = pd.DataFrame.from_dict(res_dict)
    json_file = os.path.join(pwd_path, 'chatglm_6b_llm_benchmark_test_result.json')
    df.to_json(json_file, force_ascii=False, orient='records', lines=True)
    df.to_excel(json_file + '.xlsx', index=False)


def test_chatglm_6b_lora():
    m = ChatGlmModel('chatglm', "THUDM/chatglm-6b", peft_name='shibing624/chatglm-6b-belle-zh-lora',
                     args={'use_peft': True}, )
    predict_sentences = [chatglm_generate_prompt(s) for s in sentences]
    res = m.predict(predict_sentences)
    for s, i in zip(sentences, res):
        print('input:', s, '\noutput:', i)
        print()

    res_dict = {'input': sentences, 'output': res}
    df = pd.DataFrame.from_dict(res_dict)
    json_file = os.path.join(pwd_path, 'chatglm_6b_lora_llm_benchmark_test_result.json')
    df.to_json(json_file, force_ascii=False, orient='records', lines=True)
    df.to_excel(json_file + '.xlsx', index=False)
