# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append('..')
from textgen import GptModel


def generate_prompt(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:{instruction}\n\n### Response:"""


sents = [
    "失眠怎么办？",
    "介绍下南京景点？",
    '问：用一句话描述地球为什么是独一无二的。\n答：',
    '问：给定两个数字，计算它们的平均值。 数字: 25, 36\n答：',
    '问：基于以下提示填写以下句子的空格。 空格应填写一个形容词 句子: ______出去享受户外活动，包括在公园里散步，穿过树林或在海岸边散步。\n答：',
    "Tell me about alpacas.",
    "Tell me about the president of Mexico in 2019.",
    "Tell me about the king of France in 2019.",
    "List all Canadian provinces in alphabetical order.",
    "Write a Python program that prints the first 10 Fibonacci numbers.",
    "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
    "Tell me five words that rhyme with 'shock'.",
    "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    "Count up from 1 to 500.",
]


def test_origin_7b():
    m = GptModel('llama', "decapoda-research/llama-7b-hf", args={'use_peft': False})
    predict_sentence = generate_prompt("失眠怎么办？")
    r = m.predict([predict_sentence])
    print(r)
    assert len(r) > 0
    response, history = m.chat("你好", history=None)
    print(response, history)
    assert len(response) > 0
