# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import argparse
import sys

sys.path.append('../..')
from textgen import GptModel

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default='llama', type=str)
parser.add_argument('--base_model', default='shibing624/chinese-alpaca-plus-7b-hf', type=str)
parser.add_argument('--lora_model', default="", type=str, help="If not set, perform inference on the base model")
parser.add_argument('--interactive', action='store_true', help="run in the instruction mode")
parser.add_argument('--single_round', action='store_true',
                    help="Whether to generate single round dialogue, default is multi-round dialogue")
args = parser.parse_args()
print(args)
model = GptModel(args.model_type, args.base_model, peft_name=args.lora_model)
sents = [
    "失眠怎么办？",
    '用一句话描述地球为什么是独一无二的。',
    "Tell me about alpacas.",
    "Tell me about the president of Mexico in 2019.",
]
if args.interactive:
    print(f"Start inference with interactive mode. enable multi round: {not args.single_round}")
    history = []
    while True:
        raw_input_text = input("Input:")
        if len(raw_input_text.strip()) == 0:
            break
        if args.single_round:
            sents = [raw_input_text]
            response = model.predict(sents, add_system_prompt=True)[0]
        else:
            response, history = model.chat(raw_input_text, history=history, add_system_prompt=True)
        print("Response: ", response)
        print("\n")
else:
    print("Start inference.")
    responses = model.predict(sents, add_system_prompt=True)
    for index, example, response in zip(range(len(sents)), sents, responses):
        print(f"======={index}=======")
        print(f"Input: {example}\n")
        print(f"Output: {response}\n")
