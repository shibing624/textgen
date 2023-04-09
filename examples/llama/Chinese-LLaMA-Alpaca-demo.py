# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import argparse

import torch
import transformers
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, required=True, type=str, help="Please specify a base_model")
parser.add_argument('--lora_model', default=None, required=True, type=str, help="Please specify a lora_model")
parser.add_argument('--model_size', default='7B', type=str, help="Size of the LLaMA model", choices=['7B', '13B'])
parser.add_argument('--offload_dir', default=None, type=str,
                    help="(Optional) Please specify a temp folder for offloading (useful for low-RAM machines). Default None (disable offload).")
parser.add_argument('--output_dir', default='./', type=str)
args = parser.parse_args()

assert (
        "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM

BASE_MODEL = args.base_model
LORA_MODEL = args.lora_model
output_dir = args.output_dir

assert (
    BASE_MODEL
), "Please specify a BASE_MODEL in the script, e.g. 'decapoda-research/llama-7b-hf'"

tokenizer = LlamaTokenizer.from_pretrained(LORA_MODEL)
if args.offload_dir is not None:
    # Load with offloading, which is useful for low-RAM machines.
    # Note that if you have enough RAM, please use original method instead, as it is faster.
    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        offload_folder=args.offload_dir,
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
    )
else:
    # Original method without offloading
    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cuda"},
    )

base_model.resize_token_embeddings(len(tokenizer))
assert base_model.get_input_embeddings().weight.size(0) == len(tokenizer)
# tokenizer.save_pretrained(output_dir)
print(f"Extended vocabulary size: {len(tokenizer)}")

model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL,
    device_map={"": "cuda"},
    torch_dtype=torch.float16,
)
model.half().cuda()
from demo import predict, Prompter

sents = [
    '问：用一句话描述地球为什么是独一无二的。\n答：',
    '问：给定两个数字，计算它们的平均值。 数字: 25, 36\n答：',
    '问：基于以下提示填写以下句子的空格。 空格应填写一个形容词 句子: ______出去享受户外活动，包括在公园里散步，穿过树林或在海岸边散步。\n答：',
    "Tell me about alpacas.",
    "Tell me about the president of Mexico in 2019.",
    "Tell me about the king of France in 2019.",
    "List all Canadian provinces in alphabetical order.",
    "Write a Python program that prints the first 10 Fibonacci numbers.",
    "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
    # noqa: E501
    "Tell me five words that rhyme with 'shock'.",
    "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    "Count up from 1 to 500.",
]
prompter = Prompter()
for instruction in sents:
    print("Instruction:", instruction)
    r = list(predict(instruction,
                               model,
                               tokenizer,
                               prompter))
    print("Response:", r)
    print()
