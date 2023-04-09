import argparse

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import time
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--int8', action='store_true')
args = parser.parse_args()

tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)  # fast版LlamaTokenizer载入非常慢，测试建议用slow的就好
model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.half, load_in_8bit=args.int8, device_map="balanced")
batch = tokenizer(
   """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
你好！
### Response:
""",
    return_tensors="pt"
)
batch = {k: v.cuda() for k, v in batch.items()}
with torch.inference_mode(), torch.cuda.amp.autocast():
    print("Start")
    t0 = time.time()
    generated = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], max_length=200)
    t1 = time.time()
    print(f"Output generated in {(t1-t0):.2f} seconds")
    r = tokenizer.decode(generated[0])
    print(r)


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

for i in sents:
    batch = tokenizer(i, return_tensors="pt")
    batch = {k: v.cuda() for k, v in batch.items()}
    with torch.inference_mode(), torch.cuda.amp.autocast():
        generated = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], max_length=128)
        r = tokenizer.decode(generated[0])
        print(r)
    print()
