# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

code from https://github.com/QwenLM/Qwen-7B/blob/main/eval/EVALUATION.md

usage:
Get the HumanEval.jsonl file from [here](https://github.com/openai/human-eval/tree/master/data)

python eval/evaluate_chat_humaneval.py -f HumanEval.jsonl -o HumanEval_res.jsonl
git clone https://github.com/openai/human-eval
pip install -e human-eval
evaluate_functional_correctness HumanEval_res.jsonl
"""

import argparse
import re
import textwrap
from pathlib import Path

import jsonlines
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

DEVICE = "cuda:0"


def extract_code(text, entry_point):
    # 正则表达式匹配代码块
    code_block_pattern = re.compile(rf"```(?:[Pp]ython\n)?.*?def\s+{entry_point}.*?:\n(.*?)\n```", re.DOTALL)
    code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(rf"def\s+{entry_point}.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL)
        code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(rf"def.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL)
        code_block = code_block_pattern.search(text)

    if code_block is not None:
        return code_block.group(1)
    else:
        # if no code block is found, assume the LM is simply filling the code
        return textwrap.indent(text, ' ' * 4)

@torch.no_grad()
def model_predict(model, tokenizer, question, max_new_tokens=512, **kwargs):
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs['input_ids'].to(model.device)
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, **kwargs)
    generated_sequence = outputs[0][len(input_ids[0]):]
    output_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
    return output_text.strip()


def generate_sample(model, tokenizer, question, entry_point):
    response = model_predict(model, tokenizer, question)
    print(question)
    print(response)
    answer = extract_code(response, entry_point)
    return answer, response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test HF checkpoint.')
    parser.add_argument("-c", "--checkpoint-path", type=Path, help='Checkpoint path', default="Qwen/Qwen-7B-Chat")
    parser.add_argument("-f", "--sample-input-file", type=str, default=None, help="data path to HumanEval.jsonl")
    parser.add_argument("-o", "--sample-output-file", type=str, default="HumanEval_res.jsonl")

    args = parser.parse_args()
    print('Loading tokenizer ...')
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)

    print('Loading model ...')
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, device_map="auto", trust_remote_code=True,
                                                 torch_dtype=torch.float16).eval()
    try:
        model.generation_config = GenerationConfig.from_pretrained(args.checkpoint_path, trust_remote_code=True)
        model.generation_config.do_sample = False  # use greedy decoding
    except:
        print("GenerationConfig not found, use default config.")

    f_output = jsonlines.Writer(open(args.sample_output_file, 'w', encoding='utf-8'))

    f = jsonlines.open(args.sample_input_file)
    with f_output as output:
        for jobj in tqdm.tqdm(f, desc='task_idx'):
            prompt = "Help me fill the following code.\n" + jobj['prompt']
            task_id = jobj['task_id']
            answer, response = generate_sample(model, tokenizer, prompt, jobj['entry_point'])
            gen_jobjs = {'task_id': task_id, "completion": answer, 'response': response}
            output.write(gen_jobjs)
    f_output.close()
