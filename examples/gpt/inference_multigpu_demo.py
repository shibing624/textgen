# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: use deepspeed to inference with multi-gpus

usage:
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 inference_demo.py --model_type bloom --base_model bigscience/bloom-560m
"""
import argparse
import csv
import sys

import torch
import torch.distributed as dist
from loguru import logger
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
)

sys.path.append('../..')
from textgen.gpt.gpt_utils import get_conv_template

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default=None, type=str, required=True)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--prompt_template_name', default="vicuna", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan-chat, chatglm2 etc.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--data_file', default=None, type=str, help="Predict file, one example per line")
    parser.add_argument('--output_file', default='./predictions_result.jsonl', type=str)
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    args = parser.parse_args()
    logger.info(args)
    load_type = torch.float16
    if torch.cuda.is_available():
        device = 0
    else:
        raise ValueError("No GPU available, this script is only for GPU inference.")
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    logger.info(f"local_rank: {local_rank}, world_size: {world_size}")
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    base_model = model_class.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map={"": device},
        trust_remote_code=True,
    )
    try:
        base_model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)
    except OSError:
        logger.info("Failed to load generation config, use default.")
    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        logger.info(f"Vocab of the base model: {model_vocab_size}")
        logger.info(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            logger.info("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)

    if args.lora_model:
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        logger.info("Loaded lora model")
    else:
        model = base_model
    model.eval()
    # Use DataParallel for multi-GPU inference
    # DistributedDataParallel for multi-node inference (which deepspeed better supports)
    model = torch.nn.DataParallel(model, device_ids=[local_rank])
    model = model.module
    logger.info(tokenizer)
    # test data
    if args.data_file is None:
        examples = ["介绍下北京", "乙肝和丙肝的区别？"]
    else:
        with open(args.data_file, 'r', encoding='utf-8') as f:
            examples = [l.strip() for l in f.readlines()]
        logger.info(f"first 10 examples: {examples[:10]}")

    prompt_template = get_conv_template(args.prompt_template_name)
    write_batch_size = args.batch_size * world_size * 10
    generation_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )
    stop_str = tokenizer.eos_token if tokenizer.eos_token else prompt_template.stop_str
    with open(args.output_file, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['input', 'output'])
    for batch in tqdm(
            [
                examples[i: i + write_batch_size]
                for i in range(0, len(examples), write_batch_size)
            ],
            desc="Generating outputs",
    ):
        dataset = TextDataset(batch)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

        data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
        responses = []
        for texts in data_loader:
            logger.info(f'{local_rank}, texts size:{len(texts)}, {texts}')
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(local_rank)
            generated_outputs = model.generate(**inputs, **generation_kwargs)
            responses.extend(tokenizer.batch_decode(generated_outputs, skip_special_tokens=True))

        # Gather responses from all processes
        all_responses = [None] * world_size
        dist.all_gather_object(all_responses, responses)

        # Write responses only on the main process
        if local_rank <= 0:
            all_responses_flat = [response for process_responses in all_responses for response in process_responses]
            logger.info(all_responses_flat)
            with open(args.output_file, 'a', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                for text, response in zip(batch, all_responses_flat):
                    writer.writerow([text, response])


if __name__ == '__main__':
    main()
