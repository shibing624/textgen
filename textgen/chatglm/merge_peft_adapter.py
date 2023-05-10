# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, default="THUDM/chatglm-6b")
    parser.add_argument("--peft_model_path", type=str, default="/")

    return parser.parse_args()


def main():
    args = get_args()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        return_dict=True,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    model = model.merge_and_unload()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    model.save_pretrained(f"{args.base_model_name_or_path}-merged")
    tokenizer.save_pretrained(f"{args.base_model_name_or_path}-merged")
    print(f"Model saved to {args.base_model_name_or_path}-merged")


if __name__ == "__main__":
    main()
