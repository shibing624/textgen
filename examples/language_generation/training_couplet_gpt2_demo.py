# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from transformers import BertTokenizerFast
import sys

sys.path.append('../..')
from textgen.language_generation import LanguageGenerationModel
from textgen.language_modeling import LanguageModelingModel


def raw(prompts):
    model = LanguageGenerationModel("gpt2", "ckiplab/gpt2-base-chinese", args={"max_length": 64})
    for prompt in prompts:
        # Generate text using the model. Verbose set to False to prevent logging generated sequences.
        generated = model.generate(prompt, verbose=False)
        generated = generated[0]
        print("=" * 42)
        print(generated)
        print("=" * 42)


def finetune(prompts, train_path, test_path):
    train_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "block_size": 512,
        "max_seq_length": 64,
        "learning_rate": 5e-6,
        "train_batch_size": 8,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3,
        "mlm": False,
        "output_dir": f"outputs/couplet-fine-tuned/",
        # "dataset_type": "text",
    }

    model = LanguageModelingModel("gpt2", "ckiplab/gpt2-base-chinese", args=train_args)
    model.train_model(train_path, eval_file=test_path)
    print(model.eval_model(test_path))

    # Use fine-tuned model
    tokenizer = BertTokenizerFast.from_pretrained("outputs/couplet-fine-tuned")
    model = LanguageGenerationModel("gpt2", "outputs/couplet-fine-tuned", args={"max_length": 64}, tokenizer=tokenizer)
    for prompt in prompts:
        # Generate text using the model. Verbose set to False to prevent logging generated sequences.
        generated = model.generate(prompt, verbose=False)
        generated = generated[0]
        print("=" * 42)
        print(generated)
        print("=" * 42)


if __name__ == '__main__':
    prompts = [
        "晚风摇树树还挺",
        "深院落滕花，石不点头龙不语",
        "不畏鸿门传汉祚"
    ]
    raw(prompts)

    train_path = "../data/couplet/train.tsv"
    test_path = "../data/couplet/test.tsv"
    finetune(prompts, train_path, test_path)
