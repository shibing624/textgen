# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append('../..')
from textgen.language_generation import LanguageGenerationModel
from textgen.language_modeling import LanguageModelingModel


def raw(prompts):
    model = LanguageGenerationModel("gpt2", "gpt2", args={"max_length": 200})
    for prompt in prompts:
        # Generate text using the model. Verbose set to False to prevent logging generated sequences.
        generated = model.generate(prompt, verbose=False)

        generated = ".".join(generated[0].split(".")[:-1]) + "."
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
        "num_train_epochs": 1,
        "mlm": False,
        "output_dir": f"outputs/fine-tuned/",
    }

    model = LanguageModelingModel("gpt2", "gpt2", args=train_args)
    model.train_model(train_path, eval_file=test_path)
    print(model.eval_model(test_path))

    # Use finetuned model
    model = LanguageGenerationModel("gpt2", "outputs/en-fine-tuned", args={"max_length": 200})
    for prompt in prompts:
        # Generate text using the model. Verbose set to False to prevent logging generated sequences.
        generated = model.generate(prompt, verbose=False)

        generated = ".".join(generated[0].split(".")[:-1]) + "."
        print("=" * 42)
        print(generated)
        print("=" * 42)


if __name__ == '__main__':
    """
    The chief officer of the royal province was the governor, who enjoyed
    high and important powers which he naturally sought to augment at every
    turn. He enforced the laws and, usually with the consent of a council,
    appointed the civil and military officers. He granted pardons and
    reprieves; he was head of the highest court; he was commander-in-chief
    """
    prompts = [
        "The chief officer of the royal",
        "He enforced the laws and, usually with the consent of a council,",
        "He granted pardons and reprieves; he was head of the highest court; he was"
    ]
    raw(prompts)

    train_path = "../data/en_article_tail500.txt"
    finetune(prompts, train_path, train_path)
