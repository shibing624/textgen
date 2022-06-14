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

        generated = ".".join(generated[0].split(".")[:-1]) + "."
        print("=============================================================================")
        print(generated)
        print("=============================================================================")


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
        "output_dir": f"outputs/zh-fine-tuned/",
    }

    model = LanguageModelingModel("gpt2", "ckiplab/gpt2-base-chinese", args=train_args)
    model.train_model(train_path, eval_file=test_path)
    print(model.eval_model(test_path))

    # Use fine-tuned model
    tokenizer = BertTokenizerFast.from_pretrained("outputs/zh-fine-tuned")
    model = LanguageGenerationModel("gpt2", "outputs/zh-fine-tuned", args={"max_length": 64}, tokenizer=tokenizer)
    for prompt in prompts:
        # Generate text using the model. Verbose set to False to prevent logging generated sequences.
        generated = model.generate(prompt, verbose=False)

        generated = ".".join(generated[0].split(".")[:-1]) + "."
        print("=============================================================================")
        print(generated)
        print("=============================================================================")


if __name__ == '__main__':
    """
    王语嫣知道表哥神智已乱，富贵梦越做越深，不禁凄然。

    段誉见到阿碧的神情，怜惜之念大起，只盼招呼她和慕容复回去大理，妥为安顿，却见她瞧着慕容复的眼色中柔情无限，
    而慕容复也是一副志得意满之态，心中登时一凛：“各有各的缘法，慕容兄与阿碧如此，我觉得他们可怜，其实他们心中，
    焉知不是心满意足？我又何必多事？”轻轻拉了拉王语嫣的衣袖，做个手势。

    众人都悄悄退了开去。但见慕容复在土坟上南面而坐，口中兀自喃喃不休。
    """
    prompts = [
        "王语嫣知道表哥神智已乱",
        "段誉见到阿碧的神情，怜惜之念大起，只盼",
        "众人都悄悄退了开去。但见慕容复在土坟上南面而坐"
    ]
    raw(prompts)

    train_path = "zh_article.txt"
    test_path = "zh_article.txt"
    finetune(prompts, train_path, train_path)
