# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 使用GPT2模型做对联生成任务，prompt为上联，自动对下联

本示例自定义了GPT2模型的Dataset，使其完成类似seq2seq的生成任务，可以适配对话生成、对联生成、诗歌生成等任务
"""
from loguru import logger
import argparse
from transformers import BertTokenizerFast
import torch
from torch.utils.data import Dataset
import os
import pickle
import sys

sys.path.append('../..')
from textgen.language_generation import LanguageGenerationModel
from textgen.language_modeling import LanguageModelingModel


def encode(data):
    """Encode data to src trg token ids"""
    tokenizer, line = data
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    src, trg = line.split('\t')
    input_ids = [cls_id] + tokenizer.encode(src, add_special_tokens=False) + [sep_id] + \
                tokenizer.encode(trg, add_special_tokens=False) + [sep_id]
    return input_ids


class SrcTrgDataset(Dataset):
    """Custom dataset, use it by dataset_class from train args"""
    def __init__(self, tokenizer, args, file_path, mode, block_size=512, special_tokens_count=2):
        block_size = block_size - special_tokens_count
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            args.cache_dir, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not args.no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(f" Loading features from cached file {cached_features_file}")
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(f" Creating features from dataset file at {args.cache_dir}")

            with open(file_path, encoding="utf-8") as f:
                lines = [
                    (tokenizer, line) for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())
                ]

            self.examples = [encode(line) for line in lines]

            logger.info(f" Saving features into cached file {cached_features_file}")
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='../data/zh_couplet_train.tsv', type=str, help='Training data file')
    parser.add_argument('--test_file', default='../data/zh_couplet_test.tsv', type=str, help='Test data file')
    parser.add_argument('--model_type', default='gpt2', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='uer/gpt2-distil-chinese-cluecorpussmall', type=str,
                        help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs/couplet-fine-tuned/', type=str,
                        help='Model output directory')
    parser.add_argument('--max_seq_length', default=50, type=int, help='Max sequence length')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)

    if args.do_train:
        logger.info('Training...')

        train_args = {
            "dataset_class": SrcTrgDataset,
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "block_size": 512,
            "max_seq_length": args.max_seq_length,
            "learning_rate": 5e-6,
            "train_batch_size": args.batch_size,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": args.num_epochs,
            "mlm": False,
            "output_dir": args.output_dir,
            "evaluate_during_training": True,
        }
        tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
        model = LanguageModelingModel(args.model_type, args.model_name, args=train_args, tokenizer=tokenizer)
        # Train model for pair data (format: src \t trg)
        model.train_model(args.train_file, eval_file=args.test_file)
        print(model.eval_model(args.test_file))

    if args.do_predict:
        logger.info('Predict...')
        # Use fine-tuned model
        tokenizer = BertTokenizerFast.from_pretrained(args.output_dir)
        model = LanguageGenerationModel(args.model_type, args.output_dir,
                                        args={"max_length": args.max_seq_length},
                                        tokenizer=tokenizer)

        couplet_prompts = [
            "晚风摇树树还挺",
            "深院落滕花，石不点头龙不语",
            "不畏鸿门传汉祚"
        ]
        for prompt in couplet_prompts:
            # Generate text using the model. Verbose set to False to prevent logging generated sequences.
            generated = model.generate(prompt, verbose=False, add_cls_head=True)
            generated = generated[0]
            print("inputs:", prompt)
            print("outputs:", generated)


if __name__ == '__main__':
    main()
