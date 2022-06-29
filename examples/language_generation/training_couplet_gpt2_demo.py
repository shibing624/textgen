# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from loguru import logger
import argparse
from transformers import BertTokenizerFast
from torch.utils.data import Dataset
import os
import pickle
import pandas as pd
from tqdm.auto import tqdm
import sys

sys.path.append('../..')
from textgen.language_generation import LanguageGenerationModel
from textgen.language_modeling import LanguageModelingModel


def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            terms = line.split('\t')
            if len(terms) == 2:
                data.append([terms[0], terms[1]])
            else:
                logger.warning(f'line error: {line}, split size: {len(terms)}')
    return data


def preprocess_data(data):
    input_text, target_text, tokenizer, args = data

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    input_ids = [cls_id]
    input_ids += tokenizer.encode(input_text, add_special_tokens=False)
    input_ids.append(sep_id)
    input_ids += tokenizer.encode(target_text, add_special_tokens=False)
    input_ids.append(sep_id)

    return input_ids


class SrcTrgDataset(Dataset):
    def __init__(self, tokenizer, args, data, mode, block_size=512):
        cached_features_file = os.path.join(
            args.cache_dir,
            args.model_name.replace("/", "_")
            + "_cached_"
            + str(args.max_seq_length)
            + str(len(data)),
        )

        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not args.no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s" % cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s" % args.cache_dir)

            data = [
                (input_text, target_text, tokenizer, args)
                for input_text, target_text in zip(
                    data["input_text"], data["target_text"]
                )
            ]

            self.examples = [
                preprocess_data(d) for d in tqdm(data, disable=args.silent)
            ]

            if not args.no_cache:
                logger.info(
                    " Saving features into cached file %s" % cached_features_file
                )
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


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
        train_data = load_data(args.train_file)
        logger.debug('train_data: {}'.format(train_data[:10]))
        train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

        eval_data = load_data(args.test_file)[:10]
        eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

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
            "save_best_model": True,
            "output_dir": args.output_dir,
        }
        tokenizer = BertTokenizerFast.from_pretrained(args.output_dir)
        model = LanguageModelingModel(args.model_type, args.model_name, args=train_args, tokenizer=tokenizer)
        model.train_model(train_df, eval_file=eval_df)
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
