# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import argparse
import pandas as pd
from loguru import logger
import os
from transformers import BertTokenizerFast
from transformers import BartForConditionalGeneration, Text2TextGenerationPipeline
import sys

sys.path.append('../..')
from textgen.seq2seq import BartSeq2SeqModel


def model_fill_mask():
    tokenizer = BertTokenizerFast.from_pretrained("fnlp/bart-base-chinese")
    model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
    text2text_generator = Text2TextGenerationPipeline(model, tokenizer)
    r = text2text_generator("北京是[MASK]的首都", max_length=50, do_sample=False)
    print(r)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='../data/zh_dialog.tsv', type=str, help='Training data file')
    parser.add_argument('--model_type', default='bart', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='fnlp/bart-base-chinese', type=str, help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs/bart_zh/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=50, type=int, help='Max sequence length')
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)

    if args.do_train:
        logger.info('Loading data...')
        train_data = load_data(args.train_file)
        logger.debug('train_data: {}'.format(train_data[:10]))
        train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

        eval_data = load_data(args.train_file)[:10]
        eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "train_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": False,
            "evaluate_generated_text": True,
            "evaluate_during_training": True,
            "evaluate_during_training_verbose": True,
            "use_multiprocessing": True,
            "save_best_model": True,
            "output_dir": args.output_dir,
            "use_early_stopping": True,
        }
        model = BartSeq2SeqModel(
            encoder_type=args.model_type,
            encoder_decoder_type=args.model_type,
            encoder_decoder_name=args.model_name,
            args=model_args
        )

        def sim_text_chars(text1, text2):
            if not text1 or not text2:
                return 0.0
            same = set(text1) & set(text2)
            m = len(same)
            n = len(set(text1)) if len(set(text1)) > len(set(text2)) else len(set(text2))
            return m / n

        def count_matches(labels, preds):
            logger.debug(f"labels: {labels[:10]}")
            logger.debug(f"preds: {preds[:10]}")
            match = sum([sim_text_chars(label, pred) for label, pred in zip(labels, preds)]) / len(labels)
            logger.debug(f"match: {match}")
            return match

        model.train_model(train_df, eval_data=eval_df, matches=count_matches)
        print(model.eval_model(eval_df, matches=count_matches))

    if args.do_predict:
        # Use fine-tuned model
        tokenizer = BertTokenizerFast.from_pretrained(args.output_dir)
        model = BartSeq2SeqModel(
            encoder_type=args.model_type,
            encoder_decoder_type=args.model_type,
            encoder_decoder_name=args.output_dir,
            tokenizer=tokenizer)
        sentences = ["什么是ai", "你是什么类型的计算机", "你知道热力学吗"]
        print("inputs:", sentences)
        print("outputs:", model.predict(sentences))


if __name__ == '__main__':
    main()
    model_fill_mask()
