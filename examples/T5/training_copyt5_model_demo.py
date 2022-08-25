# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import argparse
from loguru import logger
import time
import pandas as pd
import sys

sys.path.append('../..')
from textgen import CopyT5Model


def load_data(prefix, file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            terms = line.split('\t')
            if len(terms) == 2:
                data.append([prefix, terms[0], terms[1]])
            else:
                logger.warning(f'line error: {line}')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='../data/zh_dialog.tsv', type=str, help='Training data file')
    parser.add_argument('--model_type', default='copyt5', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='imxly/t5-copy', type=str, help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--prefix', default='QA', type=str, help='Prefix str')
    parser.add_argument('--output_dir', default='./outputs/copyt5_zh/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=200, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=200, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)

    if args.do_train:
        logger.info('Loading data...')
        # train_data: Pandas DataFrame containing the 3 columns - `prefix`, `input_text`, `target_text`.
        #   - `prefix`: A string indicating the task to perform. (E.g. `"question"`, `"stsb"`)
        #   - `input_text`: The input text. `prefix` is prepended to form the full input. (<prefix>: <input_text>)
        #   - `target_text`: The target sequence
        train_data = load_data(args.prefix, args.train_file)
        logger.debug('train_data: {}'.format(train_data[:10]))
        train_df = pd.DataFrame(train_data, columns=["prefix", "input_text", "target_text"])

        eval_data = load_data(args.prefix, args.train_file)[:10]
        eval_df = pd.DataFrame(eval_data, columns=["prefix", "input_text", "target_text"])

        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "max_length": args.max_length,
            "train_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": False,
            "evaluate_generated_text": True,
            "evaluate_during_training": True,
            "evaluate_during_training_verbose": True,
            "use_multiprocessing": False,
            "save_best_model": True,
            "output_dir": args.output_dir,
            "use_early_stopping": True,
            "best_model_dir": os.path.join(args.output_dir, "best_model"),
        }
        model = CopyT5Model(args.model_type, args.model_name, args=model_args)

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
        model = CopyT5Model(args.model_type, args.output_dir, args={"eval_batch_size": args.batch_size})
        sentences = ["什么是ai", "你是什么类型的计算机", "你知道热力学吗"]
        sentences_add_prefix = [args.prefix + ": " + i for i in sentences]
        print("inputs:", sentences)
        print("outputs:", model.predict(sentences_add_prefix))

        eval_data = load_data(args.prefix, args.train_file)[:50]
        eval_df = pd.DataFrame(eval_data, columns=["prefix", "input_text", "target_text"])
        sentences = eval_df['input_text'].tolist()
        sentences_add_prefix = [args.prefix + ": " + i for i in sentences]
        print(sentences_add_prefix)
        t1 = time.time()
        res = model.predict(sentences_add_prefix)
        print(type(res), len(res))
        print(res)
        logger.info(f'spend time: {time.time() - t1}, size: {len(sentences_add_prefix)}')


if __name__ == '__main__':
    main()
