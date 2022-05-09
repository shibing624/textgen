# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import argparse
import pandas as pd
from loguru import logger
import sys

sys.path.append('../..')
from textgen.seq2seq import Seq2SeqModel


def load_qa_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            q = ''
            line = line.strip()
            if line.startswith('='):
                continue
            if line.startswith('Q: '):
                q = line[3:]
            if line.startswith('A: '):
                a = line[3:]
                data.append([q, a])
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='en_dialog.txt', type=str, help='Training data file')
    parser.add_argument('--model_type', default='bert', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='bert-base-cased', type=str, help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs/en/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=50, type=int, help='Max sequence length')
    parser.add_argument('--num_epochs', default=20, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)

    if args.do_train:
        logger.info('Loading data...')
        train_data = [
            ["one", "1"],
            ["two", "2"],
            ["three", "3"],
            ["four", "4"],
            ["five", "5"],
            ["six", "6"],
            ["seven", "7"],
            ["eight", "8"],
        ]
        sub_train_data = load_qa_data(args.train_file)
        train_data += sub_train_data
        train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

        eval_data = [
            ["nine", "9"],
            ["zero", "0"],
        ]
        sub_eval_data = load_qa_data(args.train_file)[:10]
        eval_data += sub_eval_data
        eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "train_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": False,
            "silent": False,
            "evaluate_generated_text": True,
            "evaluate_during_training": False,
            "evaluate_during_training_verbose": False,
            "use_multiprocessing": False,
            "save_best_model": True,
            "output_dir": args.output_dir,
        }

        # encoder_type=None, encoder_name=None, decoder_name=None, encoder_decoder_type=None, encoder_decoder_name=None,
        model = Seq2SeqModel(args.model_type, args.model_name, args.model_name, args=model_args)

        def count_matches(labels, preds):
            print(labels)
            print(preds)
            return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])

        model.train_model(train_df, eval_data=eval_df, matches=count_matches)
        print(model.eval_model(eval_df, matches=count_matches))

    if args.do_predict:
        model = Seq2SeqModel(args.model_type, args.output_dir, args.output_dir)

        print(model.predict(["one", "four", "five"]))
        print(model.predict(
            ["that 's the kind of guy she likes ? Pretty ones ?",
             "Not the hacking and gagging and spitting part .",
             ]
        ))


if __name__ == '__main__':
    main()
