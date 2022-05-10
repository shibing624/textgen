# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import argparse
import pandas as pd
from loguru import logger
import os
import sys

sys.path.append('../..')
from textgen.seq2seq import BertSeq2SeqModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='bert', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='bert-base-cased', type=str, help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs/toy/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=50, type=int, help='Max sequence length')
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of training epochs')
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
        logger.debug('train_data: {}'.format(train_data[:20]))
        train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

        eval_data = [
            ["nine", "9"],
            ["zero", "0"],
        ]
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

        # encoder_type=None, encoder_name=None, decoder_name=None, encoder_decoder_type=None, encoder_decoder_name=None,
        model = BertSeq2SeqModel(args.model_type, args.model_name, args.model_name, args=model_args)

        def count_matches(labels, preds):
            logger.debug(f"labels: {labels[:10]}")
            logger.debug(f"preds: {preds[:10]}")
            match = sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])
            logger.debug(f"match: {match}")
            return match

        model.train_model(train_df, eval_data=eval_df, matches=count_matches)
        print(model.eval_model(eval_df, matches=count_matches))

    if args.do_predict:
        # model = Seq2SeqModel("bert", "outputs/encoder", "outputs/decoder", use_cuda=use_cuda)
        model = BertSeq2SeqModel(args.model_type,
                                 os.path.join(args.output_dir, "encoder"),
                                 os.path.join(args.output_dir, "decoder"))
        print('input: one', ' output:', model.predict(["one"]))
        print(model.predict(["four", "five"]))


if __name__ == '__main__':
    main()
