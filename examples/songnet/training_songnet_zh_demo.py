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
from textgen.songnet import SongNetModel


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
    parser.add_argument('--train_file', default='../data/zh_couplet_test.tsv', type=str, help='Training data file')
    parser.add_argument('--model_type', default='songnet', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='./pretrained_songnet/', type=str, help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs/couplet_songnet_zh/', type=str,
                        help='Model output directory')
    parser.add_argument('--max_length', default=128, type=int, help='Max sequence length')
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
            "max_length": args.max_length,
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
        model = SongNetModel(
            model_type=args.model_type,
            model_name=args.model_name,
            args=model_args
        )

        def count_matches(labels, preds):
            logger.debug(f"labels: {labels[:10]}")
            logger.debug(f"preds: {preds[:10]}")
            match = sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])
            logger.debug(f"match: {match}")
            return match

        model.train_model(train_df, eval_data=eval_df, matches=count_matches)
        print(model.eval_model(eval_df, matches=count_matches))

    if args.do_predict:
        # Use fine-tuned model
        model = SongNetModel(model_name=args.output_dir)
        sentences = [
            "严蕊<s1>如梦令<s2>道是梨花不是。</s>道是杏花不是。</s>白白与红红，别是东风情味。</s>曾记。</s>曾记。</s>人在武陵微醉。",
            "张抡<s1>春光好<s2>烟澹澹，雨。</s>水溶溶。</s>帖水落花飞不起，小桥东。</s>翩翩怨蝶愁蜂。</s>绕芳丛。</s>恋馀红。</s>不恨无情桥下水，恨东风。"
        ]
        print("inputs:", sentences)
        print("outputs:", model.predict(sentences))
        sentences = [
            "秦湛<s1>卜算子<s2>_____，____到。_______，____俏。_____，____报。_______，____笑。",
            "秦湛<s1>卜算子<s2>_雨___，____到。______冰，____俏。____春，__春_报。__山花___，____笑。"
        ]
        print("inputs:", sentences)
        print("outputs:", model.predict_mask(sentences))


if __name__ == '__main__':
    main()
