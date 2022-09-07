# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import argparse
from loguru import logger
import sys

sys.path.append('../..')
from textgen.songnet import SongNetModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='../data/zh_couplet_train.tsv', type=str, help='Training data file')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--model_path', default='./outputs/couplet_songnet_zh/model.ckpt', type=str, help='Model path')
    parser.add_argument('--vocab_path', default='./outputs/couplet_songnet_zh/vocab.txt', type=str, help='Vocab path')
    parser.add_argument('--max_seq_length', default=128, type=int, help='Max sequence length')
    parser.add_argument('--num_epochs', default=3, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)

    if args.do_train:
        logger.info('Loading data...')
        model = SongNetModel(model_path=args.model_path, vocab_path=args.vocab_path)
        model.train_model(args.train_file)
        print(model.eval_model(args.train_file))

    if args.do_predict:
        model = SongNetModel(model_path=args.model_path, vocab_path=args.vocab_path)
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
