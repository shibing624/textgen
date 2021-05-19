# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append('..')
import textgen
from textgen.utils.log import logger


if __name__ == '__main__':
    from textgen.augmentation.word_level_augment import word_augment, get_data_stats
    from textgen.augmentation.sent_level_augment import sent_augment

    a = '晚上一个人好孤单，想:找附近的人陪陪我.'
    a2 = '晚上一个人好孤单，.'
    b = "unlabeled example, aug_copy_num is the index of the generated augmented. you don't know"
    b2 = "unlabeled example, aug_copy_num is the index of the generated augmented. "
    b3 = "unlabeled example, aug_copy_num is the index "
    example = InputExample(guid=0, text_a=b, text_b=None, label=None)
    example2 = InputExample(guid=1, text_a=b2, text_b=None, label=None)
    example3 = InputExample(guid=2, text_a=b3, text_b=None, label=None)

    example4 = InputExample(guid=3, text_a=a, text_b=None, label=None)
    example5 = InputExample(guid=4, text_a=a2, text_b=None, label=None)
    examples = [example, example2, example3, example4, example5]

    tokenizer = Tokenizer()
    examples, word_vocab = tokenize_examples(examples, tokenizer)
    data_stats = get_data_stats(examples)
    import copy

    input_e = copy.deepcopy(examples)
    input_e2 = copy.deepcopy(examples)
    input_e3 = copy.deepcopy(examples)
    input_e4 = copy.deepcopy(examples)
    input_e5 = copy.deepcopy(examples)

    aug_examples = word_augment(examples, 'random-0.2', word_vocab, data_stats, show_example=True)
    print(len(aug_examples))
    aug_examples = word_augment(input_e, 'insert-0.2', word_vocab, data_stats, show_example=True)
    print(len(aug_examples))
    aug_examples = word_augment(input_e2, 'delete-0.2', word_vocab, data_stats, show_example=True)
    print(len(aug_examples))
    aug_ops = 'tfidf-0.2'
    aug_examples = word_augment(input_e3, aug_ops, word_vocab, data_stats, show_example=True)
    print(len(aug_examples))
    aug_examples = word_augment(input_e4, 'mix-0.2', word_vocab, data_stats, show_example=True)
    print(len(aug_examples))

    logger.info("getting sent augmented examples")
    aug_examples = sent_augment(input_e5, 'bt-0.2', from_lang='zh')
    print(aug_examples)
