# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from textgen.augmentation.example import InputExample
from textgen.augmentation.example import tokenize_examples
from textgen.augmentation.sent_level_augment import sent_augment
from textgen.augmentation.word_level_augment import word_augment, get_data_stats
from textgen.utils.log import logger
from textgen.utils.tokenization import Tokenizer


def augment(sentence_list, aug_ops='random-0.2', aug_type='word', from_lang='zh'):
    """
    Augment data
    :param sentence_list: query list
    :param aug_ops: word_augment for "random-0.2, insert-0.2, delete-0.2, tfidf-0.2, mix-0.2"
                    sent_augment for "bt-0.2"
    :param aug_type: word/sentence
    :return: aug_examples
    """
    examples = []
    for id, sent in enumerate(sentence_list):
        ex = InputExample(guid=id, text_a=sent, text_b=None, label=None)
        examples.append(ex)
    if aug_type == 'word':
        logger.debug('Use text augmentation of {}'.format(aug_type))
        tokenizer = Tokenizer()
        examples, word_vocab = tokenize_examples(examples, tokenizer)
        data_stats = get_data_stats(examples)
        aug_examples = word_augment(examples, aug_ops, vocab=word_vocab, data_stats=data_stats, show_example=True)
    else:
        logger.debug('Use text augmentation of sent')
        aug_examples = sent_augment(examples, aug_ops, from_lang=from_lang)
    return aug_examples


if __name__ == '__main__':
    a = ['晚上一个人好孤单，想:找附近的人陪陪我.',
         '晚上肚子好难受',
         '你会武功吗，我不会',
         '组装标题质量受限于广告主自提物料的片段质量，且表达丰富度有限'
         ]
    b = augment(a, aug_ops='tfidf-1.0', aug_type='word')
    print(b)

    b = augment(a, aug_ops='bt-0.2', aug_type='sent')
    print(b)
