# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from text2vec import Vector
from text2vec.utils.logger import set_log_level

from textgen.augment.sentence_level_augment import back_translation
from textgen.augment.word_level_augment import (get_data_idf, RandomReplace, DeleteReplace, InsertReplace,
                                                TfIdfWordReplace, MixWordReplace)
from textgen.augment.word_vocab import build_vocab
from textgen.utils.log import logger
from textgen.utils.tokenizer import Tokenizer

set_log_level('ERROR')


class TextAugment(object):
    """
    Text Data Augmentation
    """

    def __init__(self, sentence_list=None):
        """
        Init
        :param sentence_list: list, docs
        """
        self.tokenizer = Tokenizer()
        vec = Vector()
        vec.load_model()
        self.w2v = vec.model.w2v
        tokenized_sentence_list = [self.tokenizer.tokenize(i) for i in sentence_list]
        self.data_idf = get_data_idf(tokenized_sentence_list)
        word_list = []
        for i in tokenized_sentence_list:
            word_list.extend(i)
        self.vocab = build_vocab(word_list)

    def augment(self, query, aug_ops='tfidf-0.2', **kwargs):
        """
        Augment data
        :param query:
        :param aug_ops: word_augment for "random-0.2, insert-0.2, delete-0.2, tfidf-0.2, mix-0.2"
                        sent_augment for "bt-0.2"
        :return: str, new_sentence
        """
        logger.debug('Use text augmentation operation: {}'.format(aug_ops))
        details = []
        # Sentence augmentation
        if aug_ops.startswith("bt"):
            new_query = back_translation(query, from_lang='zh', use_min_length=10,
                                   use_max_length_diff_ratio=0.5)
            return new_query, details
        # Word augmentation
        tokens = self.tokenizer.tokenize(query)

        prob = float(aug_ops.split("-")[1])
        if aug_ops.startswith("random"):
            op = RandomReplace(prob, self.vocab)
        elif aug_ops.startswith("insert"):
            op = InsertReplace(prob, self.vocab)
        elif aug_ops.startswith("delete"):
            op = DeleteReplace(prob, self.vocab)
        elif aug_ops.startswith("tfidf"):
            op = TfIdfWordReplace(self.w2v, prob, self.data_idf, similar_prob=0.7,
                                  random_prob=0.1, delete_prob=0.1, insert_prob=0.1)
        elif aug_ops.startswith("mix"):
            op = MixWordReplace(self.w2v, prob, self.data_idf, similar_prob=0.7,
                                random_prob=0.1, delete_prob=0.1, insert_prob=0.1)
        else:
            raise ValueError('error aug_ops.')
        new_tokens, details = op(tokens)
        new_query = ''.join(new_tokens)
        return new_query, details
