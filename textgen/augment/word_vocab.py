# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: InputExample
"""

import collections
from codecs import open

from textgen.utils.log import logger


def convert_tokens_to_ids(vocab, tokens):
    """Converts a sequence of tokens into ids using the vocab."""
    ids = []
    for token in tokens:
        ids.append(vocab[token])
    return ids


def convert_ids_to_tokens(inv_vocab, ids):
    """Converts a sequence of ids into tokens using the vocab."""
    output = []
    for item in ids:
        output.append(inv_vocab[item])
    return output


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for token in f:
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def build_vocab(word_list, vocab_file=None):
    vocab = {}
    for word in word_list:
        if word not in vocab:
            vocab[word] = len(vocab)
    if vocab_file:
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for w, idx in vocab.items():
                f.write(w + '\n')
        logger.info('save vocab to %s' % vocab_file)
    return vocab
