# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: InputExample
"""

import collections
from codecs import open

from textgen.utils.logger import logger
from textgen.utils.tokenization import Tokenizer


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        """print"""
        res = str(self.guid) + self.text_a
        if self.text_b:
            res = str(self.guid) + self.text_a + self.text_b
        if self.label:
            res = str(self.guid) + self.text_a + str(self.label)
        if self.text_b and self.label:
            res = str(self.guid) + self.text_a + self.text_b + str(self.label)
        return res


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def tokenize_examples(examples, tokenizer, vocab_file=None):
    logger.debug("tokenizing examples")
    vocab = {}

    def add_to_vocab(word_list):
        for word in word_list:
            if word not in vocab:
                vocab[word] = len(vocab)

    for i in range(len(examples)):
        examples[i].word_list_a = tokenizer.tokenize(examples[i].text_a)
        add_to_vocab(examples[i].word_list_a)
        if examples[i].text_b:
            examples[i].word_list_b = tokenizer.tokenize(examples[i].text_b)
            add_to_vocab(examples[i].word_list_b)
        if i % 10000 == 0:
            logger.debug("finished tokenizing example {:d}".format(i))
    logger.info("finished tokenizing example {:d}".format(len(examples)))
    if vocab_file:
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for w, idx in vocab.items():
                f.write(w + '\n')
        logger.info('save vocab to %s' % vocab_file)
    return examples, vocab


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
    aug_examples = sent_augment(input_e5, 'bt-0.2')
    print(aug_examples)
