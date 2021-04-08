# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from textgen.augmentation import word_level_augment
from textgen.utils.logger import logger


def get_data_for_worker(examples, replicas, worker_id):
    data_per_worker = len(examples) // replicas
    remainder = len(examples) - replicas * data_per_worker
    if worker_id < remainder:
        start = (data_per_worker + 1) * worker_id
        end = (data_per_worker + 1) * (worker_id + 1)
    else:
        start = data_per_worker * worker_id + remainder
        end = data_per_worker * (worker_id + 1) + remainder
    if worker_id == replicas - 1:
        assert end == len(examples)
    logger.info("processing data from {:d} to {:d}".format(start, end))
    examples = examples[start: end]
    return examples, start, end


def build_vocab(examples):
    vocab = {}

    def add_to_vocab(word_list):
        for word in word_list:
            if word not in vocab:
                vocab[word] = len(vocab)

    for i in range(len(examples)):
        add_to_vocab(examples[i].word_list_a)
        if examples[i].text_b:
            add_to_vocab(examples[i].word_list_b)
    return vocab


def tokenize_examples(examples, tokenizer):
    logger.info("tokenizing examples")
    for i in range(len(examples)):
        examples[i].word_list_a = tokenizer.tokenize_to_word(examples[i].text_a)
        if examples[i].text_b:
            examples[i].word_list_b = tokenizer.tokenize_to_word(examples[i].text_b)
        if i % 10000 == 0:
            logger.debug("finished tokenizing example {:d}".format(i))
    logger.info("finished tokenizing example {:d}".format(len(examples)))
    return examples


def convert_examples_to_features(
        examples, label_list, seq_length, tokenizer, trunc_keep_right,
        data_stats=None, aug_ops=None):
    """convert examples to features."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    logger.info("number of examples to process: {}".format(len(examples)))

    features = []

    if aug_ops:
        logger.info("building vocab")
        word_vocab = build_vocab(examples)
        examples = word_level_augment.word_level_augment(
            examples, aug_ops, word_vocab, data_stats
        )

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("processing {:d}".format(ex_index))
        tokens_a = tokenizer.tokenize_to_wordpiece(example.word_list_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize_to_wordpiece(example.word_list_b)

        if tokens_b:
            pass
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                if trunc_keep_right:
                    tokens_a = tokens_a[-(seq_length - 2):]
                else:
                    tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        label_id = label_map[example.label]
        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            # st = " ".join([str(x) for x in tokens])
            st = ""
            for x in tokens:
                st += str(x) + " "
            logger.info("tokens: %s" % st)
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                label_id=label_id))
    return features


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_type_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.label_id = label_id

    def get_dict_features(self):
        return {
            "input_ids": self.input_ids,
            "input_mask": self.input_mask,
            "input_type_ids": self.input_type_ids,
            "label_ids": [self.label_id]
        }


if __name__ == '__main__':
    aug_ops = 'tf_idf-0.9'
    from textgen.augmentation.sent_level_augment import InputExample
    from textgen.augmentation.word_level_augment import word_level_augment, get_data_stats
    from textgen.utils.tokenization import FullTokenizer

    vocab_path = '../../extra_data/b_vocab.txt'
    tokenizer = FullTokenizer(vocab_file=vocab_path, is_chinese2char=False)
    a = '晚上一个人好孤单，想:找附近的人陪陪我.'
    a2 = '晚上一个人好孤单，.'
    b = "unlabeled example, aug_copy_num is the index of the generated augmented. you don't know"
    b2 = "unlabeled example, aug_copy_num is the index of the generated augmented. "
    b3 = "unlabeled example, aug_copy_num is the index "
    example = InputExample(guid=0, text_a=b, text_b=None, label=None)
    example2 = InputExample(guid=1, text_a=b2, text_b=None, label=None)
    example3 = InputExample(guid=2, text_a=b3, text_b=None, label=None)
    examples = [example, example2, example3]
    for i in range(2):
        examples.append(InputExample(guid=i + 10, text_a=b3 + " " + str(i)))

    examples = tokenize_examples(examples, tokenizer)
    data_stats = get_data_stats(examples)

    aug_examples = word_level_augment(examples, aug_ops, vocab_path, data_stats, show_example=True)
    print(len(aug_examples))

    # chinese text
    example = InputExample(guid=0, text_a=a, text_b=None, label=None)
    example2 = InputExample(guid=1, text_a=a2, text_b=None, label=None)
    examples = [example, example2]
    for i in range(2):
        examples.append(InputExample(guid=i + 10, text_a=a2 + " " + str(i)))
    vocab_path = '../../extra_data/a_vocab.txt'
    tokenizer = FullTokenizer(vocab_file=vocab_path, is_chinese2char=False)
    examples = tokenize_examples(examples, tokenizer)
    data_stats = get_data_stats(examples)

    aug_examples = word_level_augment(examples, aug_ops, vocab_path, data_stats, show_example=True)
    print(len(aug_examples))
    # char cut
    tokenizer = FullTokenizer(vocab_file=vocab_path, is_chinese2char=True)
    examples = tokenize_examples(examples, tokenizer)
    data_stats = get_data_stats(examples)

    aug_examples = word_level_augment(examples, aug_ops, vocab_path, data_stats, show_example=True)
    print(len(aug_examples))
