# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Word level augmentations including Replace words with uniform
random words or TF-IDF based word replacement.
"""

import collections
import copy
import math

import numpy as np

from textgen.utils.logger import logger


class EfficientRandomGen(object):
    """A base class that generate multiple random numbers at the same time."""

    def reset_random_prob(self):
        """Generate many random numbers at the same time and cache them."""
        cache_len = 100000
        self.random_prob_cache = np.random.random(size=(cache_len,))
        self.random_prob_ptr = cache_len - 1

    def get_random_prob(self):
        """Get a random number."""
        value = self.random_prob_cache[self.random_prob_ptr]
        self.random_prob_ptr -= 1
        if self.random_prob_ptr == -1:
            self.reset_random_prob()
        return value

    def get_random_token(self):
        """Get a Random token."""
        token = self.token_list[self.token_ptr]
        self.token_ptr -= 1
        if self.token_ptr == -1:
            self.reset_token_list()
        return token

    def get_insert_token(self, word):
        """Get a replace token."""
        # Insert word choose
        return ' '.join([word] * 2)

    def get_delete_token(self):
        """Get a replace token."""
        # Insert word choose
        return ''


class RandomReplace(EfficientRandomGen):
    """Uniformly replace word with random words in the vocab."""

    def __init__(self, token_prob, vocab, show_example=False):
        self.token_prob = token_prob
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.reset_token_list()
        self.reset_random_prob()
        self.show_example = show_example

    def __call__(self, example):
        example.word_list_a = self.replace_tokens(example.word_list_a)
        if example.text_b:
            example.word_list_b = self.replace_tokens(example.word_list_b)
        return example

    def replace_tokens(self, tokens):
        """Replace tokens randomly."""
        if len(tokens) >= 3:
            if self.show_example:
                logger.debug("before Random replace word augment: {:s}".format(
                    " ".join(tokens)))
            for i in range(len(tokens)):
                if self.get_random_prob() < self.token_prob:
                    tokens[i] = self.get_random_token()
            if self.show_example:
                logger.debug("after  Random replace word augment: {:s}".format(
                    " ".join(tokens)))
        return tokens

    def reset_token_list(self):
        """Generate many random tokens at the same time and cache them."""
        self.token_list = list(self.vocab.keys())
        self.token_ptr = len(self.token_list) - 1
        np.random.shuffle(self.token_list)


class InsertReplace(EfficientRandomGen):
    """Uniformly replace word with insert repeat words in the vocab."""

    def __init__(self, token_prob, vocab, show_example=False):
        self.token_prob = token_prob
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.reset_token_list()
        self.reset_random_prob()
        self.show_example = show_example

    def __call__(self, example):
        example.word_list_a = self.replace_tokens(example.word_list_a)
        if example.text_b:
            example.word_list_b = self.replace_tokens(example.word_list_b)
        return example

    def replace_tokens(self, tokens):
        """Replace tokens randomly."""
        if len(tokens) >= 3:
            if self.show_example:
                logger.debug("before Insert replace word augment: {:s}".format(
                    " ".join(tokens)))
            for i in range(len(tokens)):
                if self.get_random_prob() < self.token_prob:
                    tokens[i] = self.get_insert_token(tokens[i])
            if self.show_example:
                logger.debug("after  Insert replace word augment: {:s}".format(
                    " ".join(tokens)))
        return tokens

    def reset_token_list(self):
        """Generate many random tokens at the same time and cache them."""
        self.token_list = list(self.vocab.keys())
        self.token_ptr = len(self.token_list) - 1
        np.random.shuffle(self.token_list)


class DeleteReplace(EfficientRandomGen):
    """Uniformly replace word with delete words in the vocab."""

    def __init__(self, token_prob, vocab, show_example=False):
        self.token_prob = token_prob
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.reset_token_list()
        self.reset_random_prob()
        self.show_example = show_example

    def __call__(self, example):
        example.word_list_a = self.replace_tokens(example.word_list_a)
        if example.text_b:
            example.word_list_b = self.replace_tokens(example.word_list_b)
        return example

    def replace_tokens(self, tokens):
        """Replace tokens randomly."""
        if len(tokens) >= 3:
            if self.show_example:
                logger.debug("before Delete replace word augment: {:s}".format(
                    " ".join(tokens)))
            for i in range(len(tokens)):
                if self.get_random_prob() < self.token_prob:
                    tokens[i] = self.get_delete_token()
            if self.show_example:
                logger.debug("after  Delete replace word augment: {:s}".format(
                    " ".join(tokens)))
        return tokens

    def reset_token_list(self):
        """Generate many random tokens at the same time and cache them."""
        self.token_list = list(self.vocab.keys())
        self.token_ptr = len(self.token_list) - 1
        np.random.shuffle(self.token_list)


def get_data_stats(examples):
    """Compute the IDF score for each word. Then compute the TF-IDF score."""
    word_doc_freq = collections.defaultdict(int)
    # Compute IDF
    for i in range(len(examples)):
        cur_word_dict = {}
        cur_sent = copy.deepcopy(examples[i].word_list_a)
        if examples[i].text_b:
            cur_sent += examples[i].word_list_b
        for word in cur_sent:
            cur_word_dict[word] = 1
        for word in cur_word_dict:
            word_doc_freq[word] += 1
    idf = {}
    for word in word_doc_freq:
        idf[word] = math.log(len(examples) * 1. / word_doc_freq[word])
    # Compute TF-IDF
    tf_idf = {}
    for i in range(len(examples)):
        cur_sent = copy.deepcopy(examples[i].word_list_a)
        if examples[i].text_b:
            cur_sent += examples[i].word_list_b
        for word in cur_sent:
            if word not in tf_idf:
                tf_idf[word] = 0
            tf_idf[word] += 1. / len(cur_sent) * idf[word]
    return {
        "idf": idf,
        "tf_idf": tf_idf,
    }


class MixEfficientRandomGen(EfficientRandomGen):
    """Add word2vec to Random Gen"""

    def __init__(self,
                 similar_prob=0.7,
                 random_prob=0.1,
                 delete_prob=0.1,
                 insert_prob=0.1):
        from text2vec import Vector
        super(MixEfficientRandomGen, self).__init__()
        vec = Vector()
        vec.load_model()
        self.word2vec_model = vec.model.w2v
        # Insert replace prob
        self.insert_prob = insert_prob
        # Delete replace prob
        self.delete_prob = delete_prob
        # Random replace prob
        self.random_prob = random_prob
        # Similar replace prob
        self.similar_prob = similar_prob

    def get_similar_token(self, word):
        """Get a Similar replace token."""
        if word in self.word2vec_model.key_to_index:
            target_candidate = self.word2vec_model.similar_by_word(word, topn=3)
            target_words = [w for w, p in target_candidate if w]
            if len(target_words) > 1:
                word = np.random.choice(target_words, size=1).tolist()[0]
                return word
        return word

    def get_replace_token(self, word):
        """Get a replace token."""
        r_prob = np.random.rand()
        # Similar choose prob
        if r_prob < self.similar_prob:
            word = self.get_similar_token(word)
        elif r_prob - self.similar_prob < self.random_prob:
            word = self.get_random_token()
        elif r_prob - self.similar_prob - self.random_prob < self.delete_prob:
            word = self.get_delete_token()
        else:
            word = self.get_insert_token(word)
        return word


class TfIdfWordReplace(MixEfficientRandomGen):
    """TF-IDF Based Word Replacement."""

    def __init__(self, token_prob,
                 data_stats,
                 show_example=False,
                 similar_prob=0.7,
                 random_prob=0.1,
                 delete_prob=0.1,
                 insert_prob=0.1):
        super(TfIdfWordReplace,
              self).__init__(similar_prob=similar_prob,
                             random_prob=random_prob,
                             delete_prob=delete_prob,
                             insert_prob=insert_prob)
        self.token_prob = token_prob
        self.data_stats = data_stats
        self.idf = data_stats["idf"]
        self.tf_idf = data_stats["tf_idf"]
        data_stats = copy.deepcopy(data_stats)
        tf_idf_items = data_stats["tf_idf"].items()
        tf_idf_items = sorted(tf_idf_items, key=lambda item: -item[1])
        self.tf_idf_keys = []
        self.tf_idf_values = []
        for key, value in tf_idf_items:
            self.tf_idf_keys += [key]
            self.tf_idf_values += [value]
        self.normalized_tf_idf = np.array(self.tf_idf_values)
        self.normalized_tf_idf = self.normalized_tf_idf.max() - self.normalized_tf_idf
        self.normalized_tf_idf = self.normalized_tf_idf / self.normalized_tf_idf.sum()
        self.reset_token_list()
        self.reset_random_prob()
        self.show_example = show_example

    def get_replace_prob(self, all_words):
        """Compute the probability of replacing tokens in a sentence."""
        cur_tf_idf = collections.defaultdict(int)
        for word in all_words:
            cur_tf_idf[word] += 1. / len(all_words) * self.idf[word]
        replace_prob = []
        for word in all_words:
            replace_prob += [cur_tf_idf[word]]
        replace_prob = np.array(replace_prob)
        replace_prob = np.max(replace_prob) - replace_prob
        if replace_prob.sum() != 0.0:
            replace_prob = replace_prob / replace_prob.sum() * self.token_prob * len(all_words)
        return replace_prob

    def __call__(self, example):
        all_words = copy.deepcopy(example.word_list_a)
        if example.text_b:
            all_words += example.word_list_b

        if self.show_example:
            logger.debug("before tfidf aug: {:s}".format(
                " ".join(all_words)))

        replace_prob = self.get_replace_prob(all_words)
        example.word_list_a = self.replace_tokens(
            example.word_list_a,
            replace_prob[:len(example.word_list_a)]
        )
        if example.text_b:
            example.word_list_b = self.replace_tokens(
                example.word_list_b,
                replace_prob[len(example.word_list_a):]
            )

        if self.show_example:
            all_words = copy.deepcopy(example.word_list_a)
            if example.text_b:
                all_words += example.word_list_b
            logger.debug("after  tfidf aug: {:s}".format(
                " ".join(all_words)))
        return example

    def replace_tokens(self, word_list, replace_prob):
        """Replace tokens in a sentence."""
        for i in range(len(word_list)):
            if self.get_random_prob() < replace_prob[i]:
                # Use Random: word_list[i] = self.get_random_token()
                word_list[i] = self.get_similar_token(word_list[i])
        return word_list

    def reset_token_list(self):
        cache_len = len(self.tf_idf_keys)
        token_list_idx = np.random.choice(
            cache_len, (cache_len,), p=self.normalized_tf_idf)
        self.token_list = []
        for idx in token_list_idx:
            self.token_list += [self.tf_idf_keys[idx]]
        self.token_ptr = len(self.token_list) - 1
        logger.info("sampled token list: {}".format(self.token_list))


class MixWordReplace(TfIdfWordReplace):
    """Multi Method Based Word Replacement."""

    def __init__(self, token_prob,
                 data_stats,
                 show_example=False,
                 similar_prob=0.7,
                 random_prob=0.1,
                 delete_prob=0.1,
                 insert_prob=0.1):
        super(MixWordReplace,
              self).__init__(token_prob,
                             data_stats,
                             show_example=show_example,
                             similar_prob=similar_prob,
                             random_prob=random_prob,
                             delete_prob=delete_prob,
                             insert_prob=insert_prob)

    def replace_tokens(self, word_list, replace_prob):
        """Replace tokens in a sentence."""
        for i in range(len(word_list)):
            if self.get_random_prob() < replace_prob[i]:
                word_list[i] = self.get_replace_token(word_list[i])
        return word_list


def word_augment(examples, aug_ops, vocab, data_stats, show_example=False):
    """Word level augmentations. Used before augmentation."""
    if aug_ops:
        logger.info("\n>>Using augmentation {}".format(aug_ops))
        token_prob = float(aug_ops.split("-")[1])
        if aug_ops.startswith("random"):
            op = RandomReplace(token_prob, vocab, show_example)
        elif aug_ops.startswith("insert"):
            op = InsertReplace(token_prob, vocab, show_example)
        elif aug_ops.startswith("delete"):
            op = DeleteReplace(token_prob, vocab, show_example)
        elif aug_ops.startswith("tfidf"):
            op = TfIdfWordReplace(token_prob, data_stats, show_example)
        else:
            op = MixWordReplace(token_prob, data_stats, show_example)
        for i in range(len(examples)):
            examples[i] = op(examples[i])
    return examples
