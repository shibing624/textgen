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

from textgen.utils.log import logger

min_token_num = 3


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
        return ''.join([word] * 2)

    def get_delete_token(self):
        """Get a replace token."""
        # Insert word choose
        return ''


class RandomReplace(EfficientRandomGen):
    """Uniformly replace word with random words in the vocab."""

    def __init__(self, token_prob, vocab):
        self.token_prob = token_prob
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.reset_token_list()
        self.reset_random_prob()

    def __call__(self, tokens):
        return self.replace_tokens(tokens)

    def replace_tokens(self, tokens):
        """
        Replace tokens randomly.
        :param tokens: list
        :return: tokens, details
        tokens, list
        details, list eg: [(old_token, new_token, start_idx, end_idx), ...]
        """
        details = []
        idx = 0
        if len(tokens) >= min_token_num:
            for token in tokens:
                old_token = token
                if self.get_random_prob() < self.token_prob:
                    token = self.get_random_token()
                    details.append((old_token, token, idx, idx + len(token)))
                idx += len(token)
        return tokens, details

    def reset_token_list(self):
        """Generate many random tokens at the same time and cache them."""
        self.token_list = list(self.vocab.keys())
        self.token_ptr = len(self.token_list) - 1
        np.random.shuffle(self.token_list)


class InsertReplace(EfficientRandomGen):
    """Uniformly replace word with insert repeat words in the vocab."""

    def __init__(self, token_prob, vocab):
        self.token_prob = token_prob
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.reset_token_list()
        self.reset_random_prob()

    def __call__(self, tokens):
        return self.replace_tokens(tokens)

    def replace_tokens(self, tokens):
        """
        Replace tokens with insert data.
        :param tokens: list
        :return: tokens, details
        tokens, list
        details, list eg: [(old_token, new_token, start_idx, end_idx), ...]
        """
        details = []
        idx = 0
        if len(tokens) >= min_token_num:
            for token in tokens:
                old_token = token
                if self.get_random_prob() < self.token_prob:
                    token = self.get_insert_token(token)
                    details.append((old_token, token, idx, idx + len(token)))
                idx += len(token)
        return tokens, details

    def reset_token_list(self):
        """Generate many random tokens at the same time and cache them."""
        self.token_list = list(self.vocab.keys())
        self.token_ptr = len(self.token_list) - 1
        np.random.shuffle(self.token_list)


class DeleteReplace(EfficientRandomGen):
    """Uniformly replace word with delete words in the vocab."""

    def __init__(self, token_prob, vocab):
        self.token_prob = token_prob
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.reset_token_list()
        self.reset_random_prob()

    def __call__(self, tokens):
        return self.replace_tokens(tokens)

    def replace_tokens(self, tokens):
        """
        Replace tokens with insert data.
        :param tokens: list
        :return: tokens, details
        tokens, list
        details, list eg: [(old_token, new_token, start_idx, end_idx), ...]
        """
        details = []
        idx = 0
        if len(tokens) >= min_token_num:
            for token in tokens:
                old_token = token
                if self.get_random_prob() < self.token_prob:
                    token = self.get_delete_token()
                    details.append((old_token, token, idx, idx + len(token)))
                idx += len(token)
        return tokens, details

    def reset_token_list(self):
        """Generate many random tokens at the same time and cache them."""
        self.token_list = list(self.vocab.keys())
        self.token_ptr = len(self.token_list) - 1
        np.random.shuffle(self.token_list)


def get_data_idf(tokenized_sentence_list):
    """Compute the IDF score for each word. Then compute the TF-IDF score."""
    word_doc_freq = collections.defaultdict(int)
    # Compute IDF
    for cur_sent in tokenized_sentence_list:
        cur_word_dict = {}
        for word in cur_sent:
            cur_word_dict[word] = 1
        for word in cur_word_dict:
            word_doc_freq[word] += 1
    idf = {}
    for word in word_doc_freq:
        idf[word] = math.log(len(tokenized_sentence_list) * 1. / word_doc_freq[word])
    # Compute TF-IDF
    tf_idf = {}
    for cur_sent in tokenized_sentence_list:
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
                 w2v,
                 similar_prob=0.7,
                 random_prob=0.1,
                 delete_prob=0.1,
                 insert_prob=0.1):
        super(MixEfficientRandomGen, self).__init__()
        self.word2vec_model = w2v
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

    def __init__(self,
                 w2v,
                 token_prob,
                 data_idf,
                 similar_prob=0.7,
                 random_prob=0.1,
                 delete_prob=0.1,
                 insert_prob=0.1):
        super(TfIdfWordReplace, self).__init__(w2v,
                                               similar_prob=similar_prob,
                                               random_prob=random_prob,
                                               delete_prob=delete_prob,
                                               insert_prob=insert_prob)
        self.token_prob = token_prob
        self.idf = data_idf["idf"]
        self.tf_idf = data_idf["tf_idf"]
        if not self.idf:
            logger.error('sentence_list must set in tfidf word replace.')
            raise ValueError("idf is None.")
        data_idf = copy.deepcopy(data_idf)
        tf_idf_items = data_idf["tf_idf"].items()
        tf_idf_items = sorted(tf_idf_items, key=lambda item: -item[1])
        self.tf_idf_keys = []
        self.tf_idf_values = []
        for key, value in tf_idf_items:
            self.tf_idf_keys += [key]
            self.tf_idf_values += [value]
        self.normalized_tf_idf = np.array(self.tf_idf_values)
        self.normalized_tf_idf = max(self.normalized_tf_idf) - self.normalized_tf_idf
        self.normalized_tf_idf = self.normalized_tf_idf / self.normalized_tf_idf.sum()
        self.reset_token_list()
        self.reset_random_prob()

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

    def __call__(self, tokens):
        """
        Replace tokens with tfidf data.
        :param tokens: list
        :return: tokens, details
        tokens, list
        details, list eg: [(old_token, new_token, start_idx, end_idx), ...]
        """
        new_tokens = []
        details = []
        if len(tokens) >= min_token_num:
            replace_prob = self.get_replace_prob(tokens)
            new_tokens, details = self.replace_tokens(tokens, replace_prob[:len(tokens)])
        return new_tokens, details

    def replace_tokens(self, word_list, replace_prob):
        """Replace tokens with tfidf similar word"""
        details = []
        idx = 0
        for i in range(len(word_list)):
            old_token = word_list[i]
            if self.get_random_prob() < replace_prob[i]:
                # Use Tfidf find similar token
                word_list[i] = self.get_similar_token(word_list[i])
                details.append((old_token, word_list[i], idx, idx + len(word_list[i])))
            idx += len(word_list[i])
        return word_list, details

    def reset_token_list(self):
        cache_len = len(self.tf_idf_keys)
        token_list_idx = np.random.choice(
            cache_len, (cache_len,), p=self.normalized_tf_idf)
        self.token_list = []
        for idx in token_list_idx:
            self.token_list += [self.tf_idf_keys[idx]]
        self.token_ptr = len(self.token_list) - 1
        logger.debug("sampled token list: {}".format(self.token_list))


class MixWordReplace(TfIdfWordReplace):
    """Multi Method Based Word Replacement."""

    def __init__(self,
                 w2v,
                 token_prob,
                 data_idf,
                 similar_prob=0.7,
                 random_prob=0.1,
                 delete_prob=0.1,
                 insert_prob=0.1):
        super(MixWordReplace, self).__init__(w2v,
                                             token_prob,
                                             data_idf,
                                             similar_prob=similar_prob,
                                             random_prob=random_prob,
                                             delete_prob=delete_prob,
                                             insert_prob=insert_prob)

    def replace_tokens(self, word_list, replace_prob):
        """Replace tokens with mix method."""
        details = []
        idx = 0
        for i in range(len(word_list)):
            old_token = word_list[i]
            if self.get_random_prob() < replace_prob[i]:
                word_list[i] = self.get_replace_token(word_list[i])
                details.append((old_token, word_list[i], idx, idx + len(word_list[i])))
            idx += len(word_list[i])
        return word_list, details
