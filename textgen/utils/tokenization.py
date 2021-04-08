# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
@author:XuMing(xuming624@qq.com)
@description: Tokenization
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
from codecs import open

import six

re_han = re.compile("([\u4E00-\u9F5a-zA-Z0-9+#&]+)", re.U)


def split_2_short_text(text, include_symbol=True):
    """
    长句切分为短句
    :param text: str
    :param include_symbol: bool
    :return: (sentence, idx)
    """
    result = []
    blocks = re_han.split(text)
    start_idx = 0
    for blk in blocks:
        if not blk:
            continue
        if include_symbol:
            result.append((blk, start_idx))
        else:
            if re_han.match(blk):
                result.append((blk, start_idx))
        start_idx += len(blk)
    return result


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    return '\u4e00' <= uchar <= '\u9fa5'


def is_chinese_string(string):
    """判断是否全为汉字"""
    return all(is_chinese(c) for c in string)


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


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class FullTokenizer(object):
    """Runs end-to-end tokenization."""

    def __init__(self, vocab_file, do_lower_case=True, is_chinese2char=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                              is_chinese2char=is_chinese2char)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def tokenize_to_word(self, text):
        return self.basic_tokenizer.tokenize(text)

    def tokenize_to_wordpiece(self, tokens):
        split_tokens = []
        for token in tokens:
            split_tokens += self.wordpiece_tokenizer.tokenize(token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_tokens_to_ids(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_ids_to_tokens(self.inv_vocab, ids)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True, is_chinese2char=True):
        """Constructs a BasicTokenizer.
        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.is_chinese2char = is_chinese2char

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.is_chinese2char:
            text = self._tokenize_chinese_chars(text)
        else:
            text = self._tokenize_chinese_words(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _tokenize_chinese_words(self, text):
        """Chinese word segmentation"""
        import jieba
        output = []
        sentences = split_2_short_text(text, include_symbol=True)
        for sentence, idx in sentences:
            if is_chinese_string(sentence):
                output.append(" ".join(jieba.lcut(sentence)))
            else:
                output.append(sentence)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.
        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


if __name__ == '__main__':
    a = '晚上一个人好孤单，想:找附近的人陪陪我.'
    b = "unlabeled example, aug_copy_num is the index of the generated augmented. you don't know"
    import jieba
    from nltk.tokenize import word_tokenize

    a_vocab = set(jieba.lcut(a))
    a_vocab_path = '../../extra_data/a_vocab.txt'
    with open(a_vocab_path, 'w', encoding='utf-8') as f:
        for line in a_vocab:
            f.write(line + '\n')

    tokenizer = FullTokenizer(
        vocab_file=a_vocab_path, do_lower_case=False, is_chinese2char=True)
    word_list_a = tokenizer.tokenize_to_word(a)
    print('FullTokenizer:', word_list_a)

    tokenizer = FullTokenizer(
        vocab_file=a_vocab_path, do_lower_case=False, is_chinese2char=False)
    word_list_a = tokenizer.tokenize_to_word(a)
    print('FullTokenizer:', word_list_a)

    word_list_a = tokenizer.tokenize(a)
    print('FullTokenizer-tokenize:', word_list_a)

    tokenizer = BasicTokenizer(do_lower_case=False)
    word_list_a = tokenizer.tokenize(a)
    print('BasicTokenizer:', word_list_a)

    tokenizer = WordpieceTokenizer(a_vocab_path)
    word_list_a = tokenizer.tokenize(a)
    print('WordpieceTokenizer:', word_list_a)

    b_vocab = set(word_tokenize(b))
    b_vocab_path = '../../extra_data/b_vocab.txt'
    with open(b_vocab_path, 'w', encoding='utf-8') as f:
        for line in b_vocab:
            f.write(line + '\n')

    tokenizer = FullTokenizer(
        vocab_file=b_vocab_path, do_lower_case=False)
    word_list_b = tokenizer.tokenize_to_word(b)
    print('FullTokenizer:', word_list_b)

    word_list_b = tokenizer.tokenize(b)
    print('FullTokenizer-tokenize:', word_list_b)

    tokenizer = BasicTokenizer(do_lower_case=False)
    word_list_b = tokenizer.tokenize(b)
    print('BasicTokenizer:', word_list_b)

    tokenizer = WordpieceTokenizer(b_vocab_path)
    word_list_b = tokenizer.tokenize(b)
    print('WordpieceTokenizer:', word_list_b)
