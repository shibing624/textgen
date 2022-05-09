# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tokenization
"""

import re


def tokenize_words(text):
    """Word segmentation"""
    output = []
    sentences = split_2_short_text(text, include_symbol=True)
    for sentence, idx in sentences:
        if is_chinese_string(sentence):
            import jieba
            output.extend(jieba.lcut(sentence))
        else:
            output.extend(whitespace_tokenize(sentence))
    return output


class Tokenizer(object):
    """Given Full tokenization."""

    def __init__(self, lower=True):
        self.lower = lower

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        res = []
        if len(text) == 0:
            return res

        if self.lower:
            text = text.lower()
        # for the multilingual (include: Chinese and English)
        res = tokenize_words(text)
        return res


def split_2_short_text(text, include_symbol=True):
    """
    长句切分为短句
    :param text: str
    :param include_symbol: bool
    :return: (sentence, idx)
    """
    re_han = re.compile("([\u4E00-\u9F5a-zA-Z0-9+#&]+)", re.U)
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


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def segment(sentence, is_cut2char=False, enable_pos=False):
    """
    切词
    :param sentence:
    :param is_cut2char: False use jieba.lcut; True use list(sentence)
    :param enable_pos: bool, enable POS
    :return: list
    """
    import jieba
    from jieba import posseg

    jieba.setLogLevel(log_level="ERROR")
    if enable_pos:
        if not is_cut2char:
            word_pos_seq = posseg.lcut(sentence)
            word_seq, pos_seq = [], []
            for w, p in word_pos_seq:
                word_seq.append(w)
                pos_seq.append(p)
            return word_seq, pos_seq
        else:
            word_seq = list(sentence)
            pos_seq = []
            for w in word_seq:
                w_p = posseg.lcut(w)
                pos_seq.append(w_p[0].flag)
            return word_seq, pos_seq
    else:
        if not is_cut2char:
            return jieba.lcut(sentence)
        else:
            return list(sentence)


if __name__ == '__main__':
    a = '晚上一个人好孤单，想:找附近的人陪陪我.'
    b = "unlabeled example, aug_copy_num is the index of the generated augmented. you don't know"

    t = Tokenizer()
    word_list_a = t.tokenize(a + b)
    print('VocabTokenizer-word:', word_list_a)
