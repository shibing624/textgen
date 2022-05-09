# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Sentence level augmentations: back translation.
"""

import math

from textgen.augment import translate_api
from loguru import logger


def replace_with_length_check(
        ori_text,
        new_text,
        use_min_length=10,
        use_max_length_diff_ratio=0.5):
    """Use new_text if the text length satisfies several constraints."""
    if len(ori_text) < use_min_length or len(new_text) < use_min_length:
        logger.debug("not replacing due to short text: ori: {} new: {}".format(ori_text, new_text))
        return ori_text
    length_diff_ratio = 1.0 * (len(new_text) - len(ori_text)) / len(ori_text)
    if math.fabs(length_diff_ratio) > use_max_length_diff_ratio:
        logger.debug("not replacing due to too different text length: ori: {} new: {}".format(ori_text, new_text))
        return ori_text
    return new_text


def back_translation(sentence, from_lang='zh', use_min_length=10, use_max_length_diff_ratio=0.5):
    """
    Run back translation with prob
    :param sentence:
    :param from_lang:
    :param use_min_length:
    :param use_max_length_diff_ratio:
    :return:
    """
    bt_result = translate_api.back_translate(sentence, from_lang=from_lang)
    if bt_result:
        bt_text = bt_result[0][0]
        sentence = replace_with_length_check(
            sentence,
            bt_text,
            use_min_length,
            use_max_length_diff_ratio)
    return sentence
