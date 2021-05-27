# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Sentence level augmentations: back translation.
"""

import math
import random

from textgen.augmentation import translate_api
from textgen.augmentation.example import InputExample
from textgen.utils.log import logger

use_min_length = 10
use_max_length_diff_ratio = 0.5


def replace_with_length_check(
        ori_text,
        new_text,
        use_min_length,
        use_max_length_diff_ratio):
    """Use new_text if the text length satisfies several constraints."""
    if len(ori_text) < use_min_length or len(new_text) < use_min_length:
        logger.debug("not replacing due to short text: ori: {} new: {}".format(ori_text, new_text))
        return ori_text
    length_diff_ratio = 1.0 * (len(new_text) - len(ori_text)) / len(ori_text)
    if math.fabs(length_diff_ratio) > use_max_length_diff_ratio:
        logger.debug("not replacing due to too different text length: ori: {} new: {}".format(ori_text, new_text))
        return ori_text
    return new_text


def back_translation(examples, bt_prob=0.1, from_lang='zh'):
    """Run back translation."""
    logger.info("running back_translation augmentation")
    assert 0 <= bt_prob <= 1, "prob must be float num"
    aug_examples = []
    aug_cnt = 0
    for i in range(len(examples)):
        ori_example = examples[i]
        text_a = ori_example.text_a
        text_b = ori_example.text_b
        if random.random() < bt_prob:
            q = ori_example.text_a
            if ori_example.text_b:
                q += '\n' + ori_example.text_b
            bt_result = translate_api.back_translate(q, from_lang=from_lang)
            if bt_result:
                bt_text_a = bt_result[0][0]
                text_a = replace_with_length_check(
                    ori_example.text_a,
                    bt_text_a,
                    use_min_length,
                    use_max_length_diff_ratio)
                aug_cnt += 1
                if ori_example.text_b and len(bt_result) > 1:
                    bt_text_b = bt_result[0][1]
                    text_b = replace_with_length_check(
                        ori_example.text_b,
                        bt_text_b,
                        use_min_length,
                        use_max_length_diff_ratio)

        example = InputExample(
            guid=ori_example.guid,
            text_a=text_a,
            text_b=text_b,
            label=ori_example.label)

        if i % 10000 == 0:
            logger.debug("ori:{} {}".format(ori_example.text_a, ori_example.text_b))
            logger.debug("new:{} {}".format(example.text_a, example.text_b))
            logger.debug("processing example # {:d}".format(i))
        aug_examples.append(example)

    logger.debug("applied back translation for {:.1f} percent of data".format(
        aug_cnt * 1. / len(examples) * 100))
    logger.info("finishing running back translation augmentation")
    return aug_examples


def sent_augment(examples, aug_ops, from_lang='zh'):
    """Sentence level augmentations. Used before augmentation."""
    aug_examples = examples
    if aug_ops:
        if aug_ops.startswith("bt"):
            bt_prob = float(aug_ops.split("-")[1])
            aug_examples = back_translation(examples, bt_prob, from_lang=from_lang)
    return aug_examples
