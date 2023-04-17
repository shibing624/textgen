# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys

sys.path.append('..')
from textgen.language_modeling import SongNetModel

if __name__ == '__main__':
    # Use fine-tuned model
    model = SongNetModel(model_type='songnet', model_name='shibing624/songnet-base-chinese-couplet')
    sentences = [
        "严蕊<s1>如梦令<s2>道是梨花不是。</s>道是杏花不是。</s>白白与红红，别是东风情味。</s>曾记。</s>曾记。</s>人在武陵微醉。",
        "<s1><s2>一句相思吟岁月</s>千杯美酒醉风情",
        "<s1><s2>几树梅花数竿竹</s>一潭秋水半屏山"
        "<s1><s2>未舍东江开口咏</s>且施妙手点睛来",
        "<s1><s2>一去二三里</s>烟村四五家",
    ]
    print("inputs:", sentences)
    print("outputs:", model.generate(sentences))
    sentences = [
        "<s1><s2>一句____月</s>千杯美酒__情",
        "<s1><s2>一去二三里</s>烟村__家</s>亭台__座</s>八__枝花",
    ]
    print("inputs:", sentences)
    print("outputs:", model.fill_mask(sentences))
