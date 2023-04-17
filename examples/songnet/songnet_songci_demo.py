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
    model = SongNetModel(model_type='songnet', model_name='shibing624/songnet-base-chinese-songci')
    sentences = [
        "严蕊<s1>如梦令<s2>道是梨花不是。</s>道是杏花不是。</s>白白与红红，别是东风情味。</s>曾记。</s>曾记。</s>人在武陵微醉。",
        "张抡<s1>春光好<s2>烟澹澹，雨。</s>水溶溶。</s>帖水落花飞不起，小桥东。</s>翩翩怨蝶愁蜂。</s>绕芳丛。</s>恋馀红。</s>不恨无情桥下水，恨东风。"
    ]
    print("inputs:", sentences)
    print("outputs:", model.generate(sentences))
    sentences = [
        "秦湛<s1>卜算子<s2>_____，____到。_______，____俏。_____，____报。_______，____笑。",
        "秦湛<s1>卜算子<s2>_雨___，____到。______冰，____俏。____春，__春_报。__山花___，____笑。"
    ]
    print("inputs:", sentences)
    print("outputs:", model.fill_mask(sentences))

