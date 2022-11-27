# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys

sys.path.append('..')
from textgen.t5 import T5Model

if __name__ == '__main__':
    model = T5Model('t5', 'shibing624/t5-chinese-couplet', args={"eval_batch_size": 64})
    sentences = ["对联：丹枫江冷人初去", "对联：春回大地，对对黄莺鸣暖树", "对联：书香醉我凌云梦"]
    print("inputs:", sentences)
    print("outputs:", model.predict(sentences))
