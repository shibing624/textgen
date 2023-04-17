# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('../..')
from textgen.augment import TextAugment

if __name__ == '__main__':
    m = TextAugment()
    sentences = [
        '主要研究机器学习、深度学习、计算机视觉、智能对话系统相关内容',
        '希望你们好好的跳舞',
        '少先队员应该为老人让座',
        '一只小鱼船浮在平静的河面上',
        '我的家乡是有名的渔米之乡',
    ]
    for sent in sentences:
        b = m.augment(sent, aug_ops='tfidf-0.2', similar_prob=1.0,
                      random_prob=0, delete_prob=0, insert_prob=0)
        print('tfidf-0.2:', b)
