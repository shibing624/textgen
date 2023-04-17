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
    a = '主要研究机器学习、深度学习、计算机视觉、智能对话系统相关内容'
    print(a)

    b = m.augment(a, aug_ops='random-0.2')
    print('random-0.2:', b)

    b = m.augment(a, aug_ops='insert-0.2')
    print('insert-0.2:', b)

    b = m.augment(a, aug_ops='delete-0.2')
    print('delete-0.2:', b)

    b = m.augment(a, aug_ops='tfidf-0.2')
    print('tfidf-0.2:', b)

    b = m.augment(a, aug_ops='mix-0.2')
    print('mix-0.2:', b)
