# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append('..')
from textgen.augmentation.text_augment import augment

if __name__ == '__main__':
    a = ['主要研究机器学习、深度学习、计算机视觉、智能对话系统相关内容',
         '晚上肚子好难受',
         '你会武功吗，我不会',
         '组装标题质量受限于广告主自提物料的片段质量，且表达丰富度有限'
         ]
    b = augment(a, aug_ops='tfidf-0.8', aug_type='word')
    print(a)
    for i in b:
        print(i)

    b = augment(a, aug_ops='bt-0.9', aug_type='sentence')
    print(a)
    for i in b:
        print(i)
