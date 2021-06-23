# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from textgen.augment import TextAugment

if __name__ == '__main__':
    docs = ['主要研究机器学习、深度学习、计算机视觉、智能对话系统相关内容',
            '晚上肚子好难受',
            '你会武功吗，我不会',
            '组装标题质量受限于广告主自提物料的片段质量，且表达丰富度有限',
            '晚上一个人好孤单，想:找附近的人陪陪我.',
            ]
    m = TextAugment(sentence_list=docs)
    a = docs[0]
    print(a)

    b = m.augment(a, aug_ops='random-0.1')
    print('random-0.1:', b)

    b = m.augment(a, aug_ops='insert-0.1')
    print('insert-0.1:', b)

    b = m.augment(a, aug_ops='tfidf-0.2')
    print('tfidf-0.2:', b)

    b = m.augment(a, aug_ops='mix-0.1', similar_prob=0.1,
                  random_prob=0.4, delete_prob=0.3, insert_prob=0.2)
    print('mix-0.1:', b)

    b = m.augment(a, aug_ops='bt')
    print('bt:', b)
