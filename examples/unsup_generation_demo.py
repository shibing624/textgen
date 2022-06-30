# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys

sys.path.append('..')
from textgen.unsup_generation import TglsModel
from textgen.unsup_generation.phrase import load_list

pwd_path = os.path.abspath(os.path.dirname(__file__))

samples = load_list(os.path.join(pwd_path, './data/ecommerce_comments.txt'))
docs_text = [
    ["挺好的，速度很快，也很实惠，不知效果如何",
     "产品没得说，买了以后就降价，心情不美丽。",
     "刚收到，包装很完整，不错",
     "发货速度很快，物流也不错，同一时间买的两个东东，一个先到一个还在路上。这个水水很喜欢，不过盖子真的开了。盖不牢了现在。",
     "包装的很好，是正品",
     "被种草兰蔻粉水三百元一大瓶囤货，希望是正品好用，收到的时候用保鲜膜包裹得严严实实，只敢买考拉自营的护肤品",
     ],
    ['很温和，清洗的也很干净，不油腻，很不错，会考虑回购，第一次考拉买护肤品，满意',
     '这款卸妆油我会无限回购的。即使我是油痘皮，也不会闷痘，同时在脸部按摩时，还能解决白头的脂肪粒的问题。用清水洗完脸后，非常的清爽。',
     '自从用了fancl之后就不用其他卸妆了，卸的舒服又干净',
     '买贵了，大润发才卖79。9。',
     ],
    samples
]
m = TglsModel(docs_text)
r = m.generate(samples[:500])
print('size:', len(r))
for review in r:
    print('\t' + review)
