# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys

sys.path.append('..')
from textgen.unsup_generation import TglsModel, load_list

pwd_path = os.path.abspath(os.path.dirname(__file__))

samples = load_list(os.path.join(pwd_path, './data/ecommerce_comments.txt'))
m = TglsModel()
r = m.generate(samples[:500])
print('generated size:', len(r))
for review in r:
    print('\t' + review)
