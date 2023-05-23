# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import pytest
import shutil
import os
from loguru import logger
from torch.utils.data import Dataset

sys.path.append('..')
from textgen import ChatGlmArgs


class MyTestDataset(Dataset):
    pass


def test_save_args():
    args = ChatGlmArgs()
    os.makedirs('outputs/', exist_ok=True)
    print('old', args)
    logger.info(args)
    args.adafactor_clip_threshold = 2.0
    args.dataset_class = MyTestDataset()
    print('new', args)
    try:
        args.save('outputs/')
    except Exception as e:
        print(e)
    m = ChatGlmArgs()
    m.load('outputs/')
    print('new', m)
    logger.info(m)
    assert m.adafactor_clip_threshold == 2.0
    shutil.rmtree('outputs/')
