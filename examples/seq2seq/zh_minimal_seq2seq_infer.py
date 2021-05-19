# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

import torch

sys.path.append('../..')
from textgen.seq2seq import Seq2SeqModel

use_cuda = torch.cuda.is_available()

# encoder_type=None, encoder_name=None, decoder_name=None, encoder_decoder_type=None, encoder_decoder_name=None,
model = Seq2SeqModel("bert", "outputs/encoder", "outputs/decoder", use_cuda=use_cuda, )

print(model.predict(["one", "four", "five"]))
print(model.predict(["two"]))
print(model.predict(["什么是ai", "你是什么类型的计算机"]))
