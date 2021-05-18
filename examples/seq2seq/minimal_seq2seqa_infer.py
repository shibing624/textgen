# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import torch
import sys

sys.path.append('../..')
from textgen.seq2seq import Seq2SeqModel

use_cuda = torch.cuda.is_available()

# encoder_type=None, encoder_name=None, decoder_name=None, encoder_decoder_type=None, encoder_decoder_name=None,
model = Seq2SeqModel("bert", "outputs/best_model/encoder", "outputs/best_model/decoder", use_cuda=use_cuda, )

print(model.predict(["one", "four", "five"]))
print(model.predict(["one"]))
print(model.predict(["two"]))
