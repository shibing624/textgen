# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('../..')
from textgen import LlamaModel

model = LlamaModel("llama", "decapoda-research/llama-7b-hf", lora_name="ziqingyang/chinese-alpaca-lora-7b")
r = model.predict(["失眠怎么办？"])
print(r)
