# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 中文句子续写
refer https://github.com/renmada/t5-pegasus-pytorch
原文:蓝蓝的天上有一朵白白的云
torch预测     	《蓝蓝的天上有一朵白白的云》是蓝蓝的天上有一朵白白的云创作的网络小说，发表于...
"""

import sys

from transformers.models.mt5 import MT5ForConditionalGeneration

sys.path.append('../..')
from textgen.t5.t5_utils import ZHTokenizer

# 模型名	MODEL_NAME
# t5-pegasus-base	imxly/t5-pegasus
# t5-pegasus-small	imxly/t5-pegasus-small
# t5-copy	imxly/t5-copy
# t5-copy-summary	imxly/t5-copy-summary

# torch版本
model_name = 'imxly/t5-pegasus'
tokenizer = ZHTokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

sents = [
    "蓝蓝的天上有一朵白白的云,",
    "五一假期欢乐多，我这几天也玩开心了，放假第二天，我就和姐姐一起去了外公家，现在这季节正是插秧的好时机，",
    "我的心爱之物 我有一只可爱的珍珠鳖，我们有一段特别的感情。 记得它是几个月前，叔叔从河边给我带来的。这只小鳖是灰黑色的，"
]


# 文本摘要任务
for text in sents:
    ids = tokenizer.encode(text, return_tensors='pt')
    output = model.generate(ids,
                            decoder_start_token_id=tokenizer.cls_token_id,
                            eos_token_id=tokenizer.sep_token_id,
                            top_k=1,
                            max_length=250).numpy()[0]
    r = ''.join(tokenizer.decode(output[1:])).replace(' ', '')
    print(r)
