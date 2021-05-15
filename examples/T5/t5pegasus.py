# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: refer https://github.com/renmada/t5-pegasus-pytorch
"""

from transformers.models.mt5 import MT5Config, MT5ForConditionalGeneration
from textgen.t5.t5_utils import ZHTokenizer

# config_path = 'D:\\BaiduNetdiskDownload\\chinese_t5_pegasus_base\\chinese_t5_pegasus_base\\config.json'
# checkpoint_path = 'D:\\BaiduNetdiskDownload\\chinese_t5_pegasus_base\\chinese_t5_pegasus_base\\model.ckpt'
# dict_path = 'D:\\BaiduNetdiskDownload\\chinese_t5_pegasus_base\\chinese_t5_pegasus_base\\vocab.txt'
# torch_model = './'

# 模型名	MODEL_NAME
# t5-pegasus-base	imxly/t5-pegasus
# t5-pegasus-small	imxly/t5-pegasus-small

# torch版本
tokenizer = ZHTokenizer.from_pretrained("imxly/t5-pegasus")
model = MT5ForConditionalGeneration.from_pretrained("imxly/t5-pegasus")

sents = [
    "蓝蓝的天上有一朵白白的云，",
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
    torch_res = ''.join(tokenizer.decode(output[1:])).replace(' ', '')
    print(torch_res)
