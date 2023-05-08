# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import pytest

sys.path.append('..')
from textgen import ChatGlmArgs, ChatGlmModel


def test_csc():
    from pycorrector.utils import eval
    model = ChatGlmModel(
        'chatglm', "THUDM/chatglm-6b", peft_name="shibing624/chatglm-6b-csc-zh-lora",
        args={'use_lora': True, 'eval_batch_size': 8, "max_length": 128}
    )
    sents = ['问：对下面中文拼写纠错：\n少先队员因该为老人让坐。\n答：',
             '问：对下面中文拼写纠错：\n下个星期，我跟我朋唷打算去法国玩儿。\n答：']

    def batch_correct(sentences):
        prompts = [f"问：对下面中文拼写纠错：\n{s}\n答：" for s in sentences]
        r = model.predict(prompts)
        return [s.split('\n')[0] for s in r]

    response = batch_correct(sents)
    print(response)

    eval.eval_sighan2015_by_model_batch(batch_correct)
    # Sentence Level: acc:0.5264, precision:0.5263, recall:0.4052, f1:0.4579, cost time:253.49 s, total num: 1100
    # 虽然F1值低于macbert4csc(f1:0.7742)等模型，但这个纠错结果带句子润色的效果，看结果case大多数是比ground truth结果句子更通顺流畅，
    # 我觉得是当前效果最好的纠错模型之一，比较像ChatGPT效果。


def test_origin():
    m = ChatGlmModel('chatglm', "THUDM/chatglm-6b", args={'use_lora': False})
    response, history = m.chat("你好", history=[])
    print(response)
    assert len(response) > 0
    response, history = m.chat("晚上睡不着应该怎么办", history=history)
    print(response)
    assert len(response) > 0


def test_origin_int4():
    m = ChatGlmModel('chatglm', "THUDM/chatglm-6b-int4", args={'use_lora': False, "quantization_bit": None},
                     cuda_device=0)
    response, history = m.chat("你好", history=[], max_length=20)
    print(response)
    assert len(response) > 0


def test_origin_int4_cpu():
    m = ChatGlmModel('chatglm', "THUDM/chatglm-6b-int4", use_cuda=False,
                     args={'use_lora': False, "quantization_bit": None},
                     cuda_device=0)
    response, history = m.chat("你好", history=[], max_length=20)
    print(response)
    assert len(response) > 0
