# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys

sys.path.append('../..')
from textgen import T5Model

if __name__ == '__main__':
    model = T5Model('t5', 'shibing624/prompt-t5-base-chinese', args={"eval_batch_size": 64})
    sentences = [
        "阅读下列对话。_女：小李，听说你的毕业设计主题是环保？男：对，我的作品所用的材料大都是一些废弃的日用品。女：都用了什么东西？_听者会怎么说？",
        "这篇新闻会出现在哪个栏目？吴绮莉独自返家神情落寞 再被问小龙女只说了7个字_选项：故事，文化，娱乐，体育，财经，房产，汽车，教育，科技，军事，旅游，国际，股票，农业，游戏_答案：",
        "我想知道下面两句话的意思是否相同。“怎么把借呗的钱转到余额宝”，“借呗刚刚才转钱到余额宝，可以重新扣一次款吗”是相同的吗？。选项：是的，不是。答案："
    ]
    print("inputs:", sentences)
    print("outputs:", model.predict(sentences))
