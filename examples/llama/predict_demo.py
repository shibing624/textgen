# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('../..')
from textgen import LlamaModel


def generate_prompt(instruction):
    return f"""请根据我给出的app、app的描述信息，生成1条最合适的关于app下载的搜索广告标题和描述。期望的结果是：\n
- 格式以”标题：”，“描述：”开头，标题大于20字.\n\n### Instruction:{instruction}\n\n### Response:"""

model = LlamaModel("llama", "/apdcephfs_cq3/share_2973545/data/models/shibing624/chinese-alpaca-plus-7b-hf")
# model = LlamaModel("llama", "decapoda-research/llama-13b-hf", lora_name="shibing624/llama-13b-belle-zh-lora")

predict_sentence = generate_prompt("失眠怎么办？")
r = model.predict([predict_sentence])
print(r)

sents = [
    "失眠怎么办？",
    '问：用一句话描述地球为什么是独一无二的。\n答：',
    '问：给定两个数字，计算它们的平均值。 数字: 25, 36\n答：',
    '问：基于以下提示填写以下句子的空格。 空格应填写一个形容词 句子: ______出去享受户外活动，包括在公园里散步，穿过树林或在海岸边散步。\n答：',
    "Tell me about alpacas.",
    "Tell me about the president of Mexico in 2019.",
    "Tell me about the king of France in 2019.",
    "List all Canadian provinces in alphabetical order.",
    "Write a Python program that prints the first 10 Fibonacci numbers.",
    "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
    "Tell me five words that rhyme with 'shock'.",
    "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    "Count up from 1 to 500.",
]

ad_gen_sents = [
    "app:嘉睿驰鑫\napp描述:人们通过扫码的方式来取袋，在商场、农贸市场、便利店等各个场所都可以使用，也能够为环保出一份力。 嘉睿驰鑫APP是一个专注于环保袋投放服务的手机助手，可以广泛应用于生活中的不同场景，让大家随时都可以轻松的进行环保袋的获取。提供更加便捷的服务"
    "app:柠檬轻断食\napp描述:恶~龙~咆~哮~ 喵呜~~ 咦？胖友你来了？ 你最近有听大家讨论从XXL到S码的秘诀吗？据说，不节食、不运动也能减肥哦！好像就是这个《柠檬轻断食》 你说什么是轻断食？那本喵就告诉你 ‘轻断食’又叫‘间歇性断食’，是近些年来风靡一时的一种饮食方法，这种方法既能减肥，又能提高胰岛素敏感度、增强活力、预防阿尔茨海默病、延缓衰老、消除炎症哦！"

]

predict_sentences = [generate_prompt(s) for s in ad_gen_sents]
res = model.predict(sents)
for s, i in zip(sents, res):
    print(s, i)
    print()
