# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install gradio
"""
import sys
import gradio as gr
sys.path.append('../..')
from textgen import T5Model

# 中文对联生成模型(shibing624/t5-chinese-couplet)
model = T5Model("t5", "shibing624/t5-chinese-couplet")


def ai_text(sentence):
    out_sentences = model.predict([sentence])
    print("{} \t out: {}".format(sentence, out_sentences[0]))
    return out_sentences[0]


if __name__ == '__main__':
    examples = [
        ['对联：丹枫江冷人初去'],
        ['对联：春回大地，对对黄莺鸣暖树'],
        ['对联：书香醉我凌云梦'],
        ['对联：灵蛇出洞千山秀'],
        ['对联：晚风摇树树还挺'],
        ['对联：幸福体彩彩民喜爱，玩出幸福'],
        ['对联：光华照眼来，谁敢歌吟？诗仙诗圣空千古'],

    ]
    input = gr.inputs.Textbox(lines=4, placeholder="Enter Sentence")

    output_text = gr.outputs.Textbox()
    gr.Interface(ai_text,
                 inputs=[input],
                 outputs=[output_text],
                 # theme="grass",
                 title="Chinese Couplet Generation Model",
                 description="Copy or input Chinese text here. Submit and the machine will generate left text.",
                 article="Link to <a href='https://github.com/shibing624/textgen' style='color:blue;' target='_blank\'>Github REPO</a>",
                 examples=examples
                 ).launch()
