# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install gradio
"""

import gradio as gr
from textgen.seq2seq import ConvSeq2SeqModel

# 中文生成模型(ConvSeq2Seq)
model = ConvSeq2SeqModel(model_dir='seq2seq/outputs/convseq2seq_zh/', max_length=50)


def ai_text(sentence):
    out_sentences = model.predict([sentence])
    print("{} \t out: {}".format(sentence, out_sentences[0]))

    return out_sentences[0]


if __name__ == '__main__':
    examples = [
        ['什么是ai'],
        ['你是什么类型的计算机'],
    ]
    input = gr.inputs.Textbox(lines=4, placeholder="Enter Sentence")

    output_text = gr.outputs.Textbox()
    gr.Interface(ai_text,
                 inputs=[input],
                 outputs=[output_text],
                 # theme="grass",
                 title="Chinese Text Generation Model",
                 description="Copy or input Chinese text here. Submit and the machine will generate left text.",
                 article="Link to <a href='https://github.com/shibing624/textgen' style='color:blue;' target='_blank\'>Github REPO</a>",
                 examples=examples
                 ).launch()
