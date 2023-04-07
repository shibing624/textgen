# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install gradio
"""
import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, device_map='auto')
model = PeftModel.from_pretrained(model, "shibing624/chatglm-6b-csc-zh-lora")
if torch.cuda.is_available():
    model = model.half().cuda()
else:
    model = model.cpu().float()
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

sents = ['对下面中文拼写纠错：\n少先队员因该为老人让坐。\n答：',
         '对下面中文拼写纠错：\n下个星期，我跟我朋唷打算去法国玩儿。\n答：']
for s in sents:
    response = model.chat(tokenizer, s, max_length=128, eos_token_id=tokenizer.eos_token_id)
    print(response)


def ai_text(text):
    outputs = model.chat(tokenizer, text, max_length=128, eos_token_id=tokenizer.eos_token_id)
    return outputs


if __name__ == '__main__':
    examples = [
        ['对下面中文拼写纠错：\n真麻烦你了。希望你们好好的跳无\n答：'],
        ['对下面中文拼写纠错：\n机七学习是人工智能领遇最能体现智能的一个分知\n答：'],
        ['对下面中文拼写纠错：\n今天心情很好\n答：'],
        ['对下面中文拼写纠错：\n他法语说的很好，的语也不错\n答：'],
        ['对下面中文拼写纠错：\n他们的吵翻很不错，再说他们做的咖喱鸡也好吃\n答：'],
        ['对下面中文拼写纠错：\n少先队员因该为老人让坐。\n答：'],
        ['对下面中文拼写纠错：\n下个星期，我跟我朋唷打算去法国玩儿。\n答：']
    ]

    gr.Interface(
        ai_text,
        inputs="textbox",
        outputs=[
            gr.outputs.Textbox()
        ],
        theme="grass",
        title="Chinese Spelling Correction LoRA Model chatglm-6b-csc-zh-lora",
        description="Copy or input error Chinese text. Submit and the machine will correct text.",
        article="Link to <a href='https://github.com/shibing624/lmft' style='color:blue;' target='_blank\'>Github REPO</a>",
        examples=examples
    ).launch()
