# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install gradio
"""
import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
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
        ['介绍下北京：\n答：'],
        ['你能干嘛：\n答：'],
        ['帮我写个青岛旅游的路线规划：\n答：'],
        ['写10个网易云热评文案：\n歌曲是关于失恋的\n答：'],
        ['写10个网易云热评文案：\n歌曲是关于快乐的\n答：'],
        ['帮我写个请假条：\n说明我今天要带家里小猫去做结扎，休一天事假\n答：'],
        ['提取下面的人名、时间和职位：\n常建良，男，1963年出生，工科学士，高级工程师，北京物资学院客座副教授\n答：'],
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
        article="Link to <a href='https://github.com/shibing624/textgen' style='color:blue;' target='_blank\'>Github REPO</a>",
        examples=examples
    ).launch()
