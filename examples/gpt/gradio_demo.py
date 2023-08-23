# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys
from threading import Thread

import gradio as gr
import mdtex2html
import torch
from transformers import TextIteratorStreamer

sys.path.append('../..')
from textgen import GptModel, get_conv_template


@torch.inference_mode()
def stream_generate_answer(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.0,
        context_len=2048
):
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    input_ids = tokenizer(prompt).input_ids
    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]
    generation_kwargs = dict(
        input_ids=torch.as_tensor([input_ids]).to(device),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    yield from streamer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='llama', type=str)
    parser.add_argument('--base_model', default='shibing624/chinese-alpaca-plus-7b-hf', type=str)
    parser.add_argument('--lora_model', default="", type=str, help="If not set, perform inference on the base model")
    parser.add_argument('--template_name', default="vicuna", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan-chat, chatglm2 etc.")
    args = parser.parse_args()
    print(args)
    m = GptModel(args.model_type, args.base_model, peft_name=args.lora_model)

    def postprocess(self, y):
        if y is None:
            return []
        for i, (message, response) in enumerate(y):
            y[i] = (
                None if message is None else mdtex2html.convert((message)),
                None if response is None else mdtex2html.convert(response),
            )
        return y

    gr.Chatbot.postprocess = postprocess

    def reset_user_input():
        return gr.update(value='')

    def reset_state():
        return [], []

    prompt_template = get_conv_template(args.template_name)

    def predict(
            input,
            chatbot,
            history,
            max_new_tokens,
            temperature,
            top_p
    ):
        now_input = input
        chatbot.append((input, ""))
        history = history or []
        history.append([now_input, ''])

        prompt = prompt_template.get_prompt(messages=history)
        response = ""
        for new_text in stream_generate_answer(
                m.model,
                m.tokenizer,
                prompt,
                m.device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
        ):
            response += new_text
            new_history = history + [(now_input, response)]
            chatbot[-1] = (now_input, response)
            yield chatbot, new_history

    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">TextGen</h1>""")
        gr.Markdown(
            "> TextGen gradio demo")
        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                        container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(
                    0, 4096, value=512, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.8, step=0.01,
                                  label="Top P", interactive=True)
                temperature = gr.Slider(
                    0, 1, value=0.7, step=0.01, label="Temperature", interactive=True)
        history = gr.State([])
        submitBtn.click(predict, [user_input, chatbot, history, max_length, temperature, top_p], [chatbot, history],
                        show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])
        emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)
    demo.queue().launch(share=False, inbrowser=True, server_name='0.0.0.0', server_port=8082)


if __name__ == '__main__':
    main()
