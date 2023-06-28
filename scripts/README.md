# Tools 

以医疗为例生成领域数据集，给出数据生成脚本： 
- ChatGPT对话调用脚本 [answer_by_chatgpt.py](https://github.com/shibing624/textgen/blob/main/scripts/answer_by_chatgpt.py)
- SFT指令微调数据集生成脚本 [generate_sft_data.py](https://github.com/shibing624/textgen/blob/main/scripts/generate_sft_data.py)
- SFT指令数据集GPT评分脚本 [evaluate_sft_data.py](https://github.com/shibing624/textgen/blob/main/scripts/evaluate_sft_data.py)

## 字段

指令微调数据集使用统一的字段

```json
instruction: 指令
input: 输入
output: 输出
```

## 使用方法
#### ChatGPT对话调用脚本 answer_by_chatgpt.py
[answer_by_chatgpt.py](https://github.com/shibing624/textgen/blob/main/scripts/answer_by_chatgpt.py) 利用ChatGPT生成医疗问题的答案，并输出为指令数据集的格式。

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=YOUR_API_KEY
python answer_by_chatgpt.py --input_file ./medical_question.txt --output_file ./medical_question_result.jsonl
```

#### SFT指令微调数据集生成脚本 generate_sft_data.py
沿用Alpaca的方式构建self-instruct数据集，
通用问答指令微调数据集构建可以参考[BELLE的脚本](https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M)，

下面是医疗领域的指令微调数据构建方法：

[generate_sft_data.py](https://github.com/shibing624/textgen/blob/main/scripts/generate_sft_data.py) 利用few-shot技术，使用ChatGPT生成指令微调数据集。

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=YOUR_API_KEY
python generate_sft_data.py --seed_file ./seed_medical_sft_data.jsonl --output_file ./medical_sft_result.jsonl --num_instructions_to_generate 3
```

默认使用`Chat` API，模型`gpt-3.5-turbo`。

输出文件在`medical_sft_result.jsonl`，可以人工筛选后再使用。


#### SFT指令数据集GPT评分脚本 evaluate_sft_data.py

[evaluate_sft_data.py](https://github.com/shibing624/textgen/blob/main/scripts/evaluate_sft_data.py) 中可以使用ChatGPT和GPT4模型评分，评分结果保存在`scores.jsonl`中。

```bash 
pip install -r requirements.txt
export OPENAI_API_KEY=YOUR_API_KEY
python evaluate_sft_data.py --input_file ./seed_medical_sft_data.jsonl --output_file ./scores.jsonl
```

## 局限性和使用限制

我们要求开发者仅将我们开源的代码、数据、模型及后续衍生物用于研究目的，不得用于商业，以及其他会对社会带来危害的用途。

由于数据是由*ChatGPT*生成的，未经严格验证，在事实性和其他方面还存在一些不足。因此，在使用此数据集时，请务必注意甄别。

本数据集不代表任何一方的立场、利益或想法，无关任何团体的任何类型的主张。因使用本数据集带来的任何损害、纠纷，本项目的开发者不承担任何责任。
