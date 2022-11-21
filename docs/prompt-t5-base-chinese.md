---
language: 
- zh
tags:
- t5
- pytorch
- prompt
- zh
- Text2Text-Generation
license: "apache-2.0"
widget:
- text: "中文改错：为了让人们遵守交通规律，警查叔叔不分昼夜在忙碌。"
- text: "请推理出上下文的关系：_前提：对不起事情就是这样。_假设：事情就是这样，不需要道歉。_选项：中立，蕴涵，矛盾_答案："
---

# Chinese Prompt(prompt-t5-base-chinese) Model
中文NLP的Prompt模型[shibing624/prompt-t5-base-chinese](https://huggingface.co/shibing624/prompt-t5-base-chinese)，One model For All nlp task(OFA)


1. 在[ClueAI/PromptCLUE-base](https://huggingface.co/ClueAI/PromptCLUE-base)预训练模型上fine-tuned
了[pCLUE中文prompt数据集](https://github.com/CLUEbenchmark/pCLUE)和[SIGHAN+Wang271K中文纠错数据集](https://github.com/shibing624/pycorrector#Dataset)
2. 模型用[textgen](https://github.com/shibing624/textgen)的`T5Model`训练，复现脚本参考[training_zh_prompt_model_demo.py](https://github.com/shibing624/textgen/blob/main/examples/T5/training_zh_prompt_model_demo.py)


`prompt-t5-base-chinese` evaluate public test data：

The overall performance of T5 on `pCLUE_test_public.json` **test**:

|model|classify_score|nli_score|generate_score|mrc_f1_score|avg_score|
|:-- |:--- |:--- |:--- |:--- |:--- |
|ClueAI/PromptCLUE-base|0.2417|0.0|0.1731|0.2371|0.1549|
|shibing624/prompt-t5-base-chinese|0.5494|0.525|0.2751|0.2259|0.3893|

## Feature

PromptCLUE：大规模多任务Prompt预训练中文开源模型。

千亿中文token上大规模预训练，累计学习1.5万亿中文token，支持几十个不同类型的NLP任务，具有较好的零样本学习能力和少样本学习能力。针对理解类任务，如分类、情感分析、抽取等，可以自定义标签体系；针对生成任务，可以进行多样性的文本生成。

中文上的三大统一：统一模型框架，统一任务形式，统一应用方式：
- 统一模型框架：采用Text-to-Text的生成式预训练模型进行统一建模。
- 统一任务形式：Prompt统一不同的NLP任务间的差异，转化为统一的text-to-text数据形式。
- 统一应用方式：对目标任务形成拿来即用的模型，下游应用时都可转化为统一的prompt自适应方式，进行zero-shot/few-shot测试。


![arch](https://github.com/shibing624/textgen/blob/main/docs/promptclue.png)


Fine-tuned的数据集包括：

1. 单分类tnews 
2. 单分类iflytek 
3. 自然语言推理ocnli 
4. 语义匹配afqmc 
5. 指代消解-cluewsc2020 
6. 关键词识别-csl 
7. 阅读理解-自由式c3 
8. 阅读理解-抽取式cmrc2018 
9. 阅读理解-成语填空chid 
10. 中文纠错数据集-sighan+wang271k

## Usage

本项目开源在文本生成项目：[textgen](https://github.com/shibing624/textgen)，可支持T5模型，通过如下命令调用：

Install package:
```shell
pip install -U textgen
```

```python
from textgen import T5Model
model = T5Model("t5", "shibing624/prompt-t5-base-chinese")
r = model.predict(["中文改错：为了让人们遵守交通规律，警查叔叔不分昼夜在忙碌。"])
print(r) # ['为了让人们遵守交通规律,警察叔叔不分昼夜在忙碌。']
```

## Usage (HuggingFace Transformers)
Without [textgen](https://github.com/shibing624/textgen), you can use the model like this: 

First, you pass your input through the transformer model, then you get the generated sentence.

Install package:
```
pip install transformers 
```

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("shibing624/prompt-t5-base-chinese")
model = T5ForConditionalGeneration.from_pretrained("shibing624/prompt-t5-base-chinese")
def batch_generate(input_texts, max_length=64):
    features = tokenizer(input_texts, return_tensors='pt')
    outputs = model.generate(input_ids=features['input_ids'],
                             attention_mask=features['attention_mask'],
                             max_length=max_length)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
r = batch_generate(["中文改错：为了让人们遵守交通规律，警查叔叔不分昼夜在忙碌。"])
print(r)
```

output:
```shell
['为了让人们遵守交通规律,警察叔叔不分昼夜在忙碌。']
```

模型文件组成：
```
prompt-t5-base-chinese
    ├── config.json
    ├── model_args.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── spiece.model
    └── vocab.txt
```
 
## 预测示例
#### 中文改错(correction)
```bash
Input:
中文改错：为了让人们遵守交通规律，警查叔叔不分昼夜在忙碌。
Model output:
为了让人们遵守交通规律,警察叔叔不分昼夜在忙碌。
```

#### 新闻分类(classify)
```bash
Input:
分类任务：
折价率过低遭抛售基金泰和跌7.15%，证券时报记者 朱景锋本报讯 由于折价率在大盘封基中处于最低水平，基金泰和昨日遭到投资者大举抛售，跌幅达到7.15%，远超大盘。盘面显示，基金泰和随大盘高开，之后开始震荡走低，午后开始加速下行，几乎没有像样反弹。截至收盘时，在沪深300指数仅下跌2.56%的情况下，基金泰和收盘跌幅高达7.15%，在所有封基中跌幅最大，而昨日多数封基跌幅在2%左右。
选项：财经，娱乐，时政，股票
答案：
Model output:
财经
```

#### 意图分类(classify)
```bash
Input:
意图分类：
帮我定一个周日上海浦东的房间
选项：闹钟，文学，酒店，艺术，体育，健康，天气，其他
答案：
Model output:
酒店
```

#### 情感分析(classify)
```bash
Input:
情感分析：
这个看上去还可以，但其实我不喜欢
选项：积极，消极
答案：
Model output:
消极
```

#### 推理(generate)
```bash
Input:
请推理出上下文的关系：
前提：对不起事情就是这样。
假设：事情就是这样，不需要道歉。
选项：中立，蕴涵，矛盾
答案：
Model output:
矛盾
```

#### 阅读理解(generate)
```bash
Input:
阅读文章，给出答案：
段落：
港汇指数，全称港元实际汇兑指数（Effective Exchange Rate Index for the Hong Kong Dollar）是由香港政府统计处编制的一项指数，以反映港元与香港主要贸易伙伴之货币的名义有效汇率加权平均数的变动情况。加权比重是按1999年至2000年平均贸易模式所制定，但政府并未有公布详细的计算公式。旧港汇指数基准日为2000年1月1日，基数为100点。由2012年1月3日起，新系列港汇指数 (包括15种货币及以2010年1月 = 100) 已取代旧港汇指数系列。港汇指数的作用，主要是用于反映香港的货品及服务的价格相对于其主要贸易伙伴的变动，并通常被视作反映香港价格竞争力的指标。
问题：港汇指数的加权比重如何制定？
答案：
Model output:
按1999年至2000年平均贸易模式所制定
```
#### 阅读理解-自由式(generate)
```bash
Input:
阅读以下对话并回答问题。
男：今天怎么这么晚才来上班啊？女：昨天工作到很晚，而且我还感冒了。男：那你回去休息吧，我帮你请假。女：谢谢你。
问题：女的怎么样？
选项：正在工作，感冒了，在打电话，要出差。
答案：
Model output:
感冒了
```

#### 摘要(generate)
```bash
Input:
为下面的文章生成摘要：
北京时间9月5日12时52分，四川甘孜藏族自治州泸定县发生6.8级地震。地震发生后，领导高度重视并作出重要指示，要求把抢救生命作为首要任务，全力救援受灾群众，最大限度减少人员伤亡
答案：
Model output:
四川甘孜发生6.8级地震
```

#### 通用信息抽取(generate)
```bash
Input:
信息抽取：
据新华社电广东省清远市清城区政府昨日对外发布信息称,日前被实名举报涉嫌勒索企业、说“分分钟可以搞垮一间厂”的清城区环保局局长陈柏,已被免去清城区区委委员
问题：机构名，人名，职位
答案：
Model output:
机构名：新华社，清城区政府，清城区环保局，清城区区委
人名：陈柏
职位：局长，区委委员
```


#### 指代消解(generate)
```bash
Input:
指代消解：
段落：
少平跟润叶进了她二爸家的院子，润生走过来对他（代词）说：“我到宿舍找了你两回，你到哪里去了？”
问题：代词“他”指代的是？
答案：
Model output:
少平
```

#### 关键词抽取(generate)
```bash
Input:
抽取关键词：
当地时间21日，美国联邦储备委员会宣布加息75个基点，将联邦基金利率目标区间上调到3.00%至3.25%之间，符合市场预期。这是美联储今年以来第五次加息，也是连续第三次加息，创自1981年以来的最大密集加息幅度。
关键词：
Model output:
美联储，利率目标区间，加息，基点
```


#### 情感倾向(classify)
```bash
文字中包含了怎样的情感：
超可爱的帅哥，爱了。。。
选项：厌恶，喜欢，开心，悲伤，惊讶，生气，害怕
答案：
Model output:
喜欢
```

## 训练数据集
#### 中文Prompt数据集

- 数据：[pCLUE中文prompt数据集](https://github.com/CLUEbenchmark/pCLUE)
- 相关内容
  - [Huggingface](https://huggingface.co/)
  - [PromptCLUE-base Model](https://huggingface.co/ClueAI/PromptCLUE-base)
  - [textgen](https://github.com/shibing624/textgen)
  
  
数据格式：

```text
{"input": "哪个类别最好的描述了这篇新闻？扣篮王拉文：精彩暴扣表演！炸\n选项：故事，文化，娱乐，体育，财经，房产，汽车，教育，科技，军事，旅游，国际，股票，农业，游戏\n答案：", "target": "电竞", "answer_choices": ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "股票", "农业", "游戏"], "type": "classify"}
{"input": "“现在婴儿的健康状况仍很严重”记住上面的文字,考虑:“婴儿已经完全康复了。”这是总是,绝不,或有时正确的？\n答案：", "target": "绝不", "answer_choices": ["总是", "绝不", "有时"], "type": "nli"}
```


如果需要训练Prompt模型，请参考[https://github.com/shibing624/textgen/blob/main/examples/T5/training_zh_prompt_model_demo.py](https://github.com/shibing624/textgen/blob/main/examples/T5/training_zh_prompt_model_demo.py)

附上我的训练参数：
```
epoch=5
batch_size=50
max_length=512 # input text length
max_seq_length=128 # output text length
```
V100单卡训练大概48小时。


## Citation

```latex
@software{textgen,
  author = {Xu Ming},
  title = {textgen: Implementation of Text Generation models},
  year = {2022},
  url = {https://github.com/shibing624/textgen},
}
```
