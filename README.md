# textgen
textgen, Text Generation models. 文本生成，包括：UDA，Seq2Seq，ERNIE-GEN，BERT，XLNet，GPT-2等模型实现，开箱即用。


**Guide**

- [Question](#Question)
- [Solution](#Solution)
- [Feature](#Feature)
- [Install](#install)
- [Usage](#usage)
- [Contact](#Contact)
- [Cite](#Cite)
- [Reference](#reference)

# Question

文本生成，文本数据增强怎么做？

# Solution

1. UDA，非核心词替换
2. EDA，简单数据增广技术：相似词、同义词替换，随机词插入、删除、替换
3. 回译（bt, back translate），中文-英文-中文
4. 生成模型，seq2seq，gpt


# Features
### UDA(非核心词替换)

基于Google提出的UDA(非核心词替换)算法，将文本中一定比例的不重要词替换为同义词，从而产生新的文本。

### BT(回译)

基于百度翻译API，把中文句子翻译为英文，再把英文翻译为新的中文。

### Seq2Seq

基于Encoder-Decoder结构，序列到序列生成新的文本。

### GPT2

基于Transformer的decode结果的自回归生成模型。

# Install
```
pip3 install textgen
```

or

```
git clone https://github.com/shibing624/textgeneration.git
cd textgeneration
python3 setup.py install
```

# Usage

1. download pretrained vector file


以下词向量，任选一个下载：

- 轻量版腾讯词向量 [百度云盘-密码:tawe](https://pan.baidu.com/s/1La4U4XNFe8s5BJqxPQpeiQ) 或 [谷歌云盘](https://drive.google.com/u/0/uc?id=1iQo9tBb2NgFOBxx0fA16AZpSgc-bG_Rp&export=download)，二进制，111MB放到 `~/.text2vec/datasets/light_Tencent_AILab_ChineseEmbedding.bin`
- [腾讯词向量-官方全量](https://ai.tencent.com/ailab/nlp/data/Tencent_AILab_ChineseEmbedding.tar.gz), 6.78G放到： `~/.text2vec/datasets/Tencent_AILab_ChineseEmbedding.txt`


2. download pretrained language model file

bert模型

3. text generation base rule

```
import textgen

a = '晚上一个人好孤单，想找附近人陪陪我'
b = textgen.rule(a)
print(b)

```

output:

```
晚上一个人好寂寞，想找附近人陪伴我
```

4. text generation base seq2seq

```
import textgen

a = '你这么早就睡呀，'
b = textgen.seq2seq(a)
print(b)
```

output:
```
你这么早就睡呀，我还没写完作业呢，你陪我看看这个题怎么写吧。
```

5. text generation base ernie-gen

```
import textgen

a = '你这么早就睡呀，'
b = textgen.erniegen(a)
print(b)
```

output:
```
你这么早就睡呀，我还没写完作业呢，你陪我看看这个题怎么写吧。求求你了！
```

# TODO

* bert
* ernie-gen
* xlnet

# Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/textgen.svg)](https://github.com/shibing624/textgen/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：
加我*微信号：xuming624, 备注：个人名称-NLP* 进NLP交流群。

<img src="docs/wechat.jpeg" width="200" />


# Cite

如果你在研究中使用了textgen，请按如下格式引用：

```latex
@software{textgen,
  author = {Xu Ming},
  title = {textgen: A Tool for Text generation},
  year = {2021},
  url = {https://github.com/shibing624/textgen},
}
```

# License


授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。请在产品说明中附加textgen的链接和授权协议。


# Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

 - 在`tests`添加相应的单元测试
 - 使用`python setup.py test`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。


## Reference

- [ERNIE](https://github.com/PaddlePaddle/ERNIE)
- [textgenrnn](https://github.com/minimaxir/textgenrnn)
- [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple)
- [texar](https://github.com/asyml/texar)
- [GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)
- [TextGAN-PyTorch](https://github.com/williamSYSU/TextGAN-PyTorch)
- [TextBox](https://github.com/RUCAIBox/TextBox)
- [bert_score](https://github.com/Tiiiger/bert_score)
