# textgen
textgen, Text Generation models. 文本生成，包括：非核心词替换，seq2seq，ernie-gen，bert，xlnet，gpt2等模型实现，开箱即用。

## Features
### 非核心词替换

基于Google提出的UDA算法，将文本中一定比例的不重要词替换为同义词，从而产生新的文本。


### Seq2Seq

基于encoder-decoder结构，序列到序列生成新的文本。


## Install
```
pip3 install textgen
```

or

```
git clone https://github.com/shibing624/text-generation.git
cd text-generation
python3 setup.py install
```

## Usage

1. download pretrained vector file

以下词向量，任选一个：

轻量版腾讯词向量，二进制，111MB放到 `~/.text2vec/datasets/light_Tencent_AILab_ChineseEmbedding.bin` 

腾讯词向量, 6.78G放到： `~/.text2vec/datasets/Tencent_AILab_ChineseEmbedding.txt`

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

## TODO

* seq2seq
* bert
* ernie-gen
* xlnet

## License

Apache License 2.0

## Reference

1. [ERNIE](https://github.com/PaddlePaddle/ERNIE)
2. [textgenrnn](https://github.com/minimaxir/textgenrnn)
3. [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple)
4. [texar](https://github.com/asyml/texar)
5. [GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)
6. [TextGAN-PyTorch](https://github.com/williamSYSU/TextGAN-PyTorch)
7. [TextBox](https://github.com/RUCAIBox/TextBox)
8. [bert_score](https://github.com/Tiiiger/bert_score)
