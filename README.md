[![PyPI version](https://badge.fury.io/py/textgen.svg)](https://badge.fury.io/py/textgen)
[![Downloads](https://pepy.tech/badge/textgen)](https://pepy.tech/project/textgen)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/textgen.svg)](https://github.com/shibing624/textgen/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/textgen.svg)](https://github.com/shibing624/textgen/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

# TextGen
🌈 Implementation of Text Generation models. 

**textgen**实现了多种文本生成模型，包括：UDA、Seq2Seq、BERT、XLNet、GPT2、ERNIE-GEN等模型，开箱即用。


**Guide**
- [Question](#Question)
- [Solution](#Solution)
- [Feature](#Feature)
- [Install](#install)
- [Usage](#usage)
- [Contact](#Contact)
- [Reference](#reference)

# Question

文本生成，文本数据增强怎么做？

# Solution

1. UDA，非核心词替换
2. EDA，简单数据增强技术：相似词、同义词替换，随机词插入、删除、替换
3. 回译（BT, Back Translate），中文-英文-中文
4. 生成模型，Seq2Seq，GPT2等


# Feature
### UDA(非核心词替换)

基于Google提出的UDA(非核心词替换)算法，将文本中一定比例的不重要词替换为同义词，从而产生新的文本。

### BT(回译)

基于百度翻译API，把中文句子翻译为英文，再把英文翻译为新的中文。

### Seq2Seq

基于Encoder-Decoder结构，序列到序列生成新的文本。

### GPT2

基于Transformer的decode结果的自回归生成模型。

### TGLS
无监督文本生成模型：本文提出[TGLS](https://www.jiqizhixin.com/articles/2020-08-11-5)——一种基于“先搜索后学习”的无监督文本生成方法，
模型反复迭代，最终能生成较高质量的文本。


# Demo

Demo: https://huggingface.co/spaces/shibing624/textgen

![](docs/hf.png)

run example: [examples/gradio_demo.py](examples/gradio_demo.py) to see the demo:
```shell
python examples/gradio_demo.py
```

 
# Install
```
pip3 install torch # conda install pytorch
pip3 install -U textgen
```

or

```
git clone https://github.com/shibing624/textgen.git
cd textgen
python3 setup.py install
```


# Usage

### 1. 文本数据增强


example: [examples/text_augmentation_demo.py](examples/text_augmentation_demo.py)
```python
import sys

sys.path.append('..')
from textgen.augment import TextAugment

if __name__ == '__main__':
    docs = ['主要研究机器学习、深度学习、计算机视觉、智能对话系统相关内容',
            '晚上肚子好难受',
            '你会武功吗，我不会',
            '组装标题质量受限于广告主自提物料的片段质量，且表达丰富度有限',
            '晚上一个人好孤单，想:找附近的人陪陪我.',
            ]
    m = TextAugment(sentence_list=docs)
    a = docs[0]
    print(a)

    b = m.augment(a, aug_ops='random-0.1')
    print('random-0.1:', b)

    b = m.augment(a, aug_ops='insert-0.1')
    print('insert-0.1:', b)

    # tfidf
    b = m.augment(a, aug_ops='tfidf-0.2')
    print('tfidf-0.2:', b)

    b = m.augment(a, aug_ops='mix-0.1', similar_prob=0.1,
                  random_prob=0.4, delete_prob=0.3, insert_prob=0.2)
    print('mix-0.1:', b)

    b = m.augment(a, aug_ops='bt')
    print('bt:', b)
```

output:

```bash
主要研究机器学习、深度学习、计算机视觉、智能对话系统相关内容
random-0.1: ('主要的机器学习、深度学习吗计算机视觉、好孤单对话系统相关内容', [('研究', '的', 2, 3), ('、', '吗', 12, 13), ('智能', '好孤单', 19, 22)])
insert-0.1: ('主要研究机器机器学习、深度学习、计算机视觉、智能对话对话系统系统相关内容', [('机器', '机器机器', 4, 8), ('对话', '对话对话', 24, 28), ('系统', '系统系统', 28, 32)])
tfidf-0.2: ('主要原因研究机器学习、深度学习、计算机硬件视觉、智能化对话系统相关内容', [('主要', '主要原因', 0, 4), ('计算机', '计算机硬件', 16, 21), ('智能', '智能化', 24, 27)])
mix-0.1: ('主要受限于机器学习、深度学习、计算机视觉、智能对话系统相关内容', [('研究', '受限于', 2, 5)])
bt: ('主要研究机器学习、深度学习、计算机视觉和智能对话系统', [])
```

### 2. Seq2Seq 模型

训练并预测ConvSeq2Seq模型：

example: [examples/seq2sesq/training_convseq2seq_model_demo.py](examples/seq2seq/training_convseq2seq_model_demo.py)
```python
import argparse
from loguru import logger
import sys

sys.path.append('../..')
from textgen.seq2seq.conv_seq2seq_model import ConvSeq2SeqModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='../data/zh_dialog.tsv', type=str, help='Training data file')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs/convseq2seq_zh/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=50, type=int, help='Max sequence length')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)

    if args.do_train:
        logger.info('Loading data...')
        model = ConvSeq2SeqModel(epochs=args.num_epochs, batch_size=args.batch_size,
                                 model_dir=args.output_dir, max_length=args.max_seq_length)
        model.train_model(args.train_file)
        print(model.eval_model(args.train_file))

    if args.do_predict:
        model = ConvSeq2SeqModel(epochs=args.num_epochs, batch_size=args.batch_size,
                                 model_dir=args.output_dir, max_length=args.max_seq_length)
        sentences = ["什么是ai", "你是什么类型的计算机", "你知道热力学吗"]
        print("inputs:", sentences)
        print(model.predict(sentences))


if __name__ == '__main__':
    main()
```

output:
```bash
["什么是ai", "你是什么类型的计算机", "你知道热力学吗"]
['人工智能是工程和科学的分支,致力于构建思维的机器。', '我的程序运行在python,所以我在任何运脑上工作！', '我不能错热是一个疯狂的人工智能"200年。']
```

### 3. GPT2 模型

example: [examples/language_generation/gpt2_demo.py](https://github.com/shibing624/textgen/blob/main/examples/language_generation/training_zh_gpt2_demo.py)

### 4. ByT5 模型

example: [examples/t5/training_byt5_model_demo.py](https://github.com/shibing624/textgen/blob/main/examples/t5/training_byt5_model_demo.py)


### 5. 无监督生成模型

无监督的中文电商评论生成：从**电商评论**中提取用户表达观点的短句并进行组合来生成仿真评论。


example: [examples/unsup_generation_demo.py](examples/unsup_generation_demo.py)

## 效果
[美迪惠尔 N.M.F针剂水库保湿面膜](https://goods.kaola.com/product/2227311.html)有如下的20句评论，其中有10句是真实用户评论，10句是生成的评论，能看出来么?😂
```
还不错还不错还不错还不错。
东西到了，不知道好不好用。试用过后再来评价。到时看网评都还可以。
哺乳期唯一使用的护肤品，每天都是素颜，脸面全靠面膜吊着😄补水💦不粘腻一如既往的支持，喜欢💕
搞活动时买的面膜，不知道这个面膜是真是假敷在脸上面膜纸都有小水泡鼓起来。
很不错，非常补水，用过的都知道，性价比之王，好用又不贵，正品，用着放心，物流也很快。
面膜非常好用哦。面膜薄薄的。好像是蚕丝面膜啊。精华很多呢。敷在脸上很舒服。感觉挺保湿的，味道也挺好闻的。就是里面只有单纯的面膜直接敷脸上有点不好弄，哈哈哈
还可以保湿效果不错水润润的每天贴一片脸也不干了用完了在买点，不错还会继续回购的。
快递很快，东西很赞！想要得点考拉豆不容易，还要三十个字。时间宝贵，废话不说！用过了就知道了
挺好用的，朋友推荐来的
挺好用的，淡淡的，虽然不是很浓精华的感觉，但是效果也蛮好的。划算
不得不说美迪惠尔的面膜是我用过的最好的面膜之一😎补水效果非常好，没想到这么便宜的价格竟真的能买到真品。
保湿效果挺好的，面膜很好用。
期待好的产品。
一打开包装里面的精华刚刚好，用了补水补水效果不错，物流非常快。
皮肤很光滑😇比上去速度快三天就到了。
前两天皮肤干燥连续敷了两个晚上感觉还不错😂补水效果明显！可想而知精华液又多充足😍敷上以后凉凉的很舒服。
补水效果一般吧～但是我用的韩国背回来的面膜纸不算薄，希望好用会回购的，敷上脸感觉比较清爽～价格还不便宜。
希望好用，面膜用过了很好用，皮肤水嫩光滑白皙，补水不错，价格也合适。
就是精华液太少了，保湿效果不错。
面膜的补水效果非常好，保湿效果确实很赞，这个面膜相对于胶原蛋白和美白的那两款的面膜纸要厚一些，看着价格合适。
```
（前10句是真实用户评论，后10句是生成的😆



# Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/textgen.svg)](https://github.com/shibing624/textgen/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：
加我*微信号：xuming624, 备注：姓名-公司名-NLP* 进NLP交流群。

<img src="docs/wechat.jpeg" width="200" />


# License

授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。请在产品说明中附加textgen的链接和授权协议。


# Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

 - 在`tests`添加相应的单元测试
 - 使用`python -m pytest`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。


## Reference

- [PaddlePaddle/ERNIE](https://github.com/PaddlePaddle/ERNIE)
- [minimaxir/textgenrnn](https://github.com/minimaxir/textgenrnn)
- [minimaxir/gpt-2-simple](https://github.com/minimaxir/gpt-2-simple)
- [asyml/texar](https://github.com/asyml/texar)
- [yangjianxin1/GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)
- [williamSYSU/TextGAN-PyTorch](https://github.com/williamSYSU/TextGAN-PyTorch)
- [RUCAIBox/TextBox](https://github.com/RUCAIBox/TextBox)
- [Tiiiger/bert_score]()
- [1YCxZ/Fake-review-generation](https://github.com/1YCxZ/Fake-review-generation)
