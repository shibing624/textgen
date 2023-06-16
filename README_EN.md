[**🇨🇳中文**](https://github.com/shibing624/textgen/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/textgen/blob/main/README_EN.md) | [**📖文档/Docs**](https://github.com/shibing624/textgen/wiki) | [**🤖模型/Models**](https://huggingface.co/shibing624) 

<div align="center">
  <a href="https://github.com/shibing624/textgen">
    <img src="https://github.com/shibing624/textgen/blob/main/docs/logo.svg" alt="Logo">
  </a>
</div>

-----------------

# TextGen: Implementation of Text Generation models
[![PyPI version](https://badge.fury.io/py/textgen.svg)](https://badge.fury.io/py/textgen)
[![Downloads](https://pepy.tech/badge/textgen)](https://pepy.tech/project/textgen)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.8%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/textgen.svg)](https://github.com/shibing624/textgen/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

## 📖 Introduction

**TextGen** implements a variety of text generation models, including: LLaMA, ChatGLM, UDA, GPT2, Seq2Seq, BART, T5, SongNet and other models, out of the box.

## 😊 Feature

- [ChatGLM](textgen/chatglm): This project implements the LoRA fine-tuning training and prediction of the ChatGLM-6B model based on PyTorch, which can be used for text generation tasks such as sentence error correction and dialogue
- [LLaMA](textgen/llama): This project implements the LLaMA model LoRA fine-tuning training and prediction based on PyTorch, which can be used for dialogue generation tasks and domain fine-tuning training
- [BLOOM](textgen/bloom): This project implements the BLOOM model LoRA fine-tuning training and prediction based on PyTorch, which can be used for dialogue generation tasks and domain fine-tuning training
- [UDA/EDA](textgen/augment/word_level_augment.py): This project implements UDA (non-core word replacement), EDA and Back Translation (back translation) algorithms, and replaces some unimportant words in sentences based on TF-IDF For synonyms, random word insertion, deletion, replacement, etc., generate new text and realize text amplification
- [Seq2Seq](textgen/seq2seq): This project implements the training and prediction of Seq2Seq, ConvSeq2Seq, and BART models based on PyTorch, which can be used for text generation tasks such as text translation, dialogue generation, and abstract generation
- [T5](textgen/t5): This project implements T5 and CopyT5 model training and prediction based on PyTorch, which can be used for text generation tasks such as text translation, dialogue generation, couplet generation, and copywriting
- [GPT2](textgen/language_modeling): This project implements GTP2 model training and prediction based on PyTorch, which can be used for text generation tasks such as article generation and couplet generation
- [SongNet](textgen/language_modeling/songnet_model.py): This project implements SongNet model training and prediction based on PyTorch, which can be used for text generation tasks such as poems and lyrics in standardized formats
- [TGLS](textgen/unsup_generation): This project implements the [TGLS](https://www.jiqizhixin.com/articles/2020-08-11-5) unsupervised similar text generation model, which is a "first The text generation method of "learning after searching" learns the candidate set repeatedly, and the final model can generate high-quality similar text similar to the candidate set
### Release Models
The release is based on the Chinese model trained by `textgen`. The model has been released to HuggingFace models. Specifying the model name `textgen` will automatically download the model and can be used directly.

| Model                                                                                                     | Arch       | Introduce                                                                                                                                                                | Training                                                                                                                                     | Inference                                                                                                             | 
|:----------------------------------------------------------------------------------------------------------|:-----------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------|
| [shibing624/prompt-t5-base-chinese](https://huggingface.co/shibing624/prompt-t5-base-chinese)             | T5         | 中文NLP多任务Prompt模型                                                                                                                                                         | [prompt-t5-base-chinese.md](https://github.com/shibing624/textgen/blob/main/docs/prompt-t5-base-chinese.md)                                  | [predict script](https://github.com/shibing624/textgen/blob/main/examples/t5/t5_prompt_demo.py)                       |
| [shibing624/t5-chinese-couplet](https://huggingface.co/shibing624/t5-chinese-couplet)                     | T5         | fine-tuned中文对联后的模型                                                                                                                                                       | [对联生成模型调研](https://github.com/shibing624/textgen/blob/main/docs/%E5%AF%B9%E8%81%94%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E5%AF%B9%E6%AF%94.md) | [predict script](https://github.com/shibing624/textgen/blob/main/examples/t5/t5_couplet_demo.py)                      |
| [shibing624/songnet-base-chinese](https://huggingface.co/shibing624/songnet-base-chinese)                 | SongNet    | SongNet预训练模型                                                                                                                                                             | -                                                                                                                                            | -                                                                                                                     |
| [shibing624/songnet-base-chinese-songci](https://huggingface.co/shibing624/songnet-base-chinese-songci)   | SongNet    | fine-tuned宋词后的模型                                                                                                                                                         | [training script](https://github.com/shibing624/textgen/blob/main/examples/songnet/training_zh_songnet_demo.py)                              | [predict script](https://github.com/shibing624/textgen/blob/main/examples/songnet/songnet_songci_demo.py)             |
| [shibing624/songnet-base-chinese-couplet](https://huggingface.co/shibing624/songnet-base-chinese-couplet) | SongNet    | fine-tuned对联后的模型                                                                                                                                                         | [training script](https://github.com/shibing624/textgen/blob/main/examples/songnet/training_zh_songnet_demo.py)                                 | [predict script](https://github.com/shibing624/textgen/blob/main/examples/songnet/songnet_couplet_demo.py)            |
| [shibing624/chatglm-6b-csc-zh-lora](https://huggingface.co/shibing624/chatglm-6b-csc-zh-lora)             | ChatGLM-6B | 在27万中文拼写纠错数据[shibing624/CSC](https://huggingface.co/datasets/shibing624/CSC)上微调了一版ChatGLM-6B，纠错效果有提升，发布微调后的LoRA权重                                                        | [training script](https://github.com/shibing624/textgen/blob/main/examples/chatglm/training_chatglm_csc_demo.py)                             | [predict script](https://github.com/shibing624/textgen/blob/main/examples/chatglm/csc_demo.py)                        |
| [shibing624/chatglm-6b-belle-zh-lora](https://huggingface.co/shibing624/chatglm-6b-belle-zh-lora)         | ChatGLM-6B | 在100万条中文ChatGPT指令Belle数据集[BelleGroup/train_1M_CN](https://huggingface.co/datasets/BelleGroup/train_1M_CN)上微调了一版ChatGLM-6B，问答效果有提升，发布微调后的LoRA权重                           | [training script](https://github.com/shibing624/textgen/blob/main/examples/chatglm/training_chatglm_hfdataset_demo.py)                       | [predict script](https://github.com/shibing624/textgen/blob/main/examples/chatglm/training_chatglm_hfdataset_demo.py) |
| [shibing624/llama-13b-belle-zh-lora](https://huggingface.co/shibing624/llama-13b-belle-zh-lora)           | LLaMA-13B  | 在100万条中文ChatGPT指令Belle数据集[BelleGroup/train_1M_CN](https://huggingface.co/datasets/BelleGroup/train_1M_CN)上微调了一版Llama-13B，问答效果有提升，发布微调后的LoRA权重                            | [training script](https://github.com/shibing624/textgen/blob/main/examples/llama/training_llama_hfdataset_demo.py)                           | [predict script](https://github.com/shibing624/textgen/blob/main/examples/llama/training_llama_hfdataset_demo.py)     |
| [shibing624/chinese-alpaca-plus-7b-hf](https://huggingface.co/shibing624/chinese-alpaca-plus-7b-hf)       | LLaMA-7B   | [中文LLaMA-Plus, Alpaca-Plus 7B版本](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v3.0)，在LLaMA-7B上扩充了中文词表并继续预训练120G文本（通用领域），在4M指令数据集上微调后得到的中文Alpaca-plus模型     | [training script](https://github.com/shibing624/textgen/blob/main/examples/llama/training_llama_demo.py)                           | [predict script](https://github.com/shibing624/textgen/blob/main/examples/llama/training_llama_demo.py)     |
| [shibing624/chinese-alpaca-plus-13b-hf](https://huggingface.co/shibing624/chinese-alpaca-plus-13b-hf)     | LLaMA-13B  | [中文LLaMA-Plus, Alpaca-Plus 13B版本](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v3.1)，在LLaMA-13B上扩充了中文词表并继续预训练120G文本（通用领域），在4.3M指令数据集上微调后得到的中文Alpaca-plus模型 | [training script](https://github.com/shibing624/textgen/blob/main/examples/llama/training_llama_demo.py)                           | [predict script](https://github.com/shibing624/textgen/blob/main/examples/llama/training_llama_demo.py)     |

### Evaluation

| Model                                                                                                                                       | Arch       | Introduce                                                                                                                                                                                                                                                                                     | Score    |
|:--------------------------------------------------------------------------------------------------------------------------------------------|:-----------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------|
| [LLaMA-7B-Chinese-Alpaca](https://huggingface.co/ziqingyang/chinese-alpaca-lora-7b)                                                         | LLaMA-7B   | 复用[ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/examples/README.md)的评估case和得分                                                                                                                                                                          | 4.92     |
| [LLaMA-13B-Chinese-Alpaca](https://huggingface.co/ziqingyang/chinese-alpaca-lora-13b)                                                       | LLaMA-13B  | 复用[ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/examples/README.md)的评估case和得分                                                                                                                                                                          | 7.05     |
| [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b)                                                                                       | ChatGLM-6B | 基于原生`THUDM/chatglm-6b`评估测试集得分                                                                                                                                                                                                                                                                 | 7.16     |
| [ChatGLM-6B-v1.1](https://huggingface.co/THUDM/chatglm-6b)                                                                                  | ChatGLM-6B | 基于原生`THUDM/chatglm-6b`v1.1英文优化版模型评估测试集得分                                                                                                                                                                                                                                                      | **7.18** |
| [shibing624/chatglm-6b-belle-zh-lora](https://huggingface.co/shibing624/chatglm-6b-belle-zh-lora)                                           | ChatGLM-6B | 基于`THUDM/chatglm-6b`加载`shibing624/chatglm-6b-belle-zh-lora`LoRA模型后评估测试集得分                                                                                                                                                                                                                     | 7.03     |
| [facat/alpaca-lora-cn-13b](https://huggingface.co/facat/alpaca-lora-cn-13b)	                                                                | LLaMA-13B  | 基于`decapoda-research/llama-13b-hf`加载`facat/alpaca-lora-cn-13b`LoRA模型后评估测试集并标注得分                                                                                                                                                                                                               | 4.13     |  
| [Chinese-Vicuna/Chinese-Vicuna-lora-13b-belle-and-guanaco](https://huggingface.co/Chinese-Vicuna/Chinese-Vicuna-lora-13b-belle-and-guanaco) | LLaMA-13B  | 基于`decapoda-research/llama-13b-hf`加载`Chinese-Vicuna/Chinese-Vicuna-lora-13b-belle-and-guanaco`LoRA模型后评估测试集并标注得分                                                                                                                                                                               | 3.98     |
| [shibing624/chinese-alpaca-plus-7b-hf](https://huggingface.co/shibing624/chinese-alpaca-plus-7b-hf)                                         | LLaMA-7B   | 使用[ymcui/Chinese-LLaMA-Alpaca 合并模型方法](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E6%89%8B%E5%8A%A8%E6%A8%A1%E5%9E%8B%E5%90%88%E5%B9%B6%E4%B8%8E%E8%BD%AC%E6%8D%A2#%E5%A4%9Alora%E6%9D%83%E9%87%8D%E5%90%88%E5%B9%B6%E9%80%82%E7%94%A8%E4%BA%8Echinese-alpaca-plus)合并HF权重后，评估测试集并标注得分 | 6.93     |
| [shibing624/chinese-alpaca-plus-13b-hf](https://huggingface.co/shibing624/chinese-alpaca-plus-13b-hf)                                       | LLaMA-13B  | 使用[ymcui/Chinese-LLaMA-Alpaca 合并模型方法](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E6%89%8B%E5%8A%A8%E6%A8%A1%E5%9E%8B%E5%90%88%E5%B9%B6%E4%B8%8E%E8%BD%AC%E6%8D%A2#%E5%A4%9Alora%E6%9D%83%E9%87%8D%E5%90%88%E5%B9%B6%E9%80%82%E7%94%A8%E4%BA%8Echinese-alpaca-plus)合并HF权重后，评估测试集并标注得分 | 7.07     |
| [TheBloke/vicuna-13B-1.1-HF](https://huggingface.co/TheBloke/vicuna-13B-1.1-HF)                                                             | LLaMA-13B  | 使用原生vicuna-13B-1.1合并后的模型，评估测试集并标注得分                                                                                                                                                                                                                                                           | 5.13     |
| [IDEA-CCNL/Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1)                                                           | LLaMA-13B  | 使用姜子牙通用大模型V1，评估测试集并标注得分                                                                                                                                                                                                                                                                       | 6.63     |

Evaluation conclusion:

- Evaluation case, see the online document for details: Chinese LLM-benchmark multi-task evaluation set (Tencent document) https://docs.qq.com/sheet/DUUpsREtWbFBsUVJE?tab=r7io7g Thanks to Han Junming, [Yang Jiaming](https:// github.com/yangjiam) and other students' annotations
- Evaluation task types include: knowledge quiz, open-ended question and answer, numerical calculation, poetry, music, sports, entertainment, article writing, text translation, code programming, ethics, refusal, multi-round question and answer, Score score is the top 100 ( 10-point scale) average score, manually scored, the higher the better
- The number of evaluations is small, the types of tasks are not comprehensive enough, the size relationship between the scores has some reference value, and the absolute value of the score is not much reference value
- Evaluation script: [tests/test_benchmark.py](https://github.com/shibing624/textgen/blob/main/tests/test_benchmark.py), using fp16 prediction, no int quantization processing, running the script can reproduce the evaluation However, the generated results are random and are affected by factors such as decoding hyperparameters and random seeds. The evaluation is not absolutely rigorous, and the test results are for reference only
- Conclusion: The performance of the Chinese derivative models of ChatGLM-6B and LLaMA-13B (including alpaca-plus, vicuna, ziya) belongs to the first echelon, and the performance of the original LLaMA-7B is slightly worse overall
- LLaMA-13B-Chinese-Alpaca is an instruction fine-tuning model that expands the Chinese vocabulary on the original LLaMA and incorporates about 20G of general Chinese corpus, which shows that LLaMA has an excellent base and strong language transfer capabilities
- ChatGLM, a native Chinese pre-training model, understands Chinese semantics better, and scores high in Chinese knowledge questions and answers and open questions and answers
- High scores in numerical calculation, Chinese-English translation, and code programming of LLaMA series models
- The Chinese-LLaMA model after Chinese pre-training and SFT fine-tuning has improved scores in Chinese poetry, entertainment, and ethics compared with the original LLaMA model

## 🚀 Demo

HuggingFace Demo: https://huggingface.co/spaces/shibing624/chinese-couplet-generate

![](docs/hf.png)

run example: [examples/gradio_demo.py](examples/gradio_demo.py) to see the demo:

```shell
python examples/gradio_demo.py
```

model trained by [examples/t5/T5_Finetune_Chinese_Couplet.ipynb](https://github.com/shibing624/textgen/blob/main/examples/t5/T5_Finetune_Chinese_Couplet.ipynb)

## 💾 Install

```shell
pip install -U textgen
```

or

install develop version:
```shell
pip install torch # conda install pytorch
git clone https://github.com/shibing624/textgen.git
cd textgen
python setup.py install
```

## ▶️ Usage

### ChatGLM-6B Model

#### Fine-tuned model using ChatGLM-6B

example: [examples/chatglm/predict_demo.py](https://github.com/shibing624/textgen/blob/main/examples/chatglm/predict_demo.py)

```python
from textgen import ChatGlmModel

model = ChatGlmModel("chatglm", "THUDM/chatglm-6b", peft_name="shibing624/chatglm-6b-csc-zh-lora")
r = model.predict(["对下面中文拼写纠错：\n少先队员因该为老人让坐。\n答："])
print(r)  # ['少先队员应该为老人让座。\n错误字：因，坐']
```

PS: Due to the use of the peft library under development, the loading of the LoRA model may fail due to the version update. It is recommended to use the following training method to train the LoRA model by yourself.

#### Train the ChatGLM-6B fine-tuning model

1. Support custom training data sets and training parameters, the data set format reference [examples/data/zh_csc_test.tsv](https://github.com/shibing624/textgen/blob/main/examples/data/zh_csc_test.tsv) Or [shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)
2. Support some parameter fine-tuning methods such as AdaLoRA, LoRA, P_Tuning, Prefix_Tuning, etc., and also support full parameter fine-tuning
3. Support multi-card training and mixed precision training

example: [examples/chatglm/training_chatglm_demo.py](https://github.com/shibing624/textgen/blob/main/examples/chatglm/training_chatglm_demo.py)

Training with Single GPU：
```shell
cd examples/chatglm
CUDA_VISIBLE_DEVICES=0 python training_chatglm_demo.py --do_train --do_predict --num_epochs 1 --output_dir outputs_chatglm
```

Training with Multi GPU：
```shell
cd examples/chatglm
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 training_chatglm_demo.py --do_train --do_predict --num_epochs 20
```

#### Continue training based on fine-tuning (LoRA) model
If you need to continue training based on the Lora model, you can use the following script to merge the model into a new base model, and then fine-tune the training.

Execute the following command:
```shell
python -m textgen/chatglm/merge_peft_adapter.py \
     --base_model_name_or_path path_to_original_base_model_dir \
     --peft_model_path path_to_peft_model_dir \
     --output_dir path_to_output_dir
```
Parameter Description:
```
--base_model_name_or_path: directory to store base model weights and configuration files in HF format
--peft_model_path: directory for storing fine-tuning model weights and configuration files in PEFT format
--output_dir: Specify the directory to save the weight of the full model, the default is ./merged
```

### LLaMA model

#### Fine-tuned model using LLaMA
example: [examples/llama/predict_demo.py](https://github.com/shibing624/textgen/blob/main/examples/llama/predict_demo.py)

<details>
<summary>show code example and result</summary>

```python
import sys

sys.path.append('../..')
from textgen import GptModel


def generate_prompt(instruction):
  return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:{instruction}\n\n### Response:"""


model = GptModel("llama", "decapoda-research/llama-7b-hf", peft_name="ziqingyang/chinese-alpaca-lora-7b")
predict_sentence = generate_prompt("问：用一句话描述地球为什么是独一无二的。\n答：")
r = model.predict([predict_sentence])
print(r)  # ['地球是唯一一颗拥有生命的行星。']
```

</details>

#### Train the LLaMA fine-tuning model
1. Support custom training data sets and training parameters, the data set format reference [examples/data/zh_csc_test.tsv](https://github.com/shibing624/textgen/blob/main/examples/data/zh_csc_test.tsv) Or [shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)
2. Support some parameter fine-tuning methods such as AdaLoRA, LoRA, P_Tuning, Prefix_Tuning, etc., and also support full parameter fine-tuning
3. Support multi-card training, support mixed precision training, use the same method as above (ChatGLM multi-GPU training)

example: [examples/llama/training_llama_demo.py](https://github.com/shibing624/textgen/blob/main/examples/llama/training_llama_demo.py)


#### Continue training based on fine-tuning (LoRA) model
If you need to continue training based on the Lora model, you can use the following script to merge the model into a new base model, and then fine-tune the training.

Single LoRA weight merging (for Chinese-LLaMA, Chinese-LLaMA-Plus, Chinese-Alpaca)

Execute the following command:
```shell
python -m textgen/llama/merge_peft_adapter.py \
    --base_model_name_or_path path_to_original_base_model_dir \
    --peft_model_path path_to_chinese_llama_or_alpaca_lora \
    --output_type [pth|huggingface]
    --output_dir path_to_output_dir 
```
Parameter Description:
```
--base_model_name_or_path: directory to store base model weights and configuration files in HF format
--peft_model_path: The directory where the Chinese LLaMA/Alpaca LoRA file is decompressed. You can also use the Lora model name on HF. For example, `ziqingyang/chinese-alpaca-lora-7b` will automatically download the corresponding model
--output_type: Specifies the output format, which can be pth or huggingface. If not specified, the default is huggingface
--output_dir: Specify the directory to save the weight of the full model, the default is ./merged
--offload_dir (optional): For low memory users need to specify an offload cache path
```

#### Training Domain Model

| Notebook | Description | |
|:----------|:------------|------:|
| [training_medical_model.ipynb](https://github.com/shibing624/textgen/blob/main/examples/llama/training_medical_model.ipynb) | Training medical large model|[![Open In Colab](https://colab .research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/textgen/blob/main/examples/llama/training_medical_model.ipynb) |

Note: In order to comprehensively introduce the process of training large medical models, a new repo has been created for the 4-stage training method (Pretraining, Supervised Finetuning, Reward Modeling and Reinforcement Learning): [shibing624/MedicalGPT](https://github.com/ shibing624/MedicalGPT), please move to this repo to view the training method.

### BLOOM model

#### Train the BLOOM fine-tuning model

example: [examples/bloom/training_bloom_demo.py](https://github.com/shibing624/textgen/blob/main/examples/bloom/training_bloom_demo.py)

### ConvSeq2Seq 模型

Train and predict the ConvSeq2Seq model:

example: [examples/seq2sesq/training_convseq2seq_model_demo.py](https://github.com/shibing624/textgen/blob/main/examples/seq2seq/training_convseq2seq_model_demo.py)

<details>
<summary>show code example and result</summary>

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
        print('outputs:', model.predict(sentences))


if __name__ == '__main__':
    main()
```

output:

```bash
inputs: ["什么是ai", "你是什么类型的计算机", "你知道热力学吗"]
outputs: ['人工智能是工程和科学的分支,致力于构建思维的机器。', '我的程序运行在python,所以我在任何运脑上工作！', '我不能错热是一个疯狂的人工智能"200年。']
```

</details>

### BART Model
Train and predict the BART model:

example: [examples/seq2sesq/training_bartseq2seq_zh_demo.py](https://github.com/shibing624/textgen/blob/main/examples/seq2seq/training_bartseq2seq_zh_demo.py)

output:

```shell
inputs: ['什么是ai', '你是什么类型的计算机', '你知道热力学吗']
outputs: ['人工智能是工程和科学的分支,致力于构', '我的程序运行在python,所以我在任何电脑上', '什么是热力学吗？']
```

### T5 Model

example: [examples/t5/training_zh_t5_model_demo.py](https://github.com/shibing624/textgen/blob/main/examples/t5/training_zh_t5_model_demo.py)

<details>
<summary>show code example and result</summary>

```python
import argparse
from loguru import logger
import pandas as pd
import sys

sys.path.append('../..')
from textgen.t5 import T5Model


def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            terms = line.split('\t')
            if len(terms) == 2:
                data.append(['QA', terms[0], terms[1]])
            else:
                logger.warning(f'line error: {line}')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='../data/zh_dialog.tsv', type=str, help='Training data file')
    parser.add_argument('--model_type', default='t5', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='Langboat/mengzi-t5-base', type=str, help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs/mengzi_t5_zh/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=50, type=int, help='Max sequence length')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)

    if args.do_train:
        logger.info('Loading data...')
        # train_data: Pandas DataFrame containing the 3 columns - `prefix`, `input_text`, `target_text`.
        #   - `prefix`: A string indicating the task to perform. (E.g. `"question"`, `"stsb"`)
        #   - `input_text`: The input text. `prefix` is prepended to form the full input. (<prefix>: <input_text>)
        #   - `target_text`: The target sequence
        train_data = load_data(args.train_file)
        logger.debug('train_data: {}'.format(train_data[:10]))
        train_df = pd.DataFrame(train_data, columns=["prefix", "input_text", "target_text"])

        eval_data = load_data(args.train_file)[:10]
        eval_df = pd.DataFrame(eval_data, columns=["prefix", "input_text", "target_text"])

        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "train_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": False,
            "evaluate_generated_text": True,
            "evaluate_during_training": True,
            "evaluate_during_training_verbose": True,
            "use_multiprocessing": True,
            "save_best_model": True,
            "output_dir": args.output_dir,
            "use_early_stopping": True,
        }
        # model_type: t5  model_name: Langboat/mengzi-t5-base
        model = T5Model(args.model_type, args.model_name, args=model_args)

        def count_matches(labels, preds):
            logger.debug(f"labels: {labels[:10]}")
            logger.debug(f"preds: {preds[:10]}")
            match = sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])
            logger.debug(f"match: {match}")
            return match

        model.train_model(train_df, eval_data=eval_df, matches=count_matches)
        print(model.eval_model(eval_df, matches=count_matches))

    if args.do_predict:
        model = T5Model(args.model_type, args.output_dir)
        sentences = ["什么是ai", "你是什么类型的计算机", "你知道热力学吗"]
        print("inputs:", sentences)
        print("outputs:", model.predict(sentences))


if __name__ == '__main__':
    main()
```

output:

```shell
inputs: ['什么是ai', '你是什么类型的计算机', '你知道热力学吗']
outputs: ['人工智能有两个广义的定义,任何拟人的机械,如在卡雷尔capeks', '我的程序运行在Python,所以我在任何电脑上工作!', '什么是热力学']
```

</details>

### GPT2 Model

#### Chinese GPT2 - Article Generation

Use the Chinese dataset (paragraph format, `\n` interval) to train the GPT2 model, which can be used for poetry generation, article generation and other tasks.

example: [examples/gpt2/training_zh_gpt2_demo.py](https://github.com/shibing624/textgen/blob/main/examples/gpt2/training_zh_gpt2_demo.py)

#### Chinese GPT2 - couplet generation

Use the Chinese couplet dataset (tsv format, `\t` interval), customize the dataset to read the Dataset, and train the GPT2 model, which can be used for couplet generation, dialogue generation and other tasks.

example: [examples/gpt2/training_couplet_gpt2_demo.py](https://github.com/shibing624/textgen/blob/main/examples/gpt2/training_couplet_gpt2_demo.py)

GPT2 vs T5：

1. Both are improved from Transformer, T5 has both encoder and decoder, GPT2 only has decoder
2. The advantage of the T5 model is to process a given input and output tasks corresponding to the output, such as translation, dialogue, question and answer, etc.
3. The advantage of the GPT2 model is free creation, such as writing a short article
4. The couplet generation effect of T5 is better than that of GPT2, and the poetry generation effect of GPT2 is better than that of T5

- [对联生成模型调研](https://github.com/shibing624/textgen/blob/main/docs/%E5%AF%B9%E8%81%94%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E5%AF%B9%E6%AF%94.md)
- [古诗生成模型调研](https://github.com/shibing624/textgen/blob/main/docs/%E5%8F%A4%E8%AF%97%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E5%AF%B9%E6%AF%94.md)

### SongNet 模型

Format-controlled text generation model, see paper [SongNet: Rigid Formats Controlled Text Generation](https://arxiv.org/abs/2004.08022),
It is suitable for tasks such as poetry, couplets, and lyrics generation that require strong rhythmic formats.

example: [examples/songnet/training_zh_songnet_demo.py](https://github.com/shibing624/textgen/blob/main/examples/songnet/training_zh_songnet_demo.py)

### Keyword Text Augmentation(EDA/UDA)

example: [examples/text_augmentation/text_augmentation_demo.py](examples/text_augmentation/text_augmentation_demo.py)

<details>
<summary>show code example and result</summary>

```python
import sys

sys.path.append('..')
from textgen.augment import TextAugment

if __name__ == '__main__':
    docs = ['主要研究机器学习、深度学习、计算机视觉、智能对话系统相关内容',
            '晚上肚子好难受',
            '你会武功吗，我不会',
            '组装标题质量受限于广告主自提物料的片段质量，且表达丰富度有限',
            ]
    m = TextAugment(sentence_list=docs)
    a = docs[0]
    print(a)

    b = m.augment(a, aug_ops='random-0.2')
    print('random-0.2:', b)

    b = m.augment(a, aug_ops='insert-0.2')
    print('insert-0.2:', b)

    b = m.augment(a, aug_ops='delete-0.2')
    print('delete-0.2:', b)

    b = m.augment(a, aug_ops='tfidf-0.2')
    print('tfidf-0.2:', b)

    b = m.augment(a, aug_ops='mix-0.2')
    print('mix-0.2:', b)
```

output:

```bash
主要研究机器学习、深度学习、计算机视觉、智能对话系统相关内容
random-0.2: ('主要陪陪机器学习、深度学习主要计算机视觉、智能对话系统受限于内容', [('研究', '陪陪', 2, 4), ('、', '主要', 13, 15), ('相关', '受限于', 27, 30)])
insert-0.2: ('主要研究机器机器学习学习、深度深度学习、计算机视觉、智能对话系统相关内容', [('机器', '机器机器', 4, 8), ('学习', '学习学习', 8, 12), ('深度', '深度深度', 13, 17)])
delete-0.2: ('主要研究机器学习、深度学习、计算机视觉、对话系统相关内容', [('智能', '', 20, 20)])
tfidf-0.2: ('一是研究机器学习、深度学习、计算机听觉、智能交谈系统密切相关内容', [('主要', '一是', 0, 2), ('视觉', '听觉', 17, 19), ('对话', '交谈', 22, 24), ('相关', '密切相关', 26, 30)])
mix-0.2: ('主要研究机器学习、深度学、计算机听觉、智能对话软件系统相关内容', [('学习', '学', 11, 12), ('视觉', '听觉', 16, 18), ('系统', '软件系统', 23, 27)])
```
</details>

### TGLS model (unsupervised similar text generation model)

Unsupervised generation of Chinese e-commerce reviews: Extract short sentences expressing opinions from users from **e-commerce reviews** and combine them to generate simulated reviews.

example: [examples/unsup_generation/unsup_generation_demo.py](examples/unsup_generation/unsup_generation_demo.py)

<details>
<summary>show code example and result</summary>

```python
import os
import sys

sys.path.append('..')
from textgen.unsup_generation import TglsModel, load_list

pwd_path = os.path.abspath(os.path.dirname(__file__))

samples = load_list(os.path.join(pwd_path, './data/ecommerce_comments.txt'))
docs_text = [
    ["挺好的，速度很快，也很实惠，不知效果如何",
     "产品没得说，买了以后就降价，心情不美丽。",
     "刚收到，包装很完整，不错",
     "发货速度很快，物流也不错，同一时间买的两个东东，一个先到一个还在路上。这个水水很喜欢，不过盖子真的开了。盖不牢了现在。",
     "包装的很好，是正品",
     "被种草兰蔻粉水三百元一大瓶囤货，希望是正品好用，收到的时候用保鲜膜包裹得严严实实，只敢买考拉自营的护肤品",
     ],
    ['很温和，清洗的也很干净，不油腻，很不错，会考虑回购，第一次考拉买护肤品，满意',
     '这款卸妆油我会无限回购的。即使我是油痘皮，也不会闷痘，同时在脸部按摩时，还能解决白头的脂肪粒的问题。用清水洗完脸后，非常的清爽。',
     '自从用了fancl之后就不用其他卸妆了，卸的舒服又干净',
     '买贵了，大润发才卖79。9。',
     ],
    samples
]
m = TglsModel(docs_text)
r = m.generate(samples[:500])
print('size:', len(r))
for review in r:
    print('\t' + review)
```

output:

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

The first 10 sentences are real user reviews, and the last 10 sentences are generated.

</details>

## 📚 Dataset 

1. Belle dataset of 500,000 Chinese ChatGPT commands: [BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
2. Belle dataset of 1 million Chinese ChatGPT commands: [BelleGroup/train_1M_CN](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
3. Alpaca dataset of 50,000 English ChatGPT commands: [50k English Stanford Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca#data-release)
4. Alpaca dataset of 20,000 Chinese ChatGPT commands: [shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)
5. Guanaco dataset with 690,000 Chinese instructions (500,000 Belle + 190,000 Guanaco): [Chinese-Vicuna/guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)
6. 2.4 million Chinese medical data sets (including pre-training data and instruction fine-tuning data sets): [shibing624/medical](https://huggingface.co/datasets/shibing624/medical)

## ✅ Todo

1. [ ] Added multi-round dialogue data fine-tuning method
2. [x] add reward model finetuning
3. [x] add rl finetuning
4. [x] add medical reward dataset
5. [x] add llama in4 training
6. [ ] add all training and predict demo in colab

## ☎️ Contact

- Issue (suggestion)
   : [![GitHub issues](https://img.shields.io/github/issues/shibing624/textgen.svg)](https://github.com/shibing624/textgen/issues)
- Email me: xuming: xuming624@qq.com
- WeChat Me: Add me* WeChat ID: xuming624, Remarks: Name-Company Name-NLP* Enter the NLP exchange group.

<img src="docs/wechat.jpeg" width="200" />

## 😇 Citation

If you use textgen in your research, please cite it in the following format:

```latex
@misc{textgen,
  title={textgen: Text Generation Tool},
  author={Ming Xu},
  year={2021},
  howpublished={\url{https://github.com/shibing624/textgen}},
}
```

## 🤗 License

The authorization agreement is [The Apache License 2.0](/LICENSE), which can be used for commercial purposes free of charge. Please attach textgen's link and license agreement in the product description.

## 😍 Contribute

The project code is still rough. If you have improved the code, you are welcome to submit it back to this project. Before submitting, please pay attention to the following two points:

- Add corresponding unit tests in `tests`
- Use `python -m pytest` to run all unit tests to ensure that all unit tests are passed

Then you can submit a PR.

## 💕 Acknowledgements 

- [PaddlePaddle/ERNIE](https://github.com/PaddlePaddle/ERNIE)
- [minimaxir/textgenrnn](https://github.com/minimaxir/textgenrnn)
- [minimaxir/gpt-2-simple](https://github.com/minimaxir/gpt-2-simple)
- [asyml/texar](https://github.com/asyml/texar)
- [yangjianxin1/GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)
- [williamSYSU/TextGAN-PyTorch](https://github.com/williamSYSU/TextGAN-PyTorch)
- [RUCAIBox/TextBox](https://github.com/RUCAIBox/TextBox)
- [Tiiiger/bert_score](https://github.com/Tiiiger/bert_score)
- [ThilinaRajapakse/simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers)
- [1YCxZ/Fake-review-generation](https://github.com/1YCxZ/Fake-review-generation)
- [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora/blob/main/finetune.py)

Thanks for their great work!
