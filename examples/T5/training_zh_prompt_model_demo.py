# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

下载pCLUE的部分数据（如，pCLUE_train_1.json）到本地
wget https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_1.json
wget https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_2.json
wget https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_3.json
wget https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_4.json
wget https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_5.json
wget https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_6.json
wget https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_7.json
wget https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_8.json
wget https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_9.json
cat pCLUE_train_*.json > pCLUE_train.json
"""
import argparse
import json
from loguru import logger
import pandas as pd
import numpy as np
import time
import os
import sys

sys.path.append('../..')
from textgen import T5Model
from textgen.t5.t5_utils import f1_sim, rouge_l_zh


def load_json_data(prefix, file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # {"input": "以下内容为真：“滁县地区专员张友道说:大都架到高处了”那么下面的陈述：“张友道对身边的官员说了话。”是真的,假的,或未知？\n答案：", "target": "未知", "answer_choices": ["真的", "假的", "未知"], "type": "nli"}
            line = line.strip()
            if line:
                json_string = json.loads(line.strip())
                input_text = json_string["input"].replace("\n", "_")
                target_text = json_string["target"]
                answer_choices = json_string.get("answer_choices", [])
                type = json_string["type"]
                data.append([prefix, input_text, target_text])
            else:
                logger.warning(f'line error: {line}')
    return data


def normalize(text):
    """简单的文本标准化"""
    return ' '.join(text.lower().split())


def evaluate_pclue_fn(test_file, model, select_top=200):
    """
    计算pclue的成绩
    :param test_file: 预测文件
    :param target_file:  正确的文件
    :return: 一个dict，包括总分score，以及各个部分的分数（mrc, generate, classify, nli）
    """
    predict_lines = open(test_file, 'r', encoding='utf8').readlines()
    predict_lines = predict_lines[:select_top]
    # 1.记录
    classify_list = []
    mrc_list = []
    generate_list = []
    nli_list = []
    for i, predict_line in enumerate(predict_lines):
        json_dict = json.loads(predict_line.strip())
        type = json_dict["type"]
        input_text = json_dict['input']
        target_line = json_dict["target"]
        target_answer = target_line.replace("，", ",")  # 正确的标签
        if isinstance(target_answer, list):  # 将列表转换为字符串，如关键词生成
            target_answer = "，".join(target_answer)
        target_answer = normalize(target_answer)
        predict_answer = model.predict(input_text)[0]  # 预测的标签
        predict_answer = normalize(predict_answer)
        if len(predict_answer) == 0:
            predict_answer = "无答案"
        if type == 'classify' or type == 'anaphora_resolution':  # 分类
            label_temp = True if target_answer == predict_answer else False
            classify_list.append(label_temp)
        elif type == 'mrc':  # 阅读理解
            em = 1 if target_answer == predict_answer else 0
            f1 = f1_sim(predict_answer, target_answer)
            mrc_list.append((em, f1))
        elif type == 'generate':  # 生成
            rouge_l = rouge_l_zh(target_answer, predict_answer)
            generate_list.append(rouge_l)
        elif type == 'nli':  # 推理
            label_temp = True if target_answer == predict_answer else False
            nli_list.append(label_temp)
        else:
            logger.error("predict_line: {predict_line}")
            break
        if i < 10:
            logger.debug(f"num: {i}, target_answer: {target_answer}; predict_answer: {predict_answer}")
    # 2.计算最后的得分
    classify_score = np.average(classify_list)
    nli_score = np.average(nli_list)
    generate_score = np.average(generate_list)
    mrc_em_score = np.average([x[0] for x in mrc_list])
    mrc_f1_score = np.average([x[1] for x in mrc_list])
    mrc_score = np.average([mrc_em_score, mrc_f1_score])
    # 计算总分
    score = np.average([classify_score, nli_score, generate_score, mrc_score])
    # 保存分数
    result_dict = {"score": score, "classify_score": classify_score, "nli_score": nli_score,
                   "generate_score": generate_score, "mrc_em_score": mrc_em_score, "mrc_f1_score": mrc_f1_score}
    return result_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='../data/pCLUE_train_1k.json', type=str, help='Training data file')
    parser.add_argument('--test_file', default='../data/pCLUE_test_500.json', type=str, help='Test data file')
    parser.add_argument('--model_type', default='t5', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='ClueAI/PromptCLUE-base', type=str, help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--prefix', default='prompt', type=str, help='Prefix str')
    parser.add_argument('--output_dir', default='./outputs/prompt_zh/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=128, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=512, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)

    if args.do_train:
        logger.info('Loading data...')
        train_data = load_json_data(args.prefix, args.train_file)
        logger.debug('train_data: {}'.format(train_data[:10]))
        train_df = pd.DataFrame(train_data, columns=["prefix", "input_text", "target_text"])

        eval_data = load_json_data(args.prefix, args.train_file)[:10]
        eval_df = pd.DataFrame(eval_data, columns=["prefix", "input_text", "target_text"])

        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "max_length": args.max_length,
            "train_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": True,
            "evaluate_generated_text": True,
            "evaluate_during_training": True,
            "evaluate_during_training_verbose": True,
            "use_multiprocessing": False,
            "save_best_model": True,
            "output_dir": args.output_dir,
            "use_early_stopping": True,
            "best_model_dir": os.path.join(args.output_dir, "best_model"),
        }
        model = T5Model(args.model_type, args.model_name, args=model_args)

        def sim_text_chars(text1, text2):
            if not text1 or not text2:
                return 0.0
            same = set(text1) & set(text2)
            m = len(same)
            n = len(set(text1)) if len(set(text1)) > len(set(text2)) else len(set(text2))
            return m / n

        def count_matches(labels, preds):
            logger.debug(f"labels: {labels[:10]}")
            logger.debug(f"preds: {preds[:10]}")
            match = sum([sim_text_chars(label, pred) for label, pred in zip(labels, preds)]) / len(labels)
            logger.debug(f"match: {match}")
            return match

        model.train_model(train_df, eval_data=eval_df, matches=count_matches)
        print(model.eval_model(eval_df, matches=count_matches))

    if args.do_predict:
        model = T5Model(args.model_type, args.output_dir, args={"eval_batch_size": args.batch_size})
        sentences = [
            "阅读下列对话。_女：小李，听说你的毕业设计主题是环保？男：对，我的作品所用的材料大都是一些废弃的日用品。女：都用了什么东西？_听者会怎么说？",
            "这篇新闻会出现在哪个栏目？吴绮莉独自返家神情落寞 再被问小龙女只说了7个字_选项：故事，文化，娱乐，体育，财经，房产，汽车，教育，科技，军事，旅游，国际，股票，农业，游戏_答案：",
            "我想知道下面两句话的意思是否相同。“怎么把借呗的钱转到余额宝”，“借呗刚刚才转钱到余额宝，可以重新扣一次款吗”是相同的吗？。选项：是的，不是。答案："
        ]
        sentences_add_prefix = [args.prefix + ": " + i for i in sentences]
        print("inputs:", sentences)
        print("outputs:", model.predict(sentences_add_prefix))

        select_top = 200
        t1 = time.time()
        # evaluate the model for multi task
        evaluate_pclue_fn(args.test_file, model, select_top=select_top)
        logger.info(f'spend time: {time.time() - t1}, size: {select_top}')


if __name__ == '__main__':
    main()
