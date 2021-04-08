# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 一种基于UDA的非核心词替换的算法
"""

import json
from codecs import open

import jieba
import numpy as np
from text2vec import Vector

from textgen.utils.logger import logger

cop_idf_file = '../extra_data/replace_data_all_idf'
cop_tfidf_file = '../extra_data/replace_data_all_tfidf'


class TfIdfAug(object):
    """
    基于google UDA的非关键词选取
    """

    def __init__(self):
        """
        init
        """
        self.idf_dict = dict()
        self.all_tfidf_dict = dict()
        self.extract_prob = dict()
        self.default_tfidf = 0.0
        self.default_idf = 0.0
        with open(cop_idf_file, "r", encoding='utf-8') as fr:
            for line in fr:
                each_list = line.strip().split("\t")
                if len(each_list) > 1:
                    word = each_list[0]
                    idf = float(each_list[1])
                    if word not in self.idf_dict:
                        self.idf_dict[word] = idf
        with open(cop_tfidf_file, "r", encoding='utf-8') as fr:
            for line in fr:
                each_list = line.strip().split("\t")
                if len(each_list) > 1:
                    word = each_list[0]
                    word_tfidf = float(each_list[1])
                    if word not in self.all_tfidf_dict:
                        self.all_tfidf_dict[word] = word_tfidf
        self.ws = jieba
        max_all_tfidf_word = sorted(self.all_tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:1][0][0]
        max_all_tfidf = sorted(self.all_tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:1][0][1]
        z_sigma = 0.0
        for key in self.all_tfidf_dict:
            z_sigma += max_all_tfidf - self.all_tfidf_dict[key]
        original_word = []
        word_prob = []
        for key, value in self.all_tfidf_dict.items():
            prob = (max_all_tfidf - value) / z_sigma
            self.extract_prob[key] = prob
            original_word.append(key)
            word_prob.append(prob)
        self.default_prob = (max_all_tfidf - self.default_tfidf) / z_sigma
        vec = Vector()
        vec.load_model()
        self.word2vec_model = vec.model.w2v

    def data_aug(self, text_list):
        """
        数据增强核心代码
        """
        # 边界判断
        if len(text_list) == 0:
            logger.info("input text list is null")
            return ""
        tf_dict = dict()
        tfidf_dict = dict()
        cur_nums = 0
        for word in text_list:
            cur_nums += 1
            tf_dict[word] = tf_dict.get(word, 0.0) + 1.0
        for k, v in tf_dict.items():
            tf_dict[k] = v / cur_nums
        # 获得tfidf
        for word in text_list:
            if word in self.idf_dict:
                tfidf_dict[word] = tf_dict[word] * self.idf_dict[word]
            else:
                tfidf_dict[word] = tf_dict[word] * self.default_idf
        # 最大的tfidf值
        max_tfidf_word = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:1][0][0]
        max_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:1][0][1]
        # 以下为UDA中采样词的概率
        Z = 0.0
        p = 0.3
        replace_prob = dict()
        for word in text_list:
            Z += (max_tfidf - tfidf_dict[word]) / len(text_list)
        for word in text_list:
            if Z == 0.0:
                replace_prob[word] = 0.0
            else:
                temp = p * (max_tfidf - tfidf_dict[word]) / Z
                if temp < 1.0:
                    replace_prob[word] = temp
                else:
                    replace_prob[word] = 1.0
        replace_prob_sorted = sorted(replace_prob.items(), key=lambda x: x[1], reverse=True)
        logger.info("".join(text_list) + " sentence replace prob is: ")
        for k, v in replace_prob_sorted:
            logger.info("%s\t%f " % (k, v))
        ret_word_list = list()
        for word in text_list:
            if word in replace_prob:
                if np.random.rand() <= replace_prob[word] and word in self.word2vec_model.wv.vocab:
                    target_candidate = self.word2vec_model.similar_by_word(word, topn=2)
                    # target_word = target_word[0][0]
                    target_word = [w for w, p in target_candidate]
                    # target_word = np.random.choice([w for w, p in target_candidate], 2)
                    # 候选词一个
                    choice_prob_list = []
                    for tar_word in target_word:
                        if tar_word in self.extract_prob:
                            choice_prob_list.append(self.extract_prob[tar_word])
                        else:
                            choice_prob_list.append(self.default_prob)
                    best_choice_word = target_word[choice_prob_list.index(max(choice_prob_list))]
                    ret_word_list.append(best_choice_word)
                else:
                    ret_word_list.append(word)
            else:
                ret_word_list.append(word)
        return "".join(ret_word_list)

    def get_data_aug(self, data_json):
        """
        数据预处理
        """
        data_dict = json.loads(data_json)
        data_list = data_dict["text_list"]
        aug_nums = data_dict["topk"]
        data_result = list()
        for text in data_list:
            aug_data_set = set()
            seg_text = self.ws.lcut(text)
            for k in range(aug_nums):
                deal_word_list = [x for x in seg_text]
                new_text = self.data_aug(deal_word_list)
                if new_text == "":
                    continue
                aug_data_set.add(new_text)
            text_result = list()
            text_result.append(text)
            aug_text_list = list()
            for aug_text in aug_data_set:
                cur_aug_list = list()
                cur_aug_list.append(aug_text)
                aug_text_list.append(cur_aug_list)
            text_result.append(aug_text_list)
            data_result.append(text_result)
        return data_result


if __name__ == "__main__":
    tfidf_aug = TfIdfAug()
    text = "晚上一个人好孤单，想找附近人陪陪我"
    data_dict = {"text_list": [text], "topk": 10}
    data_result = tfidf_aug.get_data_aug(json.dumps(data_dict))
    print(len(data_result))
    print(data_result)
