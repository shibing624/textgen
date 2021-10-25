# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
from textgen.unsup_generation.phrase import load_list, caculate_word_idf, text2review, find_word_phrase, get_seg_pos
from textgen.unsup_generation.util import text2seg_pos, get_aspect_express, get_candidate_aspect, NSDict, PairPattSort, \
    merge_aspect_express, fake_review_filter, generate_reviews

pwd_path = os.path.abspath(os.path.dirname(__file__))
default_stopwords_path = os.path.join(pwd_path, '../data/stopwords.txt')
default_pos_adj_word_path = os.path.join(pwd_path, '../data/HowNetPOSWord.txt')


class Generate:
    def __init__(self, docs):
        print('docs_text len:', len(docs))
        # 加载停用词
        self.stopwords = set(load_list(default_stopwords_path))
        # 计算除去停用词的每个词的idf值
        self.word_idf, self.seg_pos_text = caculate_word_idf(docs, self.stopwords)

        review_list, all_word = text2review(self.seg_pos_text)
        phrase_list = find_word_phrase(all_word, review_list)
        print('find new word:', phrase_list)

        # 加载正向情感词典
        self.pos_adj_word = load_list(default_pos_adj_word_path)

    def generate(self, doc):
        seg_pos_text = [get_seg_pos(l) for l in doc]
        seg_list, pos_list, seg_review_list = text2seg_pos(seg_pos_text, pattern='[。！？，～]')
        raw_aspect_list = get_candidate_aspect(seg_list, pos_list, self.pos_adj_word, self.stopwords, self.word_idf)

        # 构建候选集合
        N = NSDict(seg_list, pos_list, raw_aspect_list)
        ns_dict = N.build_nsdict()
        # 候选集合排序
        P = PairPattSort(ns_dict)
        pair_score = P.sort_pair()

        # 得到正确的观点表达候选
        pair_useful = {}
        baseline = 0.1 * len(pair_score)
        for i, item in enumerate(pair_score):
            if i <= baseline:
                aspect, opinion = item[0].split('\t')
                if aspect in pair_useful:
                    pair_useful[aspect].append(opinion)
                else:
                    pair_useful[aspect] = [opinion]

        # 从原始评论中抽取观点表达
        aspect_express = get_aspect_express(seg_review_list, pair_useful)

        # 字符匹配合并aspect
        merged_aspect_express, opinion_set = merge_aspect_express(aspect_express, pair_useful)

        # 生成假评论
        generated_raw_reviews = generate_reviews(merged_aspect_express)

        results = fake_review_filter(generated_raw_reviews, opinion_set)
        return results


if __name__ == '__main__':
    sample1 = load_list(os.path.join(pwd_path, '../data/12617.txt'))
    docs_text = [["挺好的，速度很快，也很实惠，不知效果如何",
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
                 sample1
                 ]
    m = Generate(docs_text)
    r = m.generate(sample1[:400])
    print(len(r))
