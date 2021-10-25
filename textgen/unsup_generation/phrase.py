# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import jieba
import jieba.posseg as pseg
import math

pwd_path = os.path.abspath(os.path.dirname(__file__))

WINDOW_SIZE = 5
PUNCTUATION_MARK = ['x']  # 标点
NOUN_MARK = ['n', 'ng', 'nr', 'nrfg', 'nrt', 'ns', 'nt', 'nz']  # 名词
VERB_MARK = ['v', 'vd', 'vg', 'vi', 'vn', 'vq']  # 动词
ADJECTIVE_MARK = ['a', 'ad', 'an', 'ag']  # 形容词
ADVERB_MARK = ['d', 'df', 'dg']  # 副词
ENG_MARK = ['eng']

RESERVED_MARK = NOUN_MARK + VERB_MARK + ADJECTIVE_MARK + ADVERB_MARK + ENG_MARK  # 用于发现新词

jieba.load_userdict(os.path.join(pwd_path, '../data/user_dict.txt'))


def text2review(seg_pos_text):
    """
    经过分词的文档，得到原始用户的每条评论
    """
    review_list = []  # 保存全部的按照指定标点切分的句子
    all_word = set()  # 全部单词
    for seg_pos in seg_pos_text:
        cur_review = []
        for term in seg_pos:
            word, flag = term.split('/')
            cur_review.append(word)
            if flag in RESERVED_MARK:
                all_word.add(word)
        review_list.append(cur_review)

    return review_list, all_word


def find_word_phrase(all_word, seg_list):
    """
    根据点互信息以及信息熵发现词组，主要目的是提升分词效果
    """
    res = []
    word_count = {k: 0 for k in all_word}  # 记录全部词出现的次数

    all_word_count = 0
    all_bi_gram_count = 0
    for sentence in seg_list:
        all_word_count += len(sentence)
        all_bi_gram_count += len(sentence) - 1
        for idx, word in enumerate(sentence):
            if word in word_count:
                word_count[word] += 1

    bi_gram_count = {}
    bi_gram_lcount = {}
    bi_gram_rcount = {}
    for sentence in seg_list:
        for idx, _ in enumerate(sentence):
            left_word = sentence[idx - 1] if idx != 0 else ''
            right_word = sentence[idx + 2] if idx < len(sentence) - 2 else ''

            first = sentence[idx]
            second = sentence[idx + 1] if idx + 1 < len(sentence) else ''
            if first in word_count and second in word_count:
                if (first, second) in bi_gram_count:
                    bi_gram_count[(first, second)] += 1
                else:
                    bi_gram_count[(first, second)] = 1
                    bi_gram_lcount[(first, second)] = {}
                    bi_gram_rcount[(first, second)] = {}

                if left_word in bi_gram_lcount[(first, second)]:
                    bi_gram_lcount[(first, second)][left_word] += 1
                elif left_word != '':
                    bi_gram_lcount[(first, second)][left_word] = 1

                if right_word in bi_gram_rcount[(first, second)]:
                    bi_gram_rcount[(first, second)][right_word] += 1
                elif right_word != '':
                    bi_gram_rcount[(first, second)][right_word] = 1

    bi_gram_count = dict(filter(lambda x: x[1] >= 5, bi_gram_count.items()))

    bi_gram_le = {}  # 全部bi_gram的左熵
    bi_gram_re = {}  # 全部bi_gram的右熵
    for phrase in bi_gram_count:
        le = 0
        for l_word in bi_gram_lcount[phrase]:
            p_aw_w = bi_gram_lcount[phrase][l_word] / bi_gram_count[phrase]  # P(aW | W)
            le += p_aw_w * math.log2(p_aw_w)
        le = -le
        bi_gram_le[phrase] = le

    for phrase in bi_gram_count:
        re = 0
        for r_word in bi_gram_rcount[phrase]:
            p_wa_w = bi_gram_rcount[phrase][r_word] / bi_gram_count[phrase]  # P(Wa | W)
            re += p_wa_w * math.log2(p_wa_w)
        re = -re
        bi_gram_re[phrase] = re

    PMI = {}
    for phrase in bi_gram_count:
        p_first = word_count[phrase[0]] / all_word_count
        p_second = word_count[phrase[1]] / all_word_count
        p_bi_gram = bi_gram_count[phrase] / all_bi_gram_count
        PMI[phrase] = math.log2(p_bi_gram / (p_first * p_second))

    phrase_score = []
    for phrase in PMI:
        le = bi_gram_le[phrase]
        re = bi_gram_re[phrase]
        score = PMI[phrase] + le + re
        phrase_score.append((phrase, score))

    phrase_score = sorted(phrase_score, key=lambda x: x[1], reverse=True)

    for item in phrase_score:
        res.append('{}:{}'.format(''.join(item[0]), item[1]))

    return res


def load_list(path):
    return [l for l in open(path, 'r', encoding='utf-8').read().split()]


def caculate_word_idf(docs, stopwords):
    """
    计算所有文档中的每个词的idf
    docs: list(list(str)), 数据集
    stop_word: list, 停用词list

    return: 所有词的idf值
    """
    word_IDF = {}  # word-IDF 记录每个word在不同的doc出现过的次数,然后计算IDF
    num_doc = len(docs)  # 商品数量
    seg_pos_text = []
    for doc in docs:
        cur_doc_word_set = set()  # 记录当前文档中出现的不同的词
        for line in doc:
            line = line.strip()
            seg_pos_list = get_seg_pos(line, type='word')
            seg_pos_text.append(seg_pos_list)
            word_list = [term.split('/')[0] for term in seg_pos_list]
            for w in word_list:
                # 如果这个词在停用词表中就不添加
                if w in stopwords:
                    continue
                cur_doc_word_set.add(w)
        for w in cur_doc_word_set:
            if w in word_IDF:
                word_IDF[w] += 1
            else:
                word_IDF[w] = 1
    for w in word_IDF:
        word_IDF[w] = math.log10(num_doc / word_IDF[w])
    return word_IDF, seg_pos_text


def get_seg_pos(line, type='word'):
    """
    获取文档的分词以及词性标注结果，分词的方式可以为按词切分或者按字切分
    """
    if type == 'word':
        line_cut = pseg.cut(line.strip())
        wordlist = []
        for term in line_cut:
            wordlist.append('%s/%s' % (term.word, term.flag))
        res = wordlist
    else:
        res = list(line.strip())
    return res


if __name__ == '__main__':
    default_stopwords_path = os.path.join(pwd_path, '../data/stopwords.txt')
    sample1 = load_list(os.path.join(pwd_path, '../data/10475.txt'))
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
    print('docs_text len:', len(docs_text))
    # 加载停用词
    stopwords = set(load_list(default_stopwords_path))
    # 计算除去停用词的每个词的idf值
    word_idf, seg_pos_text = caculate_word_idf(docs_text, stopwords)

    review_list, all_word = text2review(seg_pos_text)

    phrase_list = find_word_phrase(all_word, review_list)
    print(phrase_list)
