# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 无监督抽取用户观点
参考: https://github.com/rainarch/SentiBridge

jieba分词的效果还不足以支持电商评论，例如"痘痘肌"、"炒鸡棒"、"t字区"等词是jieba无法处理的。
新增新词发现功能(PMI+左右熵)的方法来找出新词，参考：https://www.matrix67.com/blog/archives/5044
"""
import random
from loguru import logger
import os
import jieba
import jieba.posseg
import math
import re

pwd_path = os.path.abspath(os.path.dirname(__file__))
jieba.setLogLevel(log_level="ERROR")
WINDOW_SIZE = 5
PUNCTUATION_MARK = ['x']  # 标点
PUNCTUATION = ['。', '！', '？', '，', '～']
NOUN_MARK = ['n', 'ng', 'nr', 'nrfg', 'nrt', 'ns', 'nt', 'nz']  # 名词
VERB_MARK = ['v', 'vd', 'vg', 'vi', 'vn', 'vq']  # 动词
ADJECTIVE_MARK = ['a', 'ad', 'an', 'ag']  # 形容词
ADVERB_MARK = ['d', 'df', 'dg']  # 副词
ENG_MARK = ['eng']
EMOJI = ['😀', '😁', '😂', '😃', '😄', '😆', '😉', '😊',
         '😋', '😎', '😍', '😘', '😗', '😙', '😚', '😇',
         '😏', '😝']
YANWENZI = ['ヽ(✿ﾟ▽ﾟ)ノ', 'φ(≧ω≦*)♪', '╰(*°▽°*)╯', 'o(￣▽￣)ｄ', 'o( =•ω•= )m']
ILLEGAL_WORD = ['考拉', '网易', '淘宝', '京东', '拼多多', '不过', '因为', '而且', '但是', '但', '所以', '因此', '如果']  # 过滤词

RESERVED_MARK = NOUN_MARK + VERB_MARK + ADJECTIVE_MARK + ADVERB_MARK + ENG_MARK  # 用于发现新词
ASPECT_MARK = NOUN_MARK + VERB_MARK

PUNCTUATION_MAP = {".": "。", ",": "，", "!": "！", "?": "？", "~": "～"}
keep_p = ['，', '。', '！', '？', '～', '、']


def convert(content):
    """转化标点符号为中文符号"""
    nc = []
    for c in content:
        if c in PUNCTUATION_MAP:
            nc.append(PUNCTUATION_MAP[c])
            continue
        nc.append(c)
    return "".join(nc)


def clean(line):
    """清洗无意义字符"""
    if line == "":
        return
    line = convert(line)
    c_content = []
    for char in line:
        if re.search("[\u4e00-\u9fa5]", char):
            c_content.append(char)
        elif re.search("[a-zA-Z0-9]", char):
            c_content.append(char)
        elif char in keep_p:
            c_content.append(char)
        elif char == ' ':  # 很多用户喜欢用空格替代标点
            c_content.append('，')
        else:
            c_content.append('')
    nc_content = []
    c = 0
    for char in c_content:
        if char in keep_p:
            c += 1
        else:
            c = 0
        if c < 2:
            nc_content.append(char)
    result = ''.join(nc_content)
    result = result.strip()
    result = result.lower()  # 所有英文转成小写字母
    return result


def clean_review(text):
    """
    对原始评论进行清理，删去非法字符，统一标点，删去无用评论
    """
    review_set = []
    for line in text:
        line = line.lstrip()
        line = line.rstrip()
        line = clean(line)
        if len(line) < 7:  # 过于短的评论需要删除
            continue
        if line and line not in ['该用户没有填写评论。', '用户晒单。']:
            review_set.append(line)

    return review_set


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
        line_cut = jieba.posseg.cut(line.strip())
        wordlist = []
        for term in line_cut:
            wordlist.append('%s/%s' % (term.word, term.flag))
        res = wordlist
    else:
        res = list(line.strip())
    return res


def text2seg_pos(seg_pos_text, pattern='[。！？]'):
    """
    经过分词的文档，原始一条用户评论通过指定的标点符号分成多个句子
    """
    seg_list = []  # 保存全部按标点切分的seg
    pos_list = []  # 保存全部按标点切分的pos
    seg_review_list = []  # 用户完整的一条评论
    for seg_pos in seg_pos_text:
        seg_sub_list = []
        pos_sub_list = []
        cur_review = []
        for term in seg_pos:
            word, flag = term.split('/')
            cur_review.append(word)
            if word in pattern:
                seg_sub_list.append(word)
                pos_sub_list.append(flag)
                seg_list.append(list(seg_sub_list))
                pos_list.append(list(pos_sub_list))
                seg_sub_list = []
                pos_sub_list = []
            else:
                seg_sub_list.append(word)
                pos_sub_list.append(flag)
        seg_review_list.append(list(cur_review))

    return seg_list, pos_list, seg_review_list


def get_candidate_aspect(seg_list, pos_list, adj_word, stop_word, word_idf):
    """
    输入的数据为用逗号隔开的短句，
    利用开窗口的方式，根据情感词典抽名词得到候选的aspect
    """
    aspect_dict = {}
    for i, sentence in enumerate(seg_list):
        for j, word in enumerate(sentence):
            if word in adj_word and pos_list[i][j] in ADJECTIVE_MARK:  # 当前的词属于情感词且词性为形容词
                startpoint = j - WINDOW_SIZE
                startpoint = startpoint if startpoint >= 0 else 0
                for k in range(startpoint, j):
                    if pos_list[i][k] in ASPECT_MARK:
                        if seg_list[i][k] in aspect_dict:
                            aspect_dict[seg_list[i][k]] += 1
                        else:
                            aspect_dict[seg_list[i][k]] = 1

    candidates = aspect_dict.items()
    candidates = list(filter(lambda x: len(x[0]) > 1, candidates))  # 经过词组发现之后，删去一个字的词
    candidates = [item[0] for item in candidates if item[0] not in stop_word]  # 删去停用词
    candidates = [item if (item in word_idf and word_idf[item] != 0) else item for item in candidates]  # 删去IDF值为0的词
    logger.debug(f"Extract {len(candidates)} aspect candidates, top10: {candidates[:10]}")
    return candidates


class NSDict:
    """
    用来构建候选集（aspect，opinion，pattern）
    """

    def __init__(self, seg_list, pos_list, raw_aspect_list):
        self.seg_list = seg_list
        self.pos_list = pos_list
        self.raw_aspect_list = raw_aspect_list
        self.ns_dict = {}
        self.aspect_do_not_use = []
        self.opinion_do_not_use = ["最", "不", "很"]
        self.pattern_do_not_use = ["的-", "和-", "和+", "而+", "而-", "又+", "又-", "而且+", "而且-"]

    def _seg2nsd(self, aspect_for_filter):
        for x, clue in enumerate(self.seg_list):
            N_list = []
            S_list = []
            word_list = clue
            for y, word in enumerate(clue):
                if word in aspect_for_filter:
                    N_list.append(y)
                elif self.pos_list[x][y] in ADJECTIVE_MARK:
                    S_list.append(y)
            if N_list and S_list:
                self._make_nsdict(word_list, N_list, S_list)

    def _make_nsdict(self, word_list, N_list, S_list):
        for n in N_list:
            for s in S_list:
                if (1 < n - s < WINDOW_SIZE + 1) or (1 < s - n < WINDOW_SIZE + 1):  # 窗口大小是5
                    if word_list[n] not in self.ns_dict:
                        self.ns_dict[word_list[n]] = {}
                    if word_list[s] not in self.ns_dict[word_list[n]]:
                        self.ns_dict[word_list[n]][word_list[s]] = {}
                    if n > s:
                        patt = ' '.join(word_list[s + 1: n]) + '+'
                    else:
                        patt = ' '.join(word_list[n + 1: s]) + '-'
                    if patt not in self.ns_dict[word_list[n]][word_list[s]]:
                        self.ns_dict[word_list[n]][word_list[s]][patt] = 0.
                    self.ns_dict[word_list[n]][word_list[s]][patt] += 1.

    def _noise_del(self):
        for aspect in self.aspect_do_not_use:
            self._noise(aspect, self.ns_dict)
        for n in self.ns_dict:
            for opinion in self.opinion_do_not_use:
                self._noise(opinion, self.ns_dict[n])
            for s in self.ns_dict[n]:
                for pattern in self.pattern_do_not_use:
                    self._noise(pattern, self.ns_dict[n][s])

    def _noise(self, str, dict):
        if str in dict:
            del dict[str]

    def build_nsdict(self):
        """Stage 1：extract pair and pattern"""
        self._seg2nsd(self.raw_aspect_list)
        self._noise_del()
        return self.ns_dict


class PairPattSort:
    """
    Pair-Patt-Count structure
    """

    def __init__(self, ns_dict):
        self._get_map(ns_dict)

    def _get_map(self, ns_dict):
        """
        get map: [pair-patt], [patt-pair], [pair](score), [patt](score)

        :param ns_dict: Entity.str { Emotion.str { Pattern.str { Count.int (It's a three-level hash structure)
        :return:
        """
        pair_list = []
        patt_dict = {}
        patt_pair_map = {}
        pair_patt_map = {}

        aspects = list(ns_dict.keys())
        aspects.sort()

        for n in aspects:
            for s in ns_dict[n]:
                n_s = "{}\t{}".format(n, s)  # 这里存的pair是字符串，中间用\t隔开
                pair_list.append(n_s)
                pair_patt_map[n_s] = {}
                for patt in ns_dict[n][s]:
                    if patt not in patt_dict:
                        patt_dict[patt] = 1.0
                    pair_patt_map[n_s][patt] = ns_dict[n][s][patt]
                    if patt in patt_pair_map:
                        patt_pair_map[patt][n_s] = ns_dict[n][s][patt]
                    else:
                        patt_pair_map[patt] = {}
                        patt_pair_map[patt][n_s] = ns_dict[n][s][patt]
        self.patt_pair_map = patt_pair_map
        self.pair_patt_map = pair_patt_map
        self.pair_len = len(pair_list)
        self.patt_len = len(patt_dict)
        self.pair_score = dict([(word, 1.) for i, word in enumerate(pair_list)])
        self.patt_score = patt_dict

    def _norm(self, score_dict, score_len):
        """
        正则化，和为score_len
        """
        sum_score = 0.
        for s in score_dict:
            sum_score += score_dict[s]
        for s in score_dict:
            score_dict[s] = score_dict[s] / sum_score * score_len
        return score_dict

    def _patt_pair(self):
        for pair in self.pair_patt_map:  # <- 循环遍历每个pair
            value = 0.
            for patt in self.pair_patt_map[pair]:  # <- 每个pair中的pattern出现的个数 * 这个pattern的score，然后求和得到这个pair的分数
                value += self.pair_patt_map[pair][patt] * self.patt_score[patt]
            self.pair_score[pair] = value

    def _pair_patt(self):
        for patt in self.patt_pair_map:  # <- 遍历每个pattern
            value = 0.
            for pair in self.patt_pair_map[patt]:  # <- 每个被pattern修饰的pair出现的个数 * 这个pair的score，然后求和得到这个pattern1的
                value += self.patt_pair_map[patt][pair] * self.pair_score[pair]
            self.patt_score[patt] = value

    def _patt_correct(self):
        self.patt_score['的-'] = 0.0

    def _iterative(self):
        """
        A complete iteration
        [pair] = [patt-pair] * [patt]
        [patt] = [pair-patt] * [pair]
        :return:
        """
        self._patt_pair()
        self.pair_score = self._norm(self.pair_score, self.pair_len)
        self._pair_patt()
        self.patt_score = self._norm(self.patt_score, self.patt_len)

    def sort_pair(self):
        """Stage 2：pair sort"""
        for i in range(100):
            self._iterative()
        pair_score = sorted(self.pair_score.items(), key=lambda d: d[1], reverse=True)
        return pair_score


def get_aspect_express(seg_review_list, pair_useful):
    """
    抽取原始评论中的aspect作为输入，完整的评论作为输出
    """

    def check_sentence(sentence):
        """
        判断短句是否合法
        """
        _s = ''.join(sentence)
        legal = True
        if len(_s) > 30:
            legal = False
        return legal

    raw_aspect_express = {k: [] for k in pair_useful}  # 用户关于某个观点的一段原始表达
    raw_aspect_express_count = {k: 0 for k in pair_useful}  # 记录某个观点表达出现的次数
    for review in seg_review_list:  # 每个sentence就是一句完整的review
        if review[-1] not in PUNCTUATION:
            review.append('。')

        # 对于单个review进行切分
        cur_review = []
        pre_end = 0
        for i, _ in enumerate(review):
            if review[i] in ['。', '！', '？', '，', '～']:
                cur_review.append(review[pre_end:i + 1])
                pre_end = i + 1
            elif i == len(review) - 1:
                cur_review.append(review[pre_end:])

        for sentence in cur_review:  # sentence 是两个标点之间的短句
            if sentence[-1] not in PUNCTUATION:
                sentence.append('。')
            find_opinion_flag = False
            for idx, word in enumerate(sentence):
                if find_opinion_flag:  # 如果在当前的短句中已经找到了一组观点表达就结束对这个短句的搜索
                    break
                if word in pair_useful:  # 当前的word属于aspect
                    # 向前开窗口
                    startpoint = idx - WINDOW_SIZE if idx - WINDOW_SIZE > 0 else 0
                    for i in range(startpoint, idx):  # 寻找opinion word
                        cur_word = sentence[i]
                        if cur_word in pair_useful[word] and sentence[i + 1] == "的":  # eg. 超赞的一款面膜
                            if check_sentence(sentence):
                                raw_aspect_express[word].append(sentence)
                                raw_aspect_express_count[word] += 1
                                find_opinion_flag = True  # 只要找到一个opinion word就算命中一个短句了

                    # 向后开窗口
                    endpoint = idx + WINDOW_SIZE if idx + WINDOW_SIZE < len(sentence) else len(sentence)
                    for i in range(idx + 1, endpoint):
                        cur_word = sentence[i]
                        if cur_word in pair_useful[word]:
                            if check_sentence(sentence):
                                raw_aspect_express[word].append(sentence)
                                raw_aspect_express_count[word] += 1
                                find_opinion_flag = True  # 只要找到一个opinion word就算命中一个短句了
    # 筛选得到保留的aspect
    aspect_express = {}
    for aspect in raw_aspect_express:
        if raw_aspect_express_count[aspect] < 5:
            continue
        aspect_express[aspect] = raw_aspect_express[aspect]

    return aspect_express


def merge_aspect_express(aspect_express, pair_useful):
    """
    对相似的观点表达进行合并, 同时输出最终的aspect_opinion_pair
    """
    aspects = list(aspect_express.keys())
    aspects.sort()  # 排成字典序
    merged_aspects = [[aspects[0]]] if aspects else [[]]
    merged_express = {}
    opinion_set = []

    def check_is_same(word1, word2):
        """
        判断两个词当中是否存在相同的字
        """
        for i in word1:
            if i in word2:
                return True
        return False

    for i in range(1, len(aspects)):
        if check_is_same(merged_aspects[-1][-1], aspects[i]):
            merged_aspects[-1].append(aspects[i])
        else:
            merged_aspects.append([aspects[i]])
    for a_list in merged_aspects:
        # 收集全部的形容词
        for i in a_list:
            opinion_set += pair_useful[i]

        _l = ','.join(a_list)
        merged_express[_l] = []
        for i in a_list:
            merged_express[_l] += aspect_express[i]
    opinion_set = set(opinion_set)
    return merged_express, opinion_set


def build_dataset_express(seg_review_list, pair_useful):
    """
    抽取原始评论中的aspect作为输入，完整的评论作为输出
    """
    train_data = []  # 记录训练数据
    for review in seg_review_list:  # 每个sentence就是一句完整的review

        source = []  # 训练的src
        if review[-1] not in PUNCTUATION:
            review.append('。')
        target = review  # 训练的tgt

        # 对于单个review进行切分
        cur_review = []
        pre_end = 0
        for i, _ in enumerate(review):
            if review[i] in ['。', '！', '？', '，', '～']:
                cur_review.append(review[pre_end:i + 1])
                pre_end = i + 1
            elif i == len(review) - 1:
                cur_review.append(review[pre_end:])

        for sentence in cur_review:  # sentence 是两个标点之间的短
            if sentence[-1] not in PUNCTUATION:
                sentence.append('。')
            find_opinion_flag = False
            for idx, word in enumerate(sentence):
                if find_opinion_flag:  # 如果在当前的短句中已经找到了一组观点表达就结束对这个短句的搜索
                    break
                if word in pair_useful:  # 当前的word属于aspect
                    source.append(word)
                    find_opinion_flag = True  # 只要找到一个opinion word就算命中一个短句了
        train_data.append((list(source), target))
    max_source_length = 0

    # 筛选训练数据
    def check_review(item):
        """
        判断当前review是否合法
        """
        source = item[0]
        tgt = item[1]
        legal = True
        _s = ''.join(tgt)
        if len(source) == 0 or len(source) > 5:  # 不含有观点表达或者观点词太多
            legal = False
        unique_source = set(source)
        if len(unique_source) != len(source):
            legal = False
        if len(_s) > 60:
            legal = False
        return legal

    legal_train_data = []
    for item in train_data:
        if check_review(item):
            max_source_length = max(max_source_length, len(item[0]))
            legal_train_data.append(item)

    logger.debug(f'max source length: {max_source_length}')
    return legal_train_data


def generate_reviews(aspect_express, num_steps=1000):
    """
    根据候选集合生成假评论
    """
    res = []
    all_aspect = list(aspect_express.keys())
    logger.debug(f'Aspect: {all_aspect}')

    # 根据不同aspect出现的概率分配不同权重
    aspect_length_dict = {}
    for a in aspect_express:
        aspect_length_dict[a] = len(aspect_express[a])
    weight_aspect_list = []
    for aspect in aspect_length_dict:
        weight_aspect_list += [aspect] * aspect_length_dict[aspect]
    if not weight_aspect_list:
        return res
    for _ in range(num_steps):
        num_aspect = random.choice([1, 2, 3, 4, 5, 6])
        review = []
        used_aspect = []
        for _ in range(num_aspect):
            a = random.choice(weight_aspect_list)
            if a in used_aspect and len(all_aspect) > 1:
                a = random.choice(weight_aspect_list)
            used_aspect.append(a)
            a_s = random.choice(aspect_express[a])
            a_s = a_s[:-1] + ['#']  # 丢掉标点，换位#作为切分点
            review += a_s
        res.append(review)
    return res


def fake_review_filter(reviews, opinion_set, is_uniq=True):
    """
    筛去评论中不像人写的句子：如果同一个形容词重复出现两次就判定为假评论，同时筛去长度超过60的评论
    """
    results = []
    for review in reviews:
        opinion_used = {k: 0 for k in opinion_set}
        flag = True
        for word in review:
            if word in ILLEGAL_WORD:
                flag = False
            if word in opinion_used:
                opinion_used[word] += 1
                if opinion_used[word] >= 2:
                    flag = False
                    break
        if flag:
            _s = ''.join(review)
            _s = _s.split('#')  # 最后一个是空字符
            review = ''
            pu = ['，'] * 100 + ['～'] * 20 + ['！'] * 20 + EMOJI + YANWENZI
            random.shuffle(pu)
            for a_s in _s:
                if a_s:
                    review += a_s + random.choice(pu)
            if not review:
                logger.warning(f'error: {review}')
            review = review[:-1] + '。'
            if is_uniq:
                if review not in results:
                    results.append(review)
            else:
                results.append(review)
    return results


if __name__ == '__main__':
    # 使用了(PMI+左右熵)的方法来找出新词
    default_stopwords_path = os.path.join(pwd_path, '../data/stopwords.txt')
    sample1 = load_list(os.path.join(pwd_path, '../../examples/data/ecommerce_comments_100.txt'))
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
