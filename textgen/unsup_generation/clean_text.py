# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import re

FH_PUNCTUATION = [
    (u"。", u"."), (u"，", u","), (u"！", u"!"), (u"？", u"?"), (u"～", u"~"),
]

keep_p = ['，', '。', '！', '？', '～', '、']
f2h = {}
for item in FH_PUNCTUATION:
    c1 = item[0]
    c2 = item[1]
    f2h[c2] = c1


def convert(content):
    nc = []
    for c in content:
        if c in f2h:
            nc.append(f2h[c])
            continue
        nc.append(c)
    return "".join(nc)


def clean(line):
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

