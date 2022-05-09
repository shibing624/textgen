# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys

sys.path.append('..')
import textgen


def test_parse_normal():
    """测试原生parse结果"""
    company_strs = [
        "武汉海明智业电子商务有限公司",
        "泉州益念食品有限公司",
        "常州途畅互联网科技有限公司合肥分公司",
        "昆明享亚教育信息咨询有限公司",
    ]
    res = []
    for name in company_strs:
        r = textgen.__version__
        print(r)
        res.append(r)

    assert len(res) == 4


def test_parse_enable_word_segment():
    """测试带分词的parse结果"""
    company_strs = [
        "武汉海明智业电子商务有限公司",
        "泉州益念食品有限公司",
        "常州途畅互联网科技有限公司合肥分公司",
        "昆明享亚教育信息咨询有限公司",
    ]
    print(company_strs)
    assert len(company_strs) == 4
