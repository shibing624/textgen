# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys
import unittest

sys.path.append('..')
import textgen


class BaseTestCase(unittest.TestCase):
    def test_parse_normal(self):
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

        def _assert_line(linenum, p, b, t, s, sy):
            assert res[linenum]['place'] == p
            assert res[linenum]['brand'] == b
            assert res[linenum]['trade'] == t
            assert res[linenum]['suffix'] == s
            assert res[linenum]['symbol'] == sy

        _assert_line(0, '武汉', '海明智业', '电子商务', '有限公司', '')
        _assert_line(1, '泉州', '益念', '食品', '有限公司', '')
        _assert_line(2, '常州,合肥', '途畅', '互联网科技', '有限公司,分公司', '')
        _assert_line(3, '昆明', '享亚', '教育信息咨询', '有限公司', '')

    def test_parse_enable_word_segment(self):
        """测试带分词的parse结果"""
        company_strs = [
            "武汉海明智业电子商务有限公司",
            "泉州益念食品有限公司",
            "常州途畅互联网科技有限公司合肥分公司",
            "昆明享亚教育信息咨询有限公司",
        ]
        pass


if __name__ == '__main__':
    unittest.main()
