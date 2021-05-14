# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        print("hi")
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
