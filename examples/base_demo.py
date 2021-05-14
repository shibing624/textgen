# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append('..')
import textgen
from textgen.utils.log import logger


if __name__ == '__main__':
    char = '卡'
    logger.debug("hill1")
    logger.info("hill2")
    logger.warning("hill3")
    logger.error("hill4")

    from textgen.utils import log
    log.set_log_level('info')
    logger.debug("hill5")
    logger.info("hill6")
    logger.error("hill7")
