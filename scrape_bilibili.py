#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bilibili文章爬虫入口脚本

这个脚本是一个简单的入口点，用于调用DataProcess目录中的bilibili爬虫工具。
"""

import os
import sys

# 确保DataProcess目录在Python的导入路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
dataprocess_dir = os.path.join(current_dir, "DataProcess")
sys.path.insert(0, dataprocess_dir)

# 导入爬虫主模块
from DataProcess.bili_scraper import main

if __name__ == "__main__":
    # 调用主函数
    main() 