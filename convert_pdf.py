#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
麻将教材PDF转Markdown工具入口脚本

这个脚本是一个简单的入口点，用于调用DataProcess目录中的PDF转Markdown工具。
"""

import os
import sys

# 确保DataProcess目录在Python的导入路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
dataprocess_dir = os.path.join(current_dir, "DataProcess")
sys.path.insert(0, dataprocess_dir)

# 导入主转换脚本
from DataProcess.convert_mahjong_pdf import main

if __name__ == "__main__":
    # 调用主函数
    main() 