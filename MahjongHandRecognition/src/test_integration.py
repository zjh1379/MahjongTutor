#!/usr/bin/env python3
"""
Integration test for Mahjong Hand Recognition
用于测试麻将牌识别功能与现有项目集成的脚本
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mahjong_recognition.integration import integrate_with_pdf_converter, integrate_with_bilibili_scraper


def test_pdf_converter_integration():
    """
    测试与PDF转Markdown的集成
    """
    print("Testing integration with PDF converter...")
    
    # 创建一个临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建一个测试Markdown文件
        md_content = """# 麻将教程

## 基本牌型

下面是一些常见的麻将牌型：

![一萬](images/1m.jpg)
![二萬](images/2m.jpg)
![三萬](images/3m.jpg)

## 进阶牌型

复杂的牌型组合：

![牌型1](images/hand1.jpg)
![牌型2](images/hand2.jpg)
"""
        
        # 处理Markdown内容
        processed_md = integrate_with_pdf_converter(md_content, "path/to/images")
        
        print("\nOriginal Markdown:")
        print(md_content)
        print("\nProcessed Markdown:")
        print(processed_md)
        
        print("\nIntegration with PDF converter test completed.")


def test_bilibili_scraper_integration():
    """
    测试与Bilibili爬虫的集成
    """
    print("Testing integration with Bilibili scraper...")
    
    # 创建一个测试内容
    bilibili_content = """# 麻将教程视频内容

## 基本规则

在这个视频中，我们学习了基本的麻将规则。

视频中出现的牌型：
https://example.com/mahjong1.jpg
https://example.com/mahjong2.jpg

## 高级技巧

进阶的打法技巧：
https://example.com/technique.jpg
"""
    
    # 处理Bilibili内容
    processed_content = integrate_with_bilibili_scraper(bilibili_content)
    
    print("\nOriginal Bilibili content:")
    print(bilibili_content)
    print("\nProcessed Bilibili content:")
    print(processed_content)
    
    print("\nIntegration with Bilibili scraper test completed.")


def main():
    """主函数"""
    print("Running integration tests for Mahjong Hand Recognition...")
    
    # 测试与PDF转Markdown的集成
    test_pdf_converter_integration()
    
    print("\n" + "-" * 50 + "\n")
    
    # 测试与Bilibili爬虫的集成
    test_bilibili_scraper_integration()
    
    print("\nAll integration tests completed.")


if __name__ == "__main__":
    main() 