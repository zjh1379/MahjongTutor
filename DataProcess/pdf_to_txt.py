#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF转TXT工具

这个脚本用于将PDF文件转换为TXT文件，支持单个文件转换和批量转换。
所有转换的文件将保存在同一个输出目录中。
"""

import os
import sys
from PyPDF2 import PdfReader
import time

def convert_pdf_to_txt(pdf_path, output_path=None):
    """
    将PDF文件转换为TXT文件
    
    Args:
        pdf_path (str): PDF文件的路径
        output_path (str, optional): 输出TXT文件的路径。如果不指定，将在默认输出目录下创建TXT文件
    
    Returns:
        str: 输出TXT文件的路径
    """
    try:
        # 创建PDF阅读器对象
        reader = PdfReader(pdf_path)
        
        # 如果没有指定输出路径，则在默认输出目录下创建TXT文件
        if output_path is None:
            # 获取PDF文件名（不含扩展名）
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            # 设置输出文件路径
            output_path = os.path.join('Data', 'output', f'{pdf_name}.txt')
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 打开输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            # 遍历每一页
            for page in reader.pages:
                # 提取文本
                text = page.extract_text()
                # 写入文件
                f.write(text + '\n\n')
        
        print(f"转换完成！输出文件：{output_path}")
        return output_path
    
    except Exception as e:
        print(f"转换过程中出现错误：{str(e)}")
        return None

def batch_convert_pdfs(input_dir, output_dir=None):
    """
    批量转换文件夹中的所有PDF文件
    
    Args:
        input_dir (str): 输入文件夹路径，包含要转换的PDF文件
        output_dir (str, optional): 输出文件夹路径。如果不指定，将在Data/output目录下创建时间戳命名的目录
    """
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误：找不到输入目录 {input_dir}")
        return
    
    # 如果没有指定输出目录，创建一个带时间戳的目录
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('Data', 'output', timestamp)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录：{output_dir}")
    
    # 遍历输入目录中的所有文件
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                # 获取PDF文件名（不含扩展名）
                pdf_name = os.path.splitext(file)[0]
                # 设置输出文件路径
                output_path = os.path.join(output_dir, f'{pdf_name}.txt')
                
                print(f"正在转换：{pdf_path}")
                convert_pdf_to_txt(pdf_path, output_path)

def main():
    """
    主函数
    """
    if len(sys.argv) < 2:
        print("使用方法：")
        print("单个文件转换：python pdf_to_txt.py <PDF文件路径> [输出TXT文件路径]")
        print("批量转换：python pdf_to_txt.py -d <输入文件夹路径> [输出文件夹路径]")
        return
    
    # 检查是否是批量转换模式
    if sys.argv[1] == '-d':
        if len(sys.argv) < 3:
            print("错误：批量转换模式需要指定输入文件夹路径")
            return
        
        input_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        batch_convert_pdfs(input_dir, output_dir)
    else:
        # 单个文件转换模式
        pdf_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        if not os.path.exists(pdf_path):
            print(f"错误：找不到PDF文件 {pdf_path}")
            return
        
        convert_pdf_to_txt(pdf_path, output_path)

if __name__ == "__main__":
    main() 