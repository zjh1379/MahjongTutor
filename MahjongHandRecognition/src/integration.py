"""
Mahjong Hand Recognition Integration Module
用于将麻将牌识别功能与现有项目集成的模块
"""

import os
import re
import base64
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from urllib.parse import urlparse

from mahjong_recognition.tile_recognition import TileRecognizer
from mahjong_recognition.utils import is_mahjong_image


class MahjongRecognitionIntegrator:
    """
    集成器类，用于将麻将牌识别功能与现有项目集成
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化集成器
        
        Args:
            model_path: 模型路径，如果为None则使用默认模型
        """
        self.recognizer = TileRecognizer(model_path)
    
    def process_markdown_content(self, content: str, image_folder: str) -> str:
        """
        处理Markdown内容，识别并替换麻将牌图片为文本表示
        
        Args:
            content: Markdown内容
            image_folder: 图片所在的文件夹路径
            
        Returns:
            处理后的Markdown内容
        """
        # 查找Markdown中的图片标签: ![alt](path)
        image_pattern = r'!\[(.*?)\]\((.*?)\)'
        
        def replace_image(match):
            alt_text, image_path = match.groups()
            
            # 处理相对路径
            if not os.path.isabs(image_path):
                image_path = os.path.join(image_folder, image_path)
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                return match.group(0)  # 保持原样
            
            try:
                # 读取图片
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 检查是否为麻将牌图片
                if not is_mahjong_image(img):
                    return match.group(0)  # 保持原样
                
                # 识别麻将牌
                formatted_result, _ = self.recognizer.recognize_hand(image_path)
                
                # 如果识别失败或者不是麻将图片，保持原样
                if formatted_result in ["Not a mahjong image", "No tiles found"]:
                    return match.group(0)
                
                # 替换为文本表示
                return f'`{formatted_result}`'
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                return match.group(0)  # 保持原样
        
        # 替换所有匹配的图片
        processed_content = re.sub(image_pattern, replace_image, content)
        return processed_content
    
    def process_base64_image(self, base64_str: str) -> str:
        """
        处理Base64编码的图片
        
        Args:
            base64_str: Base64编码的图片字符串
            
        Returns:
            识别结果的文本表示
        """
        try:
            # 解码Base64字符串
            if ',' in base64_str:
                base64_str = base64_str.split(',', 1)[1]
            
            image_data = base64.b64decode(base64_str)
            
            # 创建临时文件保存图片
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(image_data)
                temp_path = temp_file.name
            
            # 识别麻将牌
            formatted_result, _ = self.recognizer.recognize_hand(temp_path)
            
            # 删除临时文件
            os.unlink(temp_path)
            
            return formatted_result
        except Exception as e:
            print(f"Error processing base64 image: {str(e)}")
            return "Error processing image"
    
    def process_image_url(self, url: str) -> str:
        """
        处理图片URL
        
        Args:
            url: 图片URL
            
        Returns:
            识别结果的文本表示
        """
        try:
            import requests
            
            # 下载图片
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # 创建临时文件保存图片
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_path = temp_file.name
            
            # 识别麻将牌
            formatted_result, _ = self.recognizer.recognize_hand(temp_path)
            
            # 删除临时文件
            os.unlink(temp_path)
            
            return formatted_result
        except Exception as e:
            print(f"Error processing image URL {url}: {str(e)}")
            return "Error processing image"
    
    def process_bilibili_content(self, content: str) -> str:
        """
        处理从Bilibili爬取的内容，识别并替换麻将牌图片为文本表示
        
        Args:
            content: 从Bilibili爬取的内容
            
        Returns:
            处理后的内容
        """
        # 查找图片URL
        image_pattern = r'(https?://[^\s<>"]+?\.(?:jpg|jpeg|png|gif))'
        
        def replace_image_url(match):
            url = match.group(0)
            
            try:
                # 识别麻将牌
                formatted_result = self.process_image_url(url)
                
                # 如果识别失败或者不是麻将图片，保持原样
                if formatted_result in ["Not a mahjong image", "No tiles found", "Error processing image"]:
                    return url
                
                # 替换为URL和文本表示
                return f'{url} `{formatted_result}`'
            except Exception as e:
                print(f"Error processing image URL {url}: {str(e)}")
                return url  # 保持原样
        
        # 替换所有匹配的图片URL
        processed_content = re.sub(image_pattern, replace_image_url, content)
        return processed_content


def integrate_with_pdf_converter(md_content: str, image_folder: str, model_path: Optional[str] = None) -> str:
    """
    与PDF转Markdown功能集成
    
    Args:
        md_content: Markdown内容
        image_folder: 图片所在的文件夹路径
        model_path: 模型路径，如果为None则使用默认模型
        
    Returns:
        处理后的Markdown内容
    """
    integrator = MahjongRecognitionIntegrator(model_path)
    return integrator.process_markdown_content(md_content, image_folder)


def integrate_with_bilibili_scraper(content: str, model_path: Optional[str] = None) -> str:
    """
    与Bilibili爬虫功能集成
    
    Args:
        content: 从Bilibili爬取的内容
        model_path: 模型路径，如果为None则使用默认模型
        
    Returns:
        处理后的内容
    """
    integrator = MahjongRecognitionIntegrator(model_path)
    return integrator.process_bilibili_content(content) 