#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bilibili文章爬虫

用于抓取bilibili文章内容并保存为Markdown格式
"""

import os
import re
import time
import json
import logging
import argparse
import requests
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import yaml
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 尝试导入麻将检测模块
try:
    from mahjong_detector import get_detector
    mahjong_detection_available = True
    logger.info("麻将牌检测模块已加载")
except ImportError:
    mahjong_detection_available = False
    logger.warning("麻将牌检测模块未找到，无法将图片转换为麻将牌文本表示")

class BilibiliScraper:
    def __init__(self, output_dir="Data", image_dir="images", no_image=False, detect_mahjong=True):
        """
        初始化爬虫
        :param output_dir: 输出目录
        :param image_dir: 图片保存目录
        :param no_image: 是否生成无图版本
        :param detect_mahjong: 是否在无图版本中尝试识别麻将牌并转换为文本
        """
        self.output_dir = output_dir
        self.image_dir = os.path.join(output_dir, image_dir)
        self.no_image = no_image
        self.detect_mahjong = detect_mahjong and mahjong_detection_available
        
        # 创建必要的目录
        os.makedirs(self.output_dir, exist_ok=True)
        if not no_image:
            os.makedirs(self.image_dir, exist_ok=True)
        
        # 设置请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.bilibili.com/',
            'Accept': 'application/json, text/plain, */*',
            'Origin': 'https://www.bilibili.com'
        }
        
        # 用于存储当前批次的链接及其内容
        self.current_batch = []
        
        # 初始化麻将检测器
        self.mahjong_detector = None
        if self.detect_mahjong:
            try:
                # 使用默认配置初始化检测器（使用mock模式）
                self.mahjong_detector = get_detector(use_mock=True)
                logger.info("麻将检测器初始化成功")
            except Exception as e:
                logger.error(f"初始化麻将检测器失败: {e}")
                self.detect_mahjong = False

    def extract_article_id(self, url):
        """
        从URL中提取文章ID
        :param url: Bilibili文章URL
        :return: 文章ID
        """
        try:
            url = url.strip()
            # 尝试从不同格式的URL中提取ID
            if '/opus/' in url:
                match = re.search(r'/opus/(\d+)', url)
                if match:
                    return match.group(1)
            
            # 处理 /read/cv 格式的URL
            if '/read/cv' in url:
                match = re.search(r'/read/cv(\d+)', url)
                if match:
                    return match.group(1)
            
            # 解析URL查询参数
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            
            # 可能的ID字段
            id_fields = ['id', 'aid', 'cv']
            for field in id_fields:
                if field in query_params:
                    return query_params[field][0]
            
            # 如果上述方法都失败，尝试从路径中提取
            path = parsed_url.path
            match = re.search(r'/(\d+)', path)
            if match:
                return match.group(1)
            
            logger.error(f"无法从URL中提取文章ID: {url}")
            return None
        except Exception as e:
            logger.error(f"提取文章ID时出错: {e}")
            return None

    def fetch_article(self, url):
        """
        获取文章内容
        :param url: Bilibili文章URL
        :return: 文章内容字典
        """
        try:
            article_id = self.extract_article_id(url)
            if not article_id:
                return None
            
            # 尝试使用API获取文章内容
            try:
                api_url = f"https://api.bilibili.com/opus/text/web/article?id={article_id}"
                response = requests.get(api_url, headers=self.headers)
                
                if response.status_code == 200:
                    # 解析JSON响应
                    data = response.json()
                    if data.get('code') == 0 and data.get('data'):
                        article_data = data['data']
                        
                        # 提取文章标题和作者信息
                        title = article_data.get('title', f"Bilibili文章_{article_id}")
                        
                        # 提取文章内容
                        content_text = self._process_content_api(article_data)
                        
                        # 提取图片
                        image_urls = []
                        if 'pictures' in article_data:
                            for pic in article_data['pictures']:
                                img_url = pic.get('url', '')
                                if img_url:
                                    image_urls.append(img_url)
                        
                        return {
                            'id': article_id,
                            'url': url,
                            'title': title,
                            'content_text': content_text,
                            'image_urls': image_urls
                        }
            except Exception as e:
                logger.error(f"API请求失败: {e}")
            
            # 如果API请求失败，使用备用方法
            logger.info("使用备用方法获取文章内容")
            return self._fallback_fetch_article(url, article_id)
            
        except Exception as e:
            logger.error(f"获取文章内容时出错: {e}")
            return self._fallback_fetch_article(url, article_id)
    
    def _extract_images_from_html(self, html_content):
        """
        从HTML内容中提取所有图片URL
        :param html_content: HTML内容
        :return: 图片URL列表
        """
        image_urls = []
        
        # 匹配所有图片标签
        img_patterns = [
            r'<img[^>]*src="(//[^"]+)"[^>]*>',  # 相对URL格式 //xxx.com/xxx.jpg
            r'<img[^>]*src="(https?://[^"]+)"[^>]*>',  # 完整URL格式 https://xxx.com/xxx.jpg
            r'<figure[^>]*><img[^>]*src="([^"]+)"[^>]*>',  # 嵌套在figure中的图片
            r'data-src="([^"]+)"',  # 懒加载图片
            r'img data-src="([^"]+)"',  # 另一种懒加载图片格式
            r'<div class="b-img[^>]*><img[^>]*src="([^"]+)"',  # Bilibili文章特有的图片格式
            r'<div class="opus-para-pic[^>]*>.*?<img[^>]*src="([^"]+)".*?</div>',  # opus文章中的图片
            r'<div[^>]*class="[^"]*article-img[^"]*"[^>]*>.*?<img[^>]*src="([^"]+)".*?</div>',  # article类文章中的图片
            r'<div[^>]*class="opus-module-image"[^>]*>.*?<img[^>]*src="([^"]+)".*?</div>',  # opus模块图片
        ]
        
        for pattern in img_patterns:
            for img_match in re.finditer(pattern, html_content, re.DOTALL):
                img_url = img_match.group(1)
                # 确保URL有协议头
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                # 过滤掉一些不需要的图片
                if not any(exclude in img_url for exclude in ['avatar', 'face', 'icon', 'logo', 'coin-ani']):
                    image_urls.append(img_url)
        
        # 去重
        return list(dict.fromkeys(image_urls))

    def _fallback_fetch_article(self, url, article_id):
        """
        备用的文章获取方法，直接从URL提取内容
        :param url: 文章URL
        :param article_id: 文章ID
        :return: 文章内容字典
        """
        try:            
            # 尝试直接从URL获取页面内容
            try:
                logger.info(f"尝试直接获取页面内容: {url}")
                response = requests.get(url, headers=self.headers)
                if response.status_code == 200:
                    # 提取页面中的文章内容
                    html_content = response.text
                    
                    # 从HTML中提取标题 - 根据不同格式尝试提取
                    title = self._extract_title_from_html(html_content, article_id)
                    
                    # 提取文章内容 - 根据不同格式尝试提取，包含图片位置标记
                    content = self._extract_content_from_html(html_content)
                    
                    # 使用更新的方法提取图片URL
                    image_urls = self._extract_images_from_html(html_content)
                    
                    logger.info(f"成功从页面提取内容: {title}, 找到 {len(image_urls)} 张图片")
                    return {
                        'id': article_id,
                        'url': url,
                        'title': title,
                        'content_text': content,
                        'image_urls': image_urls
                    }
            except Exception as e:
                logger.error(f"直接获取页面内容失败: {e}")
            
            # 尝试使用opus API
            try:
                logger.info(f"尝试使用opus API获取内容: {article_id}")
                api_url = f"https://api.bilibili.com/x/opus/web/detail?opus_id={article_id}"
                response = requests.get(api_url, headers=self.headers)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('code') == 0 and data.get('data'):
                        opus_data = data['data']
                        title = opus_data.get('title', f"Bilibili文章_{article_id}")
                        content = opus_data.get('text', "[未找到文章内容]")
                        
                        # 提取图片URL
                        image_urls = []
                        if 'pictures' in opus_data:
                            for pic in opus_data['pictures']:
                                img_url = pic.get('url', '')
                                if img_url:
                                    image_urls.append(img_url)
                        
                        logger.info(f"成功从opus API提取内容: {title}")
                        return {
                            'id': article_id,
                            'url': url,
                            'title': title,
                            'content_text': content,
                            'image_urls': image_urls
                        }
            except Exception as e:
                logger.error(f"使用opus API获取内容失败: {e}")
            
            # 如果以上方法都失败，尝试使用文章API
            try:
                logger.info(f"尝试使用文章API获取内容: {article_id}")
                api_url = f"https://api.bilibili.com/x/article/viewinfo?id={article_id}"
                response = requests.get(api_url, headers=self.headers)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('code') == 0 and data.get('data'):
                        article_data = data['data']
                        title = article_data.get('title', f"Bilibili文章_{article_id}")
                        
                        # 获取文章内容
                        content_url = f"https://www.bilibili.com/read/cv{article_id}"
                        content_response = requests.get(content_url, headers=self.headers)
                        
                        if content_response.status_code == 200:
                            html_content = content_response.text
                            content = self._extract_content_from_html(html_content)
                            
                            # 提取图片URL
                            image_urls = self._extract_images_from_html(html_content)
                            
                            logger.info(f"成功从文章API提取内容: {title}")
                            return {
                                'id': article_id,
                                'url': url,
                                'title': title,
                                'content_text': content,
                                'image_urls': image_urls
                            }
            except Exception as e:
                logger.error(f"使用文章API获取内容失败: {e}")
            
            # 如果以上方法都失败
            logger.error(f"无法获取文章内容，URL: {url}")
            return None
        except Exception as e:
            logger.error(f"备用文章获取方法失败: {e}")
            return None
            
    def _extract_title_from_html(self, html_content, article_id):
        """
        从HTML中提取文章标题
        :param html_content: HTML内容
        :param article_id: 文章ID（用于生成默认标题）
        :return: 标题字符串
        """
        # 尝试多种可能的标题提取方式
        title_patterns = [
            # 常规title标签
            r'<title>(.*?)</title>',
            # opus文章模块标题
            r'<div class="opus-module-title"><span class="opus-module-title__text">(.*?)</span></div>',
            # read文章标题
            r'<h1[^>]*class="title"[^>]*>(.*?)</h1>',
            # 另一种opus文章标题格式
            r'<span[^>]*class="opus-module-title__text"[^>]*>(.*?)</span>',
            # span格式的标题
            r'<span[^>]*class="[^"]*title[^"]*"[^>]*>(.*?)</span>',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, html_content, re.DOTALL)
            if match:
                title = match.group(1).strip()
                # 清理HTML标签
                title = re.sub(r'<[^>]+>', '', title)
                if title and title != "哔哩哔哩":
                    return title
        
        # 如果没有找到标题，返回默认值
        return f"Bilibili文章_{article_id}"
        
    def _extract_author_from_html(self, html_content):
        """
        从HTML中提取作者信息
        :param html_content: HTML内容
        :return: 作者字符串
        """
        # 尝试多种可能的作者提取方式
        author_patterns = [
            # span作者名
            r'<span[^>]*class="[^"]*author[^"]*"[^>]*>(.*?)</span>',
            # opus作者名
            r'<div class="opus-module-author__name"[^>]*>(.*?)</div>',
            # read作者名
            r'<a[^>]*class="[^"]*author[^"]*"[^>]*>(.*?)</a>',
            # 其他可能的格式
            r'<div[^>]*class="[^"]*author-name[^"]*"[^>]*>(.*?)</div>',
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, html_content, re.DOTALL)
            if match:
                author = match.group(1).strip()
                # 清理HTML标签
                author = re.sub(r'<[^>]+>', '', author)
                if author:
                    return author
        
        return "未知作者"
        
    def _extract_content_from_html(self, html_content):
        """
        从HTML中提取文章内容
        :param html_content: HTML内容
        :return: 文章内容字符串和图片位置信息
        """
        # 尝试多种可能的内容提取方式
        content_patterns = [
            # opus模块内容
            r'<div class="opus-module-content">(.*?)</div><div class="opus-module-bottom">',
            # 旧版文章内容
            r'<div[^>]*class="[^"]*article-content[^"]*"[^>]*>(.*?)</div>',
            # 另一种opus文章内容
            r'<div[^>]*class="opus-module-content"[^>]*>(.*?)<div class="opus-module-bottom"',
            # read类型文章正文
            r'<div[^>]*id="read-article-holder"[^>]*>(.*?)</div>',
            # 其他可能的格式
            r'<div[^>]*id="article-content"[^>]*>(.*?)</div>',
            # 动态文章正文
            r'<div[^>]*class="opus-content"[^>]*>(.*?)</div>',
        ]
        
        content = ""
        
        for pattern in content_patterns:
            match = re.search(pattern, html_content, re.DOTALL)
            if match:
                content = match.group(1).strip()
                # 提取图片位置并标记
                content = self._mark_image_positions(content)
                # 转换HTML到纯文本，处理段落和格式，但保留图片标记
                content = self._html_to_markdown(content)
                if content:
                    return content
        
        # 尝试提取所有段落文本
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html_content, re.DOTALL)
        if paragraphs:
            # 提取并标记图片位置
            joined_paragraphs = "".join([f"<p>{p}</p>" for p in paragraphs])
            content = self._mark_image_positions(joined_paragraphs)
            content = self._html_to_markdown(content)
            if content:
                return content
                
        return "[未找到文章内容]"
    
    def _mark_image_positions(self, html_content):
        """
        在HTML内容中标记图片位置，以便在转换为Markdown时保留位置
        :param html_content: HTML内容
        :return: 标记了图片位置的HTML内容
        """
        # 复制原始内容
        marked_content = html_content
        
        # 替换图片标签为特殊标记
        img_patterns = [
            r'<img[^>]*src="([^"]+)"[^>]*>',
            r'<figure[^>]*><img[^>]*src="([^"]+)"[^>]*></figure>',
            r'<div class="opus-para-pic[^>]*>.*?<img[^>]*src="([^"]+)".*?</div>',
            r'<div class="b-img[^>]*>.*?<img[^>]*src="([^"]+)".*?</div>',
            r'<div[^>]*class="[^"]*article-img[^"]*"[^>]*>.*?<img[^>]*src="([^"]+)".*?</div>',
            r'<div[^>]*class="opus-module-image"[^>]*>.*?<img[^>]*src="([^"]+)".*?</div>'
        ]
        
        # 查找所有图片位置并保存
        all_matches = []
        for pattern in img_patterns:
            for match in re.finditer(pattern, html_content, re.DOTALL):
                all_matches.append((match.start(), match.end(), match.group(1)))
        
        # 按照出现位置排序
        all_matches.sort(key=lambda x: x[0])
        
        # 从后向前替换，避免位置偏移
        for idx, (start, end, src) in enumerate(reversed(all_matches)):
            img_idx = len(all_matches) - idx - 1  # 反转索引
            placeholder = f'<img-placeholder data-index="{img_idx}" data-src="{src}"></img-placeholder>'
            marked_content = marked_content[:start] + placeholder + marked_content[end:]
        
        return marked_content

    def _html_to_markdown(self, html_content):
        """
        将HTML内容转换为Markdown格式
        :param html_content: HTML内容
        :return: Markdown格式文本
        """
        # 保存图片占位符位置和内容
        img_placeholders = []
        for match in re.finditer(r'<img-placeholder data-index="(\d+)" data-src="([^"]+)"></img-placeholder>', html_content):
            img_placeholders.append({
                'placeholder': match.group(0),
                'index': match.group(1),
                'src': match.group(2),
                'marker': f'{{IMAGE_PLACEHOLDER_{match.group(1)}}}'
            })
        
        # 先替换图片占位符为唯一标记，避免被其他处理影响
        content = html_content
        for img in img_placeholders:
            content = content.replace(img['placeholder'], img['marker'])
        
        # 替换段落
        content = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', content, flags=re.DOTALL)
        
        # 替换加粗
        content = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', content, flags=re.DOTALL)
        content = re.sub(r'<b[^>]*>(.*?)</b>', r'**\1**', content, flags=re.DOTALL)
        
        # 替换斜体
        content = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', content, flags=re.DOTALL)
        content = re.sub(r'<i[^>]*>(.*?)</i>', r'*\1*', content, flags=re.DOTALL)
        
        # 替换列表
        content = re.sub(r'<ul[^>]*>(.*?)</ul>', r'\1', content, flags=re.DOTALL)
        content = re.sub(r'<ol[^>]*>(.*?)</ol>', r'\1', content, flags=re.DOTALL)
        content = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', content, flags=re.DOTALL)
        
        # 替换标题
        content = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1\n\n', content, flags=re.DOTALL)
        content = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1\n\n', content, flags=re.DOTALL)
        content = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1\n\n', content, flags=re.DOTALL)
        
        # 移除所有HTML标签（图片占位符已替换为标记）
        content = re.sub(r'<[^>]*>', '', content, flags=re.DOTALL)
        
        # 修复可能的HTML实体
        content = content.replace('&gt;', '>')
        content = content.replace('&lt;', '<')
        content = content.replace('&amp;', '&')
        content = content.replace('&quot;', '"')
        content = content.replace('&nbsp;', ' ')
        
        # 修复多余换行和空格
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        return content.strip()

    def _process_content_api(self, article_data):
        """
        处理API返回的文章内容
        :param article_data: API返回的文章数据
        :return: Markdown格式的文章内容
        """
        markdown_content = []
        
        # 提取文章正文
        content = article_data.get('content', '')
        if content:
            # 处理图片URL，添加https:前缀
            content = re.sub(r'src="//([^"]+)"', r'src="https://\1"', content)
            markdown_content.append(content)
        
        # 处理可能的附加内容
        if 'opus_text' in article_data:
            opus_text = article_data['opus_text']
            # 也处理opus_text中的图片URL
            opus_text = re.sub(r'src="//([^"]+)"', r'src="https://\1"', opus_text)
            markdown_content.append(opus_text)
        
        # 确保内容不为空
        if not markdown_content:
            markdown_content = ["[未找到文章内容]"]
        
        return '\n\n'.join(markdown_content)
    
    def is_likely_mahjong_image(self, img_data):
        """
        判断图片是否可能是麻将牌图片
        :param img_data: 图片二进制数据
        :return: 布尔值，表示是否可能是麻将牌图片
        """
        try:
            # 转换为OpenCV格式的图像
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 检查图像尺寸（麻将牌图片通常有一定的宽高比）
            height, width = img.shape[:2]
            aspect_ratio = width / height
            
            # 检查颜色分布（麻将牌通常有白色或浅色背景）
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            # 基于简单规则的判断
            if 0.5 < aspect_ratio < 8 and avg_brightness > 150:
                logger.info(f"图片可能是麻将牌(宽高比:{aspect_ratio:.2f}, 亮度:{avg_brightness:.2f})")
                return True
            
            return False
        except Exception as e:
            logger.error(f"分析图片时出错: {e}")
            return False

    def download_images(self, image_urls, save_dir):
        """
        下载文章中的图片
        :param image_urls: 图片URL列表
        :param save_dir: 保存目录
        :return: 保存后的本地图片路径列表
        """
        if not image_urls:
            return []

        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        saved_paths = []
        for idx, img_url in enumerate(image_urls):
            try:
                # 确保URL有协议头
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                
                # 生成唯一的文件名
                img_ext = self._get_image_extension(img_url)
                img_filename = f"image_{idx + 1}{img_ext}"
                img_path = os.path.join(save_dir, img_filename)
                
                # 尝试下载图片
                logger.info(f"正在下载图片: {img_url} -> {img_path}")
                response = requests.get(img_url, headers=self.headers, stream=True, timeout=10)
                response.raise_for_status()
                
                # 保存图片
                with open(img_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # 处理webp格式：如果下载的是webp，尝试转换为png
                if img_ext.lower() == '.webp':
                    try:
                        from PIL import Image
                        webp_img = Image.open(img_path)
                        png_path = img_path.replace('.webp', '.png')
                        webp_img.save(png_path, 'PNG')
                        # 替换为转换后的路径
                        os.remove(img_path)  # 删除原webp文件
                        img_path = png_path
                        logger.info(f"已将webp转换为png: {img_path}")
                    except Exception as e:
                        logger.warning(f"无法转换webp格式: {e}")
                
                # 创建一个包含图片信息的字典，而不是仅仅保存路径
                img_info = {
                    'path': img_path,
                    'url': img_url,
                    'mahjong_text': None  # 初始化为None，后续可能会填充
                }
                
                saved_paths.append(img_info)
                logger.info(f"图片下载成功: {img_path}")
            except Exception as e:
                logger.error(f"下载图片失败 {img_url}: {e}")
                # 尝试下载备用URL
                if 'webp' in img_url.lower():
                    # 尝试获取非webp版本
                    alt_img_url = img_url.replace('.webp', '.png')
                    try:
                        logger.info(f"尝试下载备用格式: {alt_img_url}")
                        response = requests.get(alt_img_url, headers=self.headers, stream=True, timeout=10)
                        response.raise_for_status()
                        
                        # 生成新文件名
                        img_filename = f"image_{idx + 1}.png"
                        img_path = os.path.join(save_dir, img_filename)
                        
                        # 保存图片
                        with open(img_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        img_info = {
                            'path': img_path,
                            'url': alt_img_url,
                            'mahjong_text': None
                        }
                        
                        saved_paths.append(img_info)
                        logger.info(f"备用格式图片下载成功: {img_path}")
                    except Exception as e2:
                        logger.error(f"下载备用格式图片失败 {alt_img_url}: {e2}")
        
        return saved_paths
    
    def _get_image_extension(self, img_url):
        """
        从图片URL获取扩展名
        :param img_url: 图片URL
        :return: 扩展名（包括点号）
        """
        # 解析URL，获取路径
        parsed_url = urlparse(img_url)
        path = parsed_url.path
        
        # 处理路径中可能包含查询参数的情况
        if '?' in path:
            path = path.split('?')[0]
        
        # 获取扩展名
        _, ext = os.path.splitext(path)
        
        # 如果没有扩展名或扩展名不常见，使用默认扩展名
        if not ext or ext.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
            # 检查URL中是否包含格式提示
            if 'format=webp' in img_url.lower():
                ext = '.webp'
            elif 'format=jpg' in img_url.lower() or 'format=jpeg' in img_url.lower():
                ext = '.jpg'
            elif 'format=png' in img_url.lower():
                ext = '.png'
            elif 'format=gif' in img_url.lower():
                ext = '.gif'
            else:
                # 默认扩展名
                ext = '.jpg'
        
        return ext

    def save_as_markdown(self, articles, batch_name=None):
        """
        将多篇文章保存为一个Markdown文件
        :param articles: 文章列表
        :param batch_name: 批次名称，用于命名文件
        :return: 保存的文件路径
        """
        if not articles:
            logger.warning("没有文章可保存")
            return None
        
        try:
            # 生成文件名
            if not batch_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_name = f"bilibili_articles_{timestamp}"
            
            # 构建完整的输出路径
            output_filename = f"{batch_name}.md"
            if self.no_image:
                output_filename = f"{batch_name}_no_images.md"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # 写入Markdown文件
            with open(output_path, 'w', encoding='utf-8') as f:
                # 添加YAML前置元数据
                yaml_header = {
                    'title': batch_name,
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'source': 'Bilibili',
                    'articles_count': len(articles)
                }
                yaml_str = yaml.dump(yaml_header, allow_unicode=True)
                f.write(f"---\n{yaml_str}---\n\n")
                
                # 写入每篇文章
                for i, article in enumerate(articles):
                    # 检查文章数据是否有效
                    if not isinstance(article, dict):
                        logger.warning(f"跳过无效的文章数据: {type(article)}")
                        logger.warning(f"文章内容: {article}")
                        continue
                        
                    # 添加文章标题
                    article_title = article.get('title', '无标题')
                    f.write(f"# {article_title}\n\n")
                    
                    # 添加文章作者
                    author = article.get('author', '未知作者')
                    f.write(f"作者: {author}\n\n")
                    
                    # 处理内容中的图片占位符
                    content_text = article.get('content_text', '[无内容]')
                    
                    # 替换图片或麻将牌文本
                    if 'downloaded_images' in article and article['downloaded_images']:
                        content_with_images = content_text
                        
                        # 循环处理每个图片占位符
                        for j, img_info in enumerate(article['downloaded_images']):
                            # 确保img_info是字典
                            if not isinstance(img_info, dict):
                                logger.warning(f"跳过无效的图片信息: {type(img_info)}")
                                continue
                                
                            # 查找占位符                            
                            placeholder = f"{{IMAGE_PLACEHOLDER_{j}}}"
                            if placeholder in content_with_images:
                                # 替换为对应的图片或麻将牌文本
                                if self.no_image:
                                    # 如果是无图版本且有麻将牌识别结果
                                    if 'mahjong_text' in img_info and img_info['mahjong_text']:
                                        replacement = f"\n```mahjong\n{img_info['mahjong_text']}\n```\n"
                                    else:
                                        # 如果没有麻将牌识别结果，则不显示任何内容
                                        replacement = "\n[此处为图片]\n"
                                else:
                                    # 如果是有图版本，添加图片链接
                                    if 'path' in img_info:
                                        rel_path = os.path.relpath(img_info['path'], self.output_dir)
                                        replacement = f"\n![image]({rel_path})\n"
                                    else:
                                        replacement = "\n[图片加载失败]\n"
                                        
                                content_with_images = content_with_images.replace(placeholder, replacement)
                        
                        # 使用处理后的内容
                        f.write(content_with_images)
                    else:
                        # 如果没有图片，直接写入内容
                        f.write(content_text)
                    
                    # 在不同文章之间添加分隔
                    if i < len(articles) - 1:
                        f.write("\n\n")
                        f.write("---\n\n")
                        f.write("<div style='page-break-after: always;'></div>\n\n")
            
            logger.info(f"Markdown文件保存成功: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"保存Markdown文件时出错: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return None

    def add_to_batch(self, url):
        """
        将URL添加到当前批次
        :param url: Bilibili文章URL
        :return: 是否成功添加
        """
        article = self.fetch_article(url)
        if not article:
            return False
        
        # 下载图片，无论是否为no_image模式都下载以便识别麻将牌
        downloaded_images = self.download_images(article['image_urls'], self.image_dir)
        
        # 识别麻将牌图片
        if self.detect_mahjong and self.mahjong_detector:
            for img_info in downloaded_images:
                try:
                    if isinstance(img_info, dict) and 'path' in img_info and os.path.exists(img_info['path']):
                        if self.is_likely_mahjong_image_file(img_info['path']):
                            # 使用麻将检测器识别图片中的麻将牌
                            mahjong_text = self.mahjong_detector.detect_from_file(img_info['path'])
                            if mahjong_text and not mahjong_text.startswith("检测出错") and not mahjong_text.startswith("模型未加载"):
                                logger.info(f"识别到麻将牌: {mahjong_text}")
                                img_info['mahjong_text'] = mahjong_text
                except Exception as e:
                    logger.error(f"麻将牌识别失败: {e}")
        
        article['downloaded_images'] = downloaded_images
        
        # 添加到当前批次
        self.current_batch.append(article)
        return True

    def save_current_batch(self, batch_name=None):
        """
        保存当前批次的文章
        :param batch_name: 批次名称
        :return: 保存的文件路径
        """
        if not self.current_batch:
            logger.warning("当前没有待保存的文章")
            return None
        
        # 先保存无图版本，确保麻将牌文本正确替换
        original_no_image = self.no_image
        self.no_image = True
        no_image_path = self.save_as_markdown(self.current_batch, batch_name)
        
        # 然后保存带图版本
        if not original_no_image:
            self.no_image = False
            output_path = self.save_as_markdown(self.current_batch, batch_name)
        else:
            output_path = no_image_path
        
        # 恢复原来的设置
        self.no_image = original_no_image
        
        # 清空当前批次
        self.current_batch = []
        
        return output_path

    def process_urls_from_file(self, file_path, batch_name=None):
        """
        从文件中读取URL列表并处理
        :param file_path: URL列表文件路径
        :param batch_name: 批次名称
        :return: 保存的文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            return self.process_urls(urls, batch_name)
        except Exception as e:
            logger.error(f"从文件读取URL列表时出错: {e}")
            return None

    def process_urls(self, urls, batch_name=None):
        """
        处理URL列表
        :param urls: URL列表
        :param batch_name: 批次名称
        :return: 保存的文件路径
        """
        # 清空当前批次
        self.current_batch = []
        
        # 处理每个URL
        success_count = 0
        for i, url in enumerate(urls):
            logger.info(f"处理第 {i+1}/{len(urls)} 个URL: {url}")
            
            if self.add_to_batch(url):
                success_count += 1
            
            # 添加延迟，避免请求过于频繁
            if i < len(urls) - 1:
                time.sleep(1)
        
        logger.info(f"成功处理 {success_count}/{len(urls)} 个URL")
        
        # 保存当前批次
        return self.save_current_batch(batch_name)

    def save_to_db(self, articles, batch_name=None):
        """
        将抓取结果保存到数据库
        :param articles: 文章列表
        :param batch_name: 批次名称
        :return: 成功插入条数
        """
        session = self.db_session
        count = 0
        
        for article in articles:
            try:
                # 检查是否已存在
                existing = session.query(Article).filter_by(ori_url=article['url']).first()
                if existing:
                    logger.info(f"文章已存在，跳过: {article['title']}")
                    continue
                
                # 下载图片
                os.makedirs(self.image_dir, exist_ok=True)
                image_save_dir = os.path.join(self.image_dir, f"article_{article['id']}")
                downloaded_images = self.download_images(article['image_urls'], image_save_dir)
                
                # 将下载的图片路径保存到文章中
                article['local_images'] = downloaded_images
                
                # 检测图片中是否包含麻将牌
                mahjong_texts = []
                if self.detect_mahjong and not self.no_image:
                    for img_path in downloaded_images:
                        try:
                            if os.path.exists(img_path) and self.is_likely_mahjong_image_file(img_path):
                                # 使用麻将检测器识别图片中的麻将牌
                                mahjong_text = self.mahjong_detector.detect_from_file(img_path)
                                if mahjong_text and not mahjong_text.startswith("检测出错") and not mahjong_text.startswith("模型未加载"):
                                    logger.info(f"识别到麻将牌: {mahjong_text}")
                                    mahjong_texts.append(mahjong_text)
                        except Exception as e:
                            logger.error(f"麻将牌识别失败: {e}")
                
                # 创建Article对象
                new_article = Article(
                    title=article['title'],
                    content=article['content'],
                    author=article['author'],
                    publish_time=article['publish_time'],
                    ori_url=article['url'],
                    image_urls=json.dumps(article['image_urls']),
                    local_images=json.dumps(downloaded_images),
                    mahjong_texts=json.dumps(mahjong_texts),
                    source=article.get('source', 'bilibili'),
                    batch=batch_name or datetime.now().strftime('%Y%m%d%H%M%S'),
                    is_processed=False
                )
                
                # 添加到数据库
                session.add(new_article)
                session.commit()
                count += 1
                logger.info(f"成功保存文章到数据库: {article['title']}")
            except Exception as e:
                logger.error(f"保存文章到数据库失败: {e}")
                session.rollback()
        
        return count

    def is_likely_mahjong_image_file(self, img_path):
        """
        检查图片文件是否可能包含麻将牌
        :param img_path: 图片文件路径
        :return: 布尔值
        """
        try:
            from PIL import Image
            import numpy as np
            
            # 打开图片
            img = Image.open(img_path)
            
            # 检查图片尺寸，排除太小的图片
            width, height = img.size
            if width < 100 or height < 100:
                return False
            
            # 继续使用颜色分析方法
            img = img.resize((100, 100))  # 缩小图片以加快处理
            img_array = np.array(img)
            
            # 判断是否有足够的白色或浅色像素（麻将牌通常是白底）
            white_pixels = np.sum((img_array > 200).all(axis=2))
            white_ratio = white_pixels / (100 * 100)
            
            # 如果白色像素比例适中（不太少也不太多），可能是麻将牌
            return 0.3 <= white_ratio <= 0.9
        except Exception as e:
            logger.warning(f"分析图片是否包含麻将牌时出错: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Bilibili文章爬虫")
    parser.add_argument("sources", nargs='+', help="文章URL或包含URL的文件路径")
    parser.add_argument("-o", "--output-dir", default="Data", help="输出目录")
    parser.add_argument("-i", "--image-dir", default="images", help="图片保存目录")
    parser.add_argument("-n", "--name", help="输出文件名")
    parser.add_argument("--no-image", action="store_true", help="不下载图片")
    parser.add_argument("-f", "--file", action="store_true", help="指定的source是含有URL的文件")
    parser.add_argument("--no-detect", action="store_true", help="不进行麻将牌检测")
    
    args = parser.parse_args()
    
    # 初始化爬虫
    scraper = BilibiliScraper(args.output_dir, args.image_dir, args.no_image, not args.no_detect)
    
    # 处理来源
    if args.file:
        # 从文件读取URL
        for file_path in args.sources:
            scraper.process_urls_from_file(file_path, args.name)
    else:
        # 直接处理URL列表
        scraper.process_urls(args.sources, args.name)

if __name__ == "__main__":
    main() 