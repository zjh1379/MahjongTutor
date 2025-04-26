import os
import sys
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import io
import yaml
import re
import argparse
import logging
from mahjong_detector import get_detector

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MahjongPDFConverter:
    def __init__(self, pdf_path, output_md_path=None, mahjong_detector=None):
        """
        初始化PDF转换器
        :param pdf_path: PDF文件路径
        :param output_md_path: 输出的Markdown文件路径
        :param mahjong_detector: 麻将牌识别模型
        """
        self.pdf_path = pdf_path
        
        # 如果未指定输出路径，则使用与PDF同名的.md文件
        if output_md_path is None:
            base_name = os.path.basename(pdf_path)
            name_without_ext = os.path.splitext(base_name)[0]
            output_md_path = f"{name_without_ext}.md"
        
        self.output_md_path = output_md_path
        self.mahjong_detector = mahjong_detector
        self.doc = None
        self.markdown_content = []
        # 不再创建图像目录，因为不需要保存图片
    
    def open_pdf(self):
        """打开PDF文件"""
        try:
            self.doc = fitz.open(self.pdf_path)
            logger.info(f"成功打开PDF文件: {self.pdf_path}")
            logger.info(f"PDF页数: {len(self.doc)}")
            return True
        except Exception as e:
            logger.error(f"打开PDF文件时出错: {e}")
            return False
    
    def recognize_mahjong_hand(self, image_data):
        """
        识别图片中的麻将牌
        :param image_data: 图片数据
        :return: 麻将牌的文本表示，例如 "123m 456p 北中"
        """
        if self.mahjong_detector is None:
            # 模拟功能：实际应该使用外部模型
            logger.warning("没有提供麻将识别模型，返回占位符文本")
            return "[麻将牌: 此处需真实识别]"
        
        # 调用外部麻将检测模型进行识别
        return self.mahjong_detector.detect(image_data)
    
    def is_likely_mahjong_image(self, img_data):
        """
        判断图片是否可能是麻将牌图片
        :param img_data: 图片数据
        :return: 布尔值，是否可能是麻将牌
        """
        # 实际应用中可以通过图像分析来初步判断
        # 例如检测图像的颜色分布、边缘特征等
        # 这里仅作为示例实现
        try:
            # 转换为OpenCV格式的图像
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 1. 检查图像尺寸（麻将牌图片通常有一定的宽高比）
            height, width = img.shape[:2]
            aspect_ratio = width / height
            
            # 2. 检查颜色分布（麻将牌通常有白色或浅色背景）
            # 计算图像的平均亮度
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            # 基于简单规则的判断
            if 0.5 < aspect_ratio < 8 and avg_brightness > 150:
                # 可能是麻将牌图片
                logger.info(f"图片可能是麻将牌(宽高比:{aspect_ratio:.2f}, 亮度:{avg_brightness:.2f})")
                return True
            
            return False
        except Exception as e:
            logger.error(f"分析图片时出错: {e}")
            return False
    
    def process_image(self, img_data, img_index, page_num):
        """
        处理提取的图片
        :param img_data: 图片的二进制数据
        :param img_index: 图片索引
        :param page_num: 页码
        :return: Markdown格式的麻将牌表示
        """
        # 判断图片是否为麻将牌
        if self.is_likely_mahjong_image(img_data):
            # 识别麻将牌
            mahjong_text = self.recognize_mahjong_hand(img_data)
            
            # 如果识别成功且不是错误消息，则使用麻将牌表示
            if mahjong_text and not mahjong_text.startswith("检测出错") and not mahjong_text.startswith("模型未加载"):
                logger.info(f"识别到麻将牌: {mahjong_text}")
                # 直接返回麻将牌表示，不保存图片
                return f"```mahjong\n{mahjong_text}\n```"
            else:
                logger.warning(f"麻将牌识别失败: {mahjong_text}")
                # 识别失败时返回空内容
                return ""
        
        # 如果不是麻将牌或识别失败，则返回空内容
        return ""
    
    def extract_page_content(self, page_num):
        """
        提取页面内容
        :param page_num: 页码
        :return: 页面的Markdown内容
        """
        page = self.doc[page_num]
        
        # 提取文本
        text = page.get_text()
        
        # 处理文本格式（例如识别标题）
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            # 检测可能的标题（简单规则：短且不以标点结尾的行）
            if len(line.strip()) > 0 and len(line.strip()) < 50 and not re.search(r'[.,:;!?]$', line.strip()):
                # 根据行长度判断可能的标题级别
                if len(line.strip()) < 20:
                    formatted_lines.append(f"## {line.strip()}")
                else:
                    formatted_lines.append(f"### {line.strip()}")
            else:
                formatted_lines.append(line)
        
        text_content = '\n'.join(formatted_lines)
        
        # 提取页面上的图片
        image_list = page.get_images(full=True)
        
        # 按位置排序的图片和文本块
        page_elements = []
        
        # 添加文本块（简化处理，实际应按段落分块）
        text_blocks = page.get_text("blocks")
        for block in text_blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            if block_type == 0:  # 文本块
                page_elements.append({
                    "type": "text",
                    "y_pos": y0,  # 使用块的顶部位置
                    "content": text
                })
        
        # 添加图片
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = self.doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # 获取图片在页面上的位置
            img_rect = page.get_image_rects(xref)
            if img_rect:  # 确保能找到图片矩形
                y_pos = img_rect[0].y0  # 使用第一个矩形的顶部位置
            else:
                # 如果找不到图片位置，使用默认值
                y_pos = 0
            
            page_elements.append({
                "type": "image",
                "y_pos": y_pos,
                "img_index": img_index,
                "image_bytes": image_bytes
            })
        
        # 按y位置排序所有元素
        page_elements.sort(key=lambda x: x["y_pos"])
        
        # 组合排序后的内容
        markdown_content = []
        
        for element in page_elements:
            if element["type"] == "text":
                # 处理文本（可以进一步改进格式识别）
                text = element["content"].strip()
                if text:
                    # 简单的标题检测
                    if len(text) < 50 and not re.search(r'[.,:;!?]$', text):
                        if len(text) < 20:
                            markdown_content.append(f"## {text}")
                        else:
                            markdown_content.append(f"### {text}")
                    else:
                        markdown_content.append(text)
            elif element["type"] == "image":
                # 处理可能的麻将牌图片
                img_markdown = self.process_image(
                    element["image_bytes"], 
                    element["img_index"], 
                    page_num
                )
                # 只有当图片被识别为麻将牌时才添加到Markdown内容
                if img_markdown:
                    markdown_content.append(img_markdown)
        
        return "\n\n".join(markdown_content)
    
    def convert(self):
        """
        将PDF转换为Markdown
        """
        if not self.open_pdf():
            return False
        
        # 添加YAML前置元数据
        yaml_header = {
            "title": os.path.splitext(os.path.basename(self.pdf_path))[0],
            "source": self.pdf_path,
            "language": "zh-CN",  # 假设中文
            "date": f"{self.doc.metadata.get('creationDate', 'Unknown')}"
        }
        
        yaml_str = yaml.dump(yaml_header, allow_unicode=True)
        self.markdown_content.append(f"---\n{yaml_str}---\n\n")
        
        # 处理每一页
        total_pages = len(self.doc)
        for page_num in range(total_pages):
            logger.info(f"正在处理第 {page_num+1}/{total_pages} 页")
            
            page_content = self.extract_page_content(page_num)
            self.markdown_content.append(page_content)
            
            # 添加页面分隔符
            if page_num < total_pages - 1:
                self.markdown_content.append("\n---\n")
        
        # 写入Markdown文件
        with open(self.output_md_path, "w", encoding="utf-8") as md_file:
            md_file.write("\n".join(self.markdown_content))
        
        logger.info(f"转换完成，Markdown文件已保存至: {self.output_md_path}")
        return True


def main():
    parser = argparse.ArgumentParser(description="将麻将教材PDF转换为结构化Markdown")
    parser.add_argument("pdf_path", help="输入PDF文件路径")
    parser.add_argument("-o", "--output", help="输出Markdown文件路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--model", help="YOLO权重文件路径")
    parser.add_argument("--config", help="YOLO配置文件路径")
    parser.add_argument("--mock", action="store_true", help="使用模拟检测器")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # 初始化麻将检测器
    detector = get_detector(args.model, args.config, args.mock)
    
    converter = MahjongPDFConverter(
        pdf_path=args.pdf_path,
        output_md_path=args.output,
        mahjong_detector=detector
    )
    
    converter.convert()

if __name__ == "__main__":
    main() 