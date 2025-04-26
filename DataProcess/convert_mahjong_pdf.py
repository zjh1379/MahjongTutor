#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
麻将教材PDF转Markdown工具

这个脚本将麻将教材PDF转换为结构化的Markdown文件，并使用麻将牌识别模型识别文档中的麻将牌图片。
识别出的麻将牌将直接以文本形式显示在Markdown中，不会保存图片。

支持以下检测方式:
1. 本地YOLO模型检测
2. Roboflow API检测 (https://universe.roboflow.com/riichimahjongdetection/riichi-mahjong-detection/model/3)
3. 模拟检测 (仅用于测试)
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入自定义模块
try:
    from mahjong_detector import get_detector
    from pdf_to_markdown import MahjongPDFConverter
    from download_model import setup_mahjong_detection_model
except ImportError as e:
    logger.error(f"导入模块时出错: {e}")
    logger.error("请确保已安装所有依赖")
    logger.info("运行 pip install -r requirements.txt 安装依赖")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="将麻将教材PDF转换为结构化Markdown，识别麻将牌并直接显示文本表示")
    parser.add_argument("pdf_path", help="输入PDF文件路径")
    parser.add_argument("-o", "--output", help="输出Markdown文件路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--model-dir", default="mahjong_detection", help="模型目录")
    parser.add_argument("--download-model", action="store_true", help="自动下载模型")
    parser.add_argument("--mock", action="store_true", help="使用模拟检测器")
    # 添加Roboflow相关选项
    parser.add_argument("--roboflow", action="store_true", help="使用Roboflow API检测器")
    parser.add_argument("--api-key", help="Roboflow API密钥")
    parser.add_argument("--model-id", default="riichi-mahjong-detection/3", help="Roboflow模型ID")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # 检查PDF文件是否存在
    if not os.path.exists(args.pdf_path):
        logger.error(f"PDF文件不存在: {args.pdf_path}")
        sys.exit(1)
    
    # 设置麻将检测器
    if args.roboflow:
        # 使用Roboflow API检测器
        if not args.api_key:
            logger.error("使用Roboflow API检测器需要提供API密钥")
            logger.error("请使用 --api-key 参数提供Roboflow API密钥")
            sys.exit(1)
        
        logger.info(f"使用Roboflow API检测器 (模型: {args.model_id})")
        detector = get_detector(
            use_roboflow=True,
            roboflow_api_key=args.api_key,
            roboflow_model_id=args.model_id
        )
        
        if detector is None:
            logger.error("初始化Roboflow API检测器失败")
            sys.exit(1)
    elif args.mock:
        # 使用模拟检测器
        logger.info("使用模拟麻将牌检测器")
        detector = get_detector(use_mock=True)
    else:
        # 使用本地YOLO模型
        model_path = None
        config_path = None
        
        # 如果指定了下载模型，则下载模型
        if args.download_model:
            logger.info("正在下载麻将检测模型...")
            model_info = setup_mahjong_detection_model(args.model_dir)
            
            if model_info:
                model_path = model_info["model_path"]
                config_path = model_info["config_path"]
                logger.info(f"模型下载成功，路径: {model_path}")
            else:
                logger.warning("模型下载失败，将使用模拟检测器")
                detector = get_detector(use_mock=True)
        else:
            # 尝试在模型目录中查找模型
            model_dir = Path(args.model_dir)
            possible_weights = list(model_dir.glob("*.weights"))
            possible_configs = list(model_dir.glob("*.cfg"))
            
            if possible_weights and possible_configs:
                model_path = str(possible_weights[0])
                config_path = str(possible_configs[0])
                logger.info(f"使用现有模型: {model_path}")
                logger.info(f"使用现有配置: {config_path}")
            else:
                logger.warning(f"在目录 {args.model_dir} 中找不到模型文件，将使用模拟检测器")
                detector = get_detector(use_mock=True)
        
        # 如果找到了模型和配置文件，初始化检测器
        if model_path and config_path and 'detector' not in locals():
            detector = get_detector(model_path, config_path)
    
    # 设置输出路径
    output_path = args.output
    if not output_path:
        base_name = os.path.basename(args.pdf_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_path = f"{name_without_ext}.md"
    
    # 创建转换器并转换
    logger.info(f"开始转换PDF: {args.pdf_path} -> {output_path}")
    logger.info("麻将牌将直接以文本形式显示在Markdown中，不会保存图片")
    converter = MahjongPDFConverter(
        pdf_path=args.pdf_path,
        output_md_path=output_path,
        mahjong_detector=detector
    )
    
    success = converter.convert()
    
    if success:
        logger.info(f"转换完成，Markdown文件已保存至: {output_path}")
        logger.info("所有识别出的麻将牌均以文本形式显示，非麻将牌图片已被省略")
    else:
        logger.error("转换失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 