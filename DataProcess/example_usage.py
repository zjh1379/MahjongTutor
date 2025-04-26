#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例：在项目中集成inference_sdk进行麻将牌检测
"""

import os
import sys
import logging
from PIL import Image
import io

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_mahjong_tiles_in_image(image_path):
    """在图片中检测麻将牌并返回检测结果"""
    try:
        # 导入inference_sdk
        from inference_sdk import InferenceHTTPClient
        
        # 初始化客户端 - 使用预先设置的API密钥
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="t8aq85tzovOw20ril1bB"  # 项目专用API密钥
        )
        
        # 执行检测
        result = client.infer(image_path, model_id="riichi-mahjong-detection/3")
        
        # 处理检测结果
        predictions = result.get("predictions", [])
        
        # 提取麻将牌类型
        detected_tiles = [pred.get("class", "") for pred in predictions]
        logger.info(f"检测到 {len(detected_tiles)} 个麻将牌: {', '.join(detected_tiles)}")
        
        # 将检测结果转换为标准格式
        formatted_result = format_mahjong_tiles(detected_tiles)
        logger.info(f"格式化结果: {formatted_result}")
        
        return formatted_result
    
    except ImportError:
        logger.error("未安装inference_sdk，请安装依赖: pip install inference-sdk")
        return None
    
    except Exception as e:
        logger.error(f"检测麻将牌时出错: {e}")
        return None

def format_mahjong_tiles(tile_list):
    """将检测到的麻将牌列表格式化为标准表示"""
    # 按类型分组
    man_tiles = [t for t in tile_list if t.endswith('m')]
    pin_tiles = [t for t in tile_list if t.endswith('p')]
    sou_tiles = [t for t in tile_list if t.endswith('s')]
    honor_tiles = [t for t in tile_list if not (t.endswith('m') or t.endswith('p') or t.endswith('s'))]
    
    # 格式化结果
    result_parts = []
    
    # 万子
    if man_tiles:
        man_values = [t[0] for t in man_tiles]
        result_parts.append(''.join(sorted(man_values)) + 'm')
    
    # 筒子
    if pin_tiles:
        pin_values = [t[0] for t in pin_tiles]
        result_parts.append(''.join(sorted(pin_values)) + 'p')
    
    # 索子
    if sou_tiles:
        sou_values = [t[0] for t in sou_tiles]
        result_parts.append(''.join(sorted(sou_values)) + 's')
    
    # 字牌
    if honor_tiles:
        result_parts.append(' '.join(sorted(honor_tiles)))
    
    # 合并结果
    return ' '.join(result_parts)

def visualize_detection(image_path, output_path):
    """可视化麻将牌检测结果并保存"""
    try:
        from inference_sdk import InferenceHTTPClient
        
        # 初始化客户端
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="t8aq85tzovOw20ril1bB"
        )
        
        # 获取可视化结果
        visualization = client.infer(image_path, model_id="riichi-mahjong-detection/3", visualize=True)
        
        # 保存图片
        with open(output_path, "wb") as f:
            f.write(visualization)
        
        logger.info(f"可视化结果已保存到: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"可视化麻将牌检测结果时出错: {e}")
        return False

# 使用示例
if __name__ == "__main__":
    # 如果传入了参数，使用第一个参数作为图片路径
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 默认路径
        image_path = "Test/mahjong_test.jpg"
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        logger.error(f"图片文件不存在: {image_path}")
        sys.exit(1)
    
    # 输出路径
    output_path = "Test/example_result.jpg"
    
    # 执行检测
    print("检测麻将牌...")
    result = detect_mahjong_tiles_in_image(image_path)
    if result:
        print(f"检测结果: {result}")
        
        # 保存可视化结果
        visualize_detection(image_path, output_path)
    else:
        print("检测失败") 