#!/usr/bin/env python3
"""
Quick Start Script for Mahjong Hand Recognition
麻将牌识别功能的快速启动脚本
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mahjong_recognition.tile_recognition import TileRecognizer
from mahjong_recognition.utils import preprocess_image, is_mahjong_image


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Quick Start for Mahjong Hand Recognition")
    parser.add_argument("--image", "-i", type=str, help="Path to input image")
    args = parser.parse_args()
    
    if not args.image:
        print("Please provide an image path using --image or -i")
        sys.exit(1)
    
    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found")
        sys.exit(1)
    
    print(f"Processing image: {args.image}")
    
    # 步骤1: 检查是否是麻将图片
    img = preprocess_image(args.image, target_size=(800, 600))
    if not is_mahjong_image(img):
        print("The image does not appear to contain mahjong tiles.")
        sys.exit(0)
    
    print("Detected mahjong tiles in the image.")
    
    # 步骤2: 初始化识别器
    print("Initializing recognizer...")
    recognizer = TileRecognizer()
    
    # 步骤3: 识别麻将牌
    print("Recognizing tiles...")
    formatted_result, detailed_results = recognizer.recognize_hand(args.image)
    
    # 步骤4: 输出结果
    print("\nResults:")
    print(f"Formatted result: {formatted_result}")
    print("\nDetailed results:")
    for tile, conf in detailed_results:
        print(f"  {tile}: {conf:.4f}")
    
    print("\nQuick start completed!")


if __name__ == "__main__":
    main() 