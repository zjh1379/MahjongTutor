#!/usr/bin/env python3
"""
Download and transfer learning script for Mahjong Hand Recognition
下载和迁移学习脚本，用于获取和适应麻将牌识别模型
"""

import os
import sys
import argparse
import requests
from pathlib import Path
import tempfile
import zipfile
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# 定义麻将牌类别
TILE_CLASSES = [
    '1万', '2万', '3万', '4万', '5万', '6万', '7万', '8万', '9万',
    '1筒', '2筒', '3筒', '4筒', '5筒', '6筒', '7筒', '8筒', '9筒',
    '1索', '2索', '3索', '4索', '5索', '6索', '7索', '8索', '9索',
    '东', '南', '西', '北', '白', '发', '中'
]


def download_github_repo(repo_url: str, branch: str = 'master', target_dir: Optional[str] = None) -> str:
    """
    下载GitHub仓库
    
    Args:
        repo_url: GitHub仓库URL
        branch: 分支名称
        target_dir: 目标目录，如果为None则创建临时目录
        
    Returns:
        下载的仓库所在的目录路径
    """
    if target_dir is None:
        target_dir = tempfile.mkdtemp()
    
    # 获取仓库名称
    repo_name = repo_url.rstrip('/').split('/')[-1]
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]
    
    # 构建下载URL
    download_url = f"{repo_url}/archive/refs/heads/{branch}.zip"
    zip_path = os.path.join(target_dir, f"{repo_name}.zip")
    
    print(f"Downloading {download_url} to {zip_path}...")
    
    # 下载仓库
    response = requests.get(download_url, stream=True)
    response.raise_for_status()
    
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    # 解压仓库
    extract_dir = os.path.join(target_dir, repo_name)
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    
    # 解压后的目录名称通常是 "repo_name-branch"
    extracted_dir = os.path.join(target_dir, f"{repo_name}-{branch}")
    
    return extracted_dir


def create_transfer_learning_model(input_shape: tuple = (64, 64, 3), num_classes: int = len(TILE_CLASSES)) -> tf.keras.Model:
    """
    创建迁移学习模型
    
    Args:
        input_shape: 输入数据的形状
        num_classes: 类别数量
        
    Returns:
        创建的迁移学习模型
    """
    # 使用MobileNetV2作为基础模型
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # 冻结基础模型的部分层
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    # 创建新模型
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def adapt_existing_model(model_path: str, num_classes: int = len(TILE_CLASSES)) -> tf.keras.Model:
    """
    适应现有模型
    
    Args:
        model_path: 现有模型的路径
        num_classes: 类别数量
        
    Returns:
        适应后的模型
    """
    try:
        # 加载现有模型
        original_model = tf.keras.models.load_model(model_path)
        
        # 获取除了最后一层以外的所有层
        x = original_model.layers[-2].output
        
        # 添加新的分类层
        output = layers.Dense(num_classes, activation='softmax')(x)
        
        # 创建新模型
        model = tf.keras.Model(inputs=original_model.input, outputs=output)
        
        # 冻结部分层
        for layer in model.layers[:-2]:
            layer.trainable = False
        
        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        print(f"Error adapting existing model: {e}")
        print("Creating a new transfer learning model instead.")
        return create_transfer_learning_model()


def download_and_prepare_model(output_dir: str):
    """
    下载并准备模型
    
    Args:
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义模型输出路径
    model_output_path = os.path.join(output_dir, 'mahjong_classifier_model.h5')
    
    # 下载GitHub仓库
    repo_url = 'https://github.com/elise-ng/COMP4901J_Project'
    repo_dir = download_github_repo(repo_url)
    
    print(f"Downloaded repository to {repo_dir}")
    
    # 检查是否已下载了模型文件
    model_paths = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith('.h5') or file.endswith('.keras'):
                model_paths.append(os.path.join(root, file))
    
    if model_paths:
        print(f"Found {len(model_paths)} model files:")
        for path in model_paths:
            print(f"  - {path}")
        
        # 使用第一个找到的模型
        source_model_path = model_paths[0]
        
        # 适应现有模型
        model = adapt_existing_model(source_model_path)
        
        # 保存适应后的模型
        model.save(model_output_path)
        print(f"Adapted model saved to {model_output_path}")
    else:
        print("No pre-trained model found in the repository.")
        
        # 创建新的迁移学习模型
        model = create_transfer_learning_model()
        
        # 保存新模型
        model.save(model_output_path)
        print(f"New transfer learning model saved to {model_output_path}")
    
    # 清理下载的仓库
    shutil.rmtree(repo_dir)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Download and prepare model for Mahjong Hand Recognition")
    
    parser.add_argument("--output-dir", "-o", type=str, default="../models",
                        help="Output directory for the model (default: ../models)")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 获取绝对路径
    if not os.path.isabs(args.output_dir):
        # 相对于脚本所在目录
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    else:
        output_dir = args.output_dir
    
    # 下载并准备模型
    download_and_prepare_model(output_dir)


if __name__ == "__main__":
    main() 