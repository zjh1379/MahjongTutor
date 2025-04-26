import os
import requests
import zipfile
import logging
import sys
import argparse
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_file(url, destination):
    """
    下载文件
    :param url: 下载链接
    :param destination: 保存位置
    :return: 是否成功
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 如果响应状态不是200，将引发HTTPError异常
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        downloaded = 0
        
        logger.info(f"下载文件: {url}")
        logger.info(f"文件大小: {total_size / (1024 * 1024):.2f} MB")
        
        with open(destination, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
                downloaded += len(data)
                done = int(50 * downloaded / total_size)
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded / (1024 * 1024):.2f}/{total_size / (1024 * 1024):.2f} MB")
                sys.stdout.flush()
        
        sys.stdout.write('\n')
        logger.info(f"文件已下载到 {destination}")
        return True
    except Exception as e:
        logger.error(f"下载文件时出错: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """
    解压ZIP文件
    :param zip_path: ZIP文件路径
    :param extract_to: 解压目录
    :return: 是否成功
    """
    try:
        logger.info(f"解压文件: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"文件已解压到 {extract_to}")
        return True
    except Exception as e:
        logger.error(f"解压文件时出错: {e}")
        return False

def setup_mahjong_detection_model(model_dir="mahjong_detection"):
    """
    设置麻将检测模型
    :param model_dir: 模型目录
    :return: 模型路径
    """
    # 创建模型目录
    os.makedirs(model_dir, exist_ok=True)
    
    # GitHub项目的URL
    github_url = "https://github.com/lissa2077/Mahjong-Detection"
    
    # 假设项目有一个发布版本，我们可以下载它
    # 注意：这个URL可能需要根据实际情况修改
    model_url = f"{github_url}/releases/download/v0.1/yolov3-tiny-mahjong.weights"
    config_url = f"{github_url}/raw/main/darknet/cfg/yolov3-tiny-mahjong.cfg"
    
    # 下载模型文件
    model_path = os.path.join(model_dir, "yolov3-tiny-mahjong.weights")
    config_path = os.path.join(model_dir, "yolov3-tiny-mahjong.cfg")
    
    # 如果模型文件不存在，则下载
    if not os.path.exists(model_path):
        logger.info("正在下载模型文件...")
        success = download_file(model_url, model_path)
        if not success:
            logger.error("下载模型文件失败，将使用模拟检测器")
            return None
    
    # 如果配置文件不存在，则下载
    if not os.path.exists(config_path):
        logger.info("正在下载配置文件...")
        success = download_file(config_url, config_path)
        if not success:
            logger.error("下载配置文件失败，将使用模拟检测器")
            return None
    
    return {
        "model_path": model_path,
        "config_path": config_path,
        "model_dir": model_dir
    }

def clone_github_repo(repo_url, destination):
    """
    克隆GitHub仓库
    :param repo_url: 仓库URL
    :param destination: 目标目录
    :return: 是否成功
    """
    try:
        import git
        logger.info(f"克隆仓库: {repo_url}")
        git.Repo.clone_from(repo_url, destination)
        logger.info(f"仓库已克隆到 {destination}")
        return True
    except ImportError:
        logger.error("未安装GitPython库，无法克隆仓库")
        logger.info("请运行 pip install gitpython 安装依赖")
        return False
    except Exception as e:
        logger.error(f"克隆仓库时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="下载和设置麻将检测模型")
    parser.add_argument("--model-dir", default="mahjong_detection", help="模型保存目录")
    parser.add_argument("--clone-repo", action="store_true", help="克隆GitHub仓库")
    
    args = parser.parse_args()
    
    # 设置麻将检测模型
    model_info = setup_mahjong_detection_model(args.model_dir)
    
    # 如果指定了克隆仓库选项
    if args.clone_repo:
        repo_dir = os.path.join(args.model_dir, "repo")
        clone_github_repo("https://github.com/lissa2077/Mahjong-Detection.git", repo_dir)
    
    if model_info:
        logger.info("麻将检测模型设置成功")
        logger.info(f"模型路径: {model_info['model_path']}")
        logger.info(f"配置路径: {model_info['config_path']}")
    else:
        logger.warning("麻将检测模型设置失败，将使用模拟检测器")

if __name__ == "__main__":
    main() 