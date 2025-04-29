# 麻将教程 (Mahjong Tutor)

这是一个交互式的麻将学习应用，帮助初学者学习和掌握麻将的规则和策略。

## 功能

- 麻将基本规则介绍
- 牌型和组合识别
- 互动练习和测验
- 策略指导

## 技术栈

- React / Next.js
- TypeScript
- Tailwind CSS

## 项目结构

- `/src` - 源代码
- `/public` - 静态资源
- `/components` - React组件
- `/pages` - 应用页面
- `/styles` - CSS样式
- `/lib` - 辅助函数和工具
- `/types` - TypeScript类型定义

## 项目结构

- `DataProcess/`: 数据处理相关工具，包含PDF转Markdown和bilibili爬虫的主要代码
- `Data/`: 存放数据文件，如原始PDF、生成的Markdown文件和下载的图片
- `convert_pdf.py`: PDF转Markdown工具的入口脚本
- `scrape_bilibili.py`: Bilibili文章爬虫的入口脚本
- `scrape_bilibili.bat`: Windows平台下的B站爬虫便捷启动批处理脚本
- `test_inference.bat`: Windows平台下使用inference_sdk进行麻将牌检测的测试脚本

## 功能简介

### PDF转Markdown工具

将麻将教材PDF转换为结构化的Markdown文件，并能够自动识别文档中的麻将牌图片，将其转换为文本表示（如 `123m 456p 789s 东南西北中`）。

支持多种检测方式：
1. 本地YOLO模型检测
2. Roboflow API检测 (https://universe.roboflow.com/riichimahjongdetection/riichi-mahjong-detection/model/3)
3. 模拟检测 (仅用于测试)

### Bilibili文章爬虫

抓取bilibili文章内容并保存为Markdown格式，支持批量抓取和保存到同一个文件中，并提供带图片和无图片两个版本。

**新功能**：现在爬虫可以自动检测文章中的麻将牌图片，并将其转换为文本表示。在无图版本中，麻将牌将显示为文本形式，便于阅读和学习。

## 使用方法

### 1. 安装依赖

```bash
pip install -r DataProcess/requirements.txt
pip install roboflow  # 如果需要使用Roboflow API
pip install inference-sdk  # 如果需要使用Inference SDK
```

### 2. PDF转Markdown工具

```bash
# 使用模拟麻将检测器（不需要下载模型）
python convert_pdf.py 你的麻将教材.pdf --mock

# 自动下载麻将检测模型
python convert_pdf.py 你的麻将教材.pdf --download-model

# 使用已有的麻将检测模型
python convert_pdf.py 你的麻将教材.pdf --model-dir 你的模型目录

# 使用Roboflow API检测器
python convert_pdf.py 你的麻将教材.pdf --roboflow --api-key YOUR_API_KEY
```

### 3. Bilibili文章爬虫

```bash
# 直接抓取单个URL
python scrape_bilibili.py https://www.bilibili.com/opus/123456

# 批量抓取多个URL
python scrape_bilibili.py https://www.bilibili.com/opus/123456 https://www.bilibili.com/opus/789012

# 从文本文件读取URL列表进行抓取
python scrape_bilibili.py -f bilibili_urls.txt

# 指定输出目录和文件名
python scrape_bilibili.py -o Output -n mahjong_articles https://www.bilibili.com/opus/123456

# 不下载图片（只生成无图版本）
python scrape_bilibili.py --no-image https://www.bilibili.com/opus/123456

# 禁用麻将牌图片检测功能
python scrape_bilibili.py --no-detect https://www.bilibili.com/opus/123456
```


## 麻将牌文本表示

本项目使用以下格式表示麻将牌：

- 数字+m：表示万子，如 `123m` 表示一万、二万、三万
- 数字+p：表示筒子，如 `456p` 表示四筒、五筒、六筒
- 数字+s：表示索子，如 `789s` 表示七索、八索、九索
- 东南西北中发白：直接使用汉字表示字牌

例如，`123m 456p 789s 东南西` 表示一组包含一万、二万、三万、四筒、五筒、六筒、七索、八索、九索、东风、南风、西风的麻将牌组合。
