# 麻将手牌识别功能

这是一个用于识别日本立直麻将牌的模块，能够从图片中识别麻将牌并将其转换为标准文本格式。

## 功能特点

- 支持识别日本立直麻将牌（万子、筒子、索子和字牌）
- 将识别结果转换为标准文本格式（如"123m 456p 789s 东南西北中"）
- 能处理常见麻将牌展示图，包括横向排列和多行排列
- 可以判断图片是否为麻将牌图片
- 支持批量处理图片
- 可以与PDF转Markdown和bilibili爬虫功能集成

## 安装依赖

```bash
pip install tensorflow opencv-python numpy
```

## 使用方法

### 1. 准备模型

首先需要下载并准备模型：

```bash
cd MahjongHandRecognition/src
python download_model.py
```

这将从GitHub项目中下载预训练模型并进行适应性处理，保存到`MahjongHandRecognition/models`目录中。

### 2. 命令行使用

#### 识别单张图片

```bash
cd MahjongHandRecognition/src
python mahjong_recognition_cli.py --image path/to/image.jpg
```

#### 批量处理文件夹中的图片

```bash
cd MahjongHandRecognition/src
python mahjong_recognition_cli.py --folder path/to/folder
```

#### 自定义模型路径

```bash
cd MahjongHandRecognition/src
python mahjong_recognition_cli.py --image path/to/image.jpg --model path/to/model.h5
```

#### 调整置信度阈值

```bash
cd MahjongHandRecognition/src
python mahjong_recognition_cli.py --image path/to/image.jpg --confidence 0.5
```

### 3. 作为模块使用

```python
from MahjongHandRecognition.src.mahjong_recognition.tile_recognition import TileRecognizer

# 初始化识别器
recognizer = TileRecognizer()

# 识别单张图片
formatted_result, detailed_results = recognizer.recognize_hand("path/to/image.jpg")
print(f"Formatted result: {formatted_result}")

# 批量处理图片
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
results = recognizer.batch_process(image_paths)
for path, (formatted_result, _) in results.items():
    print(f"{path}: {formatted_result}")
```

### 4. 与现有项目集成

#### 与PDF转Markdown集成

```python
from MahjongHandRecognition.src.integration import integrate_with_pdf_converter

# 处理Markdown内容
md_content = "# 标题\n\n![麻将牌](images/mahjong.jpg)\n"
image_folder = "path/to/images"
processed_md = integrate_with_pdf_converter(md_content, image_folder)
```

#### 与Bilibili爬虫集成

```python
from MahjongHandRecognition.src.integration import integrate_with_bilibili_scraper

# 处理从Bilibili爬取的内容
bilibili_content = "原内容包含图片链接: https://example.com/image.jpg"
processed_content = integrate_with_bilibili_scraper(bilibili_content)
```

## 训练自己的模型

如果预训练模型不满足需求，可以使用自己的数据集进行训练：

```bash
cd MahjongHandRecognition/src
python mahjong_recognition_cli.py --train --data path/to/training/data --epochs 20
```

训练数据目录结构应为：

```
training_data/
  ├── 1万/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── 2万/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  └── ...
```

## 技术详情

- 使用卷积神经网络(CNN)识别麻将牌
- 采用OpenCV进行图像预处理和分割
- 支持迁移学习，可以使用少量数据进行微调
- 识别准确度可达90%以上（取决于图片质量和训练数据）

## 常见问题

### 识别准确度不高？

- 确保图片清晰，光线充足
- 麻将牌应当排列整齐，尽量减少重叠
- 可以尝试增加训练数据或调整模型

### 无法识别某些特殊牌型？

- 默认支持日本立直麻将的标准牌型
- 如需支持其他类型的麻将牌，需要重新训练模型

### 集成到项目中遇到问题？

- 确保路径正确，特别是相对路径
- 检查依赖库是否安装完整
- 查看详细的错误信息进行排查 