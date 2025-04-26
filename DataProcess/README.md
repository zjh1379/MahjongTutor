# 麻将教材PDF转Markdown工具

这个工具可以将麻将教材PDF转换为结构化的Markdown文件，并且能够自动识别文档中的麻将牌图片，将其转换为文本表示（如 `123m 456p 789s 东南西北中`）。识别出的麻将牌会直接以文本形式显示在Markdown中，不会保存图片文件。

## 功能特性

- 将PDF文档转换为结构化Markdown格式
- 自动识别和提取PDF中的文本和图片
- 使用YOLO模型识别图片中的麻将牌
- 直接在Markdown中以文本形式显示麻将牌，不生成额外图片文件
- 保留原始文档的文本结构和排版
- 支持YAML元数据
- 识别的麻将牌使用特殊代码块标记：```mahjong

## 安装

1. 克隆本仓库：

```bash
git clone https://github.com/yourusername/mahjong-pdf-to-markdown.git
cd mahjong-pdf-to-markdown
```

2. 创建并激活虚拟环境（可选但推荐）：

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

使用模拟麻将检测器（不需要下载模型）：

```bash
python convert_mahjong_pdf.py 你的麻将教材.pdf --mock
```

### 自动下载麻将检测模型

```bash
python convert_mahjong_pdf.py 你的麻将教材.pdf --download-model
```

### 使用已有的麻将检测模型

```bash
python convert_mahjong_pdf.py 你的麻将教材.pdf --model-dir 你的模型目录
```

### 其他选项

- `-o, --output`：指定输出Markdown文件的路径
- `--debug`：启用调试模式，输出更详细的日志
- `--model-dir`：指定模型目录，默认为"mahjong_detection"

完整的命令行选项：

```bash
python convert_mahjong_pdf.py --help
```

## 麻将牌检测模型

本工具使用了[Mahjong-Detection](https://github.com/lissa2077/Mahjong-Detection)项目的YOLO模型来识别麻将牌。

如果您想自己训练模型或者了解更多关于麻将牌检测的信息，请参考该项目。

## 使用示例

假设有一个名为"麻将入门教程.pdf"的文件，我们可以这样转换它：

```bash
python convert_mahjong_pdf.py "麻将入门教程.pdf" --download-model -o "麻将入门教程.md"
```

转换后的Markdown文件将包含类似以下内容：

```markdown
## 第一章：麻将基础知识

麻将是一种起源于中国的游戏，通常由四人玩，使用一副特殊的牌...

### 基本牌型

下面是一个简单的顺子例子：

```mahjong
123m 456p 789s
```
```

## 注意事项

- 如果模型下载或加载失败，工具会自动回退到使用模拟检测器
- 只有被识别为麻将牌的图片才会被转换并保留在Markdown中，其他图片会被忽略
- 文本识别的质量依赖于PDF文件的质量和结构

## 许可

此项目采用MIT许可证 - 有关详细信息，请查看[LICENSE](LICENSE)文件。 