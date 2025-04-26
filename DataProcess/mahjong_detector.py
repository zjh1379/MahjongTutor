import cv2
import numpy as np
import os
import logging
from io import BytesIO
from PIL import Image

# 导入Roboflow麻将检测器
try:
    from .roboflow_mahjong_detector import RoboflowMahjongDetector
except ImportError:
    try:
        from roboflow_mahjong_detector import RoboflowMahjongDetector
    except ImportError:
        # 如果导入失败，定义一个空类，后续会检查
        class RoboflowMahjongDetector:
            pass

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MahjongDetector:
    """
    麻将牌检测器类
    使用YOLO模型检测图片中的麻将牌
    """
    
    def __init__(self, config_path, weights_path, confidence_threshold=0.5):
        """
        初始化检测器
        :param config_path: YOLO配置文件路径
        :param weights_path: YOLO权重文件路径
        :param confidence_threshold: 置信度阈值
        """
        self.confidence_threshold = confidence_threshold
        self.net = None
        self.output_layers = None
        self.classes = [
            "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",  # 万子
            "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",  # 筒子
            "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",  # 索子
            "东", "南", "西", "北", "中", "发", "白"  # 字牌
        ]
        
        self.config_path = config_path
        self.weights_path = weights_path
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """
        加载YOLO模型
        """
        try:
            logger.info("正在加载YOLO模型...")
            
            # 检查文件是否存在
            if not os.path.exists(self.config_path):
                logger.error(f"配置文件不存在: {self.config_path}")
                return False
            
            if not os.path.exists(self.weights_path):
                logger.error(f"权重文件不存在: {self.weights_path}")
                return False
            
            # 加载网络
            self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
            
            # 获取输出层名称
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            logger.info("YOLO模型加载成功")
            return True
        except Exception as e:
            logger.error(f"加载YOLO模型时出错: {e}")
            return False
    
    def detect(self, image_data):
        """
        检测图片中的麻将牌
        :param image_data: 图片数据（二进制或NumPy数组）
        :return: 麻将牌的文本表示，例如 "123m 456p 789s 东南西北"
        """
        if self.net is None:
            logger.error("模型未加载成功，无法进行检测")
            return "模型未加载成功"
        
        try:
            # 将二进制数据转为NumPy数组
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = image_data
            
            # 获取图像尺寸
            height, width, channels = img.shape
            
            # 准备模型输入
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            
            # 前向传播
            outs = self.net.forward(self.output_layers)
            
            # 存储检测结果
            class_ids = []
            confidences = []
            boxes = []
            
            # 处理每个输出层的结果
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.confidence_threshold:
                        # 对象位置
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # 边界框坐标
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # 非极大值抑制，消除重复框
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
            
            # 排序检测结果（从左到右，从上到下）
            if len(indices) > 0:
                sorted_indices = []
                for i in indices:
                    if isinstance(i, list):  # OpenCV 3.x返回的是列表
                        sorted_indices.append([i[0], boxes[i[0]][0], boxes[i[0]][1]])
                    else:  # OpenCV 4.x返回的是单个值
                        sorted_indices.append([i, boxes[i][0], boxes[i][1]])
                
                # 根据y坐标分组（认为同一行的y坐标差异较小）
                y_threshold = height * 0.1  # 同一行的y坐标差异阈值
                rows = []
                current_row = [sorted_indices[0]]
                
                for i in range(1, len(sorted_indices)):
                    if abs(sorted_indices[i][2] - sorted_indices[0][2]) < y_threshold:
                        current_row.append(sorted_indices[i])
                    else:
                        # 当前行已完成，添加到rows并开始新行
                        rows.append(sorted(current_row, key=lambda x: x[1]))  # 按x坐标排序
                        current_row = [sorted_indices[i]]
                
                if current_row:
                    rows.append(sorted(current_row, key=lambda x: x[1]))  # 添加最后一行
                
                # 合并所有行
                final_indices = []
                for row in rows:
                    final_indices.extend([x[0] for x in row])
            else:
                final_indices = indices
            
            # 组织检测结果
            detected_tiles = []
            for i in final_indices:
                if isinstance(i, list):  # OpenCV 3.x
                    i = i[0]
                class_id = class_ids[i]
                tile = self.classes[class_id]
                detected_tiles.append(tile)
            
            # 按照麻将牌类型分组（万子、筒子、索子、字牌）
            man_tiles = [t for t in detected_tiles if t.endswith('m')]
            pin_tiles = [t for t in detected_tiles if t.endswith('p')]
            sou_tiles = [t for t in detected_tiles if t.endswith('s')]
            honor_tiles = [t for t in detected_tiles if not (t.endswith('m') or t.endswith('p') or t.endswith('s'))]
            
            # 格式化为麻将牌表示
            result = []
            
            # 处理万子
            if man_tiles:
                man_values = [t[0] for t in man_tiles]
                result.append(''.join(man_values) + 'm')
            
            # 处理筒子
            if pin_tiles:
                pin_values = [t[0] for t in pin_tiles]
                result.append(''.join(pin_values) + 'p')
            
            # 处理索子
            if sou_tiles:
                sou_values = [t[0] for t in sou_tiles]
                result.append(''.join(sou_values) + 's')
            
            # 处理字牌
            if honor_tiles:
                result.append(' '.join(honor_tiles))
            
            # 返回最终结果
            return ' '.join(result)
            
        except Exception as e:
            logger.error(f"检测麻将牌时出错: {e}")
            return f"检测出错: {str(e)}"
    
    def visualize_detection(self, image_data, output_path=None):
        """
        在图片上可视化麻将牌检测结果
        :param image_data: 图片数据
        :param output_path: 输出路径，若为None则不保存图片
        :return: 处理后的图片
        """
        if self.net is None:
            logger.error("模型未加载成功，无法进行检测")
            return None
        
        try:
            # 将二进制数据转为NumPy数组
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = image_data.copy()
            
            # 获取图像尺寸
            height, width, channels = img.shape
            
            # 准备模型输入
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            
            # 前向传播
            outs = self.net.forward(self.output_layers)
            
            # 存储检测结果
            class_ids = []
            confidences = []
            boxes = []
            
            # 处理每个输出层的结果
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.confidence_threshold:
                        # 对象位置
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # 边界框坐标
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # 非极大值抑制，消除重复框
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
            
            # 在图片上画框
            colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            
            for i in indices:
                if isinstance(i, list):  # OpenCV 3.x
                    i = i[0]
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                color = colors[class_id]
                
                # 画边界框
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                # 添加标签
                label = f"{self.classes[class_id]}: {confidences[i]:.2f}"
                cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 保存图片
            if output_path:
                cv2.imwrite(output_path, img)
                logger.info(f"已保存检测结果图片到 {output_path}")
            
            return img
            
        except Exception as e:
            logger.error(f"可视化检测结果时出错: {e}")
            return None

class MockMahjongDetector:
    """模拟麻将检测器，用于演示"""
    
    def detect(self, image_data):
        """模拟麻将牌识别"""
        # 返回一个随机的麻将牌组合
        import random
        
        # 万子
        man = random.sample(["1", "2", "3", "4", "5", "6", "7", "8", "9"], random.randint(1, 4))
        man_str = "".join(man) + "m" if man else ""
        
        # 筒子
        pin = random.sample(["1", "2", "3", "4", "5", "6", "7", "8", "9"], random.randint(1, 4))
        pin_str = "".join(pin) + "p" if pin else ""
        
        # 索子
        sou = random.sample(["1", "2", "3", "4", "5", "6", "7", "8", "9"], random.randint(1, 4))
        sou_str = "".join(sou) + "s" if sou else ""
        
        # 字牌
        honor = random.sample(["东", "南", "西", "北", "中", "发", "白"], random.randint(0, 3))
        honor_str = " ".join(honor) if honor else ""
        
        # 组合结果
        parts = [part for part in [man_str, pin_str, sou_str, honor_str] if part]
        return " ".join(parts)
    
    def visualize_detection(self, image_data, output_path=None):
        """模拟检测可视化"""
        return None


def get_detector(model_path=None, config_path=None, use_mock=False, use_roboflow=False, roboflow_api_key=None, roboflow_model_id="riichi-mahjong-detection/3"):
    """
    获取麻将牌检测器实例
    :param model_path: 本地YOLO模型路径
    :param config_path: 本地YOLO配置文件路径
    :param use_mock: 是否使用模拟检测器
    :param use_roboflow: 是否使用Roboflow API检测器
    :param roboflow_api_key: Roboflow API密钥
    :param roboflow_model_id: Roboflow模型ID，默认为"riichi-mahjong-detection/3"
    :return: 麻将牌检测器实例
    """
    # 如果使用模拟检测器
    if use_mock:
        logger.info("使用模拟麻将牌检测器")
        return MockMahjongDetector()
    
    # 如果使用Roboflow API检测器
    if use_roboflow:
        if not roboflow_api_key:
            logger.error("使用Roboflow API检测器需要提供API密钥")
            return None
        
        try:
            logger.info(f"使用Roboflow API麻将牌检测器 (模型: {roboflow_model_id})")
            return RoboflowMahjongDetector(roboflow_api_key, roboflow_model_id)
        except Exception as e:
            logger.error(f"初始化Roboflow API检测器失败: {e}")
            logger.warning("回退到使用模拟检测器")
            return MockMahjongDetector()
    
    # 如果使用本地YOLO模型
    if model_path and config_path:
        logger.info(f"使用本地YOLO模型 (模型: {model_path}, 配置: {config_path})")
        detector = MahjongDetector(config_path, model_path)
        if detector.net is None:
            logger.warning("加载YOLO模型失败，回退到使用模拟检测器")
            return MockMahjongDetector()
        return detector
    
    # 默认返回模拟检测器
    logger.warning("未提供模型路径或配置文件路径，使用模拟检测器")
    return MockMahjongDetector()

def test_detector(detector, image_path, output_path=None):
    """
    测试麻将检测器
    :param detector: 检测器实例
    :param image_path: 测试图片路径
    :param output_path: 输出图片路径
    """
    try:
        # 读取测试图片
        with open(image_path, 'rb') as f:
            img_data = f.read()
        
        # 进行检测
        result = detector.detect(img_data)
        logger.info(f"检测结果: {result}")
        
        # 可视化检测结果
        if output_path:
            detector.visualize_detection(img_data, output_path)
            logger.info(f"可视化结果已保存到: {output_path}")
        
        return result
    except Exception as e:
        logger.error(f"测试检测器时出错: {e}")
        return None

def main():
    """
    测试主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="麻将牌检测器")
    parser.add_argument("--image", required=True, help="要检测的图片路径")
    parser.add_argument("--model", help="模型权重文件路径")
    parser.add_argument("--config", help="模型配置文件路径")
    parser.add_argument("--output", help="输出图片路径")
    parser.add_argument("--mock", action="store_true", help="使用模拟检测器")
    parser.add_argument("--roboflow", action="store_true", help="使用Roboflow API检测器")
    parser.add_argument("--api-key", help="Roboflow API密钥")
    parser.add_argument("--model-id", default="riichi-mahjong-detection/3", help="Roboflow模型ID")
    
    args = parser.parse_args()
    
    # 使用Roboflow API检测器
    if args.roboflow:
        if not args.api_key:
            logger.error("使用Roboflow API检测器需要提供API密钥")
            return
        
        detector = get_detector(
            use_roboflow=True,
            roboflow_api_key=args.api_key,
            roboflow_model_id=args.model_id
        )
    # 使用模拟检测器
    elif args.mock:
        detector = get_detector(use_mock=True)
    # 使用本地YOLO模型
    elif args.model and args.config:
        detector = get_detector(args.model, args.config)
    else:
        logger.error("请指定检测器类型: --mock, --roboflow (需要提供API密钥), 或 --model 和 --config")
        return
    
    result = test_detector(detector, args.image, args.output)
    print(f"检测结果: {result}")

if __name__ == "__main__":
    main() 