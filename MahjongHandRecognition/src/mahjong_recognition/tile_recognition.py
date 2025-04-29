"""
Mahjong Tile Recognition Module
用于识别麻将牌的模块，集成了图像预处理、分割和识别功能
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path

from .utils import preprocess_image, segment_tiles, is_mahjong_image, format_result


class TileRecognizer:
    """
    Mahjong Tile Recognizer class
    用于识别麻将牌的类，使用预训练的CNN模型
    """
    
    # 定义麻将牌类别
    TILE_CLASSES = [
        '1万', '2万', '3万', '4万', '5万', '6万', '7万', '8万', '9万',
        '1筒', '2筒', '3筒', '4筒', '5筒', '6筒', '7筒', '8筒', '9筒',
        '1索', '2索', '3索', '4索', '5索', '6索', '7索', '8索', '9索',
        '东', '南', '西', '北', '白', '发', '中'
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the recognizer with a model
        
        Args:
            model_path: Path to the model file. If None, use the default model.
        """
        if model_path is None:
            # Use default model path
            base_dir = Path(__file__).parent.parent.parent
            model_path = str(base_dir / 'models' / 'mahjong_classifier_model.h5')
        
        self.model_path = model_path
        self.model = self._load_model()
        self.input_shape = (64, 64, 3)  # Default input shape for the model
        
    def _load_model(self) -> tf.keras.Model:
        """
        Load the model from the specified path
        
        Returns:
            Loaded TensorFlow model
        """
        try:
            model = tf.keras.models.load_model(self.model_path)
            return model
        except Exception as e:
            # If model loading fails, create a basic model
            print(f"Failed to load model from {self.model_path}: {e}")
            print("Creating a basic model instead. Please train it before use.")
            return self._create_basic_model()
            
    def _create_basic_model(self) -> tf.keras.Model:
        """
        Create a basic CNN model for tile recognition
        
        Returns:
            Basic CNN model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.TILE_CLASSES), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def recognize_tile(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize a single tile image
        
        Args:
            image: Preprocessed image of a single tile
            
        Returns:
            Tuple of (recognized tile class, confidence)
        """
        # Ensure image has correct shape
        if image.shape != self.input_shape:
            image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        
        # Normalize if needed
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
            
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        # Get predictions
        predictions = self.model.predict(image_batch, verbose=0)
        
        # Get the highest probability class and its confidence
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        return self.TILE_CLASSES[class_idx], float(confidence)
    
    def recognize_hand(self, image_path: str, min_confidence: float = 0.7) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Recognize all tiles in a hand image
        
        Args:
            image_path: Path to the image containing mahjong tiles
            min_confidence: Minimum confidence threshold for recognition
            
        Returns:
            Tuple of (formatted result string, list of (tile, confidence) tuples)
        """
        # Load and preprocess the image
        img = preprocess_image(image_path, target_size=(800, 600))
        
        # Check if it's a mahjong image
        if not is_mahjong_image(img):
            return "Not a mahjong image", []
        
        # Segment tiles from the image
        tile_images = segment_tiles(img)
        
        if not tile_images:
            return "No tiles found", []
        
        # Recognize each tile
        results = []
        for tile_img in tile_images:
            tile_class, confidence = self.recognize_tile(tile_img)
            if confidence >= min_confidence:
                results.append((tile_class, confidence))
        
        # Format results
        tile_labels = [tile for tile, _ in results]
        formatted_result = format_result(tile_labels)
        
        return formatted_result, results
    
    def batch_process(self, image_paths: List[str], min_confidence: float = 0.7) -> Dict[str, Tuple[str, List[Tuple[str, float]]]]:
        """
        Process multiple images
        
        Args:
            image_paths: List of paths to images containing mahjong tiles
            min_confidence: Minimum confidence threshold for recognition
            
        Returns:
            Dictionary mapping image paths to their recognition results
        """
        results = {}
        for image_path in image_paths:
            try:
                result = self.recognize_hand(image_path, min_confidence)
                results[image_path] = result
            except Exception as e:
                results[image_path] = (f"Error: {str(e)}", [])
        
        return results
    
    def train(self, 
              train_data_dir: str, 
              epochs: int = 10, 
              batch_size: int = 32, 
              validation_split: float = 0.2) -> None:
        """
        Train the model on custom data
        
        Args:
            train_data_dir: Directory containing training data
            epochs: Number of epochs to train
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
        """
        # Image data generator with augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,
            validation_split=validation_split
        )
        
        # Training generator
        train_generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='sparse',
            subset='training'
        )
        
        # Validation generator
        validation_generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='sparse',
            subset='validation'
        )
        
        # Train the model
        self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator
        )
        
        # Save the trained model
        model_dir = os.path.dirname(self.model_path)
        os.makedirs(model_dir, exist_ok=True)
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}") 