#!/usr/bin/env python3
"""
Command Line Interface for Mahjong Hand Recognition
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mahjong_recognition.tile_recognition import TileRecognizer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Mahjong Hand Recognition CLI")
    
    parser.add_argument("--image", "-i", type=str, help="Path to input image")
    parser.add_argument("--folder", "-f", type=str, help="Path to folder containing images")
    parser.add_argument("--model", "-m", type=str, help="Path to model file")
    parser.add_argument("--confidence", "-c", type=float, default=0.7, 
                        help="Minimum confidence threshold (default: 0.7)")
    parser.add_argument("--train", "-t", action="store_true", 
                        help="Train the model with custom data")
    parser.add_argument("--data", "-d", type=str, 
                        help="Path to training data directory (required if --train is specified)")
    parser.add_argument("--epochs", "-e", type=int, default=10, 
                        help="Number of epochs for training (default: 10)")
    parser.add_argument("--batch-size", "-b", type=int, default=32, 
                        help="Batch size for training (default: 32)")
    
    return parser.parse_args()


def process_single_image(image_path: str, model_path: Optional[str], confidence: float):
    """Process a single image"""
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found")
        return
    
    recognizer = TileRecognizer(model_path)
    formatted_result, detailed_results = recognizer.recognize_hand(image_path, confidence)
    
    print(f"\nResults for {image_path}:")
    print(f"Formatted result: {formatted_result}")
    print("\nDetailed results:")
    for tile, conf in detailed_results:
        print(f"  {tile}: {conf:.4f}")


def process_folder(folder_path: str, model_path: Optional[str], confidence: float):
    """Process all images in a folder"""
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found")
        return
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Process all images
    recognizer = TileRecognizer(model_path)
    results = recognizer.batch_process(image_paths, confidence)
    
    # Print results
    for path, (formatted_result, _) in results.items():
        print(f"{path}: {formatted_result}")


def train_model(data_dir: str, model_path: Optional[str], epochs: int, batch_size: int):
    """Train the model with custom data"""
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found")
        return
    
    recognizer = TileRecognizer(model_path)
    recognizer.train(data_dir, epochs, batch_size)


def main():
    """Main function"""
    args = parse_args()
    
    # Training mode
    if args.train:
        if not args.data:
            print("Error: Training data directory (--data) is required when using --train")
            return
        train_model(args.data, args.model, args.epochs, args.batch_size)
        return
    
    # Recognition mode
    if args.image:
        process_single_image(args.image, args.model, args.confidence)
    elif args.folder:
        process_folder(args.folder, args.model, args.confidence)
    else:
        print("Error: Either --image or --folder must be specified")


if __name__ == "__main__":
    main() 