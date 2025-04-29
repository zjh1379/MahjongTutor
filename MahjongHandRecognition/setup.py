"""
Setup script for Mahjong Hand Recognition
"""

from setuptools import setup, find_packages

setup(
    name="MahjongHandRecognition",
    version="0.1.0",
    author="MahjongTutors Team",
    author_email="example@example.com",
    description="A package for recognizing and converting mahjong tile images to text",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/MahjongHandRecognition",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "tensorflow>=2.8.0",
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "requests>=2.25.0",
    ],
    entry_points={
        "console_scripts": [
            "mahjong-recognition=MahjongHandRecognition.src.mahjong_recognition_cli:main",
            "mahjong-download-model=MahjongHandRecognition.src.download_model:main",
        ],
    },
) 