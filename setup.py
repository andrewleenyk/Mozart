#!/usr/bin/env python3
"""
Setup script for Audio Feature Extraction Module
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="audio-feature-extractor",
    version="1.0.0",
    author="Audio Feature Extraction Team",
    author_email="your.email@example.com",
    description="A comprehensive Python module for extracting audio features from local audio files",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/audio-feature-extractor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "audio-features=audio_features_simple:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="audio music analysis features librosa machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/audio-feature-extractor/issues",
        "Source": "https://github.com/yourusername/audio-feature-extractor",
        "Documentation": "https://github.com/yourusername/audio-feature-extractor#readme",
    },
) 