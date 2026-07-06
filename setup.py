"""
DeepSeek-V4 MindSpore Implementation
"""

from setuptools import setup, find_packages

setup(
    name="deepseek-v4",
    version="1.0.0",
    description="MindSpore implementation of DeepSeek-V4 architecture",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Research",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "mindspore>=2.2.0",
        "numpy>=1.20.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
