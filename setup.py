from setuptools import setup, find_packages

setup(
    name="liquid_foundation_model",
    version="0.1.0",
    description="Liquid Foundation Model - Efficient on-device foundation model",
    author="LFM Team",
    author_email="info@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "peft>=0.4.0",
        "accelerate>=0.20.0",
        "sentencepiece>=0.1.99",
        "tensorboard>=2.12.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "quantization": [
            "bitsandbytes>=0.39.0",
            "optimum>=1.8.0",
        ],
        "export": [
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)