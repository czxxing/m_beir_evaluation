from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="m-beir-evaluation",
    version="0.1.0",
    author="M-BEIR Evaluation Team",
    author_email="example@example.com",
    description="A comprehensive framework for evaluating retrieval models using the M-BEIR benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/m-beir-evaluation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.2",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.5.0",
        "requests>=2.28.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "wandb>=0.15.0",
        "accelerate>=0.20.0",
        "faiss-cpu>=1.7.0",
        "huggingface-hub>=0.15.0",
        "beir>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.0",
            "mypy>=0.900",
        ],
    },
    entry_points={
        "console_scripts": [
            "mbeir-eval=run_evaluation:main",
        ],
    },
)