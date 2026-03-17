from setuptools import setup, find_packages

try:
    from setuptools_rust import RustExtension, Binding
    rust_extensions = [RustExtension("pycleora.pycleora", binding=Binding.PyO3)]
except ImportError:
    rust_extensions = []

setup(
    name="pycleora",
    version="3.0.0",
    description="Fast CPU-only graph embedding library with Rust core",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
    ],
    extras_require={
        "viz": ["matplotlib>=3.5"],
        "networkx": ["networkx>=2.6"],
        "full": ["matplotlib>=3.5", "networkx>=2.6", "tqdm>=4.60"],
    },
    entry_points={
        "console_scripts": [
            "pycleora=pycleora.cli:main",
        ],
    },
    rust_extensions=rust_extensions,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Rust",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
