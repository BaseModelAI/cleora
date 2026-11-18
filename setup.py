import re

from setuptools import setup, find_packages

try:
    from setuptools_rust import RustExtension, Binding
    rust_extensions = [RustExtension("pycleora.pycleora", binding=Binding.PyO3)]
except ImportError:
    rust_extensions = []

# pip builds pycleora with maturin (see pyproject.toml); this file only serves
# `setup.py`-based workflows and mirrors the metadata defined there.
with open("Cargo.toml") as _cargo:
    _version = re.search(r'^version\s*=\s*"([^"]+)"', _cargo.read(), flags=re.MULTILINE).group(1)

setup(
    name="pycleora",
    version=_version,
    description="Fast CPU-only graph embedding library with Rust core",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23",
    ],
    extras_require={
        "scipy": ["scipy>=1.9"],
        "viz": ["matplotlib>=3.5"],
        "networkx": ["networkx>=2.6"],
        "full": ["scipy>=1.9", "matplotlib>=3.5", "networkx>=2.6", "tqdm>=4.60"],
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
