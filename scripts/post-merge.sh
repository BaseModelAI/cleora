#!/bin/bash
set -e

pip install flask 2>/dev/null || true

# pyo3 0.30 needs maturin 1.9.4 or newer
pip install "maturin>=1.9.4,<2.0" 2>/dev/null || true

if [ -f "Cargo.toml" ] && command -v maturin &> /dev/null; then
    echo "Rust project detected — rebuilding wheel..."
    maturin build --release -i python3 2>/dev/null || true
    # install the wheel that was just built (older builds may still be in the folder)
    WHEEL=$(ls -t target/wheels/pycleora-*.whl 2>/dev/null | head -1)
    if [ -n "$WHEEL" ]; then
        pip install "$WHEEL" --force-reinstall 2>/dev/null || true
    fi
    # The extension is built for the stable ABI and is called pycleora.abi3.so.
    # Version-specific builds of earlier releases must go: Python would import
    # them first and the Python code no longer matches them.
    SO_SRC=$(find .pythonlibs -name "pycleora.abi3.so" 2>/dev/null | head -1)
    if [ -n "$SO_SRC" ]; then
        rm -f pycleora/pycleora.cpython-*.so 2>/dev/null || true
        cp "$SO_SRC" pycleora/ 2>/dev/null || true
    fi
fi

echo "Post-merge setup complete."
