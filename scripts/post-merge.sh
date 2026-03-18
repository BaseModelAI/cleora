#!/bin/bash
set -e

pip install flask 2>/dev/null || true

if [ -f "Cargo.toml" ] && command -v maturin &> /dev/null; then
    echo "Rust project detected — rebuilding wheel..."
    maturin build --release -i python3 2>/dev/null || true
    pip install target/wheels/pycleora-*.whl --force-reinstall 2>/dev/null || true
    SO_SRC=$(find .pythonlibs -name "pycleora.cpython-*.so" 2>/dev/null | head -1)
    if [ -n "$SO_SRC" ]; then
        cp "$SO_SRC" pycleora/ 2>/dev/null || true
    fi
fi

echo "Post-merge setup complete."
