#!/bin/bash
# ============================================================
# run_inference.sh – OCR inference bằng ONNX Runtime
# ============================================================
# Cách dùng:
#   ./run_inference.sh <image_path> [onnx_dir]
#
# Ví dụ:
#   ./run_inference.sh test_images/test_image.png
#   ./run_inference.sh my_image.png ./onnx_output
# ============================================================

set -e

IMAGE="${1:?Thiếu image path. Cách dùng: $0 <image_path> [onnx_dir]}"
ONNX_DIR="${2:-./onnx_output}"

python vietocr_onnx_inference.py \
    --onnx-dir "$ONNX_DIR" \
    --image "$IMAGE"
