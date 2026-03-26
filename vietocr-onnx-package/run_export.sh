#!/bin/bash
# ============================================================
# run_export.sh – Export VietOCR PyTorch → ONNX
# ============================================================
# Cách dùng:
#   ./run_export.sh <config.yml> <weights.pth> [output_dir]
#
# Ví dụ:
#   ./run_export.sh config.yml vgg_transformer.pth ./onnx_output
# ============================================================

set -e

CONFIG="${1:?Thiếu config.yml. Cách dùng: $0 <config.yml> <weights.pth> [output_dir]}"
WEIGHTS="${2:?Thiếu weights.pth. Cách dùng: $0 <config.yml> <weights.pth> [output_dir]}"
OUTPUT="${3:-./onnx_output}"

echo "================================================"
echo "  VietOCR → ONNX Export"
echo "================================================"
echo "  Config  : $CONFIG"
echo "  Weights : $WEIGHTS"
echo "  Output  : $OUTPUT"
echo "================================================"
echo ""

# Step 1: Patch vietocr backbone
echo "[1/2] Patching vietocr backbone..."
VIETOCR_PATH=$(python -c "import vietocr; import os; print(os.path.dirname(vietocr.__file__))")
cp patches/vgg_patched.py    "$VIETOCR_PATH/model/backbone/vgg.py"
cp patches/resnet_patched.py "$VIETOCR_PATH/model/backbone/resnet.py"
echo "  Patched: $VIETOCR_PATH/model/backbone/"

# Step 2: Export
echo ""
echo "[2/2] Exporting ONNX..."
python vietocr_onnx_export.py \
    --config "$CONFIG" \
    --weights "$WEIGHTS" \
    --output "$OUTPUT"

echo ""
echo "Done! ONNX files saved to: $OUTPUT"
