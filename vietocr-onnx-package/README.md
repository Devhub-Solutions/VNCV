# VietOCR → ONNX Export & Inference

Export mô hình [VietOCR](https://github.com/pbcquoc/vietocr) (PyTorch) sang ONNX để chạy inference
mà **không cần PyTorch** — chỉ cần `onnxruntime`, nhẹ ~200MB thay vì ~2GB.

## Cấu trúc thư mục

```
vietocr-onnx-package/
├── README.md                       # File này
├── vietocr_onnx_export.py          # Script export PyTorch → ONNX (v5)
├── vietocr_onnx_inference.py       # Script inference ONNX Runtime (v4)
├── requirements-export.txt         # Dependencies cho export
├── requirements-inference.txt      # Dependencies cho inference (nhẹ)
├── Dockerfile.export               # Docker image để export
├── Dockerfile.inference            # Docker image để inference (nhẹ)
├── docker-compose.yml              # Docker Compose cho cả 2
├── test_config.yml                 # Config mẫu (vgg_transformer)
├── patches/
│   ├── vgg_patched.py              # VGG backbone đã patch cho ONNX
│   └── resnet_patched.py           # ResNet backbone đã patch cho ONNX
├── onnx_output/                    # File ONNX đã export sẵn
│   ├── model_encoder.onnx          # CNN + Transformer Encoder
│   ├── model_encoder.onnx.data     # Weights encoder (~108MB)
│   ├── model_decoder.onnx          # Transformer Decoder
│   ├── model_decoder.onnx.data     # Weights decoder (~38MB)
│   └── vocab.json                  # Vocab mapping (giữ nguyên thứ tự gốc)
└── test_images/
    ├── test_image.png              # "Xin chao"
    └── test_image_vn.png           # "Việt Nam"
```

## Kết quả test đã xác minh

| Ảnh             | PyTorch          | ONNX Runtime     | Khớp |
|-----------------|------------------|-------------------|------|
| test_image.png  | "Xin chao" 0.920 | "Xin chao" 0.920 | ✓    |
| test_image_vn.png | "Việt Nam" 0.931 | "Việt Nam" 0.931 | ✓    |

---

## 1. Inference nhanh (không cần PyTorch)

### Cài đặt
```bash
pip install onnxruntime==1.24.4 Pillow==12.1.1 numpy==2.3.5
```

### Chạy
```bash
python vietocr_onnx_inference.py \
    --onnx-dir onnx_output \
    --image path/to/your/image.png
```

### Dùng trong Python
```python
from PIL import Image
from vietocr_onnx_inference import VietOCROnnxEngine

engine = VietOCROnnxEngine(onnx_dir="onnx_output", seq_modeling="transformer")

image = Image.open("test.png")
text, prob = engine.predict(image)
print(f"OCR: {text} (confidence: {prob:.4f})")
```

---

## 2. Export ONNX từ weights của bạn

### Cài đặt (cần PyTorch)
```bash
# PyTorch CPU
pip install torch==2.11.0+cpu torchvision==0.26.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Các dependencies khác
pip install onnx==1.20.1 onnxscript==0.6.2 onnxruntime==1.24.4 \
    einops==0.8.2 vietocr==0.3.13 Pillow==12.1.1 numpy==2.3.5
```

### Patch vietocr backbone (BẮT BUỘC trước khi export)
```bash
VIETOCR_PATH=$(python -c "import vietocr; import os; print(os.path.dirname(vietocr.__file__))")
cp patches/vgg_patched.py    "$VIETOCR_PATH/model/backbone/vgg.py"
cp patches/resnet_patched.py "$VIETOCR_PATH/model/backbone/resnet.py"
```

### Export
```bash
python vietocr_onnx_export.py \
    --config  your_config.yml \
    --weights your_weights.pth \
    --output  onnx_output
```

### So sánh ONNX vs PyTorch
```bash
python vietocr_onnx_inference.py \
    --onnx-dir onnx_output \
    --image test_images/test_image.png \
    --config your_config.yml \
    --weights your_weights.pth
```

---

## 3. Docker

### Build & Export
```bash
# Build image export
docker build -f Dockerfile.export -t vietocr-onnx-export .

# Export (mount thư mục chứa config + weights)
docker run --rm \
    -v $(pwd)/weights:/data \
    -v $(pwd)/onnx_output:/output \
    vietocr-onnx-export \
    --config /data/config.yml \
    --weights /data/weights.pth \
    --output /output
```

### Build & Inference
```bash
# Build image inference (nhẹ hơn nhiều)
docker build -f Dockerfile.inference -t vietocr-onnx-inference .

# Inference
docker run --rm \
    -v $(pwd)/onnx_output:/models \
    -v $(pwd)/test_images:/images \
    vietocr-onnx-inference \
    --onnx-dir /models \
    --image /images/test_image.png
```

### Docker Compose
```bash
# Chuẩn bị: đặt config.yml + weights.pth vào ./weights/
# Export
docker compose run export

# Inference
docker compose run inference --onnx-dir /models --image /images/test_image.png

# Export + Test so sánh
docker compose run export-and-test
```

---

## 4. Chi tiết kỹ thuật

### Kiến trúc ONNX

```
Ảnh (B,3,H,W)
     │
     ▼
┌─────────────────────────┐
│   model_encoder.onnx    │  CNN (VGG19-BN) + Transformer Encoder
│   Input:  (1,3,32,W)    │  W = dynamic (chiều rộng ảnh)
│   Output: (T,1,256)     │  T = 2*(W//4), phụ thuộc pooling
└─────────────────────────┘
     │ memory
     ▼
┌─────────────────────────┐
│   model_decoder.onnx    │  Transformer Decoder (6 layers)
│   Input:  tgt_inp (S,1) │  S = dynamic (seq length, tăng dần)
│           memory  (T,1,256)
│   Output: (1,S,233)     │  233 = vocab size
└─────────────────────────┘
     │ logits
     ▼
  Greedy decode → Text
```

### Các vấn đề đã giải quyết

| # | Vấn đề | Nguyên nhân | Giải pháp |
|---|--------|-------------|-----------|
| 1 | `GuardOnDataDependentSymNode` | `_detect_is_causal_mask` trong Transformer Decoder gọi `.item()` | Gọi từng DecoderLayer riêng lẻ với `tgt_is_causal=True`, bypass TransformerDecoder |
| 2 | Reshape `{1,1,256} → {5,8,32}` | Legacy TorchScript exporter bake cứng shape từ dummy input | Dùng Dynamo exporter cho cả Encoder và Decoder |
| 3 | ONNX negative index trong Transpose | `permute(-1,0,1)` không hợp lệ trong ONNX | Patch vgg.py/resnet.py dùng index dương: `permute(2,0,1)` |
| 4 | `scaled_dot_product_attention` | Cần opset ≥ 14 | Dùng opset 18 (Dynamo yêu cầu) |
| 5 | Reshape trong MHA khi width khác | CNN + Encoder tách riêng → MHA reshape bake cứng | Hợp nhất CNN + Encoder thành 1 ONNX |

### Chiến lược export (v5)

- **Cả Encoder và Decoder**: Dynamo exporter (`torch.onnx.export(..., dynamo=True)`)
- **Decoder wrapper**: Gọi từng `TransformerDecoderLayer` với `tgt_is_causal=True`
- **Causal mask**: Tạo bằng `torch.arange` + comparison (symbolic-friendly)
- **Opset**: 18 (tự động bởi Dynamo)
- **Batch size**: 1 (chuẩn cho OCR inference)

### Versions đã test thành công

| Package | Version | Ghi chú |
|---------|---------|---------|
| Python | 3.13 | |
| torch | 2.11.0+cpu | Cần ≥ 2.9 cho Dynamo ONNX exporter |
| torchvision | 0.26.0+cpu | |
| onnx | 1.20.1 | |
| onnxscript | 0.6.2 | Cần cho Dynamo exporter |
| onnxruntime | 1.24.4 | Chỉ cần cái này cho inference |
| einops | 0.8.2 | Dependency của vietocr |
| vietocr | 0.3.13 | Cần patch backbone trước export |
| Pillow | 12.1.1 | |
| numpy | 2.3.5 | |

### Tham khảo

- [VietOCR](https://github.com/pbcquoc/vietocr)
- [VietOCR Issue #41](https://github.com/pbcquoc/vietocr/issues/41)
- [vietocr-tensorrt (NNDam)](https://github.com/NNDam/vietocr-tensorrt)
