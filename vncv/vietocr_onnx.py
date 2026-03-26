#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VietOCR ONNX Inference Example  (v4)
======================================
Chạy inference OCR bằng ONNX Runtime, sử dụng 2 file ONNX đã export
(model_encoder.onnx, model_decoder.onnx) cùng vocab.json.

Pipeline giống hệt VietOCR gốc:
    Ảnh PIL → Resize + Normalize → Encoder(CNN+Enc) → Decoder (greedy) → Text

Cách dùng:
    python vietocr_onnx_inference.py \\
        --onnx-dir  onnx_output \\
        --image     test.png \\
        [--max-seq-length 128] \\
        [--seq-modeling transformer]

Tham khảo:
    - https://github.com/pbcquoc/vietocr
    - https://github.com/NNDam/vietocr-tensorrt
"""

import os
import sys
import json
import math
import argparse
import time

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime chưa được cài. Hãy chạy: pip install onnxruntime")
    sys.exit(1)


# =====================================================================
#  1. Tiền xử lý ảnh – giống hệt VietOCR gốc
# =====================================================================

def resize(w, h, expected_height, image_min_width, image_max_width):
    """
    Tính lại chiều rộng ảnh, giữ tỷ lệ, round lên bội số 10.
    Giống hệt vietocr.tool.translate.resize().
    """
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)
    return new_w, expected_height


def process_image(image, image_height=32, image_min_width=32, image_max_width=512):
    """
    Tiền xử lý ảnh PIL → numpy array (C, H, W) float32 [0, 1].
    Giống hệt vietocr.tool.translate.process_image().
    """
    img = image.convert("RGB")
    w, h = img.size
    new_w, new_h = resize(w, h, image_height, image_min_width, image_max_width)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    img = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)  # (C, H, W)
    img = img / 255.0
    return img


def process_input(image, image_height=32, image_min_width=32, image_max_width=512):
    """
    Tiền xử lý ảnh PIL → numpy array (1, C, H, W) sẵn sàng cho ONNX.
    """
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]  # (1, C, H, W)
    return img


# =====================================================================
#  2. Load vocab từ vocab.json
# =====================================================================

class VocabONNX:
    """
    Vocab class cho ONNX inference – đọc từ vocab.json đã export.
    Giữ nguyên logic decode giống vietocr.model.vocab.Vocab.
    """

    PAD = 0
    SOS = 1
    EOS = 2
    MASK = 3

    def __init__(self, vocab_json_path):
        with open(vocab_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chars = data["chars"]
        self.total_size = data["total_vocab_size"]

        self.c2i = {c: i + 4 for i, c in enumerate(self.chars)}
        self.i2c = {i + 4: c for i, c in enumerate(self.chars)}
        self.i2c[0] = "<pad>"
        self.i2c[1] = "<sos>"
        self.i2c[2] = "<eos>"
        self.i2c[3] = "<mask>"

    def decode(self, ids):
        """Decode token ids → string. Bỏ <sos>, dừng tại <eos>."""
        first = 1 if self.SOS in ids else 0
        last = ids.index(self.EOS) if self.EOS in ids else None
        sent = "".join([self.i2c.get(i, "") for i in ids[first:last]])
        return sent

    def batch_decode(self, arr):
        return [self.decode(ids) for ids in arr]

    def __len__(self):
        return self.total_size


# =====================================================================
#  3. ONNX Runtime Inference Engine
# =====================================================================

class VietOCROnnxEngine:
    """
    Inference engine sử dụng ONNX Runtime cho VietOCR.
    Hỗ trợ cả Transformer và Seq2Seq.

    Shapes (Transformer):
    ─────────────────────
    Encoder (CNN+TransformerEncoder merged):
        Input  : input   – (B, 3, H, W)       float32
        Output : memory  – (T, B, D)           float32
                 T = f(W), phụ thuộc backbone pooling

    Decoder:
        Input  : tgt_inp – (T_tgt, B)          int64
                 memory  – (T, B, D)            float32
        Output : output  – (B, T_tgt, V)       float32
                 V = vocab_size

    Shapes (Seq2Seq):
    ─────────────────
    Encoder (CNN+GRU merged):
        Input  : input           – (B, 3, H, W)     float32
        Output : encoder_outputs – (T, B, enc_hid*2) float32
                 hidden          – (B, dec_hid)       float32

    Decoder:
        Input  : tgt             – (B,)              int64
                 hidden          – (B, dec_hid)       float32
                 encoder_outputs – (T, B, enc_hid*2)  float32
        Output : prediction      – (B, V)            float32
                 hidden_out      – (B, dec_hid)       float32
    """

    def __init__(self, onnx_dir, vocab_json_path=None, seq_modeling="transformer"):
        self.seq_modeling = seq_modeling

        enc_path = os.path.join(onnx_dir, "model_encoder.onnx")
        dec_path = os.path.join(onnx_dir, "model_decoder.onnx")

        if vocab_json_path is None:
            vocab_json_path = os.path.join(onnx_dir, "vocab.json")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        print(f"[INFO] Loading Encoder: {enc_path}")
        self.enc_sess = ort.InferenceSession(enc_path, sess_opts)

        print(f"[INFO] Loading Decoder: {dec_path}")
        self.dec_sess = ort.InferenceSession(dec_path, sess_opts)

        print(f"[INFO] Loading Vocab:   {vocab_json_path}")
        self.vocab = VocabONNX(vocab_json_path)

        self._print_io_info()

    def _print_io_info(self):
        print("\n[IO Info]")
        for name, sess in [("Encoder", self.enc_sess), ("Decoder", self.dec_sess)]:
            inputs = [(i.name, i.shape, i.type) for i in sess.get_inputs()]
            outputs = [(o.name, o.shape, o.type) for o in sess.get_outputs()]
            print(f"  {name}:")
            print(f"    Inputs : {inputs}")
            print(f"    Outputs: {outputs}")
        print()

    def run_encoder(self, img_np):
        """
        Chạy Encoder (CNN + Enc merged).
        img_np: (B, 3, H, W) float32
        """
        if self.seq_modeling == "transformer":
            result = self.enc_sess.run(None, {"input": img_np})
            return result[0]  # memory: (T, B, D)
        else:
            result = self.enc_sess.run(None, {"input": img_np})
            return result[0], result[1]  # encoder_outputs, hidden

    def run_decoder_transformer(self, tgt_inp, memory):
        """
        Chạy Transformer Decoder.
        tgt_inp: (T_tgt, B) int64
        memory : (T, B, D) float32
        Returns: output – (B, T_tgt, V) float32
        """
        result = self.dec_sess.run(None, {
            "tgt_inp": tgt_inp,
            "memory":  memory,
        })
        return result[0]

    def run_decoder_seq2seq(self, tgt, hidden, encoder_outputs):
        """
        Chạy Seq2Seq Decoder (1 step).
        tgt             : (B,) int64
        hidden          : (B, dec_hid) float32
        encoder_outputs : (T, B, enc_hid*2) float32
        Returns: prediction (B, V), hidden_out (B, dec_hid)
        """
        result = self.dec_sess.run(None, {
            "tgt": tgt,
            "hidden": hidden,
            "encoder_outputs": encoder_outputs,
        })
        return result[0], result[1]

    # ── Greedy decode (Transformer) ──────────────────────────────────

    def predict_transformer(self, img_np, max_seq_length=128):
        """
        Greedy decoding cho Transformer model.
        Giống hệt logic trong vietocr.tool.translate.translate().
        """
        sos_token = VocabONNX.SOS
        eos_token = VocabONNX.EOS
        batch_size = img_np.shape[0]

        # Encoder (CNN + TransformerEncoder)
        memory = self.run_encoder(img_np)

        # Greedy decode
        translated = [[sos_token] * batch_size]
        char_probs = [[1.0] * batch_size]

        max_length = 0
        while max_length <= max_seq_length:
            trans_arr = np.asarray(translated).T
            if all(np.any(trans_arr == eos_token, axis=1)):
                break

            tgt_inp = np.array(translated, dtype=np.int64)  # (T_tgt, B)

            output = self.run_decoder_transformer(tgt_inp, memory)
            # output: (B, T_tgt, V)

            # Softmax
            output_exp = np.exp(output - np.max(output, axis=-1, keepdims=True))
            output_prob = output_exp / np.sum(output_exp, axis=-1, keepdims=True)

            last_step = output_prob[:, -1, :]  # (B, V)
            indices = np.argmax(last_step, axis=-1).tolist()
            values = [last_step[b, indices[b]] for b in range(batch_size)]

            char_probs.append(values)
            translated.append(indices)
            max_length += 1

        translated = np.asarray(translated).T
        char_probs_arr = np.asarray(char_probs).T
        char_probs_arr = np.multiply(char_probs_arr, translated > 3)
        valid_counts = np.maximum((char_probs_arr > 0).sum(-1), 1)
        avg_probs = np.sum(char_probs_arr, axis=-1) / valid_counts

        texts = self.vocab.batch_decode(translated.tolist())
        return texts, avg_probs.tolist()

    # ── Greedy decode (Seq2Seq) ──────────────────────────────────────

    def predict_seq2seq(self, img_np, max_seq_length=128):
        """Greedy decoding cho Seq2Seq model."""
        sos_token = VocabONNX.SOS
        eos_token = VocabONNX.EOS
        batch_size = img_np.shape[0]

        encoder_outputs, hidden = self.run_encoder(img_np)

        translated = [[sos_token] * batch_size]
        char_probs = [[1.0] * batch_size]
        tgt = np.array([sos_token] * batch_size, dtype=np.int64)

        for step in range(max_seq_length):
            prediction, hidden = self.run_decoder_seq2seq(tgt, hidden, encoder_outputs)

            pred_exp = np.exp(prediction - np.max(prediction, axis=-1, keepdims=True))
            pred_prob = pred_exp / np.sum(pred_exp, axis=-1, keepdims=True)

            indices = np.argmax(pred_prob, axis=-1)
            values = [pred_prob[b, indices[b]] for b in range(batch_size)]

            char_probs.append(values)
            translated.append(indices.tolist())
            tgt = indices.astype(np.int64)

            trans_arr = np.asarray(translated).T
            if all(np.any(trans_arr == eos_token, axis=1)):
                break

        translated = np.asarray(translated).T
        char_probs_arr = np.asarray(char_probs).T
        char_probs_arr = np.multiply(char_probs_arr, translated > 3)
        valid_counts = np.maximum((char_probs_arr > 0).sum(-1), 1)
        avg_probs = np.sum(char_probs_arr, axis=-1) / valid_counts

        texts = self.vocab.batch_decode(translated.tolist())
        return texts, avg_probs.tolist()

    # ── Unified predict ──────────────────────────────────────────────

    def predict(self, image, image_height=32, image_min_width=32,
                image_max_width=512, max_seq_length=128):
        """
        Predict text từ ảnh PIL.

        Returns: (text: str, prob: float)
        """
        img_np = process_input(image, image_height, image_min_width, image_max_width)

        if self.seq_modeling == "transformer":
            texts, probs = self.predict_transformer(img_np, max_seq_length)
        else:
            texts, probs = self.predict_seq2seq(img_np, max_seq_length)

        return texts[0], probs[0]


# =====================================================================
#  4. So sánh kết quả ONNX vs PyTorch gốc
# =====================================================================

def compare_with_pytorch(onnx_dir, image_path, config_path, weights_path):
    """
    So sánh kết quả giữa ONNX Runtime và PyTorch gốc.
    Chỉ chạy khi có đủ dependencies (torch, vietocr).
    """
    try:
        import torch
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
    except ImportError:
        print("[SKIP] Không thể so sánh – thiếu torch/vietocr")
        return

    print("\n" + "=" * 60)
    print("[COMPARE] ONNX vs PyTorch")
    print("=" * 60)

    # PyTorch
    config = Cfg.load_config_from_file(config_path)
    config["device"] = "cpu"
    config["weights"] = weights_path
    config["predictor"]["beamsearch"] = False

    predictor = Predictor(config)
    image = Image.open(image_path)

    with torch.no_grad():
        pt_text, pt_prob = predictor.predict(image, return_prob=True)

    # ONNX
    engine = VietOCROnnxEngine(
        onnx_dir=onnx_dir,
        seq_modeling=config["seq_modeling"],
    )
    onnx_text, onnx_prob = engine.predict(
        image,
        image_height=config["dataset"]["image_height"],
        image_min_width=config["dataset"]["image_min_width"],
        image_max_width=config["dataset"]["image_max_width"],
    )

    print(f"  PyTorch: '{pt_text}' (prob={pt_prob:.4f})")
    print(f"  ONNX   : '{onnx_text}' (prob={onnx_prob:.4f})")
    print(f"  Match  : {'YES' if pt_text == onnx_text else 'NO'}")


# =====================================================================
#  5. CLI entry-point
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="VietOCR ONNX Inference")
    parser.add_argument("--onnx-dir", required=True,
                        help="Thư mục chứa model_encoder.onnx, model_decoder.onnx, vocab.json")
    parser.add_argument("--image", required=True, help="Ảnh cần OCR")
    parser.add_argument("--seq-modeling", default="transformer",
                        choices=["transformer", "seq2seq"])
    parser.add_argument("--image-height", type=int, default=32)
    parser.add_argument("--image-min-width", type=int, default=32)
    parser.add_argument("--image-max-width", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--config", default=None,
                        help="config.yml để so sánh với PyTorch (optional)")
    parser.add_argument("--weights", default=None,
                        help="weights .pth để so sánh với PyTorch (optional)")
    return parser.parse_args()


def main():
    args = parse_args()

    engine = VietOCROnnxEngine(
        onnx_dir=args.onnx_dir,
        seq_modeling=args.seq_modeling,
    )

    image = Image.open(args.image)
    print(f"\n[INFO] Image: {args.image} ({image.size[0]}x{image.size[1]})")

    t0 = time.time()
    text, prob = engine.predict(
        image,
        image_height=args.image_height,
        image_min_width=args.image_min_width,
        image_max_width=args.image_max_width,
        max_seq_length=args.max_seq_length,
    )
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"  Ket qua OCR: {text}")
    print(f"  Confidence : {prob:.4f}")
    print(f"  Thoi gian  : {elapsed:.3f}s")
    print(f"{'=' * 60}")

    # So sánh nếu có config + weights
    if args.config and args.weights:
        compare_with_pytorch(args.onnx_dir, args.image, args.config, args.weights)


if __name__ == "__main__":
    main()
