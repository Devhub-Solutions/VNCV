#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VietOCR → ONNX Export Script  (v5 – full Dynamo, hoạt động end-to-end)
============================================================
Tách mô hình VietOCR (Transformer hoặc Seq2Seq) thành 2 file ONNX:
  - model_encoder.onnx : CNN backbone + Encoder (hợp nhất)
  - model_decoder.onnx : Decoder

Đồng thời xuất vocab.json giữ nguyên thứ tự ký tự ↔ index từ config gốc.

Chiến lược export (Torch >= 2.9):
  - Encoder: Dynamo exporter → xử lý dynamic shapes tốt cho MultiheadAttention
  - Decoder: Dynamo exporter + layer-by-layer forward (bypass _detect_is_causal_mask)
             Gọi từng DecoderLayer với tgt_is_causal=True để tránh guard error.
             Causal mask dùng arange + comparison (symbolic-friendly).

Cách dùng:
    python vietocr_onnx_export.py \\
        --config  config.yml \\
        --weights weights/transformerocr.pth \\
        --output  onnx_output

Tham khảo:
    - https://github.com/pbcquoc/vietocr
    - https://github.com/pbcquoc/vietocr/issues/41
    - https://github.com/NNDam/vietocr-tensorrt
"""

import os
import sys
import json
import argparse
import math

import torch
import torch.nn as nn
import numpy as np

from vietocr.tool.config import Cfg
from vietocr.model.transformerocr import VietOCR
from vietocr.model.vocab import Vocab


# =====================================================================
#  Helper: detect exporter capability
# =====================================================================

def _has_dynamo_export():
    """Check xem torch.onnx.export hỗ trợ dynamo=True không."""
    import inspect
    sig = inspect.signature(torch.onnx.export)
    return "dynamo" in sig.parameters


def onnx_export_dynamo(model, args, path, **kwargs):
    """Export dùng Dynamo exporter (mặc định cho Torch >= 2.9)."""
    if _has_dynamo_export():
        torch.onnx.export(model, args, path, dynamo=True, **kwargs)
    else:
        torch.onnx.export(model, args, path, **kwargs)


def onnx_export_legacy(model, args, path, **kwargs):
    """Export dùng Legacy TorchScript exporter."""
    if _has_dynamo_export():
        torch.onnx.export(model, args, path, dynamo=False, **kwargs)
    else:
        torch.onnx.export(model, args, path, **kwargs)


# =====================================================================
#  1. Wrapper classes
# =====================================================================

class OCREncoderWrapper(nn.Module):
    """
    CNN + Encoder hợp nhất.

    Input  : img    – (B, 3, H, W) float32
    Output : memory – (T, B, D)    float32
    """

    def __init__(self, vietocr_model):
        super().__init__()
        self.cnn = vietocr_model.cnn
        self.transformer = vietocr_model.transformer

    def forward(self, img):
        src = self.cnn(img)
        memory = self.transformer.forward_encoder(src)
        return memory


class TransformerDecoderWrapper(nn.Module):
    """
    Decoder riêng cho Transformer model.

    Gọi từng layer riêng lẻ với tgt_is_causal=True để tránh
    _detect_is_causal_mask trong TransformerDecoder (gây lỗi
    GuardOnDataDependentSymNode khi export Dynamo).

    Causal mask được tạo bằng arange + comparison (symbolic-friendly).

    Input  : tgt_inp  – (T_tgt, B)   int64
             memory   – (T_src, B, D) float32
    Output : output   – (B, T_tgt, vocab_size) float32
    """

    def __init__(self, language_transformer):
        super().__init__()
        self.embed_tgt = language_transformer.embed_tgt
        self.pos_enc   = language_transformer.pos_enc
        self.layers    = language_transformer.transformer.decoder.layers
        self.norm      = language_transformer.transformer.decoder.norm
        self.fc        = language_transformer.fc
        self.d_model   = language_transformer.d_model

    def forward(self, tgt_inp, memory):
        seq_len = tgt_inp.shape[0]
        # Symbolic-friendly causal mask (no torch.triu, no data-dependent guard)
        rows = torch.arange(seq_len, device=tgt_inp.device).unsqueeze(1)
        cols = torch.arange(seq_len, device=tgt_inp.device).unsqueeze(0)
        tgt_mask = (cols > rows).float() * float("-inf")
        tgt_mask = torch.where(torch.isnan(tgt_mask), torch.zeros_like(tgt_mask), tgt_mask)

        tgt = self.pos_enc(self.embed_tgt(tgt_inp) * math.sqrt(self.d_model))

        # Gọi từng layer với tgt_is_causal=True
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, tgt_is_causal=True)

        if self.norm is not None:
            output = self.norm(output)

        output = output.transpose(0, 1)
        return self.fc(output)


# ── Seq2Seq wrappers ─────────────────────────────────────────────────

class Seq2SeqEncoderWrapper(nn.Module):
    """CNN + GRU Encoder cho Seq2Seq model."""

    def __init__(self, vietocr_model):
        super().__init__()
        self.cnn = vietocr_model.cnn
        self.encoder = vietocr_model.transformer.encoder

    def forward(self, img):
        src = self.cnn(img)
        encoder_outputs, hidden = self.encoder(src)
        return encoder_outputs, hidden


class Seq2SeqDecoderWrapper(nn.Module):
    """Attention + GRU Decoder cho Seq2Seq model."""

    def __init__(self, seq2seq_module):
        super().__init__()
        self.decoder = seq2seq_module.decoder

    def forward(self, tgt, hidden, encoder_outputs):
        output, hidden_out, _ = self.decoder(tgt, hidden, encoder_outputs)
        return output, hidden_out


# =====================================================================
#  2. Build model
# =====================================================================

def build_model(config):
    vocab = Vocab(config["vocab"])
    device = config["device"]
    model = VietOCR(
        len(vocab),
        config["backbone"],
        config["cnn"],
        config["transformer"],
        config["seq_modeling"],
    )
    model = model.to(device)
    return model, vocab


# =====================================================================
#  3. Export ONNX chính
# =====================================================================

def export_onnx(config_path, weights_path, output_dir, opset=18, image_height=32,
                dummy_width=160, max_seq_length=128):
    os.makedirs(output_dir, exist_ok=True)

    # ── Load config ──────────────────────────────────────────────────
    config = Cfg.load_config_from_file(config_path)
    config["device"] = "cpu"

    print(f"[INFO] Loaded config from: {config_path}")
    print(f"[INFO] seq_modeling = {config['seq_modeling']}")
    print(f"[INFO] backbone    = {config['backbone']}")

    # ── Build model & load weights ───────────────────────────────────
    model, vocab = build_model(config)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded weights from: {weights_path}")
    print(f"[INFO] vocab size  = {len(vocab)} (4 special + {len(vocab) - 4} chars)")

    if "dataset" in config and "image_height" in config["dataset"]:
        image_height = config["dataset"]["image_height"]

    is_transformer = config["seq_modeling"] == "transformer"

    # ── Dummy input ──────────────────────────────────────────────────
    dummy_img = torch.randn(1, 3, image_height, dummy_width)

    if is_transformer:
        # =============================================================
        # TRANSFORMER
        # =============================================================

        # ── STEP 1: Export CNN + Encoder (Dynamo) ────────────────────
        print("\n" + "=" * 60)
        print("[STEP 1/2] Exporting CNN + Encoder ...")
        print("=" * 60)

        enc_wrapper = OCREncoderWrapper(model)
        enc_wrapper.eval()

        with torch.no_grad():
            memory = enc_wrapper(dummy_img)
        print(f"  Input  shape: {list(dummy_img.shape)}  (B, C, H, W)")
        print(f"  Output shape: {list(memory.shape)}  (T, B, D)")

        encoder_onnx_path = os.path.join(output_dir, "model_encoder.onnx")

        # Dynamo exporter xử lý dynamic shapes đúng cho MultiheadAttention
        onnx_export_dynamo(
            enc_wrapper,
            dummy_img,
            encoder_onnx_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["memory"],
            dynamic_axes={
                "input":  {0: "batch", 3: "im_width"},
                "memory": {0: "feat_len", 1: "batch"},
            },
        )
        print(f"  -> Saved: {encoder_onnx_path}")

        # ── STEP 2: Export Decoder (Dynamo + layer-by-layer) ─────────
        print("\n" + "=" * 60)
        print("[STEP 2/2] Exporting Decoder ...")
        print("=" * 60)

        dec_wrapper = TransformerDecoderWrapper(model.transformer)
        dec_wrapper.eval()

        dummy_tgt = torch.randint(0, len(vocab), (5, 1))
        dummy_memory = memory.detach()

        with torch.no_grad():
            dec_out = dec_wrapper(dummy_tgt, dummy_memory)
        print(f"  tgt_inp shape: {list(dummy_tgt.shape)}  (T_tgt, B)")
        print(f"  memory  shape: {list(dummy_memory.shape)}  (T_src, B, D)")
        print(f"  output  shape: {list(dec_out.shape)}  (B, T_tgt, vocab_size)")

        decoder_onnx_path = os.path.join(output_dir, "model_decoder.onnx")

        # Dynamo exporter cho decoder (layer-by-layer, tgt_is_causal=True)
        onnx_export_dynamo(
            dec_wrapper,
            (dummy_tgt, dummy_memory),
            decoder_onnx_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["tgt_inp", "memory"],
            output_names=["output"],
            dynamic_axes={
                "tgt_inp": {0: "seq_len", 1: "batch"},
                "memory":  {0: "feat_len", 1: "batch"},
                "output":  {0: "batch", 1: "seq_len"},
            },
        )
        print(f"  -> Saved: {decoder_onnx_path}")

    else:
        # =============================================================
        # SEQ2SEQ
        # =============================================================

        print("\n" + "=" * 60)
        print("[STEP 1/2] Exporting CNN + Seq2Seq Encoder ...")
        print("=" * 60)

        enc_wrapper = Seq2SeqEncoderWrapper(model)
        enc_wrapper.eval()

        with torch.no_grad():
            enc_outputs, enc_hidden = enc_wrapper(dummy_img)

        encoder_onnx_path = os.path.join(output_dir, "model_encoder.onnx")
        onnx_export_legacy(
            enc_wrapper,
            dummy_img,
            encoder_onnx_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["encoder_outputs", "hidden"],
            dynamic_axes={
                "input":           {0: "batch", 3: "im_width"},
                "encoder_outputs": {0: "feat_len", 1: "batch"},
            },
        )
        print(f"  -> Saved: {encoder_onnx_path}")

        print("\n" + "=" * 60)
        print("[STEP 2/2] Exporting Seq2Seq Decoder ...")
        print("=" * 60)

        dec_wrapper = Seq2SeqDecoderWrapper(model.transformer)
        dec_wrapper.eval()

        tgt_dummy = torch.randint(0, len(vocab), (1,))
        decoder_onnx_path = os.path.join(output_dir, "model_decoder.onnx")
        onnx_export_legacy(
            dec_wrapper,
            (tgt_dummy, enc_hidden.detach(), enc_outputs.detach()),
            decoder_onnx_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["tgt", "hidden", "encoder_outputs"],
            output_names=["prediction", "hidden_out"],
            dynamic_axes={
                "encoder_outputs": {0: "feat_len", 1: "batch"},
            },
        )
        print(f"  -> Saved: {decoder_onnx_path}")

    # ── 4. Export vocab.json ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[VOCAB] Exporting vocab.json ...")
    print("=" * 60)

    vocab_path = os.path.join(output_dir, "vocab.json")
    chars_raw = config["vocab"]
    char_list = list(chars_raw)

    vocab_data = {
        "description": (
            "VietOCR vocab mapping. "
            "Special tokens: 0=<pad>, 1=<sos>, 2=<eos>, 3=<mask>. "
            "Actual chars start at index 4."
        ),
        "special_tokens": {"0": "<pad>", "1": "<sos>", "2": "<eos>", "3": "<mask>"},
        "chars": char_list,
        "char2idx": {c: i + 4 for i, c in enumerate(char_list)},
        "idx2char": {str(i + 4): c for i, c in enumerate(char_list)},
        "total_vocab_size": len(char_list) + 4,
    }

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    print(f"  -> Saved: {vocab_path}")
    print(f"  Total vocab size: {vocab_data['total_vocab_size']}")

    # ── Tóm tắt ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPORT HOAN TAT!")
    print("=" * 60)
    for f_path in [encoder_onnx_path, decoder_onnx_path, vocab_path]:
        size_mb = os.path.getsize(f_path) / 1024 / 1024
        print(f"  {f_path}  ({size_mb:.1f} MB)")
    print()


# =====================================================================
#  CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export VietOCR model to ONNX (Encoder + Decoder)"
    )
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--weights", "-w", required=True)
    parser.add_argument("--output", "-o", default="./onnx_output")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--max-seq-length", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_onnx(
        config_path=args.config,
        weights_path=args.weights,
        output_dir=args.output,
        opset=args.opset,
        max_seq_length=args.max_seq_length,
    )
