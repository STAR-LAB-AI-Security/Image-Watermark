#!/usr/bin/env python
"""攻击代码 —— Baseline：A0 组合后处理攻击（JPEG 压缩 + 缩放 + 高斯模糊）。

固定组合（赛题 §11.2，与主办方公开基线一致）：
    JPEG quality = 75
    -> 缩放至原尺寸的 90%（双线性）
    -> 双线性恢复至原尺寸
    -> 轻度 Gaussian Blur, sigma = 0.4

JPEG 步骤需要 Pillow；缺失时降级为"缩放+模糊"（不阻塞，仅削弱攻击）。
全部在内存中处理（BytesIO），不读写图片文件；参数固定（确定性），
不查询检测器、不访问防御代码/权重/密钥（赛题 §8.3 黑盒约束）。

另提供恒等攻击 identity_attack（不修改图像），作为资格对照
AUC(Id, D0) 与 Clean AUC（赛题 §11.3）。

接口（与赛题 §8.2 一致，供评测流水线调用）：
    attack(sample) -> {"image": RGB uint8 H×W×3}

用法：
    python attack.py                    # 读取基准数据，输出 data/attacked.npz
"""


import argparse
import io
import os

import numpy as np


# [IMPORTANT] Replace this line with real attack/defense code.

def identity_attack(sample: dict) -> dict:
    """恒等攻击：不做任何修改（资格对照 / Clean AUC）。"""
    return {"image": np.asarray(sample["image"]).copy()}


# --------------------------------------------------------------------------- #
# 后处理原语
# --------------------------------------------------------------------------- #
def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    try:
        from PIL import Image
    except Exception:
        return image.astype(np.float32)  # 降级：跳过 JPEG
    img = Image.fromarray(image)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    return np.asarray(out, dtype=np.float32)


def _resize_bilinear(image: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """双线性缩放。优先用 Pillow，否则用 numpy 实现。"""
    try:
        from PIL import Image
        img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        out = img.resize((new_w, new_h), Image.BILINEAR)
        return np.asarray(out, dtype=np.float32)
    except Exception:
        return _bilinear_np(image, new_w, new_h)


def _bilinear_np(image: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    src = image.astype(np.float32)
    h, w = src.shape[:2]
    ys = np.clip(np.linspace(0, h - 1, new_h), 0, h - 1)
    xs = np.clip(np.linspace(0, w - 1, new_w), 0, w - 1)
    y0, y1 = np.floor(ys).astype(int), np.ceil(ys).astype(int)
    x0, x1 = np.floor(xs).astype(int), np.ceil(xs).astype(int)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[:, None]
    if src.ndim == 3:
        wy = wy[..., None]
        wx = wx[None, ..., None]
    top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx
    bot = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image.astype(np.float32), sigma=sigma, axes=(0, 1))
    except Exception:
        # 退化：3×3 均值近似
        k = 3
        pad = k // 2
        p = np.pad(image.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        out = np.zeros_like(image.astype(np.float32))
        for dy in range(k):
            for dx in range(k):
                out += p[dy:dy + image.shape[0], dx:dx + image.shape[1]]
        return out / (k * k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A0 组合后处理攻击（Baseline）")
    parser.add_argument("--n", type=int, default=48, help="图像数量")
    parser.add_argument("--output", default="data/attacked.npz", help="输出路径")
    args = parser.parse_args()

    from dataset import load_benchmark, save_npz, to_samples

    ds = load_benchmark(n=args.n)
    samples = to_samples(ds)
    out_images = np.empty_like(ds["images"])
    for i, s in enumerate(samples):
        out_images[i] = attack(s)["image"]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_npz(args.output, out_images, ds["sample_ids"])
    print(f"[attack] A0 攻击完成：{len(samples)} 张 -> {args.output}")
