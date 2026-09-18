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

from __future__ import annotations

import argparse
import io
import os

import numpy as np


# -*- coding: utf-8 -*-
"""attack.py —— 攻击方 v3：系数域擦除方案


"""

import numpy as np

CFG = {
    "psnr_min": 29.0,        # 质控红线 28 dB，留 1 dB 余量
    "q_init": 24.0,          # 量化步长初值
    "q_max": 160.0,          # 步长上限（防止过度量化）
    "band_mid": (2, 4),      # 中频带：低频结构保留，u+v 落在该区间才量化
    "band_high": (5, 14),    # 高频带：可用更大步长
    "high_gain": 1.6,        # 高频步长相对中频的倍数
    "desync_ratio": 0.0,     # 几何扰动比例（0 表示关闭；建议 0～0.04）
    "chroma_q": 0.0,         # 色度量化步长（0 表示关闭）
    "budget_safety": 0.92,   # 预算安全系数，给取整误差留余量
}

# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------


def _to_u8(a):
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def _psnr(a, b):
    d = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(d * d))
    return float("inf") if mse <= 0.0 else 10.0 * np.log10(255.0 * 255.0 / mse)


def _dct_matrix():
    n = np.arange(8, dtype=np.float64)
    k = np.arange(8, dtype=np.float64).reshape(-1, 1)
    T = np.cos(np.pi * (2.0 * n + 1.0) * k / 16.0) * np.sqrt(2.0 / 8.0)
    T[0] *= np.sqrt(0.5)
    return T.astype(np.float32)


_T = _dct_matrix()


def _pad_plane(a):
    """把平面补成 8 的整数倍，返回补齐后的平面与原始尺寸。"""
    h, w = a.shape
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        a = np.pad(a, ((0, ph), (0, pw)), mode="edge")
    return a, (h, w)


def _to_blocks(a):
    H, W = a.shape
    return a.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3)


def _from_blocks(b, hw):
    h, w = hw
    rec = b.transpose(0, 2, 1, 3).reshape(b.shape[0] * 8, b.shape[1] * 8)
    return rec[:h, :w]


def _dct2(b):
    return np.einsum("ij,...jk,lk->...il", _T, b, _T)


def _idct2(c):
    return np.einsum("ji,...jk,kl->...il", _T, c, _T)


def _rgb2ycc(x):
    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    cb = 128.0 - 0.168736 * x[..., 0] - 0.331264 * x[..., 1] + 0.5 * x[..., 2]
    cr = 128.0 + 0.5 * x[..., 0] - 0.418688 * x[..., 1] - 0.081312 * x[..., 2]
    return y, cb, cr


def _ycc2rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    return _to_u8(np.stack([r, g, b], axis=-1))


def _step_table(q, cfg):
    """按频带给出每个频率位置的量化步长；返回全零表示该位置不量化。"""
    step = np.zeros((8, 8), dtype=np.float32)
    lo_m, hi_m = cfg["band_mid"]
    lo_h, hi_h = cfg["band_high"]
    for u in range(8):
        for v in range(8):
            s = u + v
            if u == 0 and v == 0:
                continue
            if lo_m <= s <= hi_m:
                step[u, v] = q
            elif lo_h <= s <= hi_h:
                step[u, v] = q * cfg["high_gain"]
    return step


def _quantize(coef, step):
    """按步长表量化；步长为 0 的位置保持原值。"""
    s = step.reshape(1, 1, 8, 8)
    q = np.where(s > 0, np.round(coef / np.maximum(s, 1e-6)) * s, coef)
    return q.astype(np.float32)


def _coef_mse(coef, step):
    return float(np.mean((_quantize(coef, step) - coef) ** 2))


# --------------------------------------------------------------------------
# 处理流程
# --------------------------------------------------------------------------


def _desync(img, ratio):
    """裁边后缩放回原尺寸：破坏块间对应关系，步长很小以控制代价。"""
    if ratio <= 0:
        return img
    h, w = img.shape[:2]
    k = min(max(1, int(round(min(h, w) * ratio))), max(1, min(h, w) // 8))
    core = img[k:, k:].astype(np.float32)
    if core.shape[0] < 8 or core.shape[1] < 8:
        return img
    yy = np.clip((np.arange(h) + 0.5) * (core.shape[0] / h) - 0.5, 0, core.shape[0] - 1)
    xx = np.clip((np.arange(w) + 0.5) * (core.shape[1] / w) - 0.5, 0, core.shape[1] - 1)
    i0 = np.floor(yy).astype(np.int64)
    j0 = np.floor(xx).astype(np.int64)
    fy = (yy - i0)[:, None, None]
    fx = (xx - j0)[None, :, None]
    i1 = np.minimum(i0 + 1, core.shape[0] - 1)
    j1 = np.minimum(j0 + 1, core.shape[1] - 1)
    top = core[i0][:, j0] * (1.0 - fx) + core[i0][:, j1] * fx
    bot = core[i1][:, j0] * (1.0 - fx) + core[i1][:, j1] * fx
    return _to_u8(top * (1.0 - fy) + bot * fy)


def _solve_and_apply(x, cfg):
    """在系数域反解最大可用步长，返回处理后的图像。"""
    y, cb, cr = _rgb2ycc(x.astype(np.float32))
    yp, (h, w) = _pad_plane(y)
    coef = _dct2(_to_blocks(yp))

    target_mse = (255.0 ** 2) / (10.0 ** (cfg["psnr_min"] / 10.0)) * cfg["budget_safety"]
    q = float(cfg["q_init"])
    step = None
    for _ in range(5):
        step = _step_table(q, cfg)
        mse = _coef_mse(coef, step)
        if mse <= 1e-9:
            break
        ratio = (target_mse / mse) ** 0.5
        q_next = float(np.clip(q * ratio, 1.0, cfg["q_max"]))
        if abs(q_next - q) / max(q, 1e-6) < 0.02:
            q = q_next
            step = _step_table(q, cfg)
            break
        q = q_next

    out_y = _from_blocks(_idct2(_quantize(coef, step)), (h, w))
    out = _ycc2rgb(out_y, cb, cr)

    if cfg["chroma_q"] > 0:                     # 可选：色度域抹平
        qc = float(cfg["chroma_q"])
        cc = np.stack([cb - 128.0, cr - 128.0], axis=0)
        cq = np.round(cc / qc) * qc + 128.0
        trial = _ycc2rgb(out_y, cq[0], cq[1])
        if _psnr(x, trial) >= cfg["psnr_min"]:
            out = trial

    for _ in range(4):                          # 取整误差兜底
        if _psnr(x, out) >= cfg["psnr_min"]:
            break
        q *= 0.9
        step = _step_table(q, cfg)
        out_y = _from_blocks(_idct2(_quantize(coef, step)), (h, w))
        out = _ycc2rgb(out_y, cb, cr)
    return out


def attack(sample: dict) -> dict:
    image = np.asarray(sample["image"])
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        return {"image": image.copy()}

    x = _to_u8(image)
    try:
        work = _desync(x, CFG["desync_ratio"]) if CFG["desync_ratio"] > 0 else x
        out = _solve_and_apply(work, CFG)
        if CFG["desync_ratio"] > 0 and _psnr(x, out) < CFG["psnr_min"]:
            out = _solve_and_apply(x, CFG)      # 几何扰动超预算时退回纯量化
    except Exception:
        out = x.copy()

    bad = (out.shape != x.shape) or float(np.std(out)) < 1e-6
    if bad or _psnr(x, out) < CFG["psnr_min"] - 0.5:
        out = x.copy()
    return {"image": np.ascontiguousarray(out.astype(np.uint8))}


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
