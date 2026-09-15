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
"""attack.py —— 攻击方（在线编辑版）

STAGE 1  A0 基线组合：JPEG(q=75) -> 缩放 90% -> 双线性还原 -> 高斯模糊(sigma=0.4)
STAGE 2  分块去对齐：裁掉边缘 -> 线性缩放回原尺寸，破坏 8×8 分块网格对齐
STAGE 3  质量预算回拉：朝输入二分插值，直到 PSNR / SSIM 达标（留安全余量）

依赖：numpy 必需；有 Pillow 时用真实 JPEG 编解码，否则用 numpy 等效实现
      （YCbCr + 4:2:0 + 8×8 DCT + 标准量化表）。不联网、不查检测器、无外部数据。
"""

import numpy as np

try:
    import io as _io
    from PIL import Image as _PILImage
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

FORCE_NUMPY_JPEG = False      # 调试用：强制走 numpy JPEG 路径

MODE = "STRONG"               # "A0" = 仅基线；"STRONG" = 基线 + 去对齐

CFG = {
    "jpeg_quality": 75,       # A0: 75
    "scale": 0.90,            # A0: 缩放到 90%
    "blur_sigma": 0.4,        # A0: 高斯模糊 sigma
    "desync_ratio": 0.094,    # 去对齐裁剪比例（32×32 上约等于 3 px）
    "desync_min": 2,          # 去对齐最小裁剪像素
    "psnr_min": 29.0,         # 裁判红线 28 dB，留 1 dB 余量
    "ssim_min": 0.92,         # 主赛道红线 0.92；实测抬到 0.94 会让抑制率腰斩（见方案文档）
}

# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------


def _to_u8(a):
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def _psnr(a, b):
    d = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(d * d, dtype=np.float64))
    return float("inf") if mse <= 0.0 else 10.0 * np.log10(255.0 * 255.0 / mse)


def _ssim(a, b, win=8):
    """8×8 非重叠块的平均 SSIM（矢量化的工程近似，用于质量自检）。"""
    if a.ndim == 3:
        return float(np.mean([_ssim(a[..., c], b[..., c], win) for c in range(a.shape[2])]))
    A_ = a.astype(np.float32)
    B_ = b.astype(np.float32)
    h, w = A_.shape
    bh, bw = h // win, w // win
    if bh < 1 or bw < 1:
        return 1.0
    A_ = A_[:bh * win, :bw * win].reshape(bh, win, bw, win).transpose(0, 2, 1, 3).reshape(bh, bw, win * win)
    B_ = B_[:bh * win, :bw * win].reshape(bh, win, bw, win).transpose(0, 2, 1, 3).reshape(bh, bw, win * win)
    if bh * bw > 512:                      # 大图抽样，省一半以上耗时
        A_ = A_[::2, ::2]
        B_ = B_[::2, ::2]
    ma, mb = A_.mean(-1), B_.mean(-1)
    va, vb = A_.var(-1), B_.var(-1)
    cov = ((A_ - ma[..., None]) * (B_ - mb[..., None])).mean(-1)
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    s = ((2.0 * ma * mb + c1) * (2.0 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def _resize_matrix(n, out_len):
    """(out_len, n) 重采样矩阵：缩小时面积平均（抗锯齿），放大时双线性。"""
    i = np.arange(out_len, dtype=np.float64)
    if out_len < n:
        e0 = i * n / out_len
        e1 = (i + 1.0) * n / out_len
        j = np.arange(n, dtype=np.float64)[None, :]
        W = np.clip(np.minimum(e1[:, None], j + 1.0) - np.maximum(e0[:, None], j), 0.0, None)
        W = W / (e1 - e0)[:, None]
    else:
        s = np.clip((i + 0.5) * (n / out_len) - 0.5, 0.0, n - 1.0)
        i0 = np.clip(np.floor(s).astype(np.int64), 0, n - 1)
        i1 = np.clip(i0 + 1, 0, n - 1)
        f = np.clip(s - i0, 0.0, 1.0)
        W = np.zeros((out_len, n), dtype=np.float64)
        W[np.arange(out_len), i0] += 1.0 - f
        W[np.arange(out_len), i1] += f
    return W.astype(np.float32)


def _resize2d(a, out_h, out_w):
    h, w = a.shape
    a = a.astype(np.float32)
    if out_w != w:
        a = a @ _resize_matrix(w, out_w).T
    if out_h != h:
        a = _resize_matrix(h, out_h) @ a
    return a


def _resize_rgb(a, out_h, out_w):
    return np.stack([_resize2d(a[..., c], out_h, out_w) for c in range(a.shape[2])], axis=-1)


def _gaussian_blur_rgb(a, sigma):
    if sigma <= 0:
        return a.astype(np.float32)
    r = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    k /= k.sum()

    def conv1d(v, axis):
        pad = [(0, 0)] * v.ndim
        pad[axis] = (r, r)
        p = np.pad(v, pad, mode="edge")
        out = np.zeros_like(v, dtype=np.float32)
        for idx, kv in enumerate(k):
            sl = [slice(None)] * v.ndim
            sl[axis] = slice(idx, idx + v.shape[axis])
            out += kv * p[tuple(sl)]
        return out

    return np.stack([conv1d(conv1d(a[..., c].astype(np.float32), 1), 0)
                     for c in range(a.shape[2])], axis=-1)


# --------------------------------------------------------------------------
# JPEG 往返（有 Pillow 用真实编解码，否则用等效 DCT 量化）
# --------------------------------------------------------------------------

_LUMA_Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float32)

_CHROMA_Q = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)


def _dct_matrix():
    n = np.arange(8, dtype=np.float64)
    k = np.arange(8, dtype=np.float64).reshape(-1, 1)
    T = np.cos(np.pi * (2.0 * n + 1.0) * k / 16.0) * np.sqrt(2.0 / 8.0)
    T[0] *= np.sqrt(0.5)
    return T.astype(np.float32)


_DCT = _dct_matrix()


def _quantize_plane(a, qtable):
    h, w = a.shape
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        a = np.pad(a, ((0, ph), (0, pw)), mode="edge")
    H, W = a.shape
    blocks = a.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3)
    coef = np.einsum("ij,...jk,lk->...il", _DCT, blocks, _DCT)
    coef = np.round(coef / qtable) * qtable
    rec = np.einsum("ji,...jk,kl->...il", _DCT, coef, _DCT)
    return rec.transpose(0, 2, 1, 3).reshape(H, W)[:h, :w]


def _jpeg_numpy(img, quality):
    q = float(np.clip(quality, 1, 100))
    scale = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
    ql = np.clip(np.floor((_LUMA_Q * scale + 50.0) / 100.0), 1, 255)
    qc = np.clip(np.floor((_CHROMA_Q * scale + 50.0) / 100.0), 1, 255)
    x = img.astype(np.float32)
    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    cb = 128.0 - 0.168736 * x[..., 0] - 0.331264 * x[..., 1] + 0.5 * x[..., 2]
    cr = 128.0 + 0.5 * x[..., 0] - 0.418688 * x[..., 1] - 0.081312 * x[..., 2]
    h, w = img.shape[:2]
    y = _quantize_plane(y, ql)
    hc, wc = max(1, (h + 1) // 2), max(1, (w + 1) // 2)         # 4:2:0
    cb = _resize2d(_quantize_plane(_resize2d(cb, hc, wc), qc), h, w)
    cr = _resize2d(_quantize_plane(_resize2d(cr, hc, wc), qc), h, w)
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    return _to_u8(np.stack([r, g, b], axis=-1))


def _jpeg(img, quality):
    if _HAS_PIL and not FORCE_NUMPY_JPEG:
        try:
            buf = _io.BytesIO()
            _PILImage.fromarray(img).save(buf, format="JPEG", quality=int(quality),
                                          subsampling=2, optimize=False)
            buf.seek(0)
            with _PILImage.open(buf) as im:
                return np.asarray(im.convert("RGB"), dtype=np.uint8)
        except Exception:
            pass
    return _jpeg_numpy(img, quality)


# --------------------------------------------------------------------------
# 三段流水线
# --------------------------------------------------------------------------


def _stage_a0(img, cfg):
    """A0：JPEG75 -> 缩放 90% -> 双线性还原 -> 高斯模糊 σ=0.4。"""
    x = _jpeg(img, cfg["jpeg_quality"])
    h, w = x.shape[:2]
    sh = max(1, int(round(h * cfg["scale"])))
    sw = max(1, int(round(w * cfg["scale"])))
    y = _resize_rgb(_resize_rgb(x, sh, sw), h, w)
    return _to_u8(_gaussian_blur_rgb(y, cfg["blur_sigma"]))


def _stage_desync(img, cfg):
    """去对齐：裁掉左上边缘 k 像素后缩放回原尺寸，破坏 8×8 分块相位。"""
    h, w = img.shape[:2]
    k = int(round(min(h, w) * cfg["desync_ratio"]))
    k = max(cfg["desync_min"], k)
    k = min(k, max(1, min(h, w) // 4))
    core = img[k:, k:]
    if core.shape[0] < 8 or core.shape[1] < 8:
        return img.astype(np.float32)
    return _resize_rgb(core.astype(np.float32), h, w)


def _stage_budget(x, y, cfg):
    """质量预算回拉：二分插值 alpha，取满足 PSNR/SSIM 的最大攻击强度。"""
    def ok(cand):
        if _psnr(x, cand) < cfg["psnr_min"]:
            return False
        return cfg["ssim_min"] is None or _ssim(x, cand) >= cfg["ssim_min"]

    y = _to_u8(y)
    if ok(y):
        return y
    xf, yf = x.astype(np.float32), y.astype(np.float32)
    lo, hi, best = 0.0, 1.0, x
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        cand = _to_u8(xf + mid * (yf - xf))
        if ok(cand):
            lo, best = mid, cand
        else:
            hi = mid
    return best


def attack(sample: dict) -> dict:
    """A0 组合后处理攻击：JPEG75 -> 缩放90% -> 双线性还原 -> 高斯模糊 σ=0.4。"""
    image = np.asarray(sample["image"])
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        return {"image": image.copy()}

    x = _to_u8(image)
    y = _stage_a0(x, CFG)
    if MODE == "STRONG":
        y = _stage_desync(y, CFG)
    y = _stage_budget(x, y, CFG)

    if y.shape != x.shape or float(np.std(y)) < 1e-6:      # 绝不出非法图
        y = x
    return {"image": np.ascontiguousarray(y.astype(np.uint8))}


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    for side in (32, 512):
        img = rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8)
        t0 = time.time()
        res = attack({"sample_id": "selftest", "image": img})["image"]
        print("%dx%d  PSNR=%.2f dB  SSIM=%.4f  %s %s  %.1f ms"
              % (side, side, _psnr(img, res), _ssim(img, res),
                 res.shape, res.dtype, (time.time() - t0) * 1000.0))


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
