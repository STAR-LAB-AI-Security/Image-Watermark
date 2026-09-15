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
"""attack.py —— 攻击方（在线编辑版，低耗时实现）

MODE 三档：
  "LIGHT"   去对齐 + 质量预算                 <- 最快
  "STRONG"  JPEG + 模糊 + 去对齐 + 质量预算
  "A0"      平台基线组合（JPEG75 + 缩放90% + 模糊σ0.4）+ 质量预算

依赖：numpy 必需；有 Pillow 时用真实 JPEG，否则用 numpy 等效 DCT 量化。
不联网、不查检测器、不使用外部数据，同一输入恒得同一输出。
"""

import numpy as np

try:
    import io as _io
    from PIL import Image as _PILImage
    from PIL import ImageFilter as _PILFilter
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

FORCE_NUMPY_JPEG = False     # 调试用：强制走 numpy JPEG

MODE = "LIGHT"               # LIGHT=最快最强（只守 PSNR 红线）；STRONG=守 PSNR+SSIM；A0=平台基线

CFG = {
    "jpeg_quality": 75,      # A0 参数
    "scale": 0.90,           # A0 参数
    "blur_sigma": 0.4,       # A0 参数
    "desync_ratio": 0.094,   # 去对齐裁剪比例（32×32 上约 3 px，512×512 上约 48 px）
    "desync_min": 2,
    "psnr_min": 29.0,        # 红线 28 dB，留 1 dB 余量
    "ssim_min": 0.92,        # 红线 0.92；LIGHT 档忽略此项
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


def _box_mean(a, win):
    """滑动窗口均值（前缀和），窗口完全落在图内。"""
    C = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float32)
    C[1:, 1:] = np.cumsum(np.cumsum(a, axis=0, dtype=np.float32), axis=1, dtype=np.float32)
    S = C[win:, win:] - C[:-win, win:] - C[win:, :-win] + C[:-win, :-win]
    return S / float(win * win)


def _ssim(a, b, win=8):
    """标准滑动窗口 SSIM（逐通道取平均，窗口 8×8）。"""
    return _ssim32(a.astype(np.float32), b.astype(np.float32), win)


def _ssim32(a, b, win=8):
    """同上，输入需为 float32。大图按整数抽样降采样后再算（质量自检足够）。"""
    if a.ndim == 3:
        return float(np.mean([_ssim32(a[..., c], b[..., c], win) for c in range(a.shape[2])]))
    step = 1
    while max(a.shape) // (step * 2) >= 192:
        step *= 2
    if step > 1:
        a = a[::step, ::step]
        b = b[::step, ::step]
    if min(a.shape) < win:
        return 1.0
    ma = _box_mean(a, win)
    mb = _box_mean(b, win)
    va = _box_mean(a * a, win) - ma * ma
    vb = _box_mean(b * b, win) - mb * mb
    cov = _box_mean(a * b, win) - ma * mb
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    s = ((2.0 * ma * mb + c1) * (2.0 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def _resize_last(a, out_len):
    """沿最后一维重采样 2D 数组：缩小用面积平均（前缀和），放大用双线性两抽头。"""
    n = a.shape[-1]
    if out_len == n:
        return a.astype(np.float32, copy=True)
    a = a.astype(np.float32)
    if out_len > n:
        s = np.clip((np.arange(out_len) + 0.5) * (n / out_len) - 0.5, 0.0, n - 1.0)
        i0 = np.floor(s).astype(np.int64)
        f = (s - i0).astype(np.float32)
        i1 = np.minimum(i0 + 1, n - 1)
        return a[..., i0] * (1.0 - f) + a[..., i1] * f
    cp = np.zeros(a.shape[:-1] + (n + 1,), dtype=np.float32)
    np.cumsum(a, axis=-1, out=cp[..., 1:])
    e = np.linspace(0.0, n, out_len + 1, dtype=np.float32)
    e0, e1 = e[:-1], e[1:]
    i0 = np.minimum(np.floor(e0).astype(np.int64), n - 1)
    i1 = np.minimum(np.floor(e1).astype(np.int64), n - 1)
    f0, f1 = e0 - i0, e1 - i1
    lo = cp[..., i0] + f0 * (cp[..., i0 + 1] - cp[..., i0])
    hi = cp[..., i1] + f1 * (cp[..., i1 + 1] - cp[..., i1])
    return (hi - lo) / (e1 - e0)


def _resize2d(a, out_h, out_w):
    a = _resize_last(a, out_w)
    return _resize_last(a.T, out_h).T


def _resize_rgb(a, out_h, out_w):
    return np.stack([_resize2d(a[..., c], out_h, out_w) for c in range(a.shape[2])], axis=-1)


def _blur_rgb(a, sigma):
    if sigma <= 0:
        return a.astype(np.float32)
    if _HAS_PIL:                            # Pillow 的 C 实现，比 numpy 快一个量级
        try:
            out = _PILImage.fromarray(_to_u8(a)).filter(_PILFilter.GaussianBlur(float(sigma)))
            return np.asarray(out, dtype=np.float32)
        except Exception:
            pass
    r = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    k /= k.sum()

    def conv(v, axis):
        pad = [(0, 0)] * v.ndim
        pad[axis] = (r, r)
        p = np.pad(v, pad, mode="edge")
        out = np.zeros_like(v, dtype=np.float32)
        for idx, kv in enumerate(k):
            sl = [slice(None)] * v.ndim
            sl[axis] = slice(idx, idx + v.shape[axis])
            out += kv * p[tuple(sl)]
        return out

    return np.stack([conv(conv(a[..., c].astype(np.float32), 0), 1)
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
    hc, wc = max(1, (h + 1) // 2), max(1, (w + 1) // 2)          # 4:2:0
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


def _stage_desync(img, cfg):
    """去对齐：裁掉左上边缘 k 像素后缩放回原尺寸，破坏 8×8 分块相位。"""
    h, w = img.shape[:2]
    k = int(round(min(h, w) * cfg["desync_ratio"]))
    k = min(max(cfg["desync_min"], k), max(1, min(h, w) // 4))
    core = img[k:, k:]
    if core.shape[0] < 8 or core.shape[1] < 8:
        return img.astype(np.float32)
    return _resize_rgb(core.astype(np.float32), h, w)


def _stage_budget(x, y, cfg, use_ssim):
    """质量预算回拉：解析解定 PSNR 上限，必要时在浮点域对 SSIM 做少量二分。"""
    xf = x.astype(np.float32)
    e = y.astype(np.float32) - xf
    mse = float(np.mean(e * e, dtype=np.float64))
    if mse <= 1e-12:
        return x

    target = (255.0 ** 2) / (10.0 ** (cfg["psnr_min"] / 10.0))
    alpha = min(1.0, float(np.sqrt(target / mse)) * 0.995)

    smin = cfg["ssim_min"]
    if use_ssim and smin is not None and _ssim32(xf, xf + alpha * e) < smin:
        lo, hi = 0.0, alpha
        for _ in range(6):
            mid = 0.5 * (lo + hi)
            if _ssim32(xf, xf + mid * e) >= smin:
                lo = mid
            else:
                hi = mid
        alpha = lo

    out = _to_u8(xf + alpha * e)
    for _ in range(3):                        # 量化舍入的兜底修正
        if _psnr(x, out) >= cfg["psnr_min"]:
            break
        alpha *= 0.95
        out = _to_u8(xf + alpha * e)
    return out


def attack(sample: dict) -> dict:
    """A0 组合后处理攻击：JPEG75 -> 缩放90% -> 双线性还原 -> 高斯模糊 σ=0.4。"""
    image = np.asarray(sample["image"])
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        return {"image": image.copy()}

    x = _to_u8(image)
    cfg = CFG

    if MODE == "A0":
        y = _jpeg(x, cfg["jpeg_quality"])
        h, w = y.shape[:2]
        y = _resize_rgb(y, max(1, int(round(h * cfg["scale"]))),
                        max(1, int(round(w * cfg["scale"]))))
        y = _to_u8(_blur_rgb(_resize_rgb(y, h, w), cfg["blur_sigma"]))
        y = _stage_budget(x, y, cfg, True)
    elif MODE == "LIGHT":
        y = _stage_budget(x, _stage_desync(x, cfg), cfg, False)
    else:
        y = _to_u8(_blur_rgb(_jpeg(x, cfg["jpeg_quality"]), cfg["blur_sigma"]))
        y = _stage_budget(x, _stage_desync(y, cfg), cfg, True)

    if y.shape != x.shape or float(np.std(y)) < 1e-6:      # 绝不出非法图
        y = x
    return {"image": np.ascontiguousarray(y.astype(np.uint8))}


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
