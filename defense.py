#!/usr/bin/env python
"""防御代码 —— Baseline：D0 DCT 中频扩频零比特水印（Spread-Spectrum DCT Watermark）。

原理（赛题 §11.1）：
    1. RGB -> YCbCr，仅亮度通道 Y 参与嵌入/检测；
    2. 亮度通道按 8×8 分块做 DCT（正交归一化）；
    3. 依据固定公开密钥选择 8 个中频 DCT 系数位置（避开 DC 与极低频/极高频）；
    4. 用固定种子生成取值为 {-1,+1} 的伪随机序列 PN（扩频码）；
    5. 按图像局部能量自适应强度（scale ∈ [0.2, 2.5]）修改选中系数；
    6. 逆 DCT 并恢复 RGB 图像；
    7. 检测器计算选中系数与 PN 序列的逐块归一化相关性 c_b，
       聚合 z = mean(c_b)·sqrt(K·n_blocks) ~ N(0,1)，经标准正态 CDF
       映射为连续 watermark_probability ∈ [0,1]（0=高置信不含，1=高置信含）。

接口（与赛题 §7.3 一致，供评测流水线调用）：
    embed(sample)  -> {"image": RGB uint8 H×W×3}
    detect(request)-> {"watermark_probability": float 0.0~1.0}

另提供最小对照防御 null_embed / null_detect（赛题 §21 最小防御示例）：
不嵌入任何水印、恒输出 0.5，用于验证评测流水线（AUC 应 ≈ 0.5，随机水平）。

特性：CPU 可运行、无需训练、代码简短；干净条件下 AUC 明显高于随机；
对 JPEG/缩放具有限鲁棒性，便于攻击方验证（赛题 §11.1）。

用法：
    python defense.py                    # 读取基准数据，输出 data/embedded.npz
"""


import argparse
import os

import numpy as np

try:
    from scipy.fft import dctn, idctn
except Exception:  # pragma: no cover - 兼容旧版 scipy
    from scipy.fftpack import dctn, idctn  # type: ignore

# --------------------------------------------------------------------------- #
# 固定公开参数（密钥；攻击方可见，但不得查询检测器）
# --------------------------------------------------------------------------- #
SEED = 20240607
# 8×8 块中频系数位置（避开 DC 与极低频/极高频）
SELECTED = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)]
K = len(SELECTED)
ALPHA = 2.0   # 嵌入强度（DCT 系数单位）；512px 下 SSIM≈0.99、Clean AUC≈1.0
BLOCK = 8


def _pn_sequence() -> np.ndarray:
    rng = np.random.RandomState(SEED)
    return np.where(rng.rand(K) < 0.5, -1.0, 1.0)


_PN = _pn_sequence()


# --------------------------------------------------------------------------- #
# 颜色空间转换
# --------------------------------------------------------------------------- #
def _rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128.0
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128.0
    return np.stack([Y, Cb, Cr], axis=-1)


def _ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    Y = ycbcr[..., 0]
    Cb = ycbcr[..., 1] - 128.0
    Cr = ycbcr[..., 2] - 128.0
    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb
    return np.stack([R, G, B], axis=-1)


# --------------------------------------------------------------------------- #
# 分块 DCT / 逆 DCT（向量化）
# --------------------------------------------------------------------------- #
def _dct2_blocks(img: np.ndarray) -> np.ndarray:
    """img: H×W -> (H/8, W/8, 8, 8) DCT 系数。"""
    h, w = img.shape
    H, W = h // BLOCK, w // BLOCK
    blocks = img[:H * BLOCK, :W * BLOCK].reshape(H, BLOCK, W, BLOCK).transpose(0, 2, 1, 3)
    return dctn(blocks, type=2, norm="ortho", axes=(2, 3))


def _idct2_blocks(coefs: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    H, W = coefs.shape[0], coefs.shape[1]
    blocks = idctn(coefs, type=2, norm="ortho", axes=(2, 3))
    img = blocks.transpose(0, 2, 1, 3).reshape(H * BLOCK, W * BLOCK)
    out = np.zeros((out_h, out_w), dtype=img.dtype)
    out[:H * BLOCK, :W * BLOCK] = img
    return out


# --------------------------------------------------------------------------- #
# 嵌入
# --------------------------------------------------------------------------- #
def embed(sample: dict) -> dict:
    """D0 水印嵌入：输出尺寸/通道/dtype 与输入一致（RGB uint8 H×W×3）。"""
    image = np.asarray(sample["image"])
    h, w = image.shape[:2]
    ycbcr = _rgb_to_ycbcr(image)
    Y = ycbcr[..., 0].astype(np.float32)

    coefs = _dct2_blocks(Y)  # (H, W, 8, 8)

    # 局部能量（AC 系数），用于自适应强度
    ac = coefs.copy()
    ac[..., 0, 0] = 0.0
    energy = np.sqrt(np.sum(ac ** 2, axis=(2, 3))) + 1e-6  # (H, W)
    mean_energy = energy.mean() + 1e-6
    scale = np.clip(energy / mean_energy, 0.2, 2.5)  # (H, W)

    # 修改选中中频系数：coef += ALPHA * pn * scale
    for k, (r, c) in enumerate(SELECTED):
        coefs[:, :, r, c] = coefs[:, :, r, c] + ALPHA * _PN[k] * scale

    Y_new = _idct2_blocks(coefs, h, w)
    ycbcr[..., 0] = np.clip(Y_new, 0.0, 255.0)
    out = np.clip(_ycbcr_to_rgb(ycbcr), 0, 255).astype(np.uint8)
    return {"image": out}


# --------------------------------------------------------------------------- #
# 检测
# --------------------------------------------------------------------------- #
def detect(request: dict) -> dict:
    """D0 水印检测：返回 {"watermark_probability": float 0.0~1.0}。"""
    image = np.asarray(request["image"])
    ycbcr = _rgb_to_ycbcr(image)
    Y = ycbcr[..., 0].astype(np.float32)
    coefs = _dct2_blocks(Y)  # (H, W, 8, 8)
    H, W = coefs.shape[0], coefs.shape[1]
    n_blocks = H * W

    # 选中系数 (H, W, K)
    sel = np.stack([coefs[:, :, r, c] for (r, c) in SELECTED], axis=-1)

    # 逐块归一化相关性 c_b = <sel_b, pn> / (||sel_b|| * ||pn||) ∈ [-1, 1]
    # 该形式对块能量不变，避免高纹理块的自然能量淹没水印信号。
    dot_b = (sel * _PN).sum(axis=-1)                       # (H, W)
    norm_b = np.sqrt((sel ** 2).sum(axis=-1)) * np.sqrt(K) + 1e-8
    c_b = dot_b / norm_b                                    # (H, W)

    # 聚合：零假设下 c_b 均值方差 ≈ 1/(K·n_blocks)，故 z = mean(c_b)·sqrt(K·n_blocks) ~ N(0,1)
    d = K * n_blocks
    z = float(c_b.mean()) * np.sqrt(d)
    prob = 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))
    prob = float(min(1.0, max(0.0, prob)))
    return {"watermark_probability": prob}


def _erf(x: np.ndarray) -> np.ndarray:
    """数值稳定的 erf 近似（Abramowitz & Stegun 7.1.26），避免依赖 math.erf 标量限制。"""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return sign * y


# --------------------------------------------------------------------------- #
# 最小对照防御（赛题 §21 最小防御示例：不嵌入、恒输出 0.5）
# --------------------------------------------------------------------------- #
def null_embed(sample: dict) -> dict:
    """对照：不嵌入任何水印，原样返回。"""
    return {"image": np.asarray(sample["image"]).copy()}


def null_detect(request: dict) -> dict:
    """对照：恒输出中性分数 0.5（随机水平）。"""
    return {"watermark_probability": 0.5}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D0 DCT 扩频水印防御（Baseline）")
    parser.add_argument("--n", type=int, default=48, help="图像数量")
    parser.add_argument("--output", default="data/embedded.npz", help="输出路径")
    args = parser.parse_args()

    from dataset import load_benchmark, save_npz, to_samples

    ds = load_benchmark(n=args.n)
    samples = to_samples(ds)
    out_images = np.empty_like(ds["images"])
    for i, s in enumerate(samples):
        out_images[i] = embed(s)["image"]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_npz(args.output, out_images, ds["sample_ids"])
    print(f"[defense] D0 嵌入完成：{len(samples)} 张 -> {args.output}")

    # 自检：干净图像检测分数分布（期望：带水印≈1，未带≈0.5 附近）
    probe_positive = [detect({"image": img})["watermark_probability"] for img in out_images]
    probe_negative = [detect({"image": img})["watermark_probability"] for img in ds["images"]]
    print(f"[defense] 自检 watermark_probability：带水印 mean={np.mean(probe_positive):.3f} "
          f"min={np.min(probe_positive):.3f}；未带水印 mean={np.mean(probe_negative):.3f}")
