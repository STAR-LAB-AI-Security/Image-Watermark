#!/usr/bin/env python
"""数据集代码 —— 图片隐藏水印消除与鲁棒检测基准（Benchmark）数据集处理。

职责：
1. 优先加载本地缓存（data/synth_images.npz），避免重复生成；
2. 否则按固定随机种子（默认 42）程序化合成 N 张 512×512 RGB uint8 图像，
   覆盖赛题 §5.1 所述多种视觉内容（自然场景感梯度、纹理、正弦条纹、几何形状、
   海报图表、人像感、文档截图等），防止水印算法只适用于单一内容类型；
3. 可选 --data-dir <目录> 从真实图片目录读取（需 Pillow，统一缩放至 512×512）；
4. 统一输出格式（基准输入输出规范）：

       dataset = {
           "images":     np.ndarray [N, 512, 512, 3] uint8（RGB，取值 0-255）
           "sample_ids": np.ndarray [N] <U8（如 "syn_0000"）
       }

    评测时对每张原图 I 构造同源配对（赛题 §6）：正分支 I+ = E_d(I)（嵌入水印），
    负分支 I- = I（无水印对照），同一攻击同参对称作用于正负分支。

用法：
    python dataset.py                    # 生成/加载基准数据并打印规模
    python dataset.py --data-dir ./imgs  # 从图片目录读取（覆盖合成）
"""

from __future__ import annotations

import glob
import os

import numpy as np

# --------------------------------------------------------------------------- #
# 常量（与基准文档一致）
# --------------------------------------------------------------------------- #
DEFAULT_N = 48              # 每组合采样的原图数量（默认）
DEFAULT_SIZE = 512          # 统一图像尺寸（宽=高=512）
DEFAULT_SEED = 42           # 数据生成/采样随机种子
CACHE_DIR = "data"
CACHE_PATH = os.path.join(CACHE_DIR, "synth_images.npz")


# --------------------------------------------------------------------------- #
# 程序化合成图像（7 类视觉内容，全部 numpy/scipy 实现，不读外部数据）
# --------------------------------------------------------------------------- #
_GENERATORS: list = []


def _register(fn):
    _GENERATORS.append(fn)
    return fn


@_register
def _gen_gradient(size, rng):
    """平滑色彩梯度（自然场景感的低频内容）。"""
    t = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(t, t)
    h = rng.uniform(0, 1)
    r = 0.5 + 0.5 * np.sin(2 * np.pi * (xx + h))
    g = 0.5 + 0.5 * np.sin(2 * np.pi * (yy + 0.33 + h))
    b = 0.5 + 0.5 * np.sin(2 * np.pi * ((xx + yy) * 0.5 + 0.66 + h))
    img = np.stack([r, g, b], axis=-1) * 0.6 + 0.2
    return img * 255


@_register
def _gen_natural(size, rng):
    """高斯滤波噪声 -> 自然纹理感。"""
    from scipy.ndimage import gaussian_filter
    noise = rng.randn(size, size, 3)
    smooth = gaussian_filter(noise, sigma=rng.uniform(1.0, 4.0))
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
    tint = rng.uniform(0.7, 1.3, size=3)
    return smooth * tint * 255


@_register
def _gen_sinusoid(size, rng):
    """正弦条纹纹理（高频细节）。"""
    freq = rng.uniform(2, 10)
    phase = rng.uniform(0, 2 * np.pi)
    x = np.linspace(0, freq * np.pi, size)
    xx, _ = np.meshgrid(x, x)
    pat = 0.5 + 0.5 * np.sin(xx + phase)
    base = rng.uniform(0.2, 0.8, size=3)
    img = pat[..., None] * base + (1 - pat[..., None]) * (1 - base) * 0.3
    return img * 255


@_register
def _gen_shapes(size, rng):
    """随机几何形状（建筑/商品感的强边缘）。"""
    bg = rng.randint(40, 200, size=3)
    img = np.full((size, size, 3), bg, dtype=np.float64)
    yy, xx = np.ogrid[:size, :size]
    for _ in range(rng.randint(12, 30)):
        cx, cy = rng.randint(0, size, 2)
        r = rng.randint(15, 90)
        col = rng.randint(0, 255, 3)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        img[mask] = col
    return img


@_register
def _gen_poster(size, rng):
    """海报/图表感：大色块网格 + 细网格线（高频边界但块内平滑，对 JPEG 友好）。"""
    cell = int(rng.choice([32, 64]))
    n = size // cell
    img = np.zeros((size, size, 3), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            img[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = rng.randint(40, 230, 3)
    img[::cell, :, :] *= 0.6
    img[:, ::cell, :] *= 0.6
    return np.clip(img, 0, 255)


@_register
def _gen_portrait(size, rng):
    """肤色平滑块（人像感的低纹理区域）。"""
    from scipy.ndimage import gaussian_filter
    skin = np.array([rng.randint(190, 230), rng.randint(150, 190), rng.randint(130, 170)],
                    dtype=np.float64)
    bg = np.array([rng.randint(40, 100), rng.randint(40, 100), rng.randint(60, 120)],
                  dtype=np.float64)
    field = gaussian_filter(rng.rand(size, size), sigma=rng.uniform(8, 20))
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    img = field[..., None] * skin + (1 - field[..., None]) * bg
    return img


@_register
def _gen_document(size, rng):
    """文档/图表截图感：浅底 + 网格 + 文字状暗块（高频细节）。"""
    img = np.full((size, size, 3), 245, dtype=np.float64)
    step = int(rng.choice([16, 32, 64]))
    img[::step, :, :] = 180
    img[:, ::step, :] = 180
    for _ in range(rng.randint(20, 60)):
        y = rng.randint(0, size - 8)
        x = rng.randint(0, size - 40)
        w = rng.randint(20, 140)
        h = rng.randint(2, 7)
        img[y:y + h, x:x + w, :] = rng.randint(0, 80)
    return img


def synthesize(n: int = DEFAULT_N, size: int = DEFAULT_SIZE, seed: int = DEFAULT_SEED) -> dict:
    """程序化合成 n 张 size×size RGB uint8 图像（按 7 类内容轮转）。"""
    rng = np.random.RandomState(seed)
    gens = list(_GENERATORS)
    images = np.empty((n, size, size, 3), dtype=np.uint8)
    ids = np.empty(n, dtype="<U12")
    for i in range(n):
        gen = gens[i % len(gens)]
        img = np.clip(gen(size, rng), 0, 255).astype(np.uint8)
        images[i] = img
        ids[i] = f"syn_{i:04d}"
    return {"images": images, "sample_ids": ids}


# --------------------------------------------------------------------------- #
# 目录读取（真实图片，需 Pillow；统一缩放至 size×size）
# --------------------------------------------------------------------------- #
def load_from_dir(directory: str, n: int = DEFAULT_N, size: int = DEFAULT_SIZE,
                  seed: int = DEFAULT_SEED) -> dict:
    from PIL import Image

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tif", "*.tiff")
    files: list[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(directory, e)))
        files.extend(glob.glob(os.path.join(directory, "**", e), recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"目录 {directory} 中未找到图片")

    rng = np.random.RandomState(seed)
    rng.shuffle(files)
    files = files[:n]
    images = np.empty((len(files), size, size, 3), dtype=np.uint8)
    ids = np.empty(len(files), dtype="<U12")
    for i, fp in enumerate(files):
        img = Image.open(fp).convert("RGB").resize((size, size), Image.BILINEAR)
        images[i] = np.asarray(img, dtype=np.uint8)
        ids[i] = f"real_{i:04d}"
    return {"images": images, "sample_ids": ids}


# --------------------------------------------------------------------------- #
# 基准数据加载（缓存优先 -> 合成/真实图片）
# --------------------------------------------------------------------------- #
def load_benchmark(
    n: int = DEFAULT_N,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
    data_dir: str | None = None,
    cache_dir: str = CACHE_DIR,
    use_cache: bool = True,
) -> dict:
    """加载基准图像集，返回 {"images": [N,size,size,3] uint8, "sample_ids": [N]}。

    优先命中本地缓存（data/synth_images.npz）；无缓存时从 data_dir 读取真实图片，
    否则程序化合成（固定 seed，可复现）。缓存只用于合成集，真实图片不落缓存。
    """
    if data_dir:
        ds = load_from_dir(data_dir, n=n, size=size, seed=seed)
        print(f"[dataset] 从目录加载真实图片 {len(ds['sample_ids'])} 张: {data_dir}")
        return ds

    cache_path = os.path.join(cache_dir, "synth_images.npz")
    if use_cache and os.path.exists(cache_path):
        d = np.load(cache_path)
        images, ids = d["images"], d["sample_ids"]
        if len(images) >= n:
            images, ids = images[:n], ids[:n]
            print(f"[dataset] 命中本地缓存: {cache_path}（取前 {n} 张）")
            return {"images": images, "sample_ids": ids}
        print(f"[dataset] 缓存样本数不足（{len(images)} < {n}），重新生成。")

    ds = synthesize(n=n, size=size, seed=seed)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_path, images=ds["images"], sample_ids=ds["sample_ids"])
    print(f"[dataset] 已生成并保存合成基准数据: {cache_path}（{n} 张 {size}x{size}）")
    return ds


def to_samples(dataset: dict) -> list[dict]:
    """将 {"images","sample_ids"} 转换为样本字典列表 [{"sample_id", "image"}]。"""
    return [{"sample_id": str(sid), "image": img}
            for sid, img in zip(dataset["sample_ids"], dataset["images"])]


def save_npz(path: str, images: np.ndarray, sample_ids) -> None:
    """将图像集保存为 .npz 文件（供攻击/防御结果落盘）。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, images=np.asarray(images, dtype=np.uint8),
                        sample_ids=np.asarray(sample_ids))
    print(f"[dataset] 已保存 {path}（{len(sample_ids)} 个样本）")


def load_npz(path: str) -> dict:
    """读取 .npz 文件为 {"images", "sample_ids"} 字典。"""
    d = np.load(path)
    return {"images": d["images"], "sample_ids": d["sample_ids"]}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="图片隐藏水印基准：数据集生成/加载")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="图像数量，默认 48")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="图像尺寸，默认 512")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子，默认 42")
    parser.add_argument("--data-dir", default=None, help="真实图片目录（可选）")
    args = parser.parse_args()

    ds = load_benchmark(n=args.n, size=args.size, seed=args.seed, data_dir=args.data_dir)
    print(f"images: {ds['images'].shape}  dtype={ds['images'].dtype}  "
          f"值域 [{ds['images'].min()}, {ds['images'].max()}]")
    print(f"sample_ids: {ds['sample_ids'].shape}  示例 {ds['sample_ids'][:3].tolist()}")
