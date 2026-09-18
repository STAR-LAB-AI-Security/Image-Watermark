#!/usr/bin/env python
"""测试代码 —— 图片隐藏水印消除与鲁棒检测基准（Benchmark）指标计算。

评测流程（赛题 §6/§10 同源配对 + 对称攻击）：
    对每个 (攻击 a, 防御 d) 组合，从同一批原图 I_i 构造：
        正分支: I+ = embed(I_i)  -> attack -> detect -> s+
        负分支: I- = I_i         -> attack -> detect -> s-
    y_true = [1..1, 0..0]，y_score = [s+.., s-..]
    M[a,d] = ROC-AUC(y_true, y_score)（Mann-Whitney U，保留原始方向，
             不做 max(AUC,1-AUC) 修正，赛题 §10.4）

非法处理（赛题 §9.6）：
    - 嵌入输出非法（尺寸/通道/dtype/NaN/纯色）-> 正样本用原始无水印图替代
    - 攻击输出非法 -> 用攻击前输入进检测（攻击方无收益）
    - 检测异常 -> 中性分数 0.5
    质量（SSIM/PSNR）不达标（strict）时按非法处理回退。

攻防矩阵：攻击 {Id（恒等对照）, A0} × 防御 {D0（基线）, null_defense（对照）}

攻击评测（run_attack.sh）：攻击方 Baseline A0 在 D0 上的消除能力
    指标：CleanAUC（Id×D0）/ AttackAUC（A0×D0）/ AUC Drop /
          AttackSuccess（AUC Drop ≥ δ_a=0.02）/ AttackStrength=1-R_A /
          TPR/FPR（τ=0.5）/ 攻击质量 SSIM/PSNR 与合格率
防御评测（run_defense.sh）：防御方 Baseline D0 的鲁棒检测能力
    指标：CleanAUC / RobustAUC（A0 攻击后）/ R_D=mean_a M[a,D0] /
          min AUC / TPR/FPR / 嵌入质量 SSIM/PSNR 与合格率

用法：
    python test.py --stage attack      # 攻击方 Baseline 评测（默认）
    python test.py --stage defense     # 防守方 Baseline 评测
    python test.py --stage all         # 攻击 + 防御
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from attack import attack, identity_attack
from dataset import load_benchmark, to_samples
from defense import (defense, detect, null_detect as null_defense_detect,
                     null_embed as null_defense_embed)

# --------------------------------------------------------------------------- #
# 基准固定参数（与基准文档一致）
# --------------------------------------------------------------------------- #
DEFAULT_N = 48          # 每组合采样的原图数量
IMAGE_SIZE = 512        # 统一图像尺寸
DEFAULT_SEED = 42       # 数据生成随机种子
TAU = 0.5               # 检测决策阈值（watermark_probability ≥ τ 判为含水印）
DELTA_A = 0.02          # 攻击成功判定：AUC(Id,D0) - AUC(A,D0) ≥ δ_a
QUALITY = {             # 赛题 §9.2/§9.3 质量硬约束
    "embed": {"ssim_min": 0.97, "psnr_min": 36.0},
    "attack": {"ssim_min": 0.92, "psnr_min": 28.0},
}
STRICT = True           # strict=True：质量不达标视为非法，触发 §9.6 回退


# --------------------------------------------------------------------------- #
# 指标：ROC-AUC / PSNR / SSIM / TPR-FPR
# --------------------------------------------------------------------------- #
def roc_auc(y_true, y_score) -> float:
    """二分类 ROC-AUC（Mann-Whitney U，平均秩处理并列，保留原始方向）。

    完美分离=1.0，完全反转=0.0，随机≈0.5；不做 max(AUC, 1-AUC) 修正。
    只有一类样本时返回 NaN。
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    pos_mask = (y_true == 1)
    n_pos = int(pos_mask.sum())
    n_neg = int((~pos_mask).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    n = len(y_score)
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[pos_mask].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def tpr_fpr(y_true, y_score, tau: float = TAU) -> tuple[float, float]:
    """固定决策阈值 τ 下的真正例率（检测率）与假正例率。"""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    tpr = float((pos >= tau).mean()) if len(pos) else float("nan")
    fpr = float((neg >= tau).mean()) if len(neg) else float("nan")
    return tpr, fpr


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """峰值信噪比（dB）。完全一致时返回 +inf。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return -float("inf")
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def _ssim_single(a: np.ndarray, b: np.ndarray, data_range: float, win: int) -> float:
    """单通道 SSIM（scipy.ndimage.uniform_filter 窗口统计）。"""
    from scipy.ndimage import uniform_filter
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_a = uniform_filter(a, size=win)
    mu_b = uniform_filter(b, size=win)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a2 = uniform_filter(a * a, size=win) - mu_a2
    sigma_b2 = uniform_filter(b * b, size=win) - mu_b2
    sigma_ab = uniform_filter(a * b, size=win) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)
    return float(np.mean(num / den))


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 255.0, win: int = 7) -> float:
    """结构相似性（多通道取平均）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return -1.0
    if a.ndim == 2:
        return _ssim_single(a, b, data_range, win)
    return float(np.mean([_ssim_single(a[..., c], b[..., c], data_range, win)
                          for c in range(a.shape[2])]))


def mean_quality(records: list) -> dict:
    """对 (ssim, psnr) 列表求均值；PSNR 为 inf（恒等）时保留原值。"""
    if not records:
        return {"ssim": None, "psnr": None, "n": 0}
    return {
        "ssim": float(np.mean([r[0] for r in records])),
        "psnr": float(np.mean([r[1] for r in records])),
        "n": len(records),
    }


def quality_ok(kind: str, ssim_v: float, psnr_v: float) -> bool:
    """按 §9.2(embed)/§9.3(attack) 阈值判定单样本质量是否达标。"""
    qc = QUALITY[kind]
    return bool(ssim_v >= qc["ssim_min"] and psnr_v >= qc["psnr_min"])


# --------------------------------------------------------------------------- #
# 图像合法性校验（赛题 §9.1）
# --------------------------------------------------------------------------- #
def validate_image(img, expected_shape) -> tuple[bool, str]:
    """校验尺寸/通道/dtype、无 NaN/Inf、值域 [0,255]、无大面积纯色。"""
    if img is None or not isinstance(img, np.ndarray):
        return False, "image is not ndarray"
    if img.shape != tuple(expected_shape):
        return False, f"shape {tuple(img.shape)} != {tuple(expected_shape)}"
    if img.dtype != np.uint8:
        return False, f"dtype {img.dtype} != uint8"
    f = img.astype(np.float32)
    if not np.all(np.isfinite(f)):
        return False, "contains NaN/Inf"
    if img.min() < 0 or img.max() > 255:
        return False, "value range outside [0,255]"
    # 大面积纯色覆盖：整体通道标准差极小
    if float(img.reshape(-1, img.shape[-1]).std(axis=0).max()) < 3.0:
        return False, "near-solid / large flat area"
    return True, "ok"


# --------------------------------------------------------------------------- #
# 单组合评测（赛题 §10：同源配对 + 对称攻击）
# --------------------------------------------------------------------------- #
def evaluate_pair(attack_fn, embed_fn, detect_fn, samples: list[dict],
                  attack_id: str, defense_id: str) -> dict:
    """对单个 (攻击, 防御) 组合执行全流程，返回指标字典。"""
    n = len(samples)
    y_true: list[int] = []
    y_score: list[float] = []

    embed_succ = embed_legal = embed_qok = 0
    atk_succ = atk_legal = atk_qok = 0
    det_succ = 0
    embed_q: list = []
    attack_q: list = []
    joint_q: list = []
    t_embed = t_attack = t_detect = 0.0

    for s in samples:
        I = np.asarray(s["image"])
        expected = tuple(I.shape)          # §7.1：输出尺寸/通道/dtype 与输入一致
        sid = s.get("sample_id", "")

        # ---- 1. 防御嵌入：正分支 ----
        t0 = time.perf_counter()
        ok, res = _safe(embed_fn, {"sample_id": sid, "image": I.copy()})
        t_embed += time.perf_counter() - t0
        embed_succ += int(ok)
        I_plus = I.copy()                  # 默认回退为原图（无水印）
        if ok:
            legal, qok, img_used = _process_embed(res, I, expected, embed_q)
            embed_legal += int(legal)
            embed_qok += int(qok)
            if img_used is not None:
                I_plus = img_used
        # 负分支 = 原始无水印图
        I_minus = I.copy()

        # ---- 2. 对称攻击：同一攻击同参作用于正负分支（不接收标签） ----
        t0 = time.perf_counter()
        ok_p, res_p = _safe(attack_fn, {"sample_id": sid, "image": I_plus.copy()})
        ok_n, res_n = _safe(attack_fn, {"sample_id": sid, "image": I_minus.copy()})
        t_attack += time.perf_counter() - t0
        atk_succ += int(ok_p and ok_n)

        legal_p, qok_p, atk_plus = _process_attack(res_p, ok_p, I_plus, expected, attack_q)
        legal_n, qok_n, atk_minus = _process_attack(res_n, ok_n, I_minus, expected, None)
        atk_legal += int(legal_p) + int(legal_n)
        atk_qok += int(qok_p) + int(qok_n)   # 正负分支各计一次，按 2n 归一

        # ---- 3. 防御检测 ----
        t0 = time.perf_counter()
        ok_dp, res_dp = _safe(detect_fn, {"image": atk_plus.copy()})
        ok_dn, res_dn = _safe(detect_fn, {"image": atk_minus.copy()})
        t_detect += time.perf_counter() - t0
        sp, ssp = _extract_probability(res_dp if ok_dp else None)
        sn, ssn = _extract_probability(res_dn if ok_dn else None)
        det_succ += int(ssp and ssn)

        y_true.extend([1, 0])
        y_score.extend([sp, sn])

        # 联合质量（攻击后正样本 vs 原图），仅报告
        joint_q.append((ssim(I, atk_plus), psnr(I, atk_plus)))

    auc = roc_auc(y_true, y_score)
    tpr, fpr = tpr_fpr(y_true, y_score, tau=TAU)

    return {
        "attack_id": attack_id, "defense_id": defense_id, "n": n,
        "auc": auc, "tpr": tpr, "fpr": fpr,
        "embed_success_rate": embed_succ / n if n else 0.0,
        "attack_success_rate": atk_succ / n if n else 0.0,
        "detect_success_rate": det_succ / n if n else 0.0,
        "embed_valid_rate": embed_legal / n if n else 0.0,
        "attack_valid_rate": atk_legal / (2 * n) if n else 0.0,
        "embed_quality_rate": embed_qok / n if n else 0.0,
        "attack_quality_rate": atk_qok / (2 * n) if n else 0.0,
        "embed_quality": mean_quality(embed_q),
        "attack_quality": mean_quality(attack_q),
        "joint_quality": mean_quality(joint_q),
        "time_embed": t_embed, "time_attack": t_attack, "time_detect": t_detect,
    }


def _safe(fn, *args):
    """异常隔离调用。返回 (ok, result_or_errorstr)。"""
    try:
        return True, fn(*args)
    except Exception as e:  # noqa: BLE001 - 故意宽捕获，隔离不可信提交
        return False, f"{type(e).__name__}: {e}"


def _process_embed(res, pre_image, expected, quality_log=None):
    """校验嵌入输出。返回 (legal, quality_ok, image_to_use)；非法/不达标->None（回退原图）。"""
    pre = np.asarray(pre_image)
    if not isinstance(res, dict) or "image" not in res:
        return False, False, None
    out = np.asarray(res["image"])
    vok, _ = validate_image(out, expected)
    if not vok:
        return False, False, None
    sv, pv = ssim(pre, out), psnr(pre, out)
    if quality_log is not None:
        quality_log.append((sv, pv))
    qok = quality_ok("embed", sv, pv)
    if qok or not STRICT:
        return True, qok, out
    return True, qok, None  # 合法但质量不达标 + strict -> 回退原图


def _process_attack(res, ok, pre_image, expected, quality_log=None):
    """校验攻击分支输出。返回 (legal, quality_ok, image_to_use)；非法/不达标->用攻击前图。

    quality_log 非 None 时记录该分支质量（仅正分支记录，作为代表，与赛题后端一致）。
    """
    pre = np.asarray(pre_image)
    if not ok or not isinstance(res, dict) or "image" not in res:
        return False, False, pre.copy()
    out = np.asarray(res["image"])
    vok, _ = validate_image(out, expected)
    if not vok:
        return False, False, pre.copy()
    sv, pv = ssim(pre, out), psnr(pre, out)
    if quality_log is not None:
        quality_log.append((sv, pv))
    qok = quality_ok("attack", sv, pv)
    if qok or not STRICT:
        return True, qok, out
    return True, qok, pre.copy()  # 合法但质量不达标 + strict -> 回退攻击前图


def _extract_probability(result):
    """从检测结果提取 watermark_probability；异常/缺失时返回 (0.5, False)。"""
    if not isinstance(result, dict):
        return 0.5, False
    p = result.get("watermark_probability", None)
    if p is None:
        return 0.5, False
    try:
        p = float(p)
    except (TypeError, ValueError):
        return 0.5, False
    if not math.isfinite(p):
        return 0.5, False
    return float(min(1.0, max(0.0, p))), True


# --------------------------------------------------------------------------- #
# 评测入口
# --------------------------------------------------------------------------- #
def evaluate(stage: str = "attack", n: int = DEFAULT_N, seed: int = DEFAULT_SEED,
             data_dir: str | None = None) -> dict:
    """运行攻防矩阵并计算指定阶段的指标。"""
    ds = load_benchmark(n=n, seed=seed, data_dir=data_dir)
    samples = to_samples(ds)
    print(f"[评测] 图像集: {len(samples)} 张 {ds['images'].shape[1]}x{ds['images'].shape[2]} RGB uint8\n")

    attacks = {"Id": identity_attack, "A0": attack}
    defenses = {"D0": (defense, detect),
                "null_defense": (null_defense_embed, null_defense_detect)}

    pairs: dict = {}
    total = len(attacks) * len(defenses)
    idx = 0
    for aid, afn in attacks.items():
        for did, (efn, dfn) in defenses.items():
            idx += 1
            t0 = time.time()
            pr = evaluate_pair(afn, efn, dfn, samples, aid, did)
            pairs[f"{aid}|{did}"] = pr
            print(f"  [{idx}/{total}] {aid} x {did} -> AUC={pr['auc']:.4f} "
                  f"({time.time()-t0:.1f}s)")

    def auc_of(a, d):
        return pairs[f"{a}|{d}"]["auc"]

    result = {
        "stage": stage,
        "n_samples": n,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE, 3],
        "seed": seed,
        "tau": TAU,
        "delta_a": DELTA_A,
        "pairs": pairs,
        "matrix": {
            "attacks": list(attacks),
            "defenses": list(defenses),
            "auc": {a: {d: auc_of(a, d) for d in defenses} for a in attacks},
        },
    }

    # ---- 攻击方指标（A0 相对 Id 对照；评分取对所有防御的平均） ----
    attack_aucs = [auc_of("A0", d) for d in defenses]
    r_a = float(np.mean(attack_aucs))
    clean_auc = auc_of("Id", "D0")
    attack_auc_d0 = auc_of("A0", "D0")
    auc_drop = clean_auc - attack_auc_d0
    result["attack_metrics"] = {
        "clean_auc_id_d0": clean_auc,
        "attack_auc_a0_d0": attack_auc_d0,
        "auc_drop": auc_drop,
        "attack_success": bool(auc_drop >= DELTA_A),
        "delta_a": DELTA_A,
        "attack_mean_auc_R_A": r_a,
        "attack_strength": float(1.0 - r_a),
        "tpr_fpr_id_d0": {"tpr": pairs["Id|D0"]["tpr"], "fpr": pairs["Id|D0"]["fpr"]},
        "tpr_fpr_a0_d0": {"tpr": pairs["A0|D0"]["tpr"], "fpr": pairs["A0|D0"]["fpr"]},
        "attack_quality_a0_d0": pairs["A0|D0"]["attack_quality"],
        "attack_quality_rate_a0_d0": pairs["A0|D0"]["attack_quality_rate"],
        "attack_valid_rate_a0_d0": pairs["A0|D0"]["attack_valid_rate"],
        "time_attack_total_s": pairs["A0|D0"]["time_attack"],
    }

    # ---- 防御方指标（D0 在 Id 与 A0 下的 AUC；评分取对所有攻击的平均） ----
    defense_aucs = [auc_of(a, "D0") for a in attacks]
    robust_auc = auc_of("A0", "D0")
    result["defense_metrics"] = {
        "clean_auc": clean_auc,
        "robust_auc_a0": robust_auc,
        "robustness_drop": clean_auc - robust_auc,
        "defense_mean_auc_R_D": float(np.mean(defense_aucs)),
        "min_auc": float(np.min(defense_aucs)),
        "tpr_fpr_clean": {"tpr": pairs["Id|D0"]["tpr"], "fpr": pairs["Id|D0"]["fpr"]},
        "tpr_fpr_robust": {"tpr": pairs["A0|D0"]["tpr"], "fpr": pairs["A0|D0"]["fpr"]},
        "embed_quality_id_d0": pairs["Id|D0"]["embed_quality"],
        "embed_quality_rate_id_d0": pairs["Id|D0"]["embed_quality_rate"],
        "embed_valid_rate_id_d0": pairs["Id|D0"]["embed_valid_rate"],
        "detect_success_rate_id_d0": pairs["Id|D0"]["detect_success_rate"],
        "time_embed_total_s": pairs["Id|D0"]["time_embed"],
        "time_detect_total_s": pairs["Id|D0"]["time_detect"],
    }
    return result


# --------------------------------------------------------------------------- #
# 报告打印
# --------------------------------------------------------------------------- #
def print_report(result: dict):
    m = result["matrix"]["auc"]
    print("\n" + "=" * 68)
    print(f"图片隐藏水印消除与鲁棒检测  |  阶段: {result['stage']}  |  "
          f"N={result['n_samples']}  seed={result['seed']}")
    print("=" * 68)
    print(f"攻防矩阵 M[a,d] = ROC-AUC（攻击 \\ 防御）:")
    defenses = result["matrix"]["defenses"]
    attacks = result["matrix"]["attacks"]
    header = "Attack\\Defense  " + "".join(f"{d:>14s}" for d in defenses)
    print(header)
    print("-" * len(header))
    for a in attacks:
        print(f"{a:<16s}" + "".join(f"{m[a][d]:>14.4f}" for d in defenses))

    am = result["attack_metrics"]
    print("-" * 68)
    print(f"[攻击方 A0] CleanAUC(Id×D0) = {am['clean_auc_id_d0']:.4f}")
    print(f"            AttackAUC(A0×D0) = {am['attack_auc_a0_d0']:.4f}")
    print(f"            AUC Drop        = {am['auc_drop']:.4f}  "
          f"AttackSuccess = {am['attack_success']}  (δ_a={am['delta_a']})")
    print(f"            AttackMeanAUC R_A = {am['attack_mean_auc_R_A']:.4f}  "
          f"AttackStrength = {am['attack_strength']:.4f}")
    print(f"            TPR/FPR(τ={TAU}) Id×D0: {am['tpr_fpr_id_d0']['tpr']:.3f}/{am['tpr_fpr_id_d0']['fpr']:.3f}"
          f"   A0×D0: {am['tpr_fpr_a0_d0']['tpr']:.3f}/{am['tpr_fpr_a0_d0']['fpr']:.3f}")
    aq = am["attack_quality_a0_d0"]
    print(f"            攻击质量 A0×D0: SSIM={aq['ssim']:.4f} PSNR={aq['psnr']:.1f}dB "
          f"合格率={am['attack_quality_rate_a0_d0']:.3f} 合法率={am['attack_valid_rate_a0_d0']:.3f}")

    dm = result["defense_metrics"]
    print("-" * 68)
    print(f"[防御方 D0] CleanAUC = {dm['clean_auc']:.4f}  RobustAUC(A0) = {dm['robust_auc_a0']:.4f}")
    print(f"            RobustnessDrop = {dm['robustness_drop']:.4f}  "
          f"DefenseMeanAUC R_D = {dm['defense_mean_auc_R_D']:.4f}  minAUC = {dm['min_auc']:.4f}")
    eq = dm["embed_quality_id_d0"]
    print(f"            嵌入质量: SSIM={eq['ssim']:.4f} PSNR={eq['psnr']:.1f}dB "
          f"合格率={dm['embed_quality_rate_id_d0']:.3f} 合法率={dm['embed_valid_rate_id_d0']:.3f}")
    print(f"            TPR/FPR(τ={TAU}) Clean: {dm['tpr_fpr_clean']['tpr']:.3f}/{dm['tpr_fpr_clean']['fpr']:.3f}"
          f"   Robust: {dm['tpr_fpr_robust']['tpr']:.3f}/{dm['tpr_fpr_robust']['fpr']:.3f}")
    print("=" * 68)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_result(result: dict, output: str):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"\n[test] 指标已保存: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="图片隐藏水印消除与鲁棒检测基准：Baseline 指标计算",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=["attack", "defense", "all"],
                        default="attack", help="评测阶段")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="每组合图像数量")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="数据生成种子")
    parser.add_argument("--data-dir", default=None, help="真实图片目录（可选）")
    parser.add_argument("--output", default="results/metrics.json", help="结果输出路径")
    args = parser.parse_args()

    res = evaluate(stage=args.stage, n=args.n, seed=args.seed, data_dir=args.data_dir)
    print_report(res)
    save_result(res, args.output)
    # 同时按阶段落一份便于对照（attack/defense 互不覆盖）
    if args.stage in ("attack", "all"):
        save_result(res, "results/metrics_attack.json")
    if args.stage in ("defense", "all"):
        save_result(res, "results/metrics_defense.json")
