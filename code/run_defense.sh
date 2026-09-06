#!/usr/bin/env bash
# =============================================================================
# 防守方终端脚本 —— 图片隐藏水印消除与鲁棒检测基准（Baseline：D0 DCT 中频扩频水印）
#
# 在新机器上以默认参数运行：
#     pip install -r requirements.txt
#     bash run_defense.sh
#
# 输出：攻防矩阵 M[a,d]、Clean AUC、Robust AUC（A0 攻击后）、Robustness Drop、
#       DefenseMeanAUC R_D、min AUC、TPR/FPR、嵌入质量（SSIM/PSNR 与合格率）
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# 指定解释器：默认 python，可用 PYTHON 环境变量覆盖
PY="${PYTHON:-python}"

echo ">>> [run_defense] 防守方 Baseline 评测（D0: DCT 中频扩频零比特水印，ALPHA=2.0）"
"$PY" test.py --stage defense
