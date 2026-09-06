#!/usr/bin/env bash
# =============================================================================
# 攻击方终端脚本 —— 图片隐藏水印消除与鲁棒检测基准（Baseline：A0 组合后处理攻击）
#
# 在新机器上以默认参数运行：
#     pip install -r requirements.txt
#     bash run_attack.sh
#
# 输出：攻防矩阵 M[a,d]（Id/A0 × D0/null_defense）、Clean AUC、Attack AUC、
#       AUC Drop、AttackSuccess（δ_a=0.02）、AttackStrength、TPR/FPR、攻击质量
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# 指定解释器：默认 python，可用 PYTHON 环境变量覆盖
PY="${PYTHON:-python}"

echo ">>> [run_attack] 攻击方 Baseline 评测（A0: JPEG75 -> 缩放90% -> 双线性 -> 高斯模糊σ0.4）"
"$PY" test.py --stage attack
