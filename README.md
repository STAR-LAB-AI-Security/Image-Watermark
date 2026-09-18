# 图片隐藏水印消除与鲁棒检测 —— 代码说明

本目录实现"图片隐藏水印消除与鲁棒检测"攻防对抗基准（Benchmark）的可运行代码包：
攻击方基线 **A0 组合后处理攻击**、防御方基线 **D0 DCT 中频扩频零比特水印**，
以及完整评测流水线与指标计算。全部基于 numpy / scipy / Pillow，仅需 CPU，
分钟级即可跑通。

## 目录结构

```
code/
├── dataset.py       # 基准数据集：程序化合成 512×512 RGB 图像集（7 类视觉内容）+ 缓存
├── attack.py        # 攻击 Baseline：A0（JPEG75→缩放90%→双线性→高斯模糊σ0.4）+ 恒等对照
├── defense.py       # 防御 Baseline：D0（DCT 中频扩频零比特水印）embed/detect + null 对照
├── test.py          # Benchmark 指标计算：攻防矩阵、ROC-AUC、TPR/FPR、PSNR/SSIM、评分
├── run_attack.sh    # 攻击方终端脚本：python test.py --stage attack
├── run_defense.sh   # 防守方终端脚本：python test.py --stage defense
└── requirements.txt # 最小依赖（numpy / scipy / Pillow）
```

## 快速开始

```bash
cd code
pip install -r requirements.txt

# 攻击方评测（默认 N=48，约 1-2 分钟）
bash run_attack.sh

# 防守方评测（默认 N=48，约 1-2 分钟）
bash run_defense.sh
```

结果写入 `results/metrics.json`（同时另存 `results/metrics_attack.json` 与
`results/metrics_defense.json` 便于对照）。

## 评测流程（赛题 §6/§10：同源配对 + 对称攻击）

对每个 (攻击 a, 防御 d) 组合，从同一批原图 I 构造：

```
正分支: I -> D_d.embed -> A_a.attack -> D_d.detect -> s+
负分支: I ──────────────> A_a.attack -> D_d.detect -> s-
y_true = [1..1, 0..0]，y_score = [s+.., s-..]
M[a,d] = ROC-AUC(y_true, y_score)
```

- 攻击/检测不接收标签，同一攻击同参对称作用于正负分支；
- ROC-AUC 保留原始方向（不做 `max(AUC, 1-AUC)` 修正）；
- 非法处理（§9.6）：嵌入非法 → 正样本用原图；攻击非法/质量不达标（strict）→
  用攻击前图进检测；检测异常 → 中性 0.5。

## 指标

- **单组合指标**：ROC-AUC M[a,d]（Mann-Whitney U，平均秩处理并列）；
- **检测率**：固定决策阈值 τ=0.5 下的 TPR 与 FPR；
- **攻击方**：CleanAUC（Id×D0）、AttackAUC（A0×D0）、AUC Drop、
  AttackSuccess（AUC Drop ≥ δ_a=0.02）、AttackMeanAUC R_A（对全部防御的平均，
  越低越好）、AttackStrength = 1 - R_A、攻击质量 SSIM/PSNR 与合格率
  （门槛：SSIM≥0.92 / PSNR≥28dB）；
- **防御方**：CleanAUC、RobustAUC（A0 攻击后）、DefenseMeanAUC R_D（对全部攻击
  的平均，越高越好）、min AUC、嵌入质量 SSIM/PSNR 与合格率
  （门槛：SSIM≥0.97 / PSNR≥36dB）。

## 数据集

默认数据源为**程序化合成**（`dataset.py` 内置 7 类生成器：色彩梯度、纹理、
正弦条纹、几何形状、海报图表、人像感、文档截图，固定 seed=42，可复现），
首次运行生成并缓存到 `data/synth_images.npz`。也可用真实图片目录：

```bash
python test.py --stage attack --data-dir /path/to/images --n 32
```

## 说明

- 图像统一格式：RGB uint8，512×512×3（与赛题 §5.3 一致）；
- `null_defense` 为赛题 §21 的最小对照防御（不嵌入、恒输出 0.5），其 AUC 应 ≈ 0.5，
  用于验证评测流水线与 ROC-AUC 语义；
- 若未安装 Pillow，A0 的 JPEG 步骤自动降级为"缩放+模糊"，攻击效果略减弱，流水线不中断。
