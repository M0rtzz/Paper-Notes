---
title: >-
  [论文解读] Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement
description: >-
  [ECCV 2024][图像生成] 提出了一种通用的频率解耦学习范式，通过拉普拉斯分解和低频一致性约束，将低频（光照恢复）和高频（去噪）增强解耦为两个独立子任务，仅需88K额外参数即可为6种SOTA低光增强模型带来最高7.68dB的PSNR提升。 低光图像增强（LLIE）面临低频（光照恢复）和高频（噪声去除）耦合优化的挑战…
tags:
  - "ECCV 2024"
  - "图像生成"
---

# Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement

**会议**: ECCV 2024  
**arXiv**: [2409.01641](https://arxiv.org/abs/2409.01641)  
**领域**: 图像生成

## 一句话总结

提出了一种通用的频率解耦学习范式，通过拉普拉斯分解和低频一致性约束，将低频（光照恢复）和高频（去噪）增强解耦为两个独立子任务，仅需88K额外参数即可为6种SOTA低光增强模型带来最高7.68dB的PSNR提升。

## 研究背景与动机

低光图像增强（LLIE）面临低频（光照恢复）和高频（噪声去除）耦合优化的挑战。现有方法通常使用统一框架同时处理这两类退化，但低频调整可能放大噪声，高频恢复可能影响光照强度恢复，导致次优结果。

**核心问题**：如何设计一种通用的频率解耦范式，使其能 (1) 无缝集成现有LLIE方法，(2) 提升频率恢复能力，(3) 仅需极少额外复杂度？

与现有频率分解方法不同，本文不仅分解图像频率，更关键的是**解耦了低频和高频的优化过程**，通过低频一致性损失实现有效的解耦学习。

## 方法详解

### 整体框架

两阶段框架：
1. **粗调阶段（Coarse Phase）**：ACCA模块主要恢复低频信息（光照），产生初步增强结果 $I_l$
2. **粗到精阶段（Coarse-to-Fine Phase）**：LDRM利用拉普拉斯分解表征，结合原始输入和粗调结果进行精细高频恢复

### 关键设计

**ACCA（自适应卷积组合聚合）模块**：
- 双分支结构：局部分支（W-CCA）+ 全局ISP分支
- W-CCA通过分步卷积将2D特征分割为非重叠patches，利用张量分解技术（3个1D张量组合）生成3D Omni相似度图，实现高效的空间-通道聚合
- 计算复杂度为 $O(4HWC + 2HWC^2/s)$，与图像分辨率线性增长
- 仅88K参数，在LOL-v2上超越Retinexformer 1dB（仅其5.5%参数量）

**LDRM（拉普拉斯解耦恢复模型）**：
- 利用拉普拉斯金字塔将图像分解为多尺度高频和低频分量
- 仅需修改SOTA模型的首尾卷积层即可集成
- 将原始输入和粗调结果的拉普拉斯分解图堆叠后输入，生成增强的拉普拉斯图
- 通过逆拉普拉斯变换重建最终输出，设 $K=4$ 层效果最优

### 损失函数

总损失函数：$L_{total} = L_r + \alpha \cdot L_i$

- **重建损失** $L_r$：多尺度预测与GT对应分解图的L1范数之和
- **低频一致性损失** $L_i$：约束LDRM输出的最粗级特征图与粗调阶段的低频分量一致，$L_i = \|m_d^K - m_l^K\|_1$

低频一致性损失是实现频率解耦优化的核心——它强制LDRM保留粗调阶段已恢复的低频信息，使其专注于高频增强。

## 实验关键数据

### 主实验

在5个基准上对6种SOTA模型的提升效果（PSNR(dB)/SSIM）：

| 方法 | LOL-v2 | SID | SDSD-in | SDSD-out | SMID |
|------|--------|-----|---------|----------|------|
| MIR-Net → MIR-Net-De | +4.17/+0.062 | +3.34/+0.075 | +3.76/+0.036 | +2.32/+0.047 | +1.16/+0.072 |
| Restormer → Restormer-De | +4.62/+0.066 | +2.49/+0.045 | **+6.11/+0.091** | **+7.68/+0.102** | +2.14/+0.058 |
| LLFlow → LLFlow-De | +1.70/+0.020 | +2.92/+0.064 | +5.09/+0.034 | +8.83/+0.057 | +1.53/+0.017 |
| SNR → SNR-De | +2.52/+0.023 | +0.68/+0.042 | +0.87/+0.007 | +3.32/+0.031 | +1.99/+0.017 |
| Retinexformer → Retinexformer-De | +1.41/+0.041 | +0.20/+0.014 | +0.77/+0.013 | +3.67/+0.028 | +1.70/+0.013 |
| Diff-L → Diff-L-De | +4.98/+0.081 | +2.03/+0.104 | +4.80/+0.031 | +4.14/+0.056 | +1.31/+0.034 |

额外模型开销：仅 +88K 参数、+2.53 GFLOPS、+0.008s 推理时间（256×256输入），占原模型参数的0.2%-5.5%。

### 消融实验

ACCA与低频一致性损失的有效性验证（基于Restormer，LOL-v2）：

| 配置 | PSNR/SSIM | 参数/FLOPs |
|------|-----------|------------|
| Restormer (Baseline) | 19.94/0.827 | 26.13M/144.25G |
| + ACCA, 无 $L_i$ | 20.21/0.837 | 26.22M/146.78G |
| + ACCA + $L_i$（完整方案） | **24.56/0.893** | 26.22M/146.78G |

不同粗调模型替代ACCA时对Restormer的提升：

| 粗调方法 | PSNR增益 | SSIM增益 |
|----------|----------|----------|
| ZeroDCE | +1.59 | +0.034 |
| Star | +2.46 | +0.039 |
| PairLIE | +2.42 | +0.033 |
| IAT | +4.03 | +0.060 |
| ACCA (Ours) | **+4.62** | **+0.066** |

### 关键发现

1. 低频一致性损失 $L_i$ 是性能提升的关键，去掉后PSNR从24.56降至20.21
2. 框架对CNN、Transformer、流模型、扩散模型均有效，证明了通用性
3. ACCA虽仅88K参数但可达到甚至超越大型LLIE模型的粗调效果
4. 方法可灵活替换粗调模块，不同轻量模型均能带来显著提升

## 亮点与洞察

- **即插即用范式**：几乎零额外成本地提升任意LLIE模型，框架设计极为优雅
- **频率解耦优化的关键洞察**：低频一致性约束将复杂的联合优化问题分解为两个更简单的子问题
- **ACCA的高效设计**：张量分解技术将3D相似度图的回归分解为3个1D张量，大幅降低计算量
- 最高7.68dB的PSNR提升在图像恢复领域是非常罕见的改进幅度

## 局限性

- 方法依赖于拉普拉斯金字塔的频率分解假设，对于频率特征不明显的退化可能效果有限
- 需要分两阶段训练（先ACCA再LDRM），训练流程较为繁琐
- 对于已经在频率域进行了较好优化的模型（如SNR、Retinexformer），提升幅度相对较小

## 评分

- **新颖性**: 8/10 — 频率解耦优化范式简洁有效，低频一致性损失是巧妙的设计
- **技术深度**: 7/10 — 理论分析清晰，但主要贡献在范式层面而非模型架构创新
- **实验充分性**: 9/10 — 6种基线×5个数据集的全面验证，消融实验详尽
- **应用价值**: 9/10 — 即插即用特性使得实际应用门槛极低

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Shedding More Light on Robust Classifiers under the lens of Energy-based Models](shedding_more_light_on_robust_classifiers_under_the_lens_of_energy-based_models.md)
- [\[ECCV 2024\] Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)
- [\[ECCV 2024\] ReNoise: Real Image Inversion Through Iterative Noising](renoise_real_image_inversion_through_iterative_noising.md)
- [\[ECCV 2024\] Removing Distributional Discrepancies in Captions Improves Image-Text Alignment](removing_distributional_discrepancies_in_captions_improves_image-text_alignment.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)

</div>

<!-- RELATED:END -->
