---
title: >-
  [论文解读] Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences
description: >-
  [ICML 2025][图像生成][偏好优化] 提出 SmPO-Diffusion，通过平滑偏好建模替代二元偏好标签 + ReNoise Inversion 替代前向加噪估计，在大幅降低训练成本（比 DPO 快 6.5 倍，比 KTO 快 26 倍）的同时实现了 T2I 扩散模型偏好对齐的 SOTA 性能。
tags:
  - "ICML 2025"
  - "图像生成"
  - "偏好优化"
  - "扩散模型对齐"
  - "平滑偏好分布"
  - "ReNoise Inversion"
  - "DPO"
---

# Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences

**会议**: ICML 2025  
**arXiv**: [2506.02698](https://arxiv.org/abs/2506.02698)  
**代码**: [项目页面](https://jaydenlyh.github.io/SmPO-project-page/)  
**领域**: LLM对齐/RLHF  
**关键词**: 偏好优化, 扩散模型对齐, 平滑偏好分布, ReNoise Inversion, DPO

## 一句话总结

提出 SmPO-Diffusion，通过平滑偏好建模替代二元偏好标签 + ReNoise Inversion 替代前向加噪估计，在大幅降低训练成本（比 DPO 快 6.5 倍，比 KTO 快 26 倍）的同时实现了 T2I 扩散模型偏好对齐的 SOTA 性能。

## 研究背景与动机

当前 T2I 扩散模型的偏好对齐方法（如 Diffusion-DPO）存在两个核心问题：

**偏好标签过于粗糙**：现有数据集对图像对采用二元偏好标注（winner/loser），忽视了"审美因人而异"的事实——两张质量接近的图像被强制标注为一胜一负，导致过度优化（over-optimization）。

**优化目标估计不准确**：Diffusion-DPO 在估计扩散模型的轨迹偏好分布时，用前向加噪过程 $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$ 替代实际采样过程 $p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0)$，由于噪声是随机采样而非与图像相关的，导致目标不对齐（objective misalignment），训练效率低下。

作者的核心洞察是：在扩散模型框架中，现有方法在**偏好建模**和**优化估计**两个层面都存在显著不准确，需要同时解决。

## 方法详解

### 整体框架

SmPO-Diffusion 包含两个互补的改进模块：

- **Step 1 — 平滑偏好建模**：用奖励模型（PickScore）为所有图像对自动计算平滑偏好标签，取代人工标注的二元标签。
- **Step 2 — ReNoise Inversion 优化**：用 ReNoise Inversion 估计扩散模型的采样轨迹，取代原始 DPO 中基于前向加噪的估计，使优化目标更准确。

最终损失函数在标准 Diffusion-DPO 基础上仅引入了一个自适应权重因子 $(2\alpha - \gamma)$，实现为对 DPO loss 的简单修改。

### 关键设计

1. **平滑偏好分布（Smoothed Preference Distribution）**

   核心思路：用加权平均的混合分布替代二元分布，当两张图像偏好相似时，损失函数自然趋近于零，避免强制拉开差距导致的过度优化。

   具体做法：假设 winner 和 loser 的概率密度为加权混合形式 $\tilde{p}(\mathbf{x}_0^w|\mathbf{c}) \propto p(\mathbf{x}_0^w|\mathbf{c})^\alpha \cdot p(\mathbf{x}_0^l|\mathbf{c})^{\gamma-\alpha}$，其中 $\alpha$ 是权重因子、$\gamma$ 是灵敏度因子。代入 DPO 目标后，整个平滑偏好建模等效于在原始 DPO loss 前乘一个自适应系数 $(2\alpha - \gamma)$。当 $\alpha = 1, \gamma = 1$ 时退化为标准 DPO。

   设计动机：加权平均起到平滑作用，能有效调节似然的尺度——偏好越接近的图像对约束越弱，偏好差距大的图像对约束更强。

2. **奖励模型驱动的平滑标签生成**

   核心思路：无需人工标注，利用 PickScore 奖励模型自动生成平滑偏好标签。

   具体做法：将权重-灵敏度比 $\alpha/\gamma$ 定义为 winner 的概率：
    $\frac{\alpha}{\gamma} = \frac{\exp(r'(\mathbf{x}_0^w, \mathbf{c}))}{\exp(r'(\mathbf{x}_0^w, \mathbf{c})) + \exp(r'(\mathbf{x}_0^l, \mathbf{c}))}$
   其中 $r'$ 是对奖励分数的全局归一化（min-max normalization），然后通过 Softmax 得到概率值。灵敏度 $\gamma$ 固定为常数以控制波动幅度。

   设计动机：AI 奖励模型与人类偏好高度一致（RLHF 领域的共识），可以作为专家评分或人工投票的可靠替代，且无需额外人工标注成本。

3. **ReNoise Inversion 轨迹估计**

   核心思路：用 ReNoise Inversion 代替前向加噪来估计扩散模型的采样轨迹 $p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0)$，消除目标不对齐问题。

   具体做法分两步：
    - **Step A — 少步 DDIM Inversion**（≤10步）：从 $\mathbf{x}_0$ 出发，通过 DDIM Inversion 公式迭代得到 $\hat{\mathbf{x}}_t$ 的近似估计。
    - **Step B — 单步 ReNoise 修正**：在 $\hat{\mathbf{x}}_t$ 基础上，用当前模型在 $\hat{\mathbf{x}}_t$ 处的噪声预测做一步修正，得到更精确的 $\tilde{\mathbf{x}}_t$。

   修正后的 score function 变为：$\tilde{s}_\theta^t = \|\tau_t - \epsilon_\theta^t(\tilde{\mathbf{x}}_t, \mathbf{c})\|^2 - \|\tau_t - \epsilon_{\text{ref}}^t(\tilde{\mathbf{x}}_t, \mathbf{c})\|^2$，其中 $\tau_t = (\tilde{\mathbf{x}}_t - \sqrt{\bar\alpha_t}\mathbf{x}_0)/\sqrt{1-\bar\alpha_t}$。

   设计动机：前向加噪使用的随机高斯噪声与图像无关，而 Inversion 得到的隐变量与图像高度相关，能更准确地估计优化目标，从而大幅提升训练效率。

### 损失函数 / 训练策略

最终损失函数：
$$\mathcal{L}(\theta) = -\mathbb{E}_{t,\mathcal{D}} \log \sigma\left(-(2\alpha - \gamma)\beta \left(\tilde{s}_\theta^t(\mathbf{x}_0^w, \mathbf{c}) - \tilde{s}_\theta^t(\mathbf{x}_0^l, \mathbf{c})\right)\right)$$

训练配置：
- 数据集：Pick-a-Pic v2（851K 数据对，59K unique prompts）
- 优化器：SD1.5 用 AdamW，SDXL 用 Adafactor
- 8×A800 GPU，batch size=1/GPU，128 步梯度累积 → 有效 batch size=1024
- SD1.5: $\beta=2000$；SDXL: $\beta=5000$
- DDIM Inversion 步数=9，CFG=1，$\gamma=10$

## 实验关键数据

### 主实验

| 模型 | PickScore↑ | HPSv2.1↑ | ImReward↑ | Aesthetic↑ | GPU Hours↓ |
|------|-----------|----------|-----------|-----------|-----------|
| Base-SDXL | 22.75 | 28.45 | 0.881 | 6.114 | - |
| DPO-SDXL | 23.13 | 30.06 | 1.184 | 6.112 | ~976 |
| MaPO-SDXL | 22.81 | 29.11 | 1.224 | 6.309 | ~834 |
| **SmPO-SDXL** | **23.62** | **32.53** | **1.331** | 6.264 | **~151** |
| Base-SD1.5 | 20.83 | 23.61 | -0.078 | 5.390 | - |
| DPO-SD1.5 | 21.29 | 25.11 | 0.195 | 5.530 | ~205 |
| KTO-SD1.5 | 21.54 | 28.28 | 0.706 | 5.692 | ~1056 |
| **SmPO-SD1.5** | **22.08** | **29.31** | **0.885** | **5.831** | **~41** |

HPDv2 测试集 median score。SmPO-SDXL 对 DPO-SDXL 在 HPSv2.1 上 win-rate 达 86.7%，训练时间仅为 15.5%。

### 消融实验

| 配置 | PickScore↑ | HPSv2.1↑ | ImReward↑ | 说明 |
|------|-----------|----------|-----------|------|
| DPO (baseline) | 21.29 | 25.11 | 0.195 | 标准 Diffusion-DPO |
| +DDIM Inversion | 21.72 | 28.71 | 0.761 | 用 Inversion 替代前向加噪 |
| +ReNoise | 21.87 | 29.01 | 0.778 | 加一步 ReNoise 修正 |
| +Smoothed Pref | **22.08** | **29.31** | **0.885** | 加平滑偏好建模（完整方法）|

| 超参数消融 | 最优值 | 要点 |
|-----------|--------|------|
| Inversion 步数 | 9步（平衡质量/效率） | 19步最优但 GPU 翻倍 |
| CFG during Inversion | 1 | DDIM Inversion 对 prompt 敏感 |
| 灵敏度 $\gamma$ | 10 | 太小→对 reward 不敏感；太大→过度优化 |
| 正则化 $\beta$ | 2000 | 太小→退化为纯 reward 模型；太大→KL 限制过强 |

### 关键发现

1. **DDIM Inversion → ReNoise → 平滑偏好**三个模块逐步叠加均带来显著提升，验证了两个核心假设的正确性。
2. **训练效率极其突出**：SmPO-SD1.5 仅需 41.3 GPU-hours，是 KTO 的 1/26，是 DPO 的 1/5。
3. 用 PickScore 作为奖励模型效果最好，因为它本身可视为 Pick-a-Pic 数据集的伪标签，相当于数据清洗。
4. SmPO 训练的模型可直接用于 ControlNet 条件生成（canny/depth map），无需额外训练。

## 亮点与洞察

1. **极致简洁的改进**：整个方法等效于在 DPO loss 上乘一个自适应系数 + 把加噪方式从前向过程换成 Inversion，代码改动极小但效果显著。
2. **训练效率和性能双赢**：不是用更多资源换性能，而是通过更准确的建模和估计同时提升了两者。
3. **消除标签噪声的优雅方式**：用 reward model 生成的 soft label 替代 hard label，既不需要人工标注，也能减少标签噪声带来的过度优化。
4. **Inversion 的新用途**：将图像编辑领域的 DDIM Inversion/ReNoise 技术创新性地应用于偏好优化中的轨迹估计。

## 局限与展望

1. **数据集偏见**：依赖 Pick-a-Pic v2 数据集，其中可能存在性别刻板印象等社会偏见，导致模型对中性 prompt 生成过度女性化的图像。
2. **离线学习**：当前方法是 offline 的，未来可集成到 online learning 中实现持续性能提升。
3. **奖励模型依赖**：平滑标签的质量完全取决于 PickScore 的质量，换用其他 reward model 性能有波动（Table 7）。
4. **Inversion 计算开销**：虽然总体比 DPO 快很多，但 Inversion 步数增加仍会线性增加训练时间。

## 相关工作与启发

- **Diffusion-DPO**（Wallace et al., 2023）：首次将 DPO 应用于扩散模型，优化轨迹联合分布的上界，是本文的直接基线。
- **MaPO**：联合最大化偏好/非偏好集的似然差距，无需参考模型。
- **DDIM-InPO**（Lu et al., 2025）：用 DDIM Inversion 对齐特定隐变量，与本文思路相关但只用了 Inversion 没有 ReNoise。
- **ReNoise Inversion**（Garibi et al., 2024）：原本用于图像编辑中的精确 Inversion，本文创新性地将其引入偏好优化。
- 对 LLM 对齐的启发：soft label + 更精确的分布估计的思路可推广到 LLM DPO。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 两个改进点各自不新（soft label、Inversion），但组合应用于扩散模型偏好优化是新颖的
- **实验充分度**: ⭐⭐⭐⭐⭐ — 双模型（SD1.5/SDXL）、五个评估指标、全面消融、条件生成测试
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，推导完整，数学符号规范
- **价值**: ⭐⭐⭐⭐⭐ — 训练效率提升 5-26 倍+性能 SOTA，对实际应用价值极大

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment](../../CVPR2025/image_generation/inpo_inversion_preference_optimization_diffusion_alignment.md)
- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](../../NeurIPS2025/image_generation/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[CVPR 2025\] Calibrated Multi-Preference Optimization for Aligning Diffusion Models](../../CVPR2025/image_generation/calibrated_multi-preference_optimization_for_aligning_diffusion_models.md)
- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](../../CVPR2025/image_generation/curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[AAAI 2026\] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference](../../AAAI2026/image_generation/margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)

</div>

<!-- RELATED:END -->
