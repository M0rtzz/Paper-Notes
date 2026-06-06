---
title: >-
  [论文解读] Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling
description: >-
  [ICML2026][多模态VLM][扩散模型奖励建模] 提出 DiNa-LRM，将偏好学习直接建立在扩散模型的噪声潜空间上，通过噪声校准的 Thurstone 似然和推理时多噪声集成，以远低于 VLM 奖励模型的计算开销实现接近 SOTA 的偏好预测精度。
tags:
  - "ICML2026"
  - "多模态VLM"
  - "扩散模型奖励建模"
  - "偏好对齐"
  - "噪声校准Thurstone"
  - "潜空间奖励"
  - "测试时噪声集成"
---

# Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling

**会议**: ICML2026  
**arXiv**: [2602.11146](https://arxiv.org/abs/2602.11146)  
**代码**: https://github.com/HKUST-C4G/diffusion-rm  
**领域**: 图像生成  
**关键词**: 扩散模型奖励建模, 偏好对齐, 噪声校准Thurstone, 潜空间奖励, 测试时噪声集成

## 一句话总结
提出 DiNa-LRM，将偏好学习直接建立在扩散模型的噪声潜空间上，通过噪声校准的 Thurstone 似然和推理时多噪声集成，以远低于 VLM 奖励模型的计算开销实现接近 SOTA 的偏好预测精度。

## 研究背景与动机

**领域现状**：扩散/Flow-Matching 模型的偏好对齐（如 ReFL、DPO、GRPO）依赖奖励模型提供监督信号。当前主流做法是使用 VLM（如 Qwen2VL-7B）作为奖励骨干，在像素空间对生成图像打分。

**现有痛点**：VLM 奖励模型存在两个核心问题。其一，计算和显存成本高昂，在对齐训练中需反复调用奖励评估，开销随之累积。其二，潜空间扩散生成器与像素空间 VLM 奖励之间存在 **域不匹配**（latent-to-pixel mismatch），需要额外的 VAE 解码步骤，并使基于奖励梯度的对齐方法更加复杂。

**核心矛盾**：扩散模型的生成预训练已经学到了丰富的判别性表征（已被证明可迁移到分类、对抗判别等任务），但现有工作并未充分挖掘其作为通用奖励模型的潜力——尤其是在与 VLM 相同的"对干净样本打分"场景下。

**本文目标**：构建一个直接在扩散潜空间中运行的奖励模型，使其 (1) 偏好预测精度接近 VLM 奖励、(2) 对齐训练时显存和计算更友好、(3) 提供推理时可扩展的鲁棒打分机制。

**切入角度**：作者观察到扩散模型在不同噪声水平下提供了同一样本的多个"视角"，如果能在偏好建模中显式引入噪声不确定性校准，就可以同时利用这些互补视角来增强鲁棒性。

**核心 idea**：将 Thurstone 偏好模型从干净样本扩展到扩散噪声状态，用与噪声水平成正比的比较不确定性来校准偏好似然，并在推理时通过多噪声集成实现测试时扩展。

## 方法详解

### 整体框架
输入为带文本 prompt $\bm{c}$ 的偏好对 $(\bm{x}_0^+, \bm{x}_0^-)$（在 VAE 潜空间中），通过前向加噪得到 $(\bm{x}_t^+, \bm{x}_t^-)$。预训练的扩散骨干（SD3.5-Medium）提取多层视觉/文本特征，经 FiLM 时间步调制后送入门控 Q-Former 打分头，输出标量奖励 $r_\theta(\bm{x}_t, t, \bm{c})$。训练使用噪声校准的 Thurstone 似然 + Fidelity Loss；推理时支持单噪声评估或多噪声 token 级集成。

### 关键设计

1. **噪声校准 Thurstone 偏好建模**:

    - 功能：将偏好学习从干净样本扩展到扩散噪声状态，使奖励模型的输入分布与扩散预训练保持一致
    - 核心思路：标准 Thurstone 模型假设感知质量 $u = r_\theta(\bm{x}_0, \bm{c}) + \eta$（$\eta \sim \mathcal{N}(0, \sigma_u^2)$），本文将比较不确定性设为噪声水平的函数 $\sigma_u^2(t) = k \cdot \sigma^2(t) + \sigma_u^2$，其中 $k=2$, $\sigma_u=0.1$。偏好概率变为 $\mathbb{P}(\bm{x}_t^+ \succ \bm{x}_t^-) = \Phi\big(\frac{r_\theta(\bm{x}_t^+, t, \bm{c}) - r_\theta(\bm{x}_t^-, t, \bm{c})}{\sqrt{2\sigma_u^2(t)}}\big)$，高噪声区域自动产生更保守的似然，防止无信息梯度破坏训练稳定性
    - 设计动机：扩散骨干预训练处理的是噪声状态而非干净样本，直接在 $\bm{x}_0$ 上学习会产生分布偏移；噪声校准让模型在不同噪声级别下学到多样且互补的特征，尤其有利于推理时集成

2. **时间步感知潜空间奖励架构**:

    - 功能：从预训练扩散骨干中提取多层特征，经时间步条件化适配后聚合为标量奖励
    - 核心思路：从骨干选定层集合 $\mathcal{S}$ 提取视觉和文本 token 特征，对每层特征施加 FiLM 调制（基于时间步嵌入 $t_{\text{emb}}$），投影到低维子空间后跨层拼接融合为统一视觉 $\mathbf{V}_t$ 和文本 $\mathbf{T}_t$ 序列。然后使用 $N_q$ 个可学习 query token 通过门控值交叉注意力（value-gated cross-attention）聚合两个序列，经 FFN 后均值池化 + MLP 输出标量 $r_\theta = \text{MLP}(\text{Pool}(\tilde{\mathbf{Q}}))$
    - 设计动机：FiLM 调制使打分头显式感知噪声级别；query-based 架构天然支持可变长度输入，为多噪声集成提供无缝接口

3. **推理时多噪声集成（测试时扩展）**:

    - 功能：通过聚合多个噪声水平下的特征产生更鲁棒的奖励分数，作为扩散原生的测试时扩展旋钮
    - 核心思路：对干净样本 $\bm{x}_0$ 在 $K$ 个不同时间步 $\{t_k\}_{k=1}^K$ 加噪，分别通过骨干提取特征并经 FiLM 适配，将所有时间步的 token 特征拼接为 $\mathbf{V}_{\text{ensemble}} \in \mathbb{R}^{(K \times N_v) \times C}$，然后用同一个 Q-Former 头一次性打分。默认使用 $t \in \{0.2, 0.5, 0.7\}$ 覆盖低/中/高噪声区间
    - 设计动机：不同噪声水平强调表征的不同方面（低噪声保留细节、高噪声捕获全局语义），token 级拼接比简单平均更灵活，让 Q-Former 自行学习跨噪声级别的注意力权重

### 训练策略
使用 Fidelity Loss $\mathcal{L}_{\text{fid}} = \mathbb{E}[1 - \sqrt{y\hat{p}_\theta + (1-y)(1-\hat{p}_\theta)}]$ 优化，时间步从 $\mathcal{U}(0,1)$ 均匀采样。在 HPDv3 数据集（~0.8M 偏好对）上训练 1 epoch，8 GPU，AdamW（lr=$5 \times 10^{-5}$），EMA 衰减 0.995。骨干使用 LoRA 微调。

## 实验关键数据

### 主实验

| 模型类别 | 模型 | 骨干 | ImageReward | HPDv2 | HPDv3 | GenAI-Bench | 平均 |
|---------|------|------|-------------|-------|-------|-------------|------|
| CLIP-based | MPS | CLIP | 66.37 | 83.27 | 64.33 | 68.08 | 70.51 |
| VLM-based | HPSv3 | Qwen2VL-7B | 67.03 | **85.36** | **76.03** | **70.95** | **74.84** |
| VLM-based | UnifiedReward | LLaVA-OV-7B | 63.82 | 83.10 | 71.96 | 72.38 | 72.81 |
| Diffusion-based | LRM-SDXL | SDXL | 60.35 | 71.19 | 53.80 | 61.58 | 61.73 |
| Diffusion-based | **DiNa-LRM** | SD3.5-M-2B | 60.34 | 82.13 | 75.04 | 68.43 | 71.49 |
| Diffusion-based | **DiNa-LRM*** | SD3.5-M-2B | 61.75 | 84.31 | 74.86 | 68.98 | **72.48** |

DiNa-LRM 比此前扩散奖励基线 LRM-SDXL 平均精度提升 **+9.76%**，并接近最强 VLM 奖励 HPSv3（72.48 vs 74.84）。

### 消融实验

| 配置 | HPDv2 | HPDv3 | GenAI-Bench | 平均 |
|------|-------|-------|-------------|------|
| Uniform + Noise-Calibrated（完整模型） | 82.13 | 75.04 | 68.43 | 71.49 |
| Uniform + Fixed variance | 78.72 | 75.11 | 68.01 | 70.68 |
| Const $t=0$ + Fixed | 59.20 | 74.37 | 67.55 | 64.93 |
| Uniform + Noise-Calibrated + Ensemble | **84.31** | 74.86 | **68.98** | **72.48** |
| Freeze backbone（无 LoRA） | — | 73.52 | 67.09 | 70.27 |

### 对齐效率分析（ReFL on SD3.5-M, 1024×1024）

| 指标 | HPSv3 (VLM) | DiNa-LRM | 节省 |
|------|-------------|----------|------|
| 峰值显存 | ~40 GB | ~19.4 GB | **51.4%** |
| 奖励计算 TFLOPS | ~8.5 | ~2.5 | **71.1%** |
| 优化阶段 TFLOPS | ~14 | ~7.5 | **46.4%** |

### 关键发现
- **噪声校准方差是核心贡献**：在 HPDv2 上从 78.72→82.13（+3.4%），集成后更从 78.16→84.31（+6.2%），说明噪声感知的不确定性建模让不同时间步学到了更互补的特征
- 最优推理噪声水平在 $t \in [0.3, 0.7]$，过干净（$t=0$）或过嘈杂（$t=0.8$）都会降低精度
- 分布式时间步采样（Uniform/LogitNormal）显著优于固定时间步训练，平均精度从 64.93~68.75 提升至 70.58~71.49
- 在 ReFL 对齐中，DiNa-LRM 的代理分数收敛更快，且持出金标准（PickScore）同步上升，无明显奖励劫持

## 亮点与洞察
- **扩散模型作为通用奖励骨干的可行性**：证明扩散预训练表征不仅可以生成，还能高质量判别偏好，为"一个骨干两个用途"提供了新范式，可将对齐管线全部保持在潜空间中运行
- **噪声校准 Thurstone 的巧妙之处**：通过一个简单的线性关系 $\sigma_u^2(t) = k\sigma^2(t) + \sigma_u^2$ 就将扩散噪声调度与偏好学习的不确定性建模统一起来，优雅且有效
- **token 级集成优于分数级平均**：将多时间步特征拼接后让 Q-Former 统一注意力聚合，而非简单平均多次打分，这个设计可迁移到任何需要多视角融合的判别任务

## 局限与展望
- 奖励在特定骨干的潜空间中学习和评估，跨骨干迁移性有限（SD3.5→FLUX 需要重新训练）
- 潜空间建模可能忽视某些像素级伪影（如纹理失真），长程奖励优化可能出现奖励劫持（虚假目标插入、风格漂移）
- 在 ImageReward 测试集上的精度（~61%）仍明显低于 VLM 方法（~67%），提示某些语义理解能力仍不足
- 未来可探索：(1) 在更强统一骨干上训练提升泛化性，(2) 增加轻量像素空间正则化约束，(3) 生成式或稠密奖励建模

## 相关工作与启发
- **CLIP-based RM**（ImageReward, PickScore, HPSv2）：计算高效但受限于 CLIP 表征能力上界
- **VLM-based RM**（HPSv3, UnifiedReward）：精度最高但计算昂贵且在像素空间运行
- **扩散判别性表征**（DDPMClassifier, DiffAE）：先验工作证明扩散预训练特征可迁移到分类等判别任务
- **并发工作 LRM**（Zhang et al., 2025）：在噪声中间状态上做步级奖励用于轨迹优化，而本文目标是通用偏好对齐场景下的干净样本打分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling](mitigating_perceptual_judgment_bias_in_multimodal_llm-as-a-judge_via_perceptual_.md)
- [\[ICLR 2026\] GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?](../../ICLR2026/multimodal_vlm/glyph-sr_can_we_achieve_both_high-quality_image_super-resolution_and_high-fideli.md)
- [\[ICML 2026\] Conditional Diffusion Sampling](conditional_diffusion_sampling.md)
- [\[CVPR 2026\] VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery](../../CVPR2026/multimodal_vlm/vlm-guided_group_preference_alignment_for_diffusion-based_human_mesh_recovery.md)
- [\[NeurIPS 2025\] Systematic Reward Gap Optimization for Mitigating VLM Hallucinations](../../NeurIPS2025/multimodal_vlm/systematic_reward_gap_optimization_for_mitigating_vlm_hallucinations.md)

</div>

<!-- RELATED:END -->
