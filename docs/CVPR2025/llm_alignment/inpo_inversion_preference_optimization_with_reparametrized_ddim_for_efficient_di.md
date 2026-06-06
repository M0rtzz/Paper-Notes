---
title: >-
  [论文解读] InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment
description: >-
  [CVPR 2025][LLM对齐][DDIM反演] 提出 InPO（Inversion Preference Optimization），通过 DDIM 反演的重参数化技巧将偏好优化从需要完整去噪链的长马尔可夫过程简化为单步优化，在训练效率和生成质量上同时优于现有 Diffusion-DPO 方法。
tags:
  - "CVPR 2025"
  - "LLM对齐"
  - "DDIM反演"
  - "偏好优化"
  - "扩散模型对齐"
  - "重参数化"
  - "高效训练"
---

# InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment

**会议**: CVPR 2025  
**arXiv**: 待公开  
**代码**: 无  
**领域**: 模型对齐 / 扩散模型  
**关键词**: DDIM反演, 偏好优化, 扩散模型对齐, 重参数化, 高效训练

## 一句话总结
提出 InPO（Inversion Preference Optimization），通过 DDIM 反演的重参数化技巧将偏好优化从需要完整去噪链的长马尔可夫过程简化为单步优化，在训练效率和生成质量上同时优于现有 Diffusion-DPO 方法。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**：将 DPO（Direct Preference Optimization）应用于扩散模型已成为提升 T2I 生成质量的重要方向，但扩散模型的多步采样特性给偏好优化带来了独特挑战。

**现有痛点**：

### 现有痛点

**现有痛点**：扩散模型的生成过程是长链马尔可夫过程（通常 20-50 步），DPO 需要计算整条链的对数概率，计算量巨大

### 核心矛盾

**核心矛盾**：现有的 Diffusion-DPO 方法在训练时需要对偏好样本进行完整的前向/反向过程，训练效率低

### 解决思路

**解决思路**：由于链长且依赖关系复杂，梯度回传时信号衰减严重，优化不稳定

### 补充说明

**补充说明**：生成质量提升有限 — 长链中的噪声累积影响了偏好信号的有效传递

**核心矛盾**：DPO 要求计算整条生成轨迹的对数概率，但扩散模型的长链采样使得这一计算既昂贵又不稳定。

**本文目标** 如何高效地计算扩散模型生成轨迹的偏好概率，避免完整长链采样的计算瓶颈。

**切入角度**：利用 DDIM 的确定性反演特性，将"从噪声到图像"的长链概率计算转化为"从图像到噪声"的反演 + 单步优化。

**核心 idea**：用 DDIM 反演将已知偏好图像映射回噪声空间，在反演后的噪声空间中直接进行偏好对比，避免完整的前向链采样。

## 方法详解

### 整体框架
InPO 的训练流程：(1) 对偏好数据中的 win/lose 图像用 DDIM 反演到噪声空间，得到对应的潜在噪声对；(2) 在噪声空间中通过重参数化的 DDIM 公式计算每步的偏好概率；(3) 用 DPO 损失优化模型参数，但无需模型本身进行完整的采样过程。

### 关键设计

1. **DDIM 反演重参数化**:
    - 功能：将偏好图像通过 DDIM 反演映射到初始噪声，建立图像-噪声之间的确定性对应关系
    - 核心思路：DDIM 的确定性采样意味着一张图像唯一对应一个初始噪声。通过反演，可以在噪声空间而非图像空间进行偏好比较
    - 设计动机：在噪声空间中比较避免了需要完整生成链来评估概率的计算瓶颈

2. **单步偏好损失**:
    - 功能：在每个时间步 t 独立计算偏好损失，而非需要整条链的联合概率
    - 核心思路：利用 DDIM 的重参数化，将整条链的 log-probability 分解为各步的独立贡献，每步可单独优化
    - 设计动机：独立的步级优化避免了长链梯度衰减问题，优化更稳定

3. **高效训练策略**:
    - 功能：通过预计算反演噪声和随机时间步采样，大幅降低训练开销
    - 核心思路：反演只需做一次并缓存，训练时随机采样时间步而非遍历所有步，将计算量从 O(T) 降至 O(1)
    - 设计动机：完整链的梯度计算需要 O(T) 次前向传播，随机步采样将其降至常数级

## 实验关键数据

### 主实验

| 方法 | 训练效率 (GPU hr) | 美学评分 | 提示对齐 | 生成多样性 |
|------|------------------|---------|---------|-----------|
| Diffusion-DPO | 高开销 | 中等提升 | 保持 | 较低 |
| D3PO | 高开销 | 中等提升 | 保持 | 中等 |
| **InPO** | **低开销** | **显著提升** | **提升** | **保持** |

### 关键发现
- InPO 的训练时间比标准 Diffusion-DPO 减少约 50-70%，主要来自于避免了完整链的采样
- 反演重参数化提供了更干净的偏好信号，生成质量提升更大
- 单步独立优化有效缓解了长链梯度衰减问题
- 预计算反演噪声可以离线完成，不影响在线训练效率

## 亮点与洞察
- **反演 = 免采样**：利用 DDIM 反演的确定性巧妙地绕过了完整采样链的计算瓶颈，是一个优雅的数学洞察
- **训练效率的实质性改善**：不是在工程层面减少计算量，而是从算法层面减少了必要的计算量
- **与 DDIM 特性的深度结合**：方法的有效性建立在 DDIM 的确定性反演特性上，体现了对扩散模型采样机制的深入理解

## 局限与展望
- 依赖 DDIM 的确定性反演，不适用于随机采样器（如 DDPM、Euler Ancestral）
- DDIM 反演本身存在近似误差，步数较少时反演不精确
- 仅在文本到图像生成上验证，图像编辑、视频生成等场景有待探索
- 可以结合 Consistency Models 等新型采样器探索更高效的反演优化方法

## 相关工作与启发

- **vs Diffusion-DPO**: Diffusion-DPO 直接在完整去噪链上计算偏好概率，训练开销大且梯度信号弱。InPO 通过反演重参数化将问题转化为噪声空间的单步优化，效率和效果双赢
- **vs D3PO**: D3PO 使用在线RL的方式优化扩散模型，需要在线采样和奖励模型评估，计算量更大。InPO 完全离线，无需额外的奖励模型
- **vs SPO（Step-aware Preference Optimization）**: SPO 也尝试在步级别做偏好优化，但仍需前向采样来获取各步的输出。InPO 通过反演直接获取各步的潜变量，避免了前向采样
- InPO 将 DDIM 反演的确定性与 DPO 的偏好学习优雅结合，这一思路可以迁移到视频生成、图像编辑等其他扩散模型对齐任务

## 评分

- 新颖性: ⭐⭐⭐⭐ DDIM反演+偏好优化的结合思路巧妙，但核心组件（DDIM反演、DPO）均为已有技术
- 实验充分度: ⭐⭐⭐⭐ 仅400步微调即达SOTA，效率提升显著，但缺乏大规模人类偏好评估
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，动机说服力强
- 价值: ⭐⭐⭐⭐ 显著降低了扩散模型对齐的训练成本，对T2I模型实际部署有直接影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[ICML 2025\] Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences](../../ICML2025/llm_alignment/smoothed_preference_optimization_via_renoise_inversion_for_aligning_diffusion_mo.md)
- [\[CVPR 2025\] Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation](sam_dpo_semi_supervised.md)
- [\[CVPR 2025\] Calibrated Multi-Preference Optimization for Aligning Diffusion Models](calibrated_multi-preference_optimization_for_aligning_diffusion_models.md)
- [\[CVPR 2025\] Boost Your Human Image Generation Model via Direct Preference Optimization](boost_your_human_image_generation_model_via_direct_preference_optimization.md)

</div>

<!-- RELATED:END -->
