---
title: >-
  [论文解读] Building Vision Models upon Heat Conduction
description: >-
  [CVPR 2025][LLM/NLP][vision backbone] 提出 vHeat 视觉 backbone，将图像 patch 建模为热源，利用物理热传导方程通过 DCT/IDCT 变换实现 $O(N^{1.5})$ 复杂度的信息传播…
tags:
  - "CVPR 2025"
  - "LLM/NLP"
  - "vision backbone"
  - "heat conduction"
  - "DCT"
  - "注意力机制"
  - "vHeat"
---

# Building Vision Models upon Heat Conduction

**会议**: CVPR 2025  
**arXiv**: [2405.16555](https://arxiv.org/abs/2405.16555)  
**代码**: [https://github.com/MzeroMiko/vHeat](https://github.com/MzeroMiko/vHeat)  
**领域**: LLM/NLP  
**关键词**: vision backbone, heat conduction, DCT, efficient attention, vHeat

## 一句话总结
提出 vHeat 视觉 backbone，将图像 patch 建模为热源，利用物理热传导方程通过 DCT/IDCT 变换实现 $O(N^{1.5})$ 复杂度的信息传播，在 ImageNet-1K 上以 3 倍吞吐量和 80% 更少 GPU 显存达到 84.0% top-1 准确率。

## 研究背景与动机

**领域现状**：视觉 backbone 从 CNN 演进到 ViT，再到各种高效注意力变体（Swin、线性注意力等）。Self-attention 虽然有效但 $O(N^2)$ 复杂度限制了在高分辨率输入上的应用。

**现有痛点**：现有高效注意力方法大多是 self-attention 的近似，要么牺牲全局交互能力（窗口注意力），要么牺牲精度（线性注意力的低秩问题）。缺乏从根本上不同于 attention 机制的全局信息传播范式。

**核心矛盾**：全局信息交互需要所有 token 间的交互（$O(N^2)$），但实际上物理世界中信息传播遵循偏微分方程（如热传导），天然具有全局性且可以高效求解。

**本文目标** 能否借鉴物理学中的热传导方程设计一种新的全局信息传播算子，既保持全局交互能力，又降低计算复杂度？

**切入角度**：热传导方程可通过频域（DCT）高效求解，而图像 patch 的信息传播可类比为热源之间的热量扩散。

**核心 idea**：将图像 patch 建模为热源，用可学习的频率热扩散率通过 DCT/IDCT 实现 $O(N^{1.5})$ 的全局信息传播。

## 方法详解

### 整体框架
vHeat 采用层级式架构（类似 Swin），分为 4 个阶段，分辨率逐步降低（H/4→H/8→H/16→H/32）。每个阶段由多个 vHeat Block 组成，每个 Block 包含 Heat Conduction Operator (HCO) 和 FFN。

### 关键设计

1. **Heat Conduction Operator (HCO)**

    - 功能：替代 self-attention 进行全局信息传播
    - 核心思路：将 2D 特征图视为温度场，每个 patch 是热源。利用热传导方程的频域解：先 DCT 变换到频域，乘以频率相关的热扩散系数，再 IDCT 变换回空域
    - 复杂度：$O(H \cdot W \cdot \log(H \cdot W))$，对于正方形图像约为 $O(N^{1.5})$
    - 设计动机：热传导方程的 Green 函数具有全局感受野但距离衰减的特性，天然适合建模局部优先、全局兼顾的视觉特征交互

2. **Learnable Frequency Value Embeddings (FVEs)**

    - 功能：为每个频率分量学习自适应的热扩散率
    - 核心思路：不同频率分量有不同的扩散速度，低频（全局结构）扩散快，高频（局部细节）扩散慢。FVE 预测每个频率的扩散系数 $\alpha(f)$
    - 设计动机：物理热传导中扩散率是材料常数，但视觉任务中不同特征通道和频率应该有不同的传播速度，因此设为可学习参数

3. **层级架构**

    - vHeat-Tiny: 各阶段 [2,2,6,2] 个 Block
    - vHeat-Small: 各阶段 [2,2,18,2] 个 Block
    - vHeat-Base: 各阶段 [2,2,18,2] 个 Block，更宽的通道

### 损失函数 / 训练策略
- ImageNet-1K 标准 300 epoch 训练
- AdamW 优化器，cosine 学习率调度

## 实验关键数据

### 主实验

**ImageNet-1K 分类：**

| 模型 | Top-1 | 吞吐量(img/s) | GPU显存 |
|------|-------|-------------|---------|
| vHeat-T | 82.2% | 1514 | — |
| vHeat-S | 83.6% | 945 | — |
| vHeat-B | 84.0% | 661 | — |
| Swin-B | 83.5% | ~470 | — |

vHeat-B 比 Swin-B 高 0.5%，吞吐量高 40%，GPU 显存少 80%，FLOPs 少 35%。

**COCO 目标检测（1× schedule）：**

| 模型 | mAP(box) | mAP(mask) | FPS |
|------|----------|-----------|-----|
| vHeat-B | 47.7 | 43.0 | 20.2 |
| Swin-B | 46.9 | 42.3 | 13.8 |

**ADE20K 语义分割：** vHeat-B 49.6 mIoU at 23.6 FPS

### 消融/扩展实验

| 应用 | vHeat 变体 | 对比方法 | 结果 |
|------|-----------|---------|------|
| 图像去噪 | vHeatIR | SwinIR | vHeatIR 更优 |
| JPEG去块 | vHeatIR | SwinIR | vHeatIR 更优 |
| ImageNet-A | vHeat-B | Swin-B | 鲁棒性更强 |
| ObjectNet | vHeat-B | Swin-B | 鲁棒性更强 |

### 关键发现
- 热传导算子在分类、检测、分割、低级视觉任务上全面优于 Swin
- 吞吐量优势源于 DCT/IDCT 的高效 FFT 实现
- 在分布外数据（ImageNet-A, ObjectNet）上鲁棒性更强，说明物理先验提供了有益的归纳偏置
- vHeatIR 在图像修复任务上也表现出色，说明 HCO 的通用性

## 亮点与洞察
- **物理启发的设计范式**：从热传导方程出发设计信息传播算子，是一种全新的思路，不同于对 attention 的各种近似
- **DCT 的巧妙应用**：热传导方程的频域解天然适合图像处理，DCT 本身就是图像压缩（JPEG）的核心工具
- **全局感受野 + 距离衰减**：热传导的 Green 函数自带这个特性，不需要像窗口注意力那样人为限制感受野
- 可以迁移到视频理解（时空热传导）和 3D 点云处理

## 局限与展望
- DCT/IDCT 在非正方形或非二的幂次分辨率上效率可能下降
- 热传导是各向同性的，而图像内容通常是各向异性的，可能需要方向性扩散
- 未在视频或 3D 任务上验证
- 与 attention 机制的互补性未探索（混合架构可能更好）

## 相关工作与启发
- **vs Swin Transformer**: 窗口注意力限制感受野，vHeat 天然全局交互且更快
- **vs VMamba**: 同为非 attention 的视觉 backbone，VMamba 用状态空间模型，vHeat 用热传导方程，物理先验不同
- **vs FNet**: FNet 用 FFT 替代 attention 但没有物理意义，vHeat 的热传导提供了更好的归纳偏置

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从物理方程出发设计视觉算子，思路独特
- 实验充分度: ⭐⭐⭐⭐ 分类+检测+分割+低级视觉全覆盖
- 写作质量: ⭐⭐⭐⭐ 物理动机阐述清晰
- 价值: ⭐⭐⭐⭐ 提供了 attention 之外的全新信息传播范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Learning Spatial Decay for Vision Transformers](../../AAAI2026/llm_nlp/learning_spatial_decay_for_vision_transformers.md)
- [\[AAAI 2026\] Vision Transformers are Circulant Attention Learners](../../AAAI2026/llm_nlp/vision_transformers_are_circulant_attention_learners.md)
- [\[ACL 2026\] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](../../ACL2026/llm_nlp/decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)
- [\[ICML 2025\] Defending LVLMs Against Vision Attacks through Partial-Perception Supervision](../../ICML2025/llm_nlp/defending_lvlms_against_vision_attacks_through_partial-perception_supervision.md)
- [\[CVPR 2025\] Exposure-slot: Exposure-centric Representations Learning with Slot-in-Slot Attention](exposure-slot_exposure-centric_representations_learning_with_slot-in-slot_attent.md)

</div>

<!-- RELATED:END -->
