---
title: >-
  [论文解读] SplitFlow: Flow Decomposition for Inversion-Free Text-to-Image Editing
description: >-
  [NeurIPS 2025][图像生成][矫正流] 提出 SplitFlow，将目标 prompt 语义分解为多个子 prompt，为每个子 prompt 计算独立的编辑流，再通过投影和自适应聚合机制组合成统一编辑轨迹，解决梯度纠缠问题，在无需反转的前提下实现更高保真度和可编辑性的文本引导图像编辑。
tags:
  - NeurIPS 2025
  - 图像生成
  - 矫正流
  - 无反转编辑
  - 流分解
  - 梯度冲突
  - 多任务学习
---

# SplitFlow: Flow Decomposition for Inversion-Free Text-to-Image Editing

**会议**: NeurIPS 2025  
**arXiv**: [2510.25970](https://arxiv.org/abs/2510.25970)  
**代码**: https://github.com/Harvard-AI-and-Robotics-Lab/SplitFlow  
**领域**: 图像编辑 / 扩散模型  
**关键词**: 矫正流, 无反转编辑, 流分解, 梯度冲突, 多任务学习

## 一句话总结

提出 SplitFlow，将目标 prompt 语义分解为多个子 prompt，为每个子 prompt 计算独立的编辑流，再通过投影和自适应聚合机制组合成统一编辑轨迹，解决梯度纠缠问题，在无需反转的前提下实现更高保真度和可编辑性的文本引导图像编辑。

## 研究背景与动机

基于矫正流（Rectified Flow）的图像编辑方法已成为主流：RF-Inversion、RF-Solver、FireFlow 等通过 ODE 反转步骤将真实图像映射回潜空间后进行编辑。然而反转过程不精确，累积误差导致语义漂移和视觉失真。FlowEdit 提出无反转方法直接操作速度场差异，但编辑质量仍有限。

核心矛盾：当目标 prompt 包含多个属性时（如"戴黑色墨镜、张嘴的德牧犬在草地上跳跃"），单一编辑轨迹对全部属性的引导会导致**梯度纠缠和方向冲突**，使得编辑不充分或过度失真。

本文的切入角度是借鉴多任务学习中梯度冲突解决的思想：将语义复杂的目标 prompt 分解为多个语义子 prompt，分别计算独立编辑流后自适应聚合，从而同时保证编辑多样性和全局一致性。

## 方法详解

### 整体框架

SplitFlow 包含两个阶段：
1. **流分解（Flow Decomposition）**：将目标 prompt 用 LLM 分解为 N 个子 prompt，为每个子 prompt 计算独立的无反转编辑流
2. **流聚合（Flow Composition）**：通过潜轨迹投影（LTP）和速度场聚合（VFA）将子流组合成统一编辑轨迹

### 关键设计

1. **基于 LLM 的 prompt 分解（流分解阶段）**：

    - 使用 Mistral-7B 作为 prompt 推理引擎，输入指令 + 源 prompt + 目标 prompt
    - 自动将语义差异分解为 N 个子目标 prompt（通常 N≤3）
    - 例如："德牧犬张嘴跳跃戴墨镜" → {"戴墨镜的狗跳跃", "张嘴的狗跳跃", "戴墨镜的德牧"}
    - 基于 FlowEdit 的无反转公式，为每个子 prompt 定义独立的编辑流 x_t^{FE(i)}
    - 分解阶段从 η_max=33 起持续到 η_dec=28（5 步）

2. **潜轨迹投影（Latent Trajectory Projection, LTP）**：

    - 确保全局语义一致性：将每个子目标潜表示投影到由完整目标 prompt 引导的目标潜表示方向上
    - 计算方法：x_t^{proj(i)} = ⟨x_t^{FE(i)}, x̂_t^{FE}⟩ * x̂_t^{FE}（沿通道维度内积）
    - 聚合投影潜表示：x_t^{proj} = (1/N) Σ x_t^{proj(i)}
    - 保持目标轨迹的全局一致性，同时保留子 prompt 引入的局部变化

3. **速度场聚合（Velocity Field Aggregation, VFA）**：

    - 计算子目标流之间的相对速度向量：g_i = v_θ(x_t^{proj(i)}, t, φ^{tgt(i)}) - v_θ(x_t^{src}, t, φ^{src})
    - 用余弦相似度衡量子流之间的方向一致性
    - 通过 softmax 操作得到空间自适应权重图 w∈R^{N×H×W}
    - 聚合权重 w_i(h,w) = exp(Σ_{j≠i} S_ij(h,w)) / Σ_k exp(Σ_{j≠k} S_kj(h,w))
    - 高一致性的子流获得更高权重，抑制冗余流、强调独特方向
    - 数学证明：VFA 的加权聚合在语义对齐方面严格优于简单平均

### 损失函数 / 训练策略

- SplitFlow 是一个**零样本、免训练**的方法，无需任何训练
- 基于 Stable Diffusion 3 / 3.5 的矫正流模型
- T=50 步，η_max=33（跳过前 1/3 步），η_dec=28（分解持续 5 步）
- 源 CFG=3.5，目标 CFG=13.5
- 推理时间：PIE-Bench 700 张图约 83 分钟（FlowEdit 57 分钟，额外 LLM 分解约 20 分钟）

## 实验关键数据

### 主实验（PIE-Bench）

| 方法 | 模型 | Structure Dist↓ | PSNR↑ | LPIPS↓ | SSIM↑ | CLIP Whole↑ | CLIP Edited↑ |
|------|------|--------|-------|--------|-------|-------------|-------------|
| FlowEdit | SD3 | 27.24 | 22.13 | 105.46 | 83.48 | 26.83 | 23.67 |
| iRFDS | SD3 | 62.72 | 19.61 | 186.39 | 74.59 | 24.54 | 21.67 |
| FTEdit | SD3.5 | 18.17 | 26.62 | 80.55 | 91.50 | 25.74 | 22.27 |
| **SplitFlow** | SD3 | 25.96 | 22.45 | 102.14 | 83.91 | **26.96** | **23.83** |
| **SplitFlow†** | SD3 | 14.55 | 25.22 | 68.53 | 87.54 | 26.23 | 23.01 |
| **SplitFlow** | SD3.5 | **11.68** | **27.12** | **52.93** | **89.76** | 26.29 | 22.89 |

### 消融实验

| 配置 | Structure Dist↓ | PSNR↑ | CLIP Whole↑ | CLIP Edited↑ |
|------|--------|-------|-------------|-------------|
| FlowEdit 基线 | 27.24 | 22.13 | 26.83 | 23.67 |
| 简单平均（AVG）| 22.28 | 23.36 | 26.81 | 23.67 |
| + LTP | 26.22 | 22.37 | 26.93 | 23.82 |
| + LTP + VFA (SplitFlow) | **25.96** | **22.45** | **26.96** | **23.83** |

### 关键发现

- 即使简单平均子流也能大幅提升背景保持（PSNR 22.13→23.36），但 CLIP 相似度未提升
- LTP 显著提升 CLIP 相似度（编辑质量），VFA 在此基础上进一步兼顾保真度和可编辑性
- SplitFlow† 的保真度增强设置下性能超越使用更强骨干的 FTEdit
- 不同 LLM（Mistral、Qwen2、LLaMA2）和不同 prompt 策略下均一致超越基线，体现鲁棒性
- 聚合时间步 η_dec=28 为最优平衡点；η_dec 越小分解越长，可编辑性越好但保真度降低
- 总推理步数约 48 步（3×5+33），略高于基线但质量显著提升

## 亮点与洞察

- **从多任务学习到图像编辑的类比非常新颖**：将速度场视为梯度，子 prompt 视为不同任务，引入梯度投影和聚合来解决冲突
- **数学证明 VFA 严格优于简单平均**：通过 Gibbs 不等式和 Jensen 不等式证明加权聚合的语义对齐度 ≥ 均匀平均
- **无训练、零样本的方法设计**：使用 LLM 做 prompt 分解是优缺兼备的巧妙选择——无需数据但引入了对 LLM 的依赖
- **分解-聚合框架高度模块化**：分解阶段和聚合阶段可独立调整，增强方法灵活性

## 局限与展望

- 推理时间较基线增加约 45%（83 vs 57 分钟），加上 LLM 分解的 20 分钟
- prompt 分解质量依赖 LLM 的能力和 prompt 工程
- LLM 仅作为代理实现分解，未来可探索基于 VLM 或优化的分解方法
- 在极端编辑场景下可能仍会出现失真

## 相关工作与启发

- 与 FlowEdit（无反转直接编辑）相比，SplitFlow 通过流分解解决了梯度纠缠问题
- 借鉴多任务学习中的 PCGrad、Nash-MTL 等梯度冲突解决方法
- 对未来图像编辑研究的启发：将编辑过程按语义分解是提升复杂编辑质量的有效路径

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](../../ICCV2025/image_generation/flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)
- [\[ICML 2025\] Taming Rectified Flow for Inversion and Editing](../../ICML2025/image_generation/taming_rectified_flow_for_inversion_and_editing.md)
- [\[CVPR 2025\] Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing](../../CVPR2025/image_generation/unveil_inversion_and_invariance_in_flow_transformer_for_versatile_image_editing.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] ALE: Attribute-Leakage-free Editing for Text-based Image Editing](../../ICCV2025/image_generation/ale_attribute_leakage_free_editing.md)

<!-- RELATED:END -->
