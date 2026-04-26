---
title: >-
  [论文解读] Shining Yourself: High-Fidelity Ornaments Virtual Try-on with Diffusion Model
description: >-
  [CVPR 2025][图像生成][虚拟试穿] 首次将扩散模型应用于饰品（手镯、戒指、耳环、项链）虚拟试戴任务，提出迭代式姿态感知佩戴蒙版预测和蒙版引导注意力机制，在大姿态和大尺度差异下实现高保真的几何结构保持。
tags:
  - CVPR 2025
  - 图像生成
  - 虚拟试穿
  - 饰品试戴
  - 扩散模型
  - 姿态对齐
  - 注意力引导
---

# Shining Yourself: High-Fidelity Ornaments Virtual Try-on with Diffusion Model

**会议**: CVPR 2025  
**arXiv**: [2503.16065](https://arxiv.org/abs/2503.16065)  
**代码**: [项目主页](https://shiningyourself.github.io/)  
**领域**: 图像生成  
**关键词**: 虚拟试穿, 饰品试戴, 扩散模型, 姿态对齐, 注意力引导

## 一句话总结

首次将扩散模型应用于饰品（手镯、戒指、耳环、项链）虚拟试戴任务，提出迭代式姿态感知佩戴蒙版预测和蒙版引导注意力机制，在大姿态和大尺度差异下实现高保真的几何结构保持。

## 研究背景与动机

- 服装虚拟试穿已有大量研究且趋于成熟，但饰品虚拟试戴几乎未被探索，尽管有巨大的商业需求
- 饰品试戴面临三大独特挑战：(1) 饰品具有精细微小的几何结构（如环形结构、重复子结构），保持困难；(2) 饰品通常是刚性物体，任何变形或伪影都会被立即察觉（不像服装可以被自然褶皱掩盖）；(3) 无法像服装试穿那样依赖骨架和语义图，因为饰品的轮廓蒙版因姿态遮挡而极难获取
- 现有服装试穿方法（OOTDiffusion、IDM-VTON）需要额外输入（骨架图、语义图），不适用于饰品
- 饰品参考图通常是近景特写，与模特图的尺度差异远大于服装，需要更精确的姿态对齐
- 通用图像编辑方法（AnyDoor、Paint-by-Example）虽可插入物体但缺乏姿态对齐能力

## 方法详解

### 整体框架

基于潜在扩散模型和 ReferenceNet 构建。输入仅需参考饰品图像、目标模特图像和一个粗略边界框。ReferenceNet 提取饰品特征注入去噪 U-Net。核心创新包含两个模块：(1) 迭代式姿态感知蒙版预测，从粗略边界框逐步精炼为精确的佩戴蒙版；(2) 蒙版引导注意力，利用参考蒙版到佩戴蒙版的隐式映射约束保持几何结构。

### 关键设计

**1. 迭代式姿态感知蒙版精炼**
- **功能**: 从粗略边界框逐步估计精确的佩戴蒙版，实现饰品与模特的姿态和尺度对齐
- **核心思路**: 在 ReferenceNet 中添加线性层，从中间特征预测佩戴蒙版 $\hat{M}_p^t = \text{MLP}([f_m^t \odot \hat{M}_p^{t-1}, f_o^t])$。预测蒙版与初始边界框按动态权重 $\alpha_t$ 混合后作为下一步输入，迭代精炼。蒙版预测使用 $L_2$ 损失与真值对齐
- **设计动机**: 一次性预测蒙版太粗糙（中间特征在早期阶段包含的语义信息有限），迭代精炼使蒙版质量随去噪过程逐步提升。$\alpha_t$ 从小到大渐变确保早期依赖边界框稳定性、后期依赖预测精确性

**2. 蒙版引导注意力 (Mask-guided Attention)**
- **功能**: 保持饰品的精细几何结构（如重复图案、环形结构、子组件数量）
- **核心思路**: 从 U-Net 各层提取注意力图 $\{M_a^i\}$，用参考饰品蒙版 $M_o^i$ 沿一维掩码、另一维求和，将结果上采样为同尺寸蒙版 $\tilde{M}_o$。要求 $\tilde{M}_o$ 与真值佩戴蒙版一致（$L_2$ 损失），从而隐式约束注意力图学习参考蒙版到佩戴蒙版的映射
- **设计动机**: 直接在注意力图上施加蒙版会遮蔽过多信息导致退化。间接方式通过约束注意力图的输出蒙版一致性，让注意力自动学习几何对应关系，既保持结构又不限制生成自由度

**3. 零样本推理设计**
- **功能**: 无需额外输入即可推理
- **核心思路**: 推理时仅需参考饰品图、模特图和一个边界框（指示佩戴位置）。蒙版由迭代预测模块自动估计，无需骨架、语义图或精确轮廓
- **设计动机**: 饰品佩戴位置具有用户特定性（如戒指戴哪个手指），边界框是最简单直观的交互方式

### 损失函数

$$\mathcal{L}_{total} = \mathcal{L}_1 + \lambda_1 \mathcal{L}_2 + \lambda_2 \mathcal{L}_3$$

其中 $\mathcal{L}_1$ 为扩散噪声预测损失，$\mathcal{L}_2 = \|\hat{M}_p^T - M_o^{gt}\|_2^2$ 为蒙版预测损失，$\mathcal{L}_3 = \|\tilde{M}_o - M_o^{gt}\|_2^2$ 为注意力引导的蒙版映射损失。$\lambda_1, \lambda_2$ 随训练衰减。

## 实验关键数据

### 主实验：定量对比

| 方法 | FID↓ | LPIPS↓ | CLIP Score↑ | DINO Score↑ |
|------|------|--------|-------------|-------------|
| Paint-by-Example | 23.49 | 0.0789 | 85.6 | 64.8 |
| AnyDoor | 28.28 | 0.1029 | 85.1 | 67.2 |
| IDM-VTON | 22.99 | 0.0709 | 85.9 | 65.0 |
| **Ours** | **19.00** | **0.0593** | **88.7** | **74.5** |

*在所有指标上全面领先，FID 比最佳基线低 17%*

### 消融实验

| 配置 | FID↓ | DINO Score↑ |
|------|------|-------------|
| 无蒙版预测 | ~24 | ~66 |
| 无蒙版引导注意力 | ~21 | ~70 |
| **完整方法** | **19.0** | **74.5** |

### 关键发现

- 蒙版预测模块对姿态对齐至关重要，蒙版引导注意力对几何细节保持至关重要
- 现有方法均无法保持饰品的外观和结构一致性，尤其是几何细节和组件数量
- 训练数据集约 64K 图像三元组，均匀分布于四类饰品

## 亮点与洞察

1. **开辟新任务**: 首次系统性地定义和解决饰品虚拟试戴问题，填补了虚拟试穿领域的重要空白
2. **隐式蒙版映射**: 不直接操作注意力图而通过约束其输出蒙版一致性，是保持精细结构的优雅方案
3. **最小输入需求**: 仅需边界框即可推理，比服装试穿方法（骨架+语义图+DensePose）简单得多

## 局限与展望

- 对极端遮挡和极端姿态差异的处理能力有限
- 数据集规模（64K）相对较小，可能限制泛化能力
- 多饰品同时试戴的场景未探索
- 未来可扩展到视频域的动态饰品试戴

## 相关工作与启发

- 与 OOTDiffusion、IDM-VTON 等服装方法对比，本方法无需额外输入且更适合小尺度精细物体
- 迭代蒙版精炼的思路可推广到其他需要精确空间对齐的图像编辑任务
- 蒙版引导注意力机制为扩散模型中的几何保持提供了新范式

## 评分

⭐⭐⭐⭐ — 开创性地定义了饰品虚拟试戴任务，两个核心模块（迭代蒙版预测+蒙版引导注意力）设计巧妙且互补。实验结果令人信服，实际应用价值高。但数据集和评估规模相对有限，对极端情况的鲁棒性需进一步验证。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction](pursuing_temporal-consistent_video_virtual_try-on_via_dynamic_pose_interaction.md)
- [\[CVPR 2026\] PROMO: Promptable Outfitting for Efficient High-Fidelity Virtual Try-On](../../CVPR2026/image_generation/promo_promptable_virtual_tryon_efficient.md)
- [\[CVPR 2025\] BooW-VTON: Boosting In-the-Wild Virtual Try-On via Mask-Free Pseudo Data Training](boow-vton_boosting_in-the-wild_virtual_try-on_via_mask-free_pseudo_data_training.md)
- [\[CVPR 2025\] GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing](glyphmastero_a_glyph_encoder_for_high-fidelity_scene_text_editing.md)
- [\[CVPR 2026\] Garments2Look: A Multi-Reference Dataset for High-Fidelity Outfit-Level Virtual Try-On with Clothing and Accessories](../../CVPR2026/image_generation/garments2look_a_multi-reference_dataset_for_high-fidelity_outfit-level_virtual_t.md)

<!-- RELATED:END -->
