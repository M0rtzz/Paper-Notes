---
title: >-
  [论文解读] Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models
description: >-
  [NeurIPS 2025][图像生成][文本嵌入优化] 提出 DATE（Diffusion Adaptive Text Embedding），在扩散模型采样过程中根据当前去噪中间结果动态更新文本嵌入，无需额外训练即可提升文本-图像语义对齐。
tags:
  - NeurIPS 2025
  - 图像生成
  - 文本嵌入优化
  - 扩散模型采样
  - 文本-图像对齐
  - 测试时优化
  - 自适应条件
---

# Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.23974](https://arxiv.org/abs/2510.23974)  
**代码**: [https://github.com/aailab-kaist/DATE](https://github.com/aailab-kaist/DATE)  
**领域**: 图像生成  
**关键词**: 文本嵌入优化, 扩散模型采样, 文本-图像对齐, 测试时优化, 自适应条件

## 一句话总结

提出 DATE（Diffusion Adaptive Text Embedding），在扩散模型采样过程中根据当前去噪中间结果动态更新文本嵌入，无需额外训练即可提升文本-图像语义对齐。

## 研究背景与动机

文本到图像扩散模型依赖预训练文本编码器（CLIP、T5）将提示编码为固定嵌入，并在所有采样步保持不变。但扩散过程不同时间步对图像的影响截然不同：早期步决定全局结构，后期步细化细节。使用静态嵌入无法适应这种语义演化，容易导致概念混淆（如"一个人"生成"两个人"）。

现有改进方向包括微调模型参数（代价高）、数据空间引导（Universal Guidance，需精心调 guidance scale）、提示层优化（需要训练额外语言模型）。这些方法都忽略了文本嵌入本身的优化价值——一个可以在测试时零训练代价优化的关键变量。EBCA 尝试在 cross-attention 层做 energy-based 更新，但缺乏全局语义控制，FID 严重恶化。P2L 直接优化文本嵌入用于逆问题，但局限于特定任务。DATE 提供了一个通用、无训练的框架，适用于任意文本到图像扩散模型。

## 方法详解

### 整体框架

DATE 将固定文本嵌入替换为时间步、实例自适应的动态嵌入。在每个采样步：(1) 利用 Tweedie 公式从含噪数据和当前嵌入估计均值预测图像 $\bar{\mathbf{x}}_0$；(2) 计算评估函数 $h(\bar{\mathbf{x}}_0; y)$ 对文本嵌入的梯度；(3) 沿归一化梯度方向更新嵌入；(4) 用更新后的嵌入执行标准去噪步。整个过程不修改网络参数和结构，完全在推理时运行。

### 关键设计

1. **时间步级目标分解**: 原始目标是最大化最终生成图像的评估函数值，涉及全部采样步的联合优化。DATE 将其分解为顺序式逐步优化：在时间步 $t$，约束 $\tau < t$ 的嵌入等于 $\mathbf{c}_t$，利用 Tweedie 公式（一次前向传播）估计 $\bar{\mathbf{x}}_0$ 并计算 $h$。通过 Taylor 展开将问题转化为嵌入空间的约束优化。理论上，这种分解等价于原始联合优化（Proposition 1），保证不劣于固定嵌入。

2. **归一化梯度单步更新**: 利用 Cauchy-Schwarz 不等式，推导出约束优化的闭式解：$\hat{\mathbf{c}}_t = \mathbf{c}_{\text{org}} + \rho \cdot \nabla_{\mathbf{c}} h_t / \|\nabla_{\mathbf{c}} h_t\|_2$。归一化确保更新幅度恒定（由 $\rho$ 控制），受 SAM（Sharpness-Aware Minimization）启发。理论分析（Theorem 2）表明，更新后的嵌入等价于在原始 score function 上添加一个嵌入域引导项，兼顾语义对齐与模型分布保持。

3. **计算效率策略**: 仅在部分采样步更新（如 10%），未更新步复用最近更新的嵌入。实验发现以前一步更新嵌入作为下一步起点（而非每次回到初始嵌入）可更广泛探索嵌入空间，提升 CLIP score。中后期时间步更新对对齐性能贡献更大，与细节生成阶段吻合。支持 FP16 推理进一步降低开销（时间从 7.82min → 4.40min，显存从 61.5GB → 32.9GB）。

### 损失函数 / 训练策略

DATE 无需训练。目标函数为在线最大化评估函数 $h_t$，可为任意可微文本-图像对齐指标：CLIP Score（语义对齐）、ImageReward（人类偏好）、PickScore、Aesthetic Score，以及它们的加权组合。组合优化时不同指标可产生协同增效——如 CS+IR 组合下 CS 甚至超过单独优化 CS。

## 实验关键数据

### 主实验

COCO 验证集 5000 张图，SD v1.5 + DDIM 50 步：

| 方法 | FID↓ | CLIP Score↑ | ImageReward↑ |
|------|------|-------------|-------------|
| 固定嵌入 (50步) | 18.66 | 0.3204 | 0.2132 |
| 固定嵌入 (70步) | 18.27 | 0.3199 | 0.2137 |
| EBCA | 25.85 | 0.2877 | -0.3128 |
| Universal Guidance | 18.56 | 0.3216 | 0.2221 |
| **DATE 10%更新 (CS)** | **17.90** | **0.3237** | 0.2364 |
| **DATE 10%更新 (IR)** | 18.61 | 0.3224 | **0.4792** |

跨骨干一致提升：SD3 (IR: 1.0018→1.0457)、FLUX (CS: 0.3257→0.3283)、SDXL (IR: 0.7284→0.9096)。

### 消融实验

| 方法变体 | FID↓ | CS↑ | IR↑ |
|---------|------|-----|-----|
| 固定嵌入 | 18.66 | 0.3204 | 0.2132 |
| 随机方向更新 | 18.66 | 0.3204 | 0.2136 |
| 对含噪数据计算 h | 18.80 | 0.3200 | 0.2121 |
| 非归一化梯度 | 18.46 | 0.3212 | 0.2225 |
| **DATE (归一化)** | **17.91** | **0.3220** | **0.2229** |

### 关键发现

- 随机更新与固定嵌入效果相同，证明梯度方向是核心而非简单扰动
- 85% 的时间步对余弦相似度 < 0.1，证实不同步需要不同嵌入方向
- 同一提示不同实例的更新方向近乎正交（< 0.05），说明更新是实例特异的
- 中后期更新比早期更有效——细节生成阶段嵌入调整价值更大
- DATE 在提升目标指标的同时，其他非目标指标也同步提升——全面质量提升而非过拟合单一指标

## 亮点与洞察

- 填补了扩散模型三大优化维度（参数、潜变量、文本嵌入）中嵌入维度的空白
- 理论将嵌入更新统一解释为 score function 中的 guidance 项，与 Classifier Guidance 形成类比
- 即插即用特性极强：不修改模型、兼容任意骨干和采样器、支持多评估函数
- 多概念生成（AnE 数据集）和文本引导图像编辑（DDPM Inversion）两个下游任务一致验证了有效性

## 局限与展望

- 每步更新需额外 score network 前向传播+梯度计算，10% 更新时时间增加约 39%
- GPU 显存消耗显著增加（24GB → 61.5GB），FP16 可缓解但仍高于基线
- 超参数 $\rho$ 需调整，过大会因 Taylor 近似误差导致性能下降
- 依赖评估函数 $h$ 的质量——Aesthetic Score 与语义对齐相关性低，单独用它反而降低其他指标
- 当文本编码器本身无法有效表示目标语义时，嵌入空间的优化提升有限

## 相关工作与启发

- **Universal Guidance**: 在数据空间加引导，DATE 在嵌入空间优化——后者 FID 更优（17.90 vs 18.56）
- **Textual Inversion**: 优化特殊 token 嵌入但需多步训练，DATE 单步更新、覆盖全部嵌入
- **SAM**: 归一化梯度思想从模型参数泛化迁移到文本嵌入优化，提示可在更多领域应用
- DATE 的时间步自适应嵌入概念可推广到视频生成中的帧级自适应条件

## 评分

⭐⭐⭐⭐ — 理论清晰、方法优雅（无训练的测试时优化），多骨干多任务一致有效。计算开销和显存增加是主要局限。

<!-- RELATED:START -->

## 相关论文

- [Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [Mitigating Sexual Content Generation via Embedding Distortion in Text-conditioned Diffusion Models](mitigating_sexual_content_generation_via_embedding_distortion_in_text-conditione.md)
- [Text Embedding Knows How to Quantize Text-Guided Diffusion Models](../../ICCV2025/image_generation/text_embedding_knows_how_to_quantize_text-guided_diffusion_models.md)
- [Addressing Text Embedding Leakage in Diffusion-Based Image Editing](../../ICCV2025/image_generation/addressing_text_embedding_leakage_in_diffusion-based_image_editing.md)
- [Prompt-Based Safety Guidance Is Ineffective for Unlearned Text-to-Image Diffusion Models](prompt-based_safety_guidance_is_ineffective_for_unlearned_text-to-image_diffusio.md)

<!-- RELATED:END -->
