---
title: >-
  [论文解读] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models
description: >-
  [图像生成] 提出 TRCE，通过两阶段概念擦除策略（文本语义擦除 + 去噪轨迹转向），在可靠擦除恶意概念的同时最小化对模型正常生成能力的影响。
tags:
  - 图像生成
---

# TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models

> **会议**: ICCV 2025
> **arXiv**: [2503.07389](https://arxiv.org/abs/2503.07389)
> **代码**: [GitHub](https://github.com/ddgoodgood/TRCE)
> **领域**: 扩散模型安全·概念擦除·图像生成
> **关键词**: concept erasure, text-to-image safety, adversarial robustness, cross-attention editing, denoising trajectory

## 一句话总结

提出 TRCE，通过两阶段概念擦除策略（文本语义擦除 + 去噪轨迹转向），在可靠擦除恶意概念的同时最小化对模型正常生成能力的影响。

## 研究背景与动机

文本到图像扩散模型（如 Stable Diffusion）在生成高质量图像的同时也面临生成 NSFW 内容的安全风险。概念擦除（Concept Erasure, CE）通过修改模型参数使其无法生成特定概念，但现有方法存在**可靠性与知识保持之间的根本矛盾**：

**隐含恶意语义难以擦除**：现有方法（如 ESD、UCE）主要消除特定关键词，但恶意概念常以隐喻、联想或对抗性提示间接表达（如不直接使用"nudity"但描述类似场景）。

**知识保持与擦除可靠性的权衡**：为应对对抗性提示，现有方法往往过度修改模型，严重降低无关内容的生成能力（FID 升高、CLIP-Score 下降）。

**对抗攻击脆弱性**：MMA、P4D、Ring-A-Bell 等红队工具能轻易绕过多数擦除方法。

**TRCE 的核心洞察**：恶意语义的消除和安全视觉内容的生成应在不同层面分别处理——第一阶段在文本层面消除隐含的恶意语义，第二阶段在去噪过程中将采样轨迹导向安全方向。

## 方法详解

### 整体框架（Fig. 3）

TRCE 分为两个阶段：

**阶段 1：文本语义擦除（Textual Semantic Erasure）** → 修改交叉注意力矩阵
**阶段 2：去噪轨迹转向（Denoising Trajectory Steering）** → 对比学习微调 U-Net

### 关键设计 1：[EoT] 作为映射目标

TRCE 识别出一个关键映射目标——**[EoT]（End of Text）嵌入**。不同于现有方法直接映射关键词嵌入（导致快速知识遗忘），[EoT] 的独特角色是：

- 携带整个 prompt 的语义信息
- 关注生成图像的显著区域
- 修改 [EoT] 可以改变图像内容但**保留 prompt 的整体上下文**

利用 LLM（GPT-4o）扩展恶意概念为 20 个同义词 + 15 个模板 = 300 个提示，并构造对应的安全提示集。通过闭式解优化交叉注意力 $W_K, W_V$ 矩阵：

$$W' = \left(\sum_{i=1}^n W \cdot e_i^s \cdot (e_i^m)^\top + \eta \sum_{j=1}^q W \cdot e_j^k \cdot (e_j^k)^\top\right) \cdot \left(\sum_{i=1}^n e_i^m \cdot (e_i^m)^\top + \eta \sum_{j=1}^q e_j^k \cdot (e_j^k)^\top\right)^{-1}$$

### 关键设计 2：去噪轨迹转向

基于扩散模型采样的确定性性质——ODE 轨迹的早期微调就能将最终生成内容导向安全方向。

**轨迹准备**：用原始 U-Net $\epsilon_\theta$ 和恶意提示缓存早期采样轨迹 $\{z_t^m\}$。

**引导增强**：构造语义增强的安全/不安全方向（使用 classifier-free guidance 放大）：

$$f_{safe} = \epsilon_\theta(z_t^m, \varnothing, t) + \beta(\epsilon_\theta(z_t^m, c^s, t) - \epsilon_\theta(z_t^m, \varnothing, t))$$

**对比损失**：使用 triplet margin loss 将去噪预测拉向安全方向、远离不安全方向：

$$L_{erase} = \mathbb{E}[\max(\|\hat{\epsilon}_\theta - f_{safe}\|^2 - \|\hat{\epsilon}_\theta - f_{unsafe}\|^2 + margin, 0)]$$

正则化项保持无条件预测不变：$L_{preserve} = \|\hat{\epsilon}_\theta(z_t^u, \varnothing, t) - \epsilon_\theta(z_t^u, \varnothing, t)\|^2$

仅微调视觉层（self-attention + cross-attention 的 Q 矩阵），3 个 epoch，约 300 秒。

## 实验

### 主实验：性概念擦除（Tab. 1）

| 方法 | I2P ↓ | MMA ↓ | P4D ↓ | Ring ↓ | UnDiff ↓ | FID_real ↓ | CLIP-S ↑ |
|------|-------|-------|-------|--------|----------|-----------|---------|
| SD1.4 | 34.69% | 79.00% | 83.44% | 59.49% | 57.75% | 27.18 | 30.97 |
| ESD | 31.15% | 58.50% | 82.67% | 50.63% | 77.46% | 26.88 | 31.21 |
| UCE | 8.16% | 30.80% | 43.71% | 13.92% | 19.72% | 27.20 | 30.92 |
| RECE | 6.34% | 23.10% | 32.00% | 6.33% | 15.49% | 28.26 | 30.79 |
| MACE | 7.09% | 10.60% | 7.95% | 10.13% | 11.27% | 26.98 | 28.84 |
| AdvUnlearn | 1.71% | 0.30% | 1.99% | 6.33% | 3.52% | 29.65 | 28.93 |
| **TRCE(T+V)** | **1.29%** | **1.40%** | **1.99%** | **1.27%** | **0.70%** | **26.89** | **30.71** |

TRCE(T+V) 在所有 5 种攻击下 ASR 均达到约 1%，同时 FID_real 和 CLIP-Score 保持在最优水平——**擦除可靠性和知识保持首次实现真正的兼顾**。

### 多概念擦除（Tab. 2, I2P 7 类恶意概念）

| 方法 | 整体 ↓ | FID_real ↓ | CLIP-S ↑ |
|------|--------|-----------|---------|
| MACE | 5.6% | 26.20 | 28.13 |
| TRCE(T) | 3.6% | 27.25 | 30.43 |
| **TRCE(T+V)** | **2.0%** | **27.23** | **30.48** |

关键发现：多概念擦除场景下，MACE 的 CLIP-S 从 30.97 降至 28.13（知识严重损失），而 TRCE 仅从 30.97 降至 30.48。

### 两阶段各自贡献分析

| 阶段 | I2P ↓ | MMA ↓ | P4D ↓ |
|------|-------|-------|-------|
| TRCE(T) 仅第一阶段 | 5.05% | 7.80% | 7.95% |
| TRCE(V) 仅第二阶段 | 13.86% | 35.00% | 48.00% |
| **TRCE(T+V) 两阶段** | **1.29%** | **1.40%** | **1.99%** |

关键发现：
- 仅文本擦除已很有效（[EoT] 映射目标的优势）
- 仅轨迹转向效果较差（prompt 中仍含恶意语义，后期去噪会重新引入）
- 两阶段协同产生乘法效应

## 亮点与洞察

1. **[EoT] 作为映射目标**是核心贡献，比直接映射关键词更有效且更少损害知识——因为 [EoT] 携带整体语义而非孤立概念
2. **两阶段协同**的设计哲学优雅：文本层先"拆弹"，去噪层再"保险"
3. 对抗性提示下 ASR 降至约 1% 在该领域是突破性结果
4. 微调仅需 300 秒（单卡 RTX 4090），实用性极强

## 局限性

- 基于 SD1.4 评估，对 SDXL/SD3 等新架构的泛化性待验证
- 闭式解修改交叉注意力可能在多轮迭代擦除后积累误差
- 对极端复杂的对抗性 prompt 工程仍可能存在边界情况

## 相关工作

- 概念擦除：ESD、UCE、RECE、MACE、SPM、AdvUnlearn
- 红队攻击：P4D、MMA、Ring-A-Bell、UnlearnDiff
- 推理时引导：SLD、Safree

## 评分

- **新颖性**: ★★★★☆ — [EoT] 映射和两阶段协同设计新颖实用
- **技术深度**: ★★★★★ — 对扩散模型内部机制理解深入
- **实验质量**: ★★★★★ — 5 种攻击 + 多概念 + 消融，评估极为全面
- **写作质量**: ★★★★☆ — 问题动机清晰，两阶段逻辑递进

<!-- RELATED:START -->

## 相关论文

- [FlowTok: Flowing Seamlessly Across Text and Image Tokens](flowtok_flowing_seamlessly_across_text_and_image_tokens.md)
- [EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)
- [Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [What Makes for Text to 360-degree Panorama Generation with Stable Diffusion?](what_makes_for_text_to_360-degree_panorama_generation_with_stable_diffusion.md)
- [LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models](loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)

<!-- RELATED:END -->
