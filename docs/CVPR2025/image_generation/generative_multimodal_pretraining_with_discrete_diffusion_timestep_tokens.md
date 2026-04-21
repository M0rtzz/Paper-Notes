---
title: >-
  [论文解读] Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens
description: >-
  [CVPR 2025][图像生成][多模态大语言模型] DDT-LLaMA 提出用扩散时间步编码学习具有递归结构的离散视觉 token（DDT），使视觉 token 序列像自然语言一样具有层级依赖关系，从而在统一的 next-token-prediction 框架下同时实现多模态理解和生成的 SOTA 性能。
tags:
  - CVPR 2025
  - 图像生成
  - 多模态大语言模型
  - 扩散时间步token
  - 视觉tokenizer
  - 递归结构
  - 统一理解生成
---

# Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens

**会议**: CVPR 2025  
**arXiv**: [2504.14666](https://arxiv.org/abs/2504.14666)  
**代码**: https://DDT-LLaMA.github.io/ (项目页)  
**领域**: 多模态VLM / 图像生成  
**关键词**: 多模态大语言模型, 扩散时间步token, 视觉tokenizer, 递归结构, 统一理解生成

## 一句话总结
DDT-LLaMA 提出用扩散时间步编码学习具有递归结构的离散视觉 token（DDT），使视觉 token 序列像自然语言一样具有层级依赖关系，从而在统一的 next-token-prediction 框架下同时实现多模态理解和生成的 SOTA 性能。

## 研究背景与动机

1. **领域现状**：多模态大语言模型（MLLM）追求在统一的 next-token-prediction 范式中同时完成视觉理解和生成。现有方法主要分为级联架构（如 EMU2 分别训练理解和生成模块再拼接）和 tokenization 方法（如 Emu3 将视觉量化为离散 token 再统一训练）。
2. **现有痛点**：所有现有方法都基于空间视觉 token（按光栅扫描顺序排列 image patch），但空间 token 序列缺乏人类语言的递归结构特性。实验表明打乱空间 token 的顺序对 LLM 训练收敛几乎无影响——这与语言 token 的行为截然不同，说明 LLM 无法有效建模空间 token。
3. **核心矛盾**：理解和生成存在冲突目标——理解是多对一映射（多种柯基犬照片→"柯基"），生成是一对一映射（保留所有视觉细节）。要在统一框架中同时做好两者，需要让视觉 token 具备语言般的可建模性。
4. **本文目标**：设计一种视觉 tokenizer 使视觉 token 具有递归结构，让 LLM 可以像处理语言一样自然地处理视觉信息。
5. **切入角度**：扩散模型前向过程中，随着时间步增加，高斯噪声逐渐破坏图像的视觉属性。作者利用这个"渐进属性丢失"过程来学习具有递归补偿结构的 token 序列——每个新 token 补偿当前时间步新丢失的视觉属性。
6. **核心 idea**：将扩散时间步对应的"渐进式信息丢失"编码为递归视觉 token 序列，使其具有类似自然语言的层级依赖。

## 方法详解

### 整体框架
系统包含两部分：(1) DDT Tokenizer 将图像编码为递归离散 token 序列；(2) DDT-LLaMA 基于 LLaMA3-8B 的 MLLM，用统一的 next-token-prediction 训练。推理时 LLM 预测视觉 token，再用扩散模型作为解码器还原图像。

### 关键设计

1. **DDT 编码器 (Diffusion Timestep Encoder)**:
    - 功能：将无噪图像编码为 T 个递归离散 token 序列
    - 核心思路：使用 Transformer 编码器处理图像 patch tokens 和 T 个可学习 query tokens。采用双独立 Transformer + 共享注意力的 MMDiT 风格架构（类似 SD3），最终只保留 query tokens 的输出 $(\hat{V}_1, ..., \hat{V}_T)$。关键在于递归性：$f_{t+1}(x_0) = (f_t(x_0), V_{t+1})$，即第 $t+1$ 个 token 序列在第 $t$ 个基础上追加一个新 token，补偿从 $x_t$ 到 $x_{t+1}$ 新丢失的视觉属性。
    - 设计动机：与空间 token 不同，DDT token 的顺序有明确语义——从粗粒度（全局布局、主色调）到细粒度（纹理、边缘），具有类似语言的递归嵌套结构。

2. **向量量化 (Vector Quantization)**:
    - 功能：将连续编码器输出离散化为固定词汇表中的 token
    - 核心思路：使用标准 VQ 模块，codebook 大小 65536，维度 m=16。采用 EMA 更新策略提升训练稳定性。使用两个trick提高codebook利用率：(a) 小维度 m=16（参考文献验证有效）；(b) 监控死 entry 并用训练 batch 中的随机编码器输出替换。
    - 设计动机：离散 token 可以直接复用 LLM 的词汇表扩展机制，实现真正的统一训练。

3. **扩散模型解码器 (Diffusion Decoder)**:
    - 功能：根据 DDT token 引导去噪过程还原图像
    - 核心思路：基于 MMDiT 架构。在时间步 $t$ 时，只使用前 $t$ 个 token $(V_1, ..., V_t)$ 作为引导条件（剩余 mask 为 0），解码器学习 $\hat{x}_0 = d(x_t, t, (V_1, ..., V_t))$ 的重建。训练损失为 Rectified Flow 下的重建损失加上标准 commitment loss。推理时通过 T 步 DDPM 过程从纯噪声生成图像。
    - 设计动机：扩展式 token 输入天然契合扩散模型的时间步去噪过程——噪声越大需要越多 token 来补偿信息丢失。

### 损失函数 / 训练策略
Tokenizer 训练：重建损失 $\mathcal{L} = \mathbb{E}[\|d(t\epsilon + (1-t)x_0, t, (V_1,...,V_t)) - x_0\|^2]$ + commitment loss。仅在 ImageNet 上训练。MLLM 两阶段训练：(1) 预训练阶段用 200M 图文对（Laion+Coyo）对齐 DDT token 和文本 token，512 Ascend 910B NPUs 训练约两周；(2) 指令微调阶段在公开数据上做多模态任务训练。词汇表扩展 65536 个视觉 code。

## 实验关键数据

### 主实验

**文本到图像生成 (MLLM-based Generalist)**

| Benchmark | 指标 | DDT-LLaMA | 之前SOTA (Emu3) | 提升 |
|-----------|------|-----------|----------|------|
| GenEval | Overall↑ | **0.66** | 0.54 | +0.12 |
| GenEval | Counting↑ | **0.56** | 0.34 | +0.22 |
| GenEval | Position↑ | **0.39** | 0.17 | +0.22 |
| GenEval | ColorAttri↑ | **0.48** | 0.21 | +0.27 |
| T2I-CompBench | Color↑ | **0.728** | 0.611 | +0.117 |

**零样本图像编辑**

| Benchmark | 指标 | DDT-LLaMA | UltraEdit (Specialist) |
|-----------|------|-----------|----------|
| MagicBrush | L1↓ | **5.4** | 6.6 |
| MagicBrush | CVS↑ | **92.9** | 88.4 |

### 消融实验

| 配置 | 说明 |
|------|------|
| 空间 token (打乱序列) | 训练曲线几乎不变 → 不是好的"视觉语言" |
| DDT token (打乱序列) | 训练曲线显著退化 → 具有类似语言的序列依赖 |
| 仅 ImageNet 训练 tokenizer | 已足够，证明递归结构比数据规模更重要 |

### 关键发现
- DDT token 的语言验证实验最有说服力：打乱 DDT token 顺序后 LLM 训练曲线显著退化（与语言行为一致），而空间 token 不受影响
- Tokenizer 仅在 ImageNet 上训练即可超越使用大规模数据（Laion/Coyo）的 tokenizer，证明递归结构本身是关键
- GenEval 上 Overall 0.66 已接近专门的 T2I 模型 DALL-E 3 的 0.67，且远超所有 MLLM 方法
- 在图像编辑任务上甚至超越了专门的编辑模型（UltraEdit），证明统一框架的优势

## 亮点与洞察
- **利用扩散时间步构建递归 token 是一个极具洞察力的创新**：巧妙地将扩散模型的渐进去噪过程和语言的递归结构联系起来，是一种全新的视觉量化范式。从"空间 token 不是好的视觉语言"这个观察出发，分析原因并提出解决方案，整个研究逻辑非常完整
- **语言性验证实验**：用序列扰动实验验证 token 的"语言性"是一个可复用的评估方法论，可用于评估任何试图将非文本模态统一到 LLM 的 tokenization 方案
- **仅用 ImageNet 训练的 tokenizer 就够用**：说明 token 的结构比训练数据规模更重要，这对 tokenizer 设计有重要指导意义

## 局限与展望
- Token 序列长度与扩散时间步数 T 绑定，T 的选择需要在序列长度和重建质量间权衡
- Tokenizer 基于 MMDiT 架构较重，编码效率可能不如简单的 VQ-VAE
- 当前仅处理图像模态，视频和3D等更高维模态的扩展有待探索
- MLLM 训练需要大量计算资源（512 NPUs 训练两周），实用性受限
- 视觉理解虽然有所提升，但与专门的理解模型（如 LLaVA-1.6）仍有差距

## 相关工作与启发
- **vs Emu3**: Emu3 使用标准空间 VQ token 统一训练，DDT-LLaMA 用递归时间步 token 替代，在生成和理解上都显著更优
- **vs Transfusion**: Transfusion 在单一 Transformer 中混合自回归和扩散目标，但仍用空间 token。DDT 的优势在于 token 本身具有递归结构
- **vs VILA-U**: VILA-U 尝试统一理解和生成的 token，但也受限于空间 token 的非递归性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 扩散时间步→递归视觉语言的洞察极为新颖
- 实验充分度: ⭐⭐⭐⭐ 覆盖生成、编辑、理解多个任务
- 写作质量: ⭐⭐⭐⭐⭐ 动机分析深入，语言验证实验精彩
- 价值: ⭐⭐⭐⭐⭐ 为统一多模态模型提供了新的 tokenization 范式

<!-- RELATED:START -->

## 相关论文

- [Efficient Generative Modeling with Residual Vector Quantization-Based Tokens](../../ICML2025/image_generation/efficient_generative_modeling_with_residual_vector_quantization-based_tokens.md)
- [DoraCycle: Domain-Oriented Adaptation of Unified Generative Model in Multimodal Cycles](doracycle_domain-oriented_adaptation_of_unified_generative_model_in_multimodal_c.md)
- [Timestep-Aware Diffusion Model for Extreme Image Rescaling](../../ICCV2025/image_generation/timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [VLOGGER: Multimodal Diffusion for Embodied Avatar Synthesis](vlogger_multimodal_diffusion_for_embodied_avatar_synthesis.md)
- [Constrained Discrete Diffusion](../../NeurIPS2025/image_generation/constrained_discrete_diffusion.md)

<!-- RELATED:END -->
