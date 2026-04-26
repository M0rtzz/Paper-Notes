---
title: >-
  [论文解读] JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation
description: >-
  [CVPR 2025][图像生成][unified model] 提出 JanusFlow，将 rectified flow 直接嵌入自回归 LLM 框架，通过解耦理解/生成编码器 + 表征对齐正则化，在 1.3B 参数下同时达到多模态理解和图像生成的 SOTA。
tags:
  - CVPR 2025
  - 图像生成
  - unified model
  - rectified flow
  - autoregressive LLM
  - 多模态
  - decoupled encoder
  - REPA
---

# JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation

**会议**: CVPR 2025  
**arXiv**: [2411.07975](https://arxiv.org/abs/2411.07975)  
**代码**: [deepseek-ai/Janus](https://github.com/deepseek-ai/Janus)  
**领域**: image_generation  
**关键词**: unified model, rectified flow, autoregressive LLM, multimodal understanding, decoupled encoder, REPA

## 一句话总结

提出 JanusFlow，将 rectified flow 直接嵌入自回归 LLM 框架，通过解耦理解/生成编码器 + 表征对齐正则化，在 1.3B 参数下同时达到多模态理解和图像生成的 SOTA。

## 研究背景与动机

**领域现状**: 大语言模型在多模态理解（LLaVA、Qwen-VL）和文生图（SD、DALL-E）两个方向都取得了显著进展，但两类能力通常由独立模型实现。

**现有痛点**: 现有统一方案存在两类问题：(1) 利用预训练扩散模型做外部工具（SEED-X 等），架构复杂但生成能力受限于条件生成；(2) 基于 VQ 离散 token 的自回归方法（Chameleon 等），质量受限于 tokenizer 精度。

**核心矛盾**: 理解任务需要高层语义特征，生成任务需要低层详细特征，两者在编码器层面存在任务干扰；而自回归和连续扩散两种范式的融合缺乏简洁有效的方案。

**本文目标**: 如何以最简架构将 rectified flow（连续生成范式）与自回归 LLM 结合，在单一模型中同时实现高质量的多模态理解和图像生成。

**切入角度**: 极简设计——仅需轻量编码器/解码器即可在 LLM 内训练 rectified flow，加上解耦编码器消除任务干扰，表征对齐提升生成语义性。

## 方法详解

### 整体框架

JanusFlow 基于 1.3B DeepSeek-LLM，核心思路是在同一个 LLM 中运行两种模式：
- **理解模式**: 图像经 SigLIP 编码→线性投影→与文本 token 拼接→自回归预测文本 token
- **生成模式**: 文本条件→LLM 处理→从高斯噪声出发，LLM 迭代预测速度向量→Euler 求解器更新潜在状态→VAE 解码出图

### 关键设计

**1. 解耦理解/生成编码器**
- **功能**: 理解用预训练 SigLIP-Large-Patch/16（~300M）提取语义特征；生成用从头训练的 ConvNeXt blocks（~70M）做编码器 $g_{enc}$ 和解码器 $g_{dec}$。
- **核心思路**: 语义编码器擅长高层抽象但不适合像素级重建，生成编码器需要在 VAE latent space 中操作。两者解耦后各司其职。
- **设计动机**: 先前工作（Show-o、Transfusion）共享编码器导致理解-生成任务互相干扰，消融实验证实解耦显著提升性能。

**2. 表征对齐正则化 (REPA)**
- **功能**: 在生成训练中，将 LLM 第 6 层的中间特征投影后与 SigLIP 编码器对真实图像的特征做余弦相似度对齐。
- **核心思路**: $\mathcal{L}_{REPA} = -\text{sim}(\text{stop\_grad}(f_{enc}(x^{res})), h_\varphi(q_\theta(z_t)))$，其中 $h_\varphi$ 是 3 层 MLP，梯度不回传到理解编码器。
- **设计动机**: 对齐让 LLM 在生成时的内部表征空间与理解编码器的语义空间一致，从而提升从随机噪声和文本条件生成图像时的语义质量。这是解耦设计带来的独特优势——共享编码器无法做这种跨模块对齐。

**3. Rectified Flow 集成与 CFG**
- **功能**: 在 SDXL-VAE latent space 中训练 rectified flow，LLM 的输出经 $g_{dec}$ 变换为速度向量，用 Euler 求解器迭代更新。
- **核心思路**: 训练时对时间 $t$ 采样 logit-normal 分布，10% 概率随机 drop 文本 prompt 用于 CFG 推理。推理时用 $v = w \cdot v_{cond} + (1-w) \cdot v_{uncond}$ 做 classifier-free guidance。
- **设计动机**: 使用 causal attention 即可，无需复杂 attention masking，极大简化架构。

### 损失函数 / 训练策略

三阶段训练：
1. **Stage 1（适应期）**: 10K steps，仅训练随机初始化的线性层和 $g_{enc}$、$g_{dec}$，LR=$1\times10^{-4}$
2. **Stage 2（统一预训练）**: 390K steps，训练除 SigLIP 外的全部模块，理解:生成:文本 = 14:80:6
3. **Stage 3（SFT）**: 26K steps，解冻 SigLIP，指令微调，LR=$2\times10^{-5}$

总损失：理解用 $\mathcal{L}_{AR}$（自回归交叉熵），生成用 $\mathcal{L}_{RF}$（flow matching L2）+ $\mathcal{L}_{REPA}$（表征对齐）。使用 EMA（ratio=0.99）。训练资源约 1600 A100 GPU days。

## 实验关键数据

### 主实验——图像生成

| 方法 | 类型 | 参数量 | GenEval↑ | DPG-Bench↑ | MJHQ FID↓ |
|---|---|---|---|---|---|
| SDv1.5 | 生成专用 | 0.9B | 0.43 | 63.18 | - |
| SDXL | 生成专用 | 2.6B | 0.55 | 74.65 | - |
| DALL-E 3 | 生成专用 | - | **0.67** | - | - |
| Show-o | 统一 | 1.3B | 0.53 | - | 15.18 |
| Janus | 统一 | 1.3B | 0.61 | - | 10.10 |
| **JanusFlow** | **统一** | **1.3B** | **0.63** | **80.09** | **9.51** |

### 主实验——多模态理解

| 方法 | 类型 | LLM | MMBench | SEED | GQA | VQAv2 |
|---|---|---|---|---|---|---|
| LLaVA-v1.5 | 理解专用 | 7B | 64.3 | 58.6 | 62.0 | 78.5 |
| Qwen-VL-Chat | 理解专用 | 7B | 60.6 | 58.2 | - | 78.2 |
| **JanusFlow** | **统一** | **1.3B** | **74.9** | **70.5** | **60.3** | **79.8** |

1.3B 参数的统一模型超越多个 7B 理解专用模型。

### 消融实验

| 设置 | GenEval↑ | DPG-Bench↑ |
|---|---|---|
| 共享编码器（baseline） | 0.56 | 73.02 |
| + 解耦编码器 | 0.60 | 76.98 |
| + REPA | **0.63** | **80.09** |
| 去掉 REPA | 0.60 | 76.98 |

### 关键发现

1. **Rectified flow 可在 LLM 框架内直接训练**: 无需修改 LLM 架构，仅需轻量 ConvNeXt 编解码器，causal attention 足够。
2. **解耦编码器是性能提升的关键**: GenEval 从 0.56 提升至 0.60，消除了理解-生成任务干扰。
3. **REPA 进一步显著提升**: 在解耦基础上再提升 0.03 GenEval，DPG-Bench 提升 3 个百分点。
4. **1.3B 统一模型可超越多个专用 7B 模型**: 证明自回归+flow 的统一范式潜力巨大。

## 亮点与洞察

- 极简架构设计：仅需轻量编解码器即可在 LLM 中运行 rectified flow，无需复杂 attention masking
- 解耦+对齐的思路非常优雅：先物理分离两个任务的编码器，再用正则化在语义空间对齐
- 连续生成（flow）优于离散生成（VQ）：同等参数下 FID 从 10.10 降至 9.51
- 在统一模型理解能力不降反升的前提下大幅提升生成质量

## 局限与展望

- 生成分辨率仅 384×384，远低于 SDXL（1024×1024）
- 1.3B 模型规模偏小，扩展到更大 LLM 的效果待验证
- 仅支持图像生成，未扩展到视频
- CFG 推理需要双倍 forward pass，效率有优化空间
- 训练资源消耗大（1600 A100 GPU days），可探索更高效训练策略

## 相关工作与启发

- **Janus**: JanusFlow 的前身，使用 VQ token 做生成，本文改用 rectified flow 实现连续生成
- **Show-o / Transfusion**: 统一模型使用共享编码器，JanusFlow 证明解耦更优
- **REPA（Yu et al.）**: 跨模型表征对齐的思想被引入统一框架，成为关键创新
- **Stable Diffusion 3**: logit-normal 时间分布和 flow matching 训练策略的来源

## 评分

⭐⭐⭐⭐ — 极简架构+解耦编码器+REPA 三板斧组合效果出色，1.3B 统一模型超越 7B 专用模型是亮点；但分辨率和模型规模限制了实际应用价值。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_unified_generation_understanding.md)
- [\[CVPR 2026\] CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion](../../CVPR2026/image_generation/careflow_cyclic_adaptive_rectified_flow_for_multimodal_fusion.md)
- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](../../NeurIPS2025/image_generation/coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[CVPR 2025\] WeGen: A Unified Model for Interactive Multimodal Generation as We Chat](wegen_a_unified_model_for_interactive_multimodal_generation_as_we_chat.md)

<!-- RELATED:END -->
