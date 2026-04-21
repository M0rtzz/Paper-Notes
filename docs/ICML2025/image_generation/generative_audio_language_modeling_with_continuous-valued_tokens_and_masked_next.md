---
title: >-
  [论文解读] Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction
description: >-
  [ICML 2025][图像生成][音频生成] 本文研究不使用离散 Token 的因果语言模型进行音频生成，利用 token-wise diffusion 建模连续值 next-token 分布，并提出 masked next-token prediction 任务，以 193M 参数在 AudioCaps 上达到与 SOTA 扩散模型相当的性能。
tags:
  - ICML 2025
  - 图像生成
  - 音频生成
  - 连续值 Token
  - 语言模型
  - 扩散模型
  - Masked Next-Token Prediction
---

# Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction

**会议**: ICML 2025  
**arXiv**: [2507.09834](https://arxiv.org/abs/2507.09834)  
**代码**: 有（项目页面）  
**领域**: Image Generation (Audio Generation)  
**关键词**: 音频生成, 连续值 Token, 语言模型, Token-wise Diffusion, Masked Next-Token Prediction

## 一句话总结
本文研究不使用离散 Token 的因果语言模型进行音频生成，利用 token-wise diffusion 建模连续值 next-token 分布，并提出 masked next-token prediction 任务，以 193M 参数在 AudioCaps 上达到与 SOTA 扩散模型相当的性能。

## 研究背景与动机

**领域现状**：自回归 next-token 预测结合 Transformer decoder 已成为 LLM 的事实标准，在 NLP 中取得巨大成功。将该范式扩展到音频一直是研究热点。

**现有痛点**：音频天然是连续信号，将其扩展到自回归 LM 面临独特挑战。现有方法（如 AudioGen）依赖离散化（通过 VQ-VAE 等方式将音频量化为离散 token），但量化过程不可避免地造成信息损失，限制了生成质量。

**核心矛盾**：离散 token 方便用标准 LM 建模（交叉熵损失），但会丢失音频的细微细节；直接建模连续值 token 的概率分布则面临分布复杂、训练困难等问题。

**本文目标**：探索无需离散 token 的音频生成因果语言模型，并提出新的训练任务来提升性能。

**切入角度**：用 token-wise diffusion 来建模 next continuous-valued token 的分布，同时创新性地将 masked prediction 引入因果 LM 框架。

**核心 idea**：在因果 LM 中用 token-wise 扩散建模连续 token 分布 + masked next-token prediction 双管齐下，实现高效的连续音频生成。

## 方法详解

### 整体框架

- **输入**：文本条件 + 连续音频 token 序列（由预训练编码器提取）
- **建模**：因果 Transformer 自回归预测 next token，但 token 是连续值向量
- **分布建模**：每个位置的 next-token 用一个小型 diffusion 模型建模其条件分布
- **输出**：逐 token 自回归采样后解码为音频波形

### 关键设计

1. **Token-wise Diffusion for Continuous Tokens**:

    - 不同于传统 LM 在离散 token 上用 softmax+交叉熵，本文在每个自回归步用一个小型扩散模型来建模 next continuous-valued token 的分布 $p(x_{t+1} | x_{\leq t})$
    - Transformer decoder 提供条件上下文，扩散模型在此条件下去噪生成 next token
    - **设计动机**：连续 token 空间的分布可以是多模态的、复杂的，扩散模型是建模此类分布的理想选择

2. **Masked Next-Token Prediction (MNTP)**:

    - 创新性地将 masked prediction 融入因果 LM 框架
    - 在训练时，随机 mask 部分 token 但仍保持因果结构（只能看到之前的 unmasked tokens）
    - 模型需要预测被 mask 的 token，相当于在因果框架中加入了 bidirectional 的信息
    - **设计动机**：标准 next-token prediction 只利用前文，MNTP 迫使模型利用更长距离的依赖关系，类似 BERT 式训练在因果设置下的适配

3. **轻量参数设计**:

    - Base 模型仅 193M 参数，Large 模型 462M 参数
    - 远小于 AudioGen Base (285M) 和 Large (1B)
    - **设计动机**：证明连续 token 方法的参数效率优势

### 损失函数 / 训练策略

- Token-wise diffusion 损失：标准去噪得分匹配损失，在每个 token 位置独立训练
- MNTP 损失：额外的 masked token 预测损失
- 总损失为两者的加权和
- 使用预训练的连续音频编码器（如 EnCodec 的连续版本）提取 token

## 实验关键数据

### 主实验

| 模型 | 参数量 | FAD↓ | KL↓ | 说明 |
|------|--------|------|-----|------|
| AudioGen Base | 285M | 基线 | 基线 | 离散 token |
| AudioGen Large | 1B | 基线 | 基线 | 离散 token |
| **Ours Base** | **193M** | **-20% (相对)** | **-40% (相对)** | 连续 token |
| **Ours Base + MNTP** | **193M** | **-41% (相对 AG-Base)** | - | +masked prediction |
| **Ours Large + MNTP** | **462M** | **-33% (相对 AG-Large)** | - | 与 SOTA 扩散模型持平 |

### 消融实验

| 配置 | FAD↓ | 说明 |
|------|------|------|
| 离散 token (AudioGen) | 基线 | 标准量化 |
| 连续 token (无 MNTP) | 改善 20% | 连续 token 优势 |
| 连续 token + MNTP | 改善 41% | MNTP 额外贡献 ~20% |
| 标准 next-token only | 居中 | 验证 MNTP 互补性 |

### 关键发现

- 连续 token 方法在相同参数量下明显优于离散 token 方案（FAD 相对改善 20-40%）
- MNTP 在因果 LM 框架中有效，提供额外 ~20% 相对改善
- 仅 193M 参数即可达到 285M AudioGen 级别，462M 即与 SOTA 扩散模型持平
- 参数效率显著：用不到 AudioGen Large 一半的参数达到更好性能

## 亮点与洞察

1. **打破"必须离散化"的假设**：证明因果 LM 可以直接建模连续 token
2. **MNTP 是因果 LM 的有效增强**：将 masked prediction 的信息增益引入自回归框架
3. **参数效率**：连续表示避免了维护大量离散 codebook 的开销
4. **通用性潜力**：该方法原则上可扩展到视频、音乐等其他连续模态

## 局限与展望

1. Token-wise diffusion 增加了每步采样的计算量（虽然总参数少）
2. 推理延迟可能高于纯离散 token 方法
3. 连续 token 编码器的选择和质量是性能上限
4. 尚未扩展到超大规模（>1B 参数）

## 相关工作与启发

- AudioGen/AudioLDM 系列提供了音频生成的离散和连续方法对比基线
- MAR (Masked Autoregressive) 在图像领域也探索了类似的 masked + autoregressive 结合
- 启发：该方法可能为"LM 范式生成连续数据"提供一个通用模板

## 评分
- 新颖性: ⭐⭐⭐⭐ 连续 token LM + MNTP 组合有新意
- 实验充分度: ⭐⭐⭐⭐ AudioCaps 充分验证，但数据集和任务可更多
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 为音频生成提供了参数高效的新方案

<!-- RELATED:START -->

## 相关论文

- [NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction](ntpp_generative_speech_language_modeling_for_dual-channel_spoken_dialogue_via_ne.md)
- [Continuous Diffusion Model for Language Modeling](../../NeurIPS2025/image_generation/continuous_diffusion_model_for_language_modeling.md)
- [IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation with Diffusion Modeling](impact_iterative_mask-based_parallel_decoding_for_text-to-audio_generation.md)
- [PanoLlama: Generating Endless and Coherent Panoramas with Next-Token-Prediction LLMs](../../ICCV2025/image_generation/panollama_generating_endless_and_coherent_panoramas_with_next-token-prediction_l.md)
- [Efficient Generative Modeling with Residual Vector Quantization-Based Tokens](efficient_generative_modeling_with_residual_vector_quantization-based_tokens.md)

<!-- RELATED:END -->
