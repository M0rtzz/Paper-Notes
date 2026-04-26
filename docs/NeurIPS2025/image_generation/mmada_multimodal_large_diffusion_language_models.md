---
title: >-
  [论文解读] MMaDA: Multimodal Large Diffusion Language Models
description: >-
  [NeurIPS 2025][图像生成][扩散语言模型] 提出 MMaDA，首个在统一离散扩散架构下同时实现文本推理、多模态理解和文本到图像生成的多模态基础模型，通过混合长 CoT 微调和 UniGRPO 强化学习算法弥合了扩散模型预训练与后训练之间的鸿沟。
tags:
  - NeurIPS 2025
  - 图像生成
  - 扩散语言模型
  - 多模态统一架构
  - UniGRPO
  - 混合长思维链
  - 离散扩散
---

# MMaDA: Multimodal Large Diffusion Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.15809](https://arxiv.org/abs/2505.15809)  
**代码**: [GitHub](https://github.com/Gen-Verse/MMaDA)  
**领域**: 扩散模型 / 多模态基础模型  
**关键词**: 扩散语言模型, 多模态统一架构, UniGRPO, 混合长思维链, 离散扩散

## 一句话总结

提出 MMaDA，首个在统一离散扩散架构下同时实现文本推理、多模态理解和文本到图像生成的多模态基础模型，通过混合长 CoT 微调和 UniGRPO 强化学习算法弥合了扩散模型预训练与后训练之间的鸿沟。

## 研究背景与动机

多模态大模型正在经历架构范式的演进：

**纯自回归（AR）**：如 Emu3、Janus，用 next-token prediction 统一所有模态，架构简单但视觉生成质量受限

**AR + 扩散（两个模型）**：如 DreamLLM，文本用 AR、视觉用连续扩散，需要两个独立模型

**AR + 扩散（一个模型）**：如 Show-o、Transfusion，在单模型中混合 AR 和扩散目标，引入复杂的混合机制

然而现有统一多模态模型**严重缺乏对后训练的探索**——预训练后直接部署，没有经历 CoT 微调或强化学习。这在非自回归架构中尤为突出，因为将 AR 范式的 GRPO 直接搬到扩散模型面临三大技术障碍：
1. Token 级 log-likelihood 仅在掩码区域有效
2. 策略分布依赖于掩码率的采样
3. 序列级似然无法通过自回归链式法则累积

MMaDA 的核心定位：用**纯扩散（离散）**统一所有模态，并首次系统地为扩散基础模型设计完整的后训练流程。

## 方法详解

### 整体框架

MMaDA 的训练流程包含三个阶段：
- **Stage 1（预训练）**：统一扩散目标，联合训练文本生成、类条件图像生成、图文理解（600K steps）
- **Stage 2（混合长 CoT 微调）**：在跨模态 CoT 数据上微调，建立推理能力（50K steps）
- **Stage 3（UniGRPO 强化学习）**：用多样化奖励进行 RL 训练（50K steps）

### 关键设计

1. **统一离散扩散架构**：以 LLaDA-8B 为骨干，将文本和图像都处理为离散 token 的掩码预测任务。图像用 MAGVIT-v2 量化为 $32 \times 32 = 1024$ 个离散 token（codebook 大小 8192）。统一训练目标：
    $\mathcal{L}_{\text{unify}}(\theta) = -\mathbb{E}_{t,x_0,x_t}\left[\frac{1}{t}\sum_{i=1}^L \mathbf{I}[x_t^i=\texttt{[MASK]}] \log p_\theta(x_0^i | x_t)\right]$
   核心优势：语言和视觉共享完全相同的概率公式和架构，无需模态特定组件。

2. **混合长 CoT 微调**：设计统一 CoT 格式 `|<special_token>|<reasoning_process>|<special_token>|<result>`，跨任务对齐推理过程。数据来源包括：

    - 文本推理：ReasonFlux、LIMO、OpenThoughts 等
    - 多模态推理：LMM-R1 在 GeoQA/CLEVR 上的正确响应
    - 知识感知图像生成：GPT-4.1 合成的科学/文化/地标描述对
   
   关键设计是**让文本推理能力迁移到视觉生成**——模型先进行文本推理（分析物体特征、空间关系），再生成图像。

3. **UniGRPO**：专为扩散模型设计的策略梯度 RL 算法，解决三大挑战：

    - **结构化噪声策略**：对每个响应 $o_i$ 均匀采样掩码率 $p_i \in [0,1]$，跨梯度步变化随机种子。使模型暴露在从几乎全掩码到几乎全清晰的各种去噪阶段
    - **高效 log-likelihood 近似**：
    $\pi'_\theta = \frac{1}{M}\sum_{o_{i,t} \in M} \log p_\theta(o_{i,t} | q)$
      对掩码 token 取平均，避免了 LLaDA 的 128 次 Monte Carlo 采样
    - **均匀随机掩码**：先随机采样起始步，然后均匀分布剩余去噪步（而非完全随机），近似 Monte Carlo 平均，训练更稳定

### 损失函数 / 训练策略

**多样化奖励建模**：
- 文本推理：正确性奖励 2.0 + 格式奖励 0.5（遵循 `<think>...</think>` 格式）
- 多模态推理：正确性 + 格式 + CLIP 奖励 $0.1 \cdot \text{CLIP}(\text{image}, \text{text})$
- 图像生成：CLIP 奖励 + Image Reward（人类偏好分），均缩放 0.1

**采样策略**：
- 文本生成：半自回归去噪（分 64 token 块，每步解掩码 2 个最低置信度 token）
- 图像生成：并行非自回归，余弦掩码调度，50 步去噪，CFG=3.5

## 实验关键数据

### 主实验

**多模态理解基准**

| 模型 | POPE | MME | VQAv2 | GQA | MMMU | MMB | SEED |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LLaVA-v1.5 | 85.9 | 1510.7 | 78.5 | 62.0 | 35.4 | 64.3 | 58.6 |
| Show-o | 80.0 | 1097.2 | 69.4 | 58.0 | 26.7 | - | - |
| **MMaDA** | **86.1** | **1410.7** | **76.7** | **61.3** | **30.2** | **68.5** | **64.2** |

**文本到图像生成**

| 模型 | WISE Cultural↑ | ImageReward↑ | CLIP Score↑ | GenEval Overall↑ |
|------|:---:|:---:|:---:|:---:|
| SDXL | 0.43 | 1.13 | 32.12 | 0.55 |
| Janus | 0.16 | 1.03 | 29.45 | 0.61 |
| Show-o | 0.28 | 0.92 | 28.94 | 0.53 |
| **MMaDA** | **0.67** | **1.15** | **32.46** | **0.63** |

**文本推理**

| 模型 | 架构 | MMLU | GSM8K | MATH | ARC-C |
|------|------|:---:|:---:|:---:|:---:|
| LLaMA-3-8B | AR | 64.5 | 53.1 | 15.1 | 53.1 |
| Qwen2-7B | AR | 70.3 | 80.2 | 43.5 | 60.6 |
| LLaDA-8B | Diffusion | 65.9 | 70.7 | 27.3 | 47.9 |
| **MMaDA-8B** | Diffusion | **68.4** | **73.4** | **36.0** | **57.4** |

### 消融实验

| 阶段 | GSM8K | MATH500 | GeoQA | CLEVR | CLIP Score | ImageReward |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Stage 1 (预训练) | 17.4 | 4.2 | 8.3 | 10.3 | 23.1 | 0.69 |
| + Mixed Long-CoT | 65.2 | 26.5 | 15.9 | 27.5 | 29.4 | 0.84 |
| + UniGRPO | **73.4** | **36.0** | **21.0** | **34.5** | **32.5** | **1.15** |

| 掩码策略对比 | 效果 | 说明 |
|-------------|------|------|
| d1 (问题掩码+答案全掩码) | 收敛慢、奖励低 | 忽略了扩散模型多步特性 |
| 完全随机掩码率 | 训练不稳定 | 奖励震荡大 |
| **UniGRPO (均匀随机)** | **收敛快、奖励高** | 近似 Monte Carlo 平均 |

### 关键发现

- **跨模态协同效应**：三个任务（文本/理解/生成）在 Stage 2 训练期间所有指标同步提升，文本推理能力直接提升图像生成的语义准确性
- **采样效率**：图像生成仅用 15 步去噪即可保持强劲性能（CLIP 31.7 vs 完整 1024 步的 32.8），文本生成 256 步即可
- **天然支持修复任务**：扩散模型可直接用于文本 span 预测、VQA 答案补全和图像修复，无需额外微调

## 亮点与洞察

- **首个全扩散多模态基础模型**进行系统后训练，证明扩散模型不仅能做生成，也能做理解和推理
- UniGRPO 的均匀随机掩码策略既高效又稳定，解决了将 GRPO 迁移到扩散模型的关键技术障碍
- WISE Cultural 基准上大幅领先（0.67 vs SDXL 的 0.43），表明推理增强确实帮助了知识密集型图像生成
- 扩散模型天然的并行解码 + 修复能力是相对 AR 模型的结构性优势

## 局限与展望

- 与 Qwen2-7B 等顶尖 AR 模型在纯文本任务上仍有差距（MMLU 68.4 vs 70.3，GSM8K 73.4 vs 80.2）
- 图像分辨率限于 512×512，未探索高分辨率生成
- 多模态理解在部分基准上不及 LLaVA-v1.5（如 MME 1410.7 vs 1510.7）
- 训练成本高（64 A100 GPU），可扩展性受限
- 未探索视频生成或更多模态的统一

## 相关工作与启发

- 与 Show-o（AR+扩散混合）形成直接对比，MMaDA 证明纯扩散也能达到甚至超越混合架构
- LLaDA 为文本扩散模型奠基，MMaDA 将其拓展到多模态并首次添加完整后训练
- UniGRPO 与 RLVR-World（本批另一篇）的 GRPO 互补——前者面向离散扩散模型，后者面向自回归视频模型
- 可启发探索更大规模扩散基础模型的后训练范式

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次在纯扩散架构下实现三合一（文本+理解+生成）+系统后训练，开创性工作
- **实验充分度**: ⭐⭐⭐⭐ 覆盖三类任务多个基准，消融充分，但缺乏与更大模型（如 13B/70B）的对比
- **写作质量**: ⭐⭐⭐⭐ 方法阐述清晰系统，但论文结构较冗长
- **价值**: ⭐⭐⭐⭐⭐ 为扩散模型作为通用基础模型开辟了新方向，UniGRPO 具有广泛适用性

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](reinforcing_the_diffusion_chain_of_lateral_thought_with_diffusion_language_model.md)
- [\[AAAI 2026\] DiffA: Large Language Diffusion Models Can Listen and Understand](../../AAAI2026/image_generation/diffa_large_language_diffusion_models_can_listen_and_understand.md)
- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](non-markovian_discrete_diffusion_with_causal_language_models.md)
- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [\[CVPR 2025\] Font-Agent: Enhancing Font Understanding with Large Language Models](../../CVPR2025/image_generation/font-agent_enhancing_font_understanding_with_large_language_models.md)

<!-- RELATED:END -->
