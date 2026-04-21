---
title: >-
  [论文解读] Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space
description: >-
  [NEURIPS2025][语音][speech language model] 提出 SLED，将语音波形编码为连续潜在表示序列，在连续空间中通过 energy distance 目标进行自回归建模，避免了离散化信息损失和 RVQ 所需的复杂层级架构，同时实现高效的零样本与流式语音合成。
tags:
  - NEURIPS2025
  - 语音
  - speech language model
  - continuous latent space
  - energy distance
  - zero-shot TTS
  - streaming synthesis
---

# Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space

**会议**: NEURIPS2025  
**arXiv**: [2505.13181](https://arxiv.org/abs/2505.13181)  
**代码**: [ictnlp/SLED-TTS](https://github.com/ictnlp/SLED-TTS)  
**领域**: audio_speech  
**关键词**: speech language model, continuous latent space, energy distance, zero-shot TTS, streaming synthesis

## 一句话总结

提出 SLED，将语音波形编码为连续潜在表示序列，在连续空间中通过 energy distance 目标进行自回归建模，避免了离散化信息损失和 RVQ 所需的复杂层级架构，同时实现高效的零样本与流式语音合成。

## 背景与动机

- 文本语言模型（GPT 系列）的成功激发了以类似自回归方式建模语音的研究，但语音本质上是连续高采样率信号，与离散文本存在根本差异
- 主流方法通过 residual vector quantization (RVQ) 将语音离散化为多流 token 序列，但带来两个核心问题：
    1. **信息瓶颈**：离散化不可避免地丢失原始波形中的丰富细节，降低重建质量
    2. **架构复杂性**：RVQ 产生的多流序列需要层级自回归架构（如 VALL-E 的 AR+NAR 两阶段，或 RQ-Transformer 的嵌套 Transformer），增加了建模和工程难度
- 连续潜在空间建模可以绕开上述问题，但核心挑战在于：如何构建一个**轻量、高表达力、训练稳定且采样高效**的逐步条件生成模块——理想情况下它应当像离散模型中的 softmax 一样简洁

## 核心问题

在连续潜在空间中进行语音自回归建模时，如何设计逐步分布的学习目标与生成模块，使其兼顾建模能力、训练稳定性和推理效率？

## 方法详解

### 1. 连续潜在空间编码

使用 Encodec 将原始语音波形编码为连续向量序列。具体做法是将 Encodec 八个 codebook 的 token embedding 逐帧求和，得到 75Hz 采样率、128 维的连续表示 $\bm{h} \in \mathbb{R}^{Tf_h \times 128}$，保留几乎全部信息。

### 2. 自回归网络 + 轻量生成模块

整体架构分为两个组件：

- **自回归网络 $\psi$**：12 层 LLaMA-style Transformer（RMSNorm、SwiGLU、RoPE），捕获序列依赖关系，输出条件向量 $\bm{z}_t = \psi(\bm{h}_{<t}; \theta)$
- **逐步生成模块 $g$**：轻量 MLP（6 个残差块 + AdaLN），接收条件向量 $\bm{z}_t$ 和随机噪声 $\bm{\epsilon}$，隐式定义连续分布 $p_g(\bm{h}_t | \bm{z}_t)$

$$\bm{h}_t = g(\bm{z}_t, \bm{\epsilon}; \phi)$$

AdaLN 模块将噪声通过线性变换预测 scale 和 shift 参数，对条件向量进行随机调制。采样时仅需单次前向传播，与 softmax 采样效率相当。

### 3. Energy Distance 训练目标

采用 generalized energy distance (GED) 作为训练损失，它是 MMD 的特例。对每个时间步最小化模型分布与数据分布之间的 energy distance：

$$\mathcal{L}_{\text{GED}} = \sum_t \mathbb{E}_{\bm{h}_t, \bm{h}'_t} \left[ 2 \| \bm{h}_t - \bm{h}_t^* \|_2 - \| \bm{h}_t - \bm{h}'_t \|_2 \right]$$

其中 $\bm{h}_t, \bm{h}'_t$ 是从 $p_g$ 中独立采样的两个样本，$\bm{h}_t^*$ 是目标。关键在于：

- 第一项是与目标的距离（类似 RMSE）
- 第二项是**排斥项** $\|\bm{h}_t - \bm{h}'_t\|_2$，防止模型退化为点回归——去掉此项等价于 RMSE 损失，实验表明会导致模型完全失败（WER 从 1.59 暴涨至 40.60）
- 当距离函数选择 $d(\bm{x}, \bm{y}) = \|\bm{x} - \bm{y}\|_2^\beta$（$\beta \in (0,2)$）时，GED 构成 strictly proper scoring rule，保证训练收敛到真实分布

### 4. Classifier-Free Guidance (CFG)

推理时在每一步额外进行一次 text-masked 前向传播得到无条件输出 $\bm{z}'_t$，通过线性插值增强文本对齐：

$$\bm{z}_t^{\text{cfg}} = \bm{z}'_t + \lambda (\bm{z}_t - \bm{z}'_t)$$

默认 $\lambda = 2.0$，在准确率和语音质量间取得平衡。训练时以 0.1 概率随机 mask 文本。

### 5. 流式推理

通过文本-语音位置交错实现增量合成：每接收 $n$ 个文本 subword 即生成 $m$ 个语音向量（如 5:20 或 5:45）。纯自回归架构无需任何后处理，天然支持流式。通过二分类头预测停止位置。

## 实验关键数据

训练数据：LibriHeavy（约 50,000 小时语音，6,736 说话人），BF16，batch size 512，训练 300K 步。

**零样本 TTS 性能**（LibriSpeech test-clean）：

| 设置 | WER-C (%) | WER-H (%) | SIM |
|------|-----------|-----------|-----|
| 3s 前缀提示 | 1.59 | 1.99 | 0.515 |
| 参考语音提示 | 1.51 | 1.97 | 0.664 |
| Ground Truth | 1.78 | 2.15 | 0.668/0.778 |

- WER 超过 ground truth（1.59 vs 1.78），表明极高的文本还原准确性
- 流式推理 DNSMOS（3.59）接近离线（3.58），WER 仅小幅上升（2.18 vs 1.67）

**效率对比（10秒音频推理）**：

| 模型 | 参数量 | RTF | FLOPs |
|------|--------|-----|-------|
| SLED | 0.2B | 0.8 | 280G |
| DiTAR | 0.6B | 0.66 | 2750G |

SLED 仅用 DiTAR 约 1/10 的 FLOPs 和 1/3 参数量即达到相近实时因子。

## 亮点

- **架构极简**：单层自回归 + 轻量 MLP 生成器（~35M），无需层级架构或后处理，相比 VALL-E 的 NAR 模块（~159M）更高效
- **理论扎实**：energy distance 作为 strictly proper scoring rule 有严格数学保证；论文还深入分析了 MELLE 的 flux loss 本质上近似 energy distance 的排斥项
- **流式天然支持**：纯自回归模型无需后处理即可逐步输出，适合实时语音交互系统
- **实验发现有价值**：1000 小时数据即可获得大部分生成与 in-context learning 能力

## 局限与展望

- 当前使用 Encodec（为 codec 设计），专门为连续自回归建模训练编码器应能进一步提升性能
- 语音克隆（SIM）与传统 TTS 模型（MegaTTS 3: 0.78）仍有差距
- 仅在语音合成任务上验证，尚未扩展到通用语音语言模型（语音理解、对话等）
- CFG 需要额外一次前向传播，增加了约一倍的推理计算量

## 与相关工作的对比

| 方法 | 潜在空间 | 每步采样 | 后处理 | 流式支持 |
|------|----------|----------|--------|----------|
| VALL-E | 离散 (RVQ) | softmax | NAR 模型 | 否 |
| MELLE | 连续 (mel) | 回归+flux loss | NAR 精炼 | 否 |
| FELLE | 连续 | ODE 多步积分 | 无 | 否 |
| DiTAR | 连续 (patch) | DiT 迭代 | 无 | 否 |
| **SLED** | 连续 (Encodec) | **单次 MLP** | **无** | **是** |

SLED 是唯一同时实现单次采样、无后处理和流式推理的连续语音语言模型。

## 启发与关联

- Energy distance 作为隐式生成模型的训练目标，思路可迁移到其他连续序列建模场景（视频生成、运动生成等）
- 排斥项的关键性揭示了回归损失与分布匹配损失的本质区别——这对所有连续 token 预测任务都有指导意义
- 流式文本-语音交错方案可直接用于 GPT-4o 类实时语音交互系统的 TTS 模块
- Llasa (8B) 证明离散方法的 scaling 潜力，SLED 在连续域的 scaling 值得期待

## 评分
- 新颖性: 8/10 — energy distance 用于连续语音 LM 是新颖且理论扎实的贡献
- 实验充分度: 8/10 — 零样本/流式/消融/效率分析全面，但缺少更大规模实验
- 写作质量: 9/10 — 数学推导清晰，从 MMD 到 GED 的理论链条完整
- 价值: 8/10 — 显著简化连续语音 LM 架构，为后续 scaling 和通用化奠定基础

<!-- RELATED:START -->

## 相关论文

- [Latent Space Factorization in LoRA](latent_space_factorization_in_lora.md)
- [FLAM: Frame-Wise Language-Audio Modeling](../../ICML2025/audio_speech/flam_frame-wise_language-audio_modeling.md)
- [Bridging the Language Gap: Synthetic Voice Diversity via Latent Mixup for Equitable Speech Recognition](../../ICML2025/audio_speech/bridging_the_language_gap_synthetic_voice_diversity_via_latent_mixup_for_equitab.md)
- [E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models](e-bats_efficient_backpropagation-free_test-time_adaptation_for_speech_foundation.md)
- [Multi-head Temporal Latent Attention](multi-head_temporal_latent_attention.md)

<!-- RELATED:END -->
