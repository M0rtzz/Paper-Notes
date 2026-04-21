---
title: >-
  [论文解读] MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition
description: >-
  [NeurIPS 2025][语音][音视频语音识别] MoME将稀疏MoE集成到Matryoshka表示学习框架中，用于LLM-based音视频语音识别，通过共享路由器实现跨粒度知识迁移，在单一模型权重下支持多种压缩率的弹性推理，同时达到AVSR/ASR/VSR的SOTA性能。
tags:
  - NeurIPS 2025
  - 语音
  - 音视频语音识别
  - Matryoshka表示学习
  - Mixture-of-Experts
  - 弹性推理
  - LLM
---

# MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition

**会议**: NeurIPS 2025  
**arXiv**: [2510.04136](https://arxiv.org/abs/2510.04136)  
**代码**: 无  
**领域**: Audio & Speech  
**关键词**: 音视频语音识别, Matryoshka表示学习, Mixture-of-Experts, 弹性推理, LLM

## 一句话总结

MoME将稀疏MoE集成到Matryoshka表示学习框架中，用于LLM-based音视频语音识别，通过共享路由器实现跨粒度知识迁移，在单一模型权重下支持多种压缩率的弹性推理，同时达到AVSR/ASR/VSR的SOTA性能。

## 研究背景与动机

LLM-based AVSR面临核心矛盾：**token hunger vs 计算成本**。音视频语音信号的时间分辨率远高于文本，导致输入token数量巨大。现有token压缩方法（拼接、重采样、平均池化等）需要预先固定压缩率，输出固定长度序列，无法在推理时动态平衡精度与效率。

Matryoshka表示学习(MRL)通过训练时使用多种压缩率，使单一模型支持推理时动态调整粒度。但现有Matryoshka方法存在两大不足：

**独立训练各粒度**——每个分辨率被当作独立问题，缺乏跨尺度交互，高压缩率下信息损失严重

**均匀单体表示**——所有尺度使用相同的单体网络结构，无法针对不同粒度做专业化处理

MoME的核心思想是：用MoE的稀疏专家实现跨粒度的知识迁移——同一组专家在不同压缩率下被类似地激活，低分辨率序列可以复用高分辨率序列训练出的专家路径。

## 方法详解

### 整体框架

输入音频/视频 → 预训练编码器(Whisper/AV-HuBERT) → 多种压缩率的Matryoshka token序列 → 冻结LLM(Llama 3) + MoME模块(并行插入) → 自回归解码转录文本。训练时对所有 $G \times L$ 种音视频压缩率组合联合训练，推理时可选任意压缩率。

### 关键设计

1. **MoME模块结构**：每个MoME模块包含 $N_r$ 个路由专家和 $N_s$ 个共享专家。每个专家是瓶颈结构（线性下采样 → GELU → 线性上采样），瓶颈维度可极端压缩至1。路由器为线性层，通过top-k选择稀疏激活 $K$ 个路由专家：
   $$\text{MoME}(\mathbf{H}_l^{ij}) = \sum_{n=1}^{N_s} E_n(\mathbf{H}_l^{ij}) + \sum_{n=N_s+1}^{N_s+N_r} g_n E_n(\mathbf{H}_l^{ij})$$
   其中 $g_n$ 由top-k稀疏门控决定。

2. **共享路由器与跨粒度对齐**：核心创新在于MoME模块的专家和路由器在所有Matryoshka序列间共享。这意味着路由器在训练时同时处理高分辨率（信息丰富）和低分辨率（压缩严重）的输入，自然地学会在不同粒度下激活相似的专家子集。实验验证（Figure 5）确认了这种隐式对齐——同一层的专家激活分布在不同压缩率间高度一致，同时不同层间激活模式有显著差异，实现了层级多样性。

3. **共享专家**：借鉴DeepSeekMoE和Llama 4，引入1-2个始终激活的共享专家捕获全局、跨模态、尺度不变的知识。消融实验证实共享专家对WER有明显改善。

4. **灵活插入位置**：MoME可并行插入LLM层的三个位置：MHSA模块、FFN模块、或整个Transformer层。冻结LLM骨干，仅训练MoME模块（参数高效微调）。

### 损失函数 / 训练策略

多粒度平均交叉熵损失：
$$\mathcal{L}_{LM} = -\frac{1}{G \cdot L} \sum_{i=1}^{G}\sum_{j=1}^{L} \log p(\mathbf{Y}|\mathbf{Z}^{ij}) \cdot c_{ij}$$
$c_{ij}=1$表示各粒度等权。加上负载均衡损失 $\mathcal{L}_B$ 防止路由崩塌（系数0.01）。训练时音频压缩率{4,16}、视频{2,5}，共4种组合。

## 实验关键数据

### 主实验（AVSR, WER%↓）

| 方法 | 活跃参数 | LRS2 (4,2) | (4,5) | (16,2) | (16,5) | LRS3 (4,2) | (4,5) | (16,2) | (16,5) |
|------|---------|-----------|-------|--------|--------|-----------|-------|--------|--------|
| Llama-AVSR(独立) | 27.5M | 4.1 | 4.5 | 5.3 | 8.1 | 2.4 | 2.8 | 3.3 | 4.1 |
| Llama-MTSK SS | 27.5M | 3.4 | 4.7 | 4.8 | 6.4 | 2.3 | 2.2 | 3.3 | 3.6 |
| Llama-MTSK MSS | 55.0M | 3.6 | 4.8 | 6.1 | 9.0 | 2.4 | 2.4 | 3.2 | 3.5 |
| MoME-23/4-MHSA | **12.7M** | **2.9** | **3.0** | **4.2** | **4.3** | **1.8** | **1.7** | **2.9** | **2.9** |
| MoME-23/4-LAYER | **12.7M** | **2.7** | **2.7** | **4.2** | **4.2** | **1.5** | **1.8** | **3.1** | **3.2** |

MoME在所有压缩率下全面超越基线，同时活跃参数少2-4倍。

### 消融实验（MoME-MHSA on LRS2）

| 路由专家数 | 共享专家 | 瓶颈大小 | Top-k | (4,2) | (4,5) | (16,2) | (16,5) |
|-----------|---------|---------|-------|-------|-------|--------|--------|
| 1 | 0 | 48 | / | 3.4 | 3.4 | 4.9 | 5.1 |
| 4 | 0 | 24 | 2 | 3.3 | 3.3 | 4.8 | 5.0 |
| 4 | **1** | 24 | 2 | **3.2** | **3.2** | **4.4** | **4.7** |
| 23 | 1 | 12 | 4 | **2.9** | **3.0** | **4.2** | **4.3** |
| 23 | 2 | 12 | 4 | 2.8 | 3.0 | 4.1 | 4.7 |

### 关键发现

- 噪声鲁棒性：在SNR=-5dB下，MoME (32.6% WER) 大幅优于Llama-AVSR (41.8%) 和Llama-MTSK (44.9%)
- 极端压缩：瓶颈维度降至1时（0.9M活跃参数），WER仅轻微下降（LRS3: 1.8→2.0）
- 跨模态token分析（Figure 4）：不同压缩率的音视频token呈强线性相关，高压缩token约对应2-3个低压缩token
- 计算效率：(16,5)压缩率下TFLOPs减少8倍，推理时间从12.75s降至6.74s（23秒语音）

## 亮点与洞察

- 首个统一MoE和Matryoshka表示学习的框架，巧妙利用稀疏专家实现跨粒度知识迁移
- 共享路由器设计使得跨尺度对齐成为自然涌现的特性（而非显式约束）
- "浅层大脑假说"(shallow brain hypothesis)的类比很有启发性：深层LLM + 并行浅层MoME模块
- 单一模型权重支持弹性推理，是端侧部署友好的设计

## 局限与展望

- 仅验证英文语音识别，多语言/多任务泛化性待验证
- MoME模块的最优插入位置（MHSA/FFN/LAYER）因数据集而异，缺乏自动选择机制
- 未与MMS-Llama的自适应压缩策略（基于语速）对比其灵活性
- 共享专家数量超过2个时性能反而下降，原因未深入分析

## 相关工作与启发

- 与Llama-MTSK的核心区别：后者用Multi-Scale LoRA但各尺度独立，MoME通过共享路由器实现跨尺度关联
- DeepSeekMoE的共享专家思想在多模态Matryoshka场景证明有效
- 启发：MoME框架可推广到视觉-语言等其他多模态任务，只需替换编码器和压缩策略

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （MoE+MRL的首次统一，设计巧妙）
- 实验充分度: ⭐⭐⭐⭐⭐ （LRS2+LRS3、3种任务、详细消融+可视化）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，实验图表丰富）
- 价值: ⭐⭐⭐⭐⭐ （弹性推理+SOTA性能，端侧部署价值大）

<!-- RELATED:START -->

## 相关论文

- [MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation](megadance_mixture-of-experts_architecture_for_genre-aware_3d_dance_generation.md)
- [AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time](textttavrobustbench_benchmarking_the_robustness_of_audio-visual_recognition_mode.md)
- [MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens](../../ACL2025/audio_speech/mms-llama_efficient_llm-based_audio-visual_speech_recognition_with_minimal_multi.md)
- [Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](../../ICCV2025/audio_speech/zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)
- [MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](../../ICCV2025/audio_speech/mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)

<!-- RELATED:END -->
