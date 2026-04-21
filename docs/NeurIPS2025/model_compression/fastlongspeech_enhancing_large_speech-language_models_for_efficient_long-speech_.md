---
title: >-
  [论文解读] FastLongSpeech: Enhancing Large Speech-Language Models for Efficient Long-Speech Processing
description: >-
  [NeurIPS 2025][模型压缩][speech compression] 提出 FastLongSpeech，通过迭代融合策略压缩冗余语音表征和动态压缩训练转移短语音能力到长语音场景，使 LSLM 无需长语音训练数据即可高效处理长语音，在长语音 QA 上实现最优性能且推理效率提升 70%。
tags:
  - NeurIPS 2025
  - 模型压缩
  - speech compression
  - long-speech processing
  - iterative fusion
  - CTC
  - dynamic compression training
---

# FastLongSpeech: Enhancing Large Speech-Language Models for Efficient Long-Speech Processing

**会议**: NeurIPS 2025  
**arXiv**: [2507.14815](https://arxiv.org/abs/2507.14815)  
**代码**: 有 (github.com/ictnlp/FastLongSpeech)  
**领域**: 模型压缩 / 语音模型效率  
**关键词**: speech compression, long-speech processing, iterative fusion, CTC, dynamic compression training

## 一句话总结

提出 FastLongSpeech，通过迭代融合策略压缩冗余语音表征和动态压缩训练转移短语音能力到长语音场景，使 LSLM 无需长语音训练数据即可高效处理长语音，在长语音 QA 上实现最优性能且推理效率提升 70%。

## 研究背景与动机

大语音语言模型（LSLM）如 Qwen2-Audio 在短语音任务上表现出色，但处理长语音（>30秒）面临两大挑战：

**训练数据稀缺**：长语音对齐和指令微调数据极少，生成代价高

**计算代价高**：语音表征序列通常是等价文本的 4 倍以上长度，长语音导致 LLM 的计算成本剧增

现有方案局限：
- **级联方法**（ASR→LLM）：误差传播、丢失副语言信息
- **NTK-RoPE**：扩展位置编码上限，但未压缩序列，计算量依然大（61.21 TFLOPs）
- **简单压缩**（随机采样、平均池化）：信息损失大，生成质量差
- 仅有少数 LSLM（如 Gemini）通过构建大规模长语音数据集实现 30 分钟处理，但代价极高

核心问题：如何仅用短语音数据，使 LSLM 高效处理长语音？

## 方法详解

### 整体框架

FastLongSpeech 基于 Qwen2-Audio，增加 speech extractor 模块：

```
原始语音 s → 音频编码器 (Whisper) → 语音表征 h (25Hz)
→ Extractor（CTC 解码器 + 迭代融合） → 压缩表征 h' (长度 ≤ L)
→ LLM → 文本响应 y
```

对于长语音推理：先将输入分割为 30 秒片段 → 各片段独立通过音频编码器 → 拼接为完整语音表征 → 迭代融合压缩到目标长度 L。

### 关键设计

#### 1. 迭代融合策略（Iterative Fusion）

核心思想：逐步合并冗余帧，保留信息密度高的帧。每次迭代将序列长度减半，直到达到目标长度 L。

**两个度量指标**：

**内容密度（Content Density）**：利用 CTC 解码器输出，非空白 token 概率之和衡量帧的文本信息含量：
$$d_j = \sum_{a_j \neq \epsilon} p_{ctc}(a_j \mid h_j)$$

**帧间相似度**：相邻帧的余弦相似度：
$$e_{j,j+1} = \frac{h_j h_{j+1}}{|h_j| |h_{j+1}|}$$

**迭代过程**：
1. 计算当前长度 T(m) 和目标长度 T(m+1) = ⌊T(m)/2⌋（若 T(m)>2L），否则 T(m+1)=L
2. 计算需要减少的帧数 r(m) = T(m) - T(m+1)
3. 找出 r(m) 个最相似的相邻帧对
4. 将连续识别帧分组为 span，在每个 span 内按内容密度加权融合为单帧
5. 重复直到序列长度 ≤ L

关键优势：渐进缩小感受野（每轮减半），相比一步到位的压缩更好地保留语义信息；内容密度引导保留信息量大的帧。

#### 2. 动态压缩训练（Dynamic Compression Training, DCT）

问题：LLM 只见过原始长度的语音表征，直接输入压缩表征会失配。

方案：在训练时随机采样目标长度 L，让 LLM 适应不同压缩比的压缩表征：

$$L_{dct} = -\sum_{L \sim \mathcal{U}(\mathbf{L})} \log p(\mathbf{y} \mid \mathbf{x}, \text{IF}(\mathbf{h}, L))$$

其中 L 从集合 {750, 400, 200, 100, 50, 25, 12} 中均匀采样。这样 LLM 学会处理从无压缩到 60× 压缩的各种输入。

### 训练策略

**两阶段训练**：

**Stage 1 - CTC 训练**：
- 仅训练 CTC 解码器，使其学会衡量语音帧的内容密度
- 使用 LibriSpeech 960h + MLS 3000h 的 ASR 数据
- 冻结其他所有模块

**Stage 2 - 动态压缩训练**：
- 使用 LoRA 微调 Qwen2-Audio 的 LLM 部分
- 训练数据：OpenASQA（5.9kh）+ LibriSQA（360h）+ Common Voice（1.7kh），全部 <30s 短语音
- CTC 解码器冻结
- 语音窗口 L=750（Qwen2-Audio 原始设置）

## 实验关键数据

### 主实验：长语音 Spoken QA

| 方法 | Score (↑) | 说明 |
|------|-----------|------|
| Random | 2.54 | 随机采样帧 |
| Similar (MostSim) | 3.08 | 合并最相似帧 |
| AvgPool | 3.10 | 平均池化 |
| NTK-RoPE | 3.44 | 扩展位置编码（不压缩） |
| **FastLongSpeech** | **3.55** | 迭代融合+动态压缩训练 |

在长语音理解任务上，FastLongSpeech 在保持与 NTK-RoPE 相同的语音窗口下，性能最优。

### 推理效率对比

| 方法 | Score | TFLOPs (↓) | 时间/秒 (↓) |
|------|-------|-----------|------------|
| NTK-RoPE | 3.44 | 61.21 | 4.80 |
| Cascaded (Whisper+LLM) | 3.75 | n/a | 17.23+1.38 |
| **FastLongSpeech** | **3.55** | **26.44** | **1.47** |

计算量减少 57%，推理速度提升 3.3×（vs. NTK-RoPE），7× 加速（vs. Cascaded）。

### 短语音效率（LibriTTS OpenASQA）

| 方法 | Score (↑) | TFLOPs (↓) |
|------|-----------|-----------|
| Baseline (Qwen2-Audio) | 3.73 | 9.79 |
| Ours (L=400) | **3.80** | 8.54 |
| Ours (L=200) | **3.87** | **5.64** |
| Ours (L=100) | 3.71 | 4.17 |

在短语音上也能以一半计算量匹配甚至超越基线。

### 消融实验

| 方法 | Score (↑) |
|------|-----------|
| FastLongSpeech（完整） | **3.55** |
| w/o DCT（去掉动态压缩训练） | 3.33 (-0.22) |
| w/o Iterative Fusion（单步压缩） | 3.41 (-0.14) |
| w/o Content Density（均匀权重融合） | 3.28 (-0.27) |

三个组件都有显著贡献。内容密度引导贡献最大，说明区分有信息帧和冗余帧是关键。

### 关键发现

1. 迭代（多轮减半）优于一步压缩：渐进扩大感受野更好地聚合语义
2. CTC 内容密度是有效的信息量指标，引导保留高信息帧
3. 动态压缩训练成功将短语音能力转移到长语音场景，无需长语音训练数据
4. ASR 任务在低压缩比（L=400）时 WER 仅比 Qwen2-Audio 高 0.23，但高压缩比（L=100）时 WER 显著上升，说明最优压缩比取决于任务
5. 方法可直接扩展到 Qwen2.5-Omni（无需 DCT 训练）

## 亮点与洞察

- **零长语音数据训练**：仅用 <30s 短语音数据，通过动态压缩训练实现长语音处理能力迁移
- **信息感知压缩**：CTC 输出分布自然提供帧级信息密度，比简单相似度或随机采样更有效
- **灵活的效率-性能权衡**：通过调节目标长度 L 可自由平衡推理效率和生成质量
- **LongSpeech-Eval 基准**：构建了首个长语音理解评测集，填补了领域空白

## 局限与展望

1. **仅基于 Qwen2-Audio**：对其他 LSLM 的通用性待验证
2. **长语音训练数据仍然有潜力**：随长语音数据增多，直接训练可能优于压缩迁移
3. **CTC 解码器引入额外参数**：虽然相对轻量，但增加了系统复杂度
4. **高压缩比下 ASR 退化明显**：内容完整性和效率的平衡仍待优化
5. 可探索方向：端到端联合训练 CTC+LLM、自适应压缩比选择（根据语音复杂度动态调节 L）

## 相关工作与启发

- **SpeechPrune / FastAdaSP**：token 选择/剪枝策略，FastLongSpeech 的融合策略保留了更多信息
- **StreamUni**：实时语音翻译的分段策略，本文的迭代融合可视为离线版本的高效压缩
- **NTK-RoPE**：扩展上下文长度的经典方案，但不减少计算量
- 启发：语音的高冗余特性（帧率远高于文本 token 率）为压缩提供了天然空间，CTC 的空白概率是衡量冗余度的优秀指标

## 评分

- 新颖性：★★★★☆（迭代融合+CTC 内容密度+动态压缩训练的组合新颖）
- 技术深度：★★★★☆（问题分解清晰，各组件设计有明确动机和理论支撑）
- 实验充分度：★★★★☆（多任务评估+消融+效率分析完整，但仅基于单个基础模型）
- 实用价值：★★★★★（无需长语音数据，即插即用，推理大幅加速，实际应用价值高）

<!-- RELATED:START -->

## 相关论文

- [Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)
- [Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models](../../ICML2025/model_compression/lacache_ladder-shaped_kv_caching_for_efficient_long-context_modeling_of_large_la.md)
- [Correlation Dimension of Auto-Regressive Large Language Models](correlation_dimension_of_auto-regressive_large_language_models.md)
- [Restoring Pruned Large Language Models via Lost Component Compensation](restoring_pruned_large_language_models_via_lost_component_compensation.md)

<!-- RELATED:END -->
