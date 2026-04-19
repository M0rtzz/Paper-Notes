---
title: >-
  [论文解读] ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning
description: >-
  [ACL 2025][LLM对齐][DPO] 将 DPO 的偏好优化粒度从回复级细化到句子级，通过图文相似度和文本困惑度两个维度动态计算每个句子的自适应奖励权重，在 LLaVA-1.5-7B/13B 和 InstructBLIP-13B 上分别带来平均 2.57/2.87/1.98 分提升，同时显著降低幻觉率。
tags:
  - ACL 2025
  - LLM对齐
  - DPO
  - 句子级偏好
  - 自适应奖励
  - 多模态推理
  - 幻觉缓解
---

# ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning

**会议**: ACL 2025  
**arXiv**: [2505.19100](https://arxiv.org/abs/2505.19100)  
**代码**: 无  
**领域**: 多模态LLM / 偏好优化  
**关键词**: DPO, 句子级偏好, 自适应奖励, 多模态推理, 幻觉缓解

## 一句话总结

将 DPO 的偏好优化粒度从回复级细化到句子级，通过图文相似度和文本困惑度两个维度动态计算每个句子的自适应奖励权重，在 LLaVA-1.5-7B/13B 和 InstructBLIP-13B 上分别带来平均 2.57/2.87/1.98 分提升，同时显著降低幻觉率。

## 研究背景与动机

**多模态大模型的幻觉问题日益严重。** 经过 SFT 的 MLLM 在长回复中容易产生幻觉，且增加 preferred 输出概率会附带增加 dis-preferred 输出概率。DPO 作为主流对齐方法虽然简单高效，但根本局限在于以二元制对整个回复做偏好优化——回复中哪些句子正确、哪些错误完全不区分。

**噪声数据加剧了这一问题。** 机器生成的偏好数据中，chosen 回复的句子可能部分正确部分错误。传统 DPO 将所有句子等权处理，正确句子和错误句子获得相同的奖励权重，这导致模型在包含噪声的半正确数据上收敛到次优解。

**现有细粒度方法依赖外部资源。** RLHF-V 需要人工标注段落级纠正，FiSAO 需要额外视觉编码器做验证，TLDR 需要训练额外奖励模型。这些方法要么成本高，要么引入额外参数和模型。**ASPO 的核心 idea 是利用模型自身的预测来评估每个句子的质量**，不需要任何外部模型、API 或额外标注数据，实现零额外成本的细粒度偏好优化。

## 方法详解

### 整体框架

ASPO 在标准 DPO 的基础上引入句子级自适应奖励机制。训练流程：对每个偏好对 $(x, y_c, y_r)$，将 chosen 回复 $y_c$ 拆分为句子序列 $\{s_1, s_2, \ldots, s_n\}$，计算每个句子的两个特征（图文相似度和文本困惑度），融合为自适应权重 $w_i$，然后用加权的句子级隐式奖励替代原始的回复级隐式奖励进行优化。

### 关键设计

1. **图文相似度权重 (Image-Text Similarity)**:
    - 功能：衡量每个句子与输入图像的语义相关性
    - 核心思路：用 CLIP 计算每个句子 $s_i$ 与图像 $x$ 的余弦相似度 $S_i = \text{cosine}(s_i, x)$，再做 min-max 归一化得到 $S'_i \in [0, 1]$
    - 设计动机：与图像高度相关的句子更可能是正确的描述，低相关的句子可能是幻觉。通过图文相似度赋予高相关句子更大的奖励权重，减轻幻觉影响

2. **文本困惑度权重 (Textual Perplexity)**:
    - 功能：衡量模型对每个句子的预测置信度
    - 核心思路：计算每个句子的条件困惑度 $PPL_i = \exp(-\frac{1}{N}\sum_{j=M+1}^{M+N}\log P(w_j|x, w_{<j}))$，取负后 min-max 归一化得到 $PPL'_i$（低困惑度=高置信度=高权重）
    - 设计动机：模型置信度高的句子更可能正确；困惑度反映了模型内部对该句子的确定性

3. **自适应权重融合与奖励缩放**:
    - 功能：融合两个维度的权重并归一化
    - 核心思路：$w_i = \alpha S'_i + (1-\alpha) PPL'_i$，其中 $\alpha$ 平衡两个指标的贡献。最终的自适应隐式奖励边际为 $\mathcal{M}^* = \frac{R_c}{R_c^*}\sum_{i=1}^{K}\beta(1+w_i)\log\frac{\pi_\theta(s_i^c|x)}{\pi_{ref}(s_i^c|x)} - \beta\log\frac{\pi_\theta(y_r|x)}{\pi_{ref}(y_r|x)}$
    - 设计动机：$R_c/R_c^*$ 归一化因子防止长回复因总权重增加而不成比例地获益。当回复仅有一个句子时，$w_i$ 归一化为 0，ASPO 退化为标准 DPO

### 损失函数 / 训练策略

损失函数为 $\mathcal{L}_{ASPO} = -\mathbb{E}_\mathcal{D}[\log\sigma(\mathcal{M}^*)]$，形式上与 DPO 一致但用自适应奖励边际替代了原始边际。偏好数据通过 SeVa 管线生成：对 LLaVA-Instruct-150K 中采样的约 20K 指令，给图像加扩散噪声，原始图像和噪声图像分别生成 chosen/rejected 回复，过滤掉相同的对，最终得到约 16K 偏好对。噪声步数设为 500。

## 实验关键数据

### 主实验（LLaVA-1.5-7B 基座）

| 基准 | LLaVA-1.5 | +DPO | +ASPO | 提升(vs DPO) |
|------|-----------|------|-------|-------------|
| MMVet | 30.5 | 33.3 | **35.3** | +2.0 |
| MMB-D | 64.3 | 64.7 | **65.6** | +0.9 |
| LLaVA-W | 63.4 | 65.7 | **75.7** | +10.0 |
| SQA-I | 66.8 | 66.4 | **67.7** | +1.3 |
| POPE | 85.9 | 86.2 | **86.6** | +0.4 |
| SHR↓ | 36.7 | 40.1 | **33.9** | -6.2 |
| 平均(除SHR) | 62.59 | 63.12 | **65.16** | +2.04 |

### 与其他偏好优化方法对比（LLaVA-1.5-7B）

| 方法 | 粒度 | MMVet | MMB-D | LLaVA-W | SQA-I | POPE | 平均 |
|------|------|-------|-------|---------|-------|------|------|
| POVID | 回复级 | 31.8 | 64.9 | 68.7 | 68.8 | 86.9 | 64.22 |
| CSR iter-3 | 句子级 | 33.9 | 65.4 | 71.1 | 70.7 | 85.9 | 65.40 |
| FiSAO | Token级 | 30.7 | 64.8 | - | 69.3 | 85.7 | - |
| **ASPO** | **句子级** | **35.3** | **65.6** | **75.7** | 67.7 | **86.6** | **66.18** |

### 消融实验

| 配置 | 平均分 | 说明 |
|------|--------|------|
| ASPO (完整) | 65.16 | 图文相似度+困惑度 |
| ASPO-S (仅相似度) | 64.91 | 去掉困惑度，-0.25 |
| ASPO-P (仅困惑度) | 65.00 | 去掉相似度，-0.16 |
| DPO (回复级) | 63.12 | 无细粒度权重，-2.04 |

### 关键发现

- ASPO 在 LLaVA-W 上提升最大（+10.0），该基准评估开放式长回复质量——长回复正是细粒度优化收益最大的场景
- SHR（幻觉率）从 40.1 降到 33.9，说明 ASPO 有效减少了幻觉
- 两个维度（相似度和困惑度）互补，去掉任一个都会降低性能
- 在 LLaVA-1.5-13B 上同样显著提升（平均 65.93→68.80），验证了 ASPO 的规模泛化性
- 在 InstructBLIP-13B（Q-Former 架构）上也有效（43.86→45.84），说明不依赖特定架构

## 亮点与洞察

- **句子是偏好优化的"甜蜜粒度"**：比回复级精确，比 token 级语义完整。ASPO 超越了 token 级方法 FiSAO 和大多数回复级方法
- **零额外成本的自监督信号**：图文相似度用现有 CLIP，困惑度用模型自身预测——不需要训练额外模型或调用付费 API
- **长度归一化设计精巧**：$R_c/R_c^*$ 因子确保长回复不会因为句子多而获得不成比例的总权重增加
- **退化性质优雅**：单句回复自动退化为 DPO，保持了与标准方法的兼容性

## 局限与展望

- **句子分割质量影响大**：错误的句子边界会导致奖励分配不准确，论文未讨论分割鲁棒性
- **仅在多模态场景验证**：图文相似度权重依赖视觉信息，纯文本任务需要替代方案
- **$\alpha$ 超参敏感性**：如何最优地平衡图文相似度和困惑度的权重未充分探索
- **rejected 回复未做句子级加权**：仅对 chosen 回复的句子分配自适应权重，rejected 仍为整体

## 相关工作与启发

- **vs CSR (迭代句子级)**：CSR 需要多轮迭代生成候选回复+外部评估；ASPO 单次训练即完成，且性能更优
- **vs RLHF-V (人工段落纠正)**：依赖昂贵的人工标注；ASPO 完全自监督
- **vs FiSAO (token 级)**：token 粒度过细导致语义碎片化；句子级保持了语义完整性
- **vs MDPO (回复级+图像优先)**：解决了无条件偏好问题但粒度仍然粗糙；ASPO 在粒度和信号质量上都更优

## 评分

- 新颖性: ⭐⭐⭐⭐ 句子级自适应奖励是对 DPO 的有效改进，图文相似度+困惑度双维度设计合理
- 实验充分度: ⭐⭐⭐⭐ 3种架构的基座模型+10个基准+与12种方法对比+消融
- 写作质量: ⭐⭐⭐⭐ 图示清晰，公式推导完整，退化性质分析好
- 价值: ⭐⭐⭐⭐ 对多模态 DPO 方法论有实用改进，零额外成本是重要优势

<!-- RELATED:START -->

## 相关论文

- [Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization](fine-grained_video_dubbing_duration_alignment_with_segment_supervised_preference.md)
- [DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models](../../NeurIPS2025/llm_alignment/densedpo_finegrained_temporal_preference_optimization_for_vi.md)
- [Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)
- [SDPO: Segment-Level Direct Preference Optimization for Social Agents](sdpo_segment-level_direct_preference_optimization_for_social_agents.md)
- [T-REG: Preference Optimization with Token-Level Reward Regularization](t-reg_preference_optimization_with_token-level_reward_regularization.md)

<!-- RELATED:END -->
