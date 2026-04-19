---
title: >-
  [论文解读] Your Language Model Secretly Contains Personality Subnetworks
description: >-
   本文提出通过激活引导的剪枝（activation-guided pruning）从预训练 LLM 中提取人格专用子网络，无需任何训练即可实现高效的人格切换，并引入对比剪枝策略增强对立人格间的参数分离。
tags:

---

# Your Language Model Secretly Contains Personality Subnetworks

## 基本信息

- **会议**: ICLR 2026
- **arXiv**: [2602.07164](https://arxiv.org/abs/2602.07164)
- **代码**: [GitHub](https://github.com/Ruimeng-Ye/Persona)
- **领域**: 模型压缩 / 可解释性 / 人格建模
- **关键词**: persona subnetwork, network pruning, contrastive pruning, MBTI, activation-guided masking

## 一句话总结

本文提出通过激活引导的剪枝（activation-guided pruning）从预训练 LLM 中提取人格专用子网络，无需任何训练即可实现高效的人格切换，并引入对比剪枝策略增强对立人格间的参数分离。

## 研究背景与动机

- 人类在不同社交场景中自然切换人格，LLM 同样能采用不同角色，但现有方法依赖外部知识注入
- **Prompt 方法**：简单快速，但人格保持不稳定，容易漂移
- **RAG 方法**：需要检索管道，存在干扰问题
- **微调方法**：需要额外训练，成本高（数小时到数天）
- **核心问题**：LLM 是否真的需要外部干预才能展现不同人格？还是这些行为已经嵌入在参数空间中？
- 受 Lottery Ticket Hypothesis 启发，作者假设单个预训练模型中**已包含**多个对应不同人格的"中奖彩票"子网络

## 方法详解

### 整体框架

给定预训练 LLM 和少量人格校准数据 → 收集激活统计量 → 构建二值掩码 → 隔离人格子网络 → 推理时应用掩码实现人格开关

### 问题定义

对每个人格 $p \in \mathcal{P}$，假设有小规模校准集 $\mathcal{D}_p = \{(x_i^p, y_i^p)\}_{i=1}^{N_p}$，目标是找到最大化人格对齐的掩码：

$$\max_{\mathbf{M}^p} \mathbb{E}_{(x,y) \sim \mathcal{D}_p} [\log P_{\mathcal{M}_p}(y|x)]$$

其中 $\|\mathbf{M}^p\|_0 \leq (1 - \rho) d$ 为稀疏约束，$\rho$ 为目标稀疏率。

### 基于激活的重要性打分

对每层 $l$，收集人格校准数据上的激活统计量：

$$\mathbf{A}_p^{(l)}[j] = \mathbb{E}_{(x,y) \sim \mathcal{D}_p} [|\mathbf{h}_j^{(l)}(x)|]$$

结合权重幅度计算重要性分数：

$$S_{ij}^p = |w_{ij}| \cdot \mathbf{A}_p^{(l)}[j]$$

对每个输出通道 $i$，保留 Top-K 个最重要的输入通道，得到二值掩码 $\mathbf{M}^p$。

### 对比剪枝（Contrastive Pruning）

针对对立人格对（如内向/外向），标准剪枝可能产生高度重叠的掩码。对比剪枝通过差异化激活模式最大化参数分离：

**Contrastive-Wanda 变体**：

$$S_{ij}^p = |w_{ij}| \cdot \phi\left(\frac{\mu_{ij}^{p_+} - \mu_{ij}^{p_-}}{\sqrt{\sigma_{ij}^{p_+} + \sigma_{ij}^{p_-}} + \varepsilon}\right)$$

**Contrastive-Sparse 变体**：

$$C_{ij} = |\tilde{S}_{ij}^{p_+} - \tilde{S}_{ij}^{p_-}|, \quad \tilde{S}_{ij}^p = \frac{S_{ij}^p}{\sum_k S_{ik}^p}$$

将每个参数分配给分数更大的人格，构建不相交掩码 $\mathbf{M}^{p_+}, \mathbf{M}^{p_-}$。

### 动态掩码推理

推理时直接应用掩码，无需修改原始权重：

$$\mathbf{y} = (\mathbf{W} \odot \mathbf{M}^p) \mathbf{x} + \mathbf{b}$$

支持可选的软门控 $G = \mathbf{M}^p + \gamma(1 - \mathbf{M}^p)$，$\gamma = 0$ 即标准硬掩码。

## 实验

### 数据集与模型

- **数据集**：MBTI（16 种人格类型）、AI Persona（权力寻求/财富寻求/幻觉检测）、RoleAgentBench（角色扮演）
- **模型**：LLaMA-2-13B, LLaMA-3-8B, Qwen2.5-14B

### 主实验结果

**AI Persona 分类（LLaMA-2-13B）**：

| 方法 | Power-Seeking | Wealth-Seeking | Hallucination |
|------|:---:|:---:|:---:|
| Prompt | 41.0% | 44.0% | 58.5% |
| RAG | 45.5% | 50.5% | 64.5% |
| Wanda | 51.5% | 54.5% | 89.0% |
| Contrastive Wanda | 54.0% | 66.0% | 95.0% |
| Contrastive Sparse | **56.5%** | 64.5% | **96.0%** |
| SFT（上界） | 64.0% | 71.0% | 97.5% |

对比剪枝较 Prompt 方法在 Power-Seeking 上提升 +15.5，Wealth-Seeking 上提升 +20.5。

**RoleAgentBench 角色扮演（LLaMA-3-8B）**：

| 方法 | Friends | Harry Potter | Sherlock | Big Bang | Venice |
|------|:---:|:---:|:---:|:---:|:---:|
| Prompt | 18.37 | 42.06 | 42.11 | 29.55 | 41.67 |
| Sparse | **51.02** | **53.97** | **60.53** | **61.76** | **70.83** |

### 消融实验

**掩码分析**：

| MBTI 维度 | 平均差异率(%) | Attn | MLP |
|-----------|:---:|:---:|:---:|
| I vs. E | 1.34 | 1.28 | 1.44 |
| F vs. T | 1.08 | 1.03 | 1.14 |
| N vs. S | 0.75 | 0.75 | 0.76 |
| J vs. P | 0.76 | 0.73 | 0.79 |

- I/E 和 F/T 维度差异更大 → 切换效果更好
- MLP 层差异一致大于 Attention 层 → 人格分离主要依赖 FFN 变换

**通用能力影响**（LLaMA-3-8B）：

| 方法 | MMLU | HellaSwag |
|------|:---:|:---:|
| Base Model | 0.378 | 0.675 |
| Wanda | 0.369 | 0.668 |
| Sparse | 0.362 | 0.653 |

剪枝后通用能力退化极小（≤1.6%），表明人格子网络仅占模型容量的小部分。

## 亮点

1. **全新视角**：首次从 Lottery Ticket Hypothesis 角度理解 LLM 中的人格表征，证明人格行为是嵌入式而非外部诱导的
2. **训练无关**：无需任何梯度更新，仅需小规模校准数据（几百到几千条样本）
3. **对比剪枝**：专门设计的策略有效增强对立人格间的参数解纠缠
4. **实用高效**：掩码切换仅需分钟级计算，支持快速人格切换

## 局限性

1. N/S 和 J/P 维度的掩码分离度较弱，导致这些人格维度切换效果不稳定
2. 高层（L39）部分人格对的余弦相似度仍然很高（如 INFJ-INFP 达 0.9883），表明深层纠缠难以解开
3. 目前仅在 13B 级别模型上验证，对更大或更小模型的迁移性未知
4. 校准数据的质量和代表性可能影响剪枝效果

## 相关工作

- **人格建模**：提示法（Shao et al., 2023）、RAG（Zerhoudi, 2024）、微调（Zhou et al., 2023）
- **网络剪枝**：Lottery Ticket Hypothesis（Frankle & Carlin, 2019）、Wanda（Sun et al., 2023）、SparseGPT（Frantar & Alistarh, 2023）
- **机制可解释性**：truth direction（Li et al., 2023）、activation steering（Zou et al., 2022）、FFN 键值记忆（Geva et al., 2023）

## 评分

- **新颖性**：⭐⭐⭐⭐ — 将剪枝用于人格发现而非压缩，视角新颖
- **技术贡献**：⭐⭐⭐⭐ — 对比剪枝设计合理，理论直觉清晰
- **实验充分性**：⭐⭐⭐⭐ — 三个数据集、三种模型、详细消融
- **写作质量**：⭐⭐⭐⭐ — 条理清晰，图表丰富
- **综合评分**：8/10

<!-- RELATED:START -->

## 相关论文

- [LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant](../../CVPR2025/information_retrieval/lamra_large_multimodal_model_as_your_advanced_retrieval_assistant.md)
- [Towards Open-Ended Visual Recognition with Large Language Model](../../ECCV2024/information_retrieval/towards_open-ended_visual_recognition_with_large_language_models.md)
- [HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks](hume_measuring_the_human-model_performance_gap_in_text_embedding_tasks.md)
- [SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model](../../ACL2025/information_retrieval/saferag_benchmarking_security_in_retrieval-augmented_generation_of_large_languag.md)
- [MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining](../../NeurIPS2025/information_retrieval/murating_a_high_quality_data_selecting_approach_to_multilingual_large_language_m.md)

<!-- RELATED:END -->
