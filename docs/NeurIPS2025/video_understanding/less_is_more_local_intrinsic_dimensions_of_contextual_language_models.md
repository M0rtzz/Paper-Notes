---
title: >-
  [论文解读] Less is More: Local Intrinsic Dimensions of Contextual Language Models
description: >-
  [NeurIPS 2025][视频理解][intrinsic dimension] 提出利用上下文 token 嵌入的局部内在维度（Local Intrinsic Dimension, LID）来无监督监测 LLM 训练动态——维度下降预示泛化改善，维度上升预示过拟合——在对话状态跟踪、grokking、情感识别等任务上验证了这一几何信号的实用性。
tags:
  - NeurIPS 2025
  - 视频理解
  - intrinsic dimension
  - LLM
  - fine-tuning
  - grokking
  - overfitting detection
  - embedding geometry
---

# Less is More: Local Intrinsic Dimensions of Contextual Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2506.01034](https://arxiv.org/abs/2506.01034)  
**代码**: [GitHub](https://github.com/aidos-lab/Topo_LLM_public)  
**领域**: video_understanding  
**关键词**: intrinsic dimension, LLM, fine-tuning, grokking, overfitting detection, embedding geometry

## 一句话总结

提出利用上下文 token 嵌入的局部内在维度（Local Intrinsic Dimension, LID）来无监督监测 LLM 训练动态——维度下降预示泛化改善，维度上升预示过拟合——在对话状态跟踪、grokking、情感识别等任务上验证了这一几何信号的实用性。

## 研究背景与动机

- **理解 LLM 内部机制仍然很难**：即便是"微调如何影响模型行为"这样的基础问题，通常也需要大量的经验评估。
- **缺乏无监督训练诊断工具**：大多数性能诊断依赖有标签验证集或任务特定探针，在低资源场景下不可用。
- **现有维度研究的局限**：Tulchinskii 等人发现 AI 生成文本的全局内在维度较低，但仅作用于单段文本；Aghajanyan 等人定义了基于参数空间的内在维度而非嵌入空间；Valeriani 等人研究了数据通过 LLM 后全局维度的变化，但未进行局部化分析。
- **全局维度不够细腻**：嵌入空间不是均匀维度的单一流形，而是由多个局部维度不同的区域组成（Union of Manifolds），需要局部估计。

## 方法详解

### 潜在空间建模

给定文本语料 $\mathcal{D} = (s_0, \ldots, s_D)$ 和模型 $\mathcal{M}$（深度 $l$ 层），每条序列 $s_m$ 经 tokenizer $\mathcal{T}$ 后在第 $i$ 层产生上下文嵌入：

$$\mathcal{M}_i(s_m) = (\mathcal{M}_i(t_0^m), \ldots, \mathcal{M}_i(t_{n_m}^m))$$

全部 token 嵌入构成点云 $\mathbb{T}_i = \{\mathcal{M}_i(t_j^m)\}_{m, j}$，在欧氏空间中度量距离。

### 两步采样策略

实际中 $\mathbb{T}_i$ 可达数百万向量，直接计算邻域不可行。采用：
1. 从 $\mathcal{D}$ 中采样 $M$ 条序列
2. 去重后再采样 $N$ 个 token 向量
3. 为每个 token 计算 $L$-近邻 $\mathcal{N}_L(t_j; \mathbb{T})$

### 局部 TwoNN 维度估计

利用 TwoNN 估计器，基于每个点到最近和次近邻的距离比 $r_2/r_1$（弱假设下服从 Pareto 分布）估计局部维度：

$$\text{LID}(v) = \text{TwoNN}(\mathcal{N}_L(v; \mathbb{T}))$$

对所有采样 token 得到维度向量 $\in \mathbb{R}_{\geq 0}^N$，聚合为均值 LID 作为整体几何签名。

### 跨模型比较

基础模型 $\mathcal{M}$ 和微调模型 $\mathcal{M}^\Delta$ 共享相同架构和 tokenizer，因此嵌入空间之间存在自然的逐点对应关系，可直接比较维度变化。

## 实验

### 实验1：微调引发数据集特异性维度偏移

**设置**：RoBERTa-base 在 MultiWOZ 对话数据上做 MLM 微调（5 epoch），分别在 MultiWOZ、Wikipedia、Reddit 上测量 LID。

**结果**：
- MultiWOZ（微调数据）：维度显著下降（标准化均值差 1.19）
- Wikipedia/Reddit（非微调数据）：维度几乎不变（标准化均值差 0.08/0.10）

**核心发现**：LID 下降具有**数据集特异性**——仅在微调数据分布上发生，不影响无关数据区域。

### 实验2：LID 检测 Grokking

**设置**：在加法 mod $p=197$ 任务上训练 2 层 decoder-only Transformer，训练数据比例从 10% 到 50%。

| 训练数据比例 | 是否 Grokking | 训练 LID 变化趋势 |
|:---:|:---:|:---|
| 10% | 否 | 上升后持平 |
| 15% | 否 | 上升后持平 |
| ≥20% | 是 | 上升后**显著下降** |

**关键发现**：训练集上 LID 的显著下降与验证准确率开始上升的时间点吻合——仅从训练数据即可预测 grokking 是否发生（无需验证标签）。

### 实验3：LID 检测训练能力耗竭

**设置**：TripPy-R 对话状态跟踪模型（RoBERTa 编码器）在 MultiWOZ 上训练 20 epoch。

**结果**：
- 训练集上均值 LID 与 JGA（Joint Goal Accuracy）的 Spearman 相关系数：**−0.982**
- 验证损失在 7500 step 已最小化，但 JGA 仍在提升，此时 LID 仍在下降——说明验证损失给出了错误的"训练已收敛"信号
- LID 在约 25000 step 后稳定，与 JGA 收敛同步

**关键发现**：LID 是比验证损失更可靠的训练收敛指标。

### 实验4：LID 检测过拟合

**设置**：BERT-base + 线性分类器在 EmoWOZ 情感分类上训练 8 epoch。

**结果**：
- 第 1 epoch 后 LID 从 ~9.94 骤降至 ~7.25（模型找到高效表征）
- 此后 LID 逐渐回升至 ~8（维度上升暗示记忆化）
- 验证损失在第 1 epoch 后**持续上升**——明确过拟合信号
- LID 与训练损失的 Spearman 相关：−0.952；与验证损失：+0.952

**关键发现**：LID 先降后升的模式对应"找到高效表征→过拟合"的过程，可作为无监督早停信号。

## 亮点

- **统一框架覆盖多种训练动态**：同一个 LID 指标能检测微调效果、grokking、训练收敛和过拟合四种不同现象
- **无需标签的诊断信号**：完全基于训练数据的嵌入几何，不依赖有标签验证集
- **简洁有效的启发式**：LID 下降→泛化改善；LID 上升→记忆化/过拟合，直觉清晰
- **实验设计精心**：涵盖编码器（RoBERTa/BERT）和解码器（GPT-2/tiny Transformer）、序列标注和分类等多种设定

## 局限性

- **计算成本较高**：需要大量前向传播构建嵌入 + $O(dN^2)$ 的近邻搜索，实时监控受限
- **TwoNN 假设强**：要求局部密度近似常数且来自泊松过程，对 Transformer 嵌入的适用性仅有经验验证
- **绝对值不可跨架构比较**：LID 的绝对值依赖超参数（$M$, $N$, $L$），仅有相对变化可比较
- **因果关系未建立**：LID 下降与泛化改善之间是相关而非因果，理论解释仍缺乏
- **仅在较小模型上验证**：实验集中在 RoBERTa-base、GPT-2-medium 等，对 7B+ 模型的可扩展性未知

## 相关工作

- **LLM 内在维度**（Aghajanyan+ 2021）：研究参数空间维度而非嵌入空间，发现大模型参数维度更低
- **全局嵌入维度**（Valeriani+ 2023、Tulchinskii+ 2023）：分析 AI 文本 vs 人类文本的全局维度差异，但不做局部化
- **token 级维度**（Viswanathan+ 2025）：分析单条 prompt 内的 token 维度，本文从整个数据集子采样
- **拓扑深度学习**（Papamarkou+ 2024）：几何/拓扑方法观察性地分析 ML 模型，本文是该方向在训练动态诊断上的新应用
- **LoRA 维度自适应**（Ed-dib+ 2024）：基于隐状态信息矩阵秩调整 LoRA 秩，本文的 LID 可互补

## 评分

- 新颖性: ⭐⭐⭐⭐ — 局部内在维度作为训练动态的无监督诊断信号是新颖的视角
- 实验充分度: ⭐⭐⭐⭐⭐ — 四个独立实验覆盖微调/grokking/收敛/过拟合，多种模型和任务
- 写作质量: ⭐⭐⭐⭐⭐ — 理论-实验对应清晰，每个实验都有明确的研究问题
- 价值: ⭐⭐⭐⭐ — 为 LLM 训练监控提供了有价值的几何工具，对低资源场景尤其有意义

## 实验关键数据

## 亮点

## 局限与展望

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)
- [\[NeurIPS 2025\] Steering When Necessary: Flexible Steering Large Language Models with Backtracking](steering_when_necessary_flexible_steering_large_language_models_with_backtrackin.md)
- [\[NeurIPS 2025\] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models](sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[CVPR 2025\] Video Summarization with Large Language Models](../../CVPR2025/video_understanding/video_summarization_with_large_language_models.md)

<!-- RELATED:END -->
