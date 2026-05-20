---
title: >-
  [论文解读] Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring
description: >-
  [ACL 2026][信息检索/RAG][段落检索] 提出 BAGEL，一个基于高斯过程（GP）的贝叶斯主动学习框架，在有限 LLM 预算下通过探索-利用平衡策略传播稀疏 LLM 相关性信号，实现全局嵌入空间的段落检索，显著超越传统 LLM 重排序方法。
tags:
  - "ACL 2026"
  - "信息检索/RAG"
  - "段落检索"
  - "高斯过程"
  - "主动学习"
  - "LLM重排序"
  - "贝叶斯优化"
---

# Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring

**会议**: ACL 2026  
**arXiv**: [2604.17906](https://arxiv.org/abs/2604.17906)  
**代码**: [GitHub](https://github.com/junieberry/BAGEL)  
**领域**: Information Retrieval  
**关键词**: 段落检索, 高斯过程, 主动学习, LLM重排序, 贝叶斯优化

## 一句话总结

提出 BAGEL，一个基于高斯过程（GP）的贝叶斯主动学习框架，在有限 LLM 预算下通过探索-利用平衡策略传播稀疏 LLM 相关性信号，实现全局嵌入空间的段落检索，显著超越传统 LLM 重排序方法。

## 研究背景与动机

**领域现状**: LLM 具有出色的零样本相关性建模能力，但高计算成本使段落检索成为一个预算受限的全局优化问题。主流方法采用 LLM 重排序范式：先用稠密检索器获取 top-K 候选，再用 LLM 重排序。

**现有痛点**: (1) 相关段落往往分布在语义空间的多个不同簇中，稠密检索器仅检索查询嵌入附近的邻域，无法发现远处的相关簇；(2) 现有方法无法将已评分段落的相关性信号传播到未见段落，忽视了嵌入空间的语义结构。

**核心矛盾**: 需要在有限的 LLM 推理预算下探索整个嵌入空间，但传统方法被动依赖一阶段检索器，无法进行全局探索。

**本文目标**: 利用 GP 的核函数相关性传播和不确定性估计能力，主动导航嵌入空间以发现多模态相关性分布。

**切入角度**: 将段落检索建模为贝叶斯优化问题，GP 提供预测均值和不确定性，采集函数平衡探索与利用。

**核心 idea**: GP 天然适合此任务——核函数传播相关性信号，后验方差引导主动学习探索不确定区域。

## 方法详解

### 整体框架

BAGEL 分两阶段：(1) 暖启动初始化——将查询本身作为最高相关性观测，加上 top-M 稠密检索段落的 LLM 评分；(2) 主动学习探索——通过采集函数（UCB）迭代选择下一个段落进行 LLM 评分，更新 GP 后验，最终对所有段落生成排名。

### 关键设计

1. **查询特定高斯过程**:

    - 功能：在嵌入空间上建模查询-段落相关性函数
    - 核心思路：GP 以段落嵌入 $\mathbf{x}_p$ 为输入，LLM 相关性评分为输出，后验预测均值 $\mu_q(\mathbf{x}_{p_*})$ 和方差 $\sigma_q^2(\mathbf{x}_{p_*})$ 分别提供相关性估计和不确定性
    - 设计动机：GP 的核函数（RBF）自然建模嵌入空间中的平滑相关性结构，支持信号传播

2. **UCB 采集函数引导的主动探索**:

    - 功能：平衡探索高不确定性区域和利用高预测相关性区域
    - 核心思路：$a^{\text{UCB}}(\mathbf{x}) = \mu_q(\mathbf{x}) + \sqrt{\beta}\,\sigma_q(\mathbf{x})$，$\beta$ 控制探索-利用权衡，每步选择得分最高的未标记段落
    - 设计动机：纯利用会陷入局部最优，纯探索浪费预算，UCB 自然平衡两者

3. **暖启动初始化策略**:

    - 功能：缓解冷启动问题，提供高质量初始信号
    - 核心思路：将查询嵌入 $\mathbf{x}_q$ 作为最大相关性观测，加上 top-M 稠密检索段落的 LLM 评分，构成初始观测集 $\mathcal{D}_q^{(0)}$
    - 设计动机：查询本身天然是最相关的"段落"，为 GP 提供强正信号和初始锚点

### 损失函数 / 训练策略

无需训练。GP 超参数（核长度尺度 $\ell$、噪声 $\alpha$）通过标准方法设定。支持 Expected Relevance (ER) 和 Peak Relevance (PR) 两种 LLM 评分方式。支持任意时间预测——GP 可在任意迭代后对所有段落生成排名。

## 实验关键数据

### 主实验（LLM 预算 = 50/查询）

| 数据集 | 指标 | BM25 | Dense Retr. | LLM Point. | BAGEL (Qwen3) | BAGEL (GPT-4o) |
|--------|------|------|-------------|-----------|---------------|----------------|
| Covid | N@50 | 42.8 | 48.7 | 52.9 | **61.4** | **62.1** |
| Robust04 | N@50 | 34.9 | 33.2 | 38.2 | **44.4** | **48.7** |
| TravelDest | N@10 | 21.1 | 22.3 | 45.8 | 49.8 | **57.0** |
| NFCorpus | N@50 | 27.7 | 29.0 | 32.7 | 32.8 | **35.9** |

### 消融实验

| 配置 | 发现 |
|------|------|
| RBF vs Linear vs Matérn 核 | RBF 和 Matérn 表现最优，Linear 较差 |
| UCB vs EI vs PI 采集函数 | 不确定性相关的采集函数（UCB）关键 |
| 有/无暖启动 | 暖启动显著提升初期性能 |

### 关键发现

- BAGEL 在所有四个数据集上均超越 LLM 重排序基线（相同 LLM 预算）
- TravelDest 数据集上 NDCG@50 从 29.3 提升至 41.6（+42%）
- 稳态核（RBF、Matérn）有效捕捉多模态相关性结构
- 不确定性引导的探索对发现远离查询的相关簇至关重要

## 亮点与洞察

- 将段落检索优雅地转化为贝叶斯优化问题，GP 天然契合此场景
- 解决了现有方法无法传播相关性信号和探索远处簇的两大限制
- 支持任意时间预测，适应不同预算约束
- 暖启动 + 查询作为最大相关性观测的设计简洁有效

## 局限与展望

- GP 的 $O(n^3)$ 计算复杂度限制了大规模观测集
- 假设嵌入空间中语义相近的段落具有相似相关性，可能不总成立
- 仅在英文检索上评估
- 未来可探索稀疏 GP 或神经核函数以提升可扩展性

## 相关工作与启发

- LLM 重排序（Zhuang et al., 2024; Sun et al., 2023）：主流但受限于一阶段候选集
- 贝叶斯优化/GP：经典方法在新场景（检索）中的创新应用
- 主动学习用于文档标注：但通常用于分类而非排序
- GP 在信息检索中的应用是一个值得深入探索的方向

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ GP + 主动学习用于段落检索，视角独特
- 实验充分度: ⭐⭐⭐⭐ 四个数据集、两个 LLM、核/采集函数消融
- 写作质量: ⭐⭐⭐⭐ 图示直观，GP 与检索的联系阐述清晰
- 价值: ⭐⭐⭐⭐ 在预算受限场景下显著提升检索效果

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)
- [\[ACL 2026\] An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs](an_iterative_utility_judgment_framework_inspired_by_philosophical_relevance_via_.md)
- [\[ICLR 2026\] Fine-tuning with RAG for Improving LLM Learning of New Skills](../../ICLR2026/information_retrieval/fine-tuning_with_rag_for_improving_llm_learning_of_new_skills.md)
- [\[ACL 2026\] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)
- [\[ACL 2026\] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)

</div>

<!-- RELATED:END -->
