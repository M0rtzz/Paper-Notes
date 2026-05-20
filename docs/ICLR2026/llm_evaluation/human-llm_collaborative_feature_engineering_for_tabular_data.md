---
title: >-
  [论文解读] Human-LLM Collaborative Feature Engineering for Tabular Learning
description: >-
  [ICLR 2026][LLM评测][特征工程] 提出一个人-LLM协作特征工程框架，将LLM的特征操作提议与选择过程解耦，通过贝叶斯神经网络建模操作效用和不确定性来指导选择，并选择性地引入人类偏好反馈，在18个表格数据集上平均错误率降低8.96%~11.23%。
tags:
  - "ICLR 2026"
  - "LLM评测"
  - "特征工程"
  - "人机协作"
  - "贝叶斯优化"
  - "LLM"
  - "表格数据"
---

# Human-LLM Collaborative Feature Engineering for Tabular Learning

**会议**: ICLR 2026  
**arXiv**: [2601.21060](https://arxiv.org/abs/2601.21060)  
**代码**: 无  
**领域**: AutoML / 表格学习  
**关键词**: 特征工程, 人机协作, 贝叶斯优化, LLM, 表格数据

## 一句话总结

提出一个人-LLM协作特征工程框架，将LLM的特征操作提议与选择过程解耦，通过贝叶斯神经网络建模操作效用和不确定性来指导选择，并选择性地引入人类偏好反馈，在18个表格数据集上平均错误率降低8.96%~11.23%。

## 研究背景与动机

**领域现状**：LLM在表格学习中被广泛用于自动化特征工程，通过语义理解生成有意义的特征变换操作（如CAAFE、OCTree）。

**现有痛点**：现有方法将LLM同时用作特征操作的提议者和选择者，完全依赖LLM内部启发式，缺乏对操作效用和不确定性的校准估计，导致反复探索低收益操作，在有限迭代预算下表现不佳。

**核心矛盾**：LLM擅长生成多样化的特征变换候选，但不擅长在候选中做出最优选择——提议能力强但选择能力弱的矛盾。

**本文目标** 如何将LLM的操作提议与选择解耦，并在选择过程中有效融合人类专家知识，以提升特征工程效率。

**切入角度**：借鉴贝叶斯优化思想，用显式代理模型替代LLM的隐式选择，并设计选择性人类反馈机制控制专家参与成本。

**核心 idea**：LLM只负责提议候选特征操作，选择由贝叶斯神经网络的UCB策略引导，在不确定性高时选择性引入人类偏好反馈。

## 方法详解

### 整体框架

每轮特征工程中：(1) LLM基于任务描述、特征语义和历史性能生成$N$个候选特征变换操作；(2) 贝叶斯神经网络（BNN）代理模型估计每个操作的期望效用$\mu_t(e)$和不确定性$\sigma_t^2(e)$；(3) 使用UCB策略选择操作，当满足条件时选择性查询人类偏好反馈；(4) 评估选中操作的实际效用并更新代理模型。

### 关键设计

1. **特征操作编码与BNN代理模型**:
    - 做什么：将LLM生成的自然语言特征操作映射为向量表示，并用贝叶斯神经网络估计效用
    - 核心思路：操作嵌入由语义嵌入$\phi_{\text{embedding}}(e)$（text-embedding-3-small）和列使用编码$\phi_{\text{column}}(e) \in \{0,1\}^d$拼接而成。BNN通过变分推断学习参数后验$q_t(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta}; \boldsymbol{M}_t, \boldsymbol{\Sigma}_t)$，提供预测均值$\mu_t(e)$和方差$\sigma_t^2(e)$
    - 设计动机：GP在高维语言派生特征空间中扩展性差，BNN更适合建模非平稳性；列使用编码解决了多列语义描述相似时的歧义问题

2. **选择性人类偏好反馈机制**:
    - 做什么：在UCB选出最优候选$e_t^a$后，决定是否向人类专家查询偏好反馈
    - 核心思路：需同时满足两个条件才触发查询——(C1) 置信区间重叠：$\text{UCB}_t(e_t^b) > \text{LCB}_t(e_t^a)$，确保存在不确定性空间；(C2) 不确定性足够大：$\sqrt{\beta_t}(\sigma_t(e_t^a) + \sigma_t(e_t^b)) \geq \gamma_\kappa$，确保潜在收益大于查询成本
    - 设计动机：无差别查询会产生不必要的认知负担，仅在反馈能带来显著效用增益时才值得请求人类介入

3. **基于偏好反馈的后验更新**:
    - 做什么：将人类偏好反馈$Z_t$融入代理模型的后验分布
    - 核心思路：偏好反馈通过probit似然建模$\mathcal{P}(Z_t | \boldsymbol{\theta}, e_t^a, e_t^b) = \Phi(\eta Z_t [\hat{g}(\phi(e_t^a); \boldsymbol{\theta}) - \hat{g}(\phi(e_t^b); \boldsymbol{\theta})])$，更新变分后验$q_t'(\boldsymbol{\theta})$后用新UCB值做最终选择
    - 设计动机：概率化处理人类反馈比直接采信更鲁棒，能平滑噪声反馈

### 损失函数 / 训练策略

BNN通过最小化ELBO训练：$\text{KL}(q_t(\boldsymbol{\theta}) \| \mathcal{P}(\boldsymbol{\theta})) - \mathbb{E}_{q_t(\boldsymbol{\theta})}[\log \mathcal{P}(H_t | \boldsymbol{\theta})]$。UCB选择系数$\beta_t = 2\log(|\mathcal{S}_t|\pi^2 t^2 / 3\delta)$，$\delta=0.1$。人类查询成本$\gamma_\kappa=4$。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文(w/o human) | 本文(w/ human) | 最佳基线 | 提升(w/ human) |
|--------|------|------|------|------|------|
| 13分类数据集(MLP) | AUROC(%) | 85.3 | 85.5 | 84.7(OCTree) | 错误率降低8.96% |
| 13分类数据集(XGBoost) | AUROC(%) | 87.4 | 87.4 | 86.7(OCTree) | 错误率降低11.23% |
| flight(MLP) | AUROC(%) | 96.9 | 97.3 | 94.8(OCTree) | +48.1%错误率降低 |
| conversion(XGBoost) | AUROC(%) | 93.5 | 93.9 | 92.4(OCTree) | +11.5%错误率降低 |

### 消融实验

| 配置 | 指标 | 说明 |
|------|------|------|
| 不同LLM骨干(GPT-5) | MLP平均85.9→86.5 | GPT-5骨干下Ours(w/ human)最优 |
| 不同LLM骨干(GPT-3.5) | MLP平均84.6→85.1 | 弱骨干也能保持优势 |
| 用户研究(ALG vs Control) | 性能: p=0.011 | ALG框架显著提升用户性能 |
| 用户研究(ALG vs Control) | 完成时间: p<0.001 | ALG框架显著减少完成时间 |

### 关键发现

- LLM-based方法整体优于传统AutoML（OpenFE、AutoGluon），验证了语义理解对特征工程的价值
- 显式建模效用和不确定性比纯依赖LLM启发式分别提升7.24%和9.02%的错误率降低
- 人类偏好反馈一致性地带来额外提升，且计算开销（BNN+UCB）仅占总时间的2.2%

## 亮点与洞察

- 将贝叶斯优化的思想引入LLM驱动的特征工程，解耦提议与选择是一个优雅的工程设计。UCB平衡探索/利用的理论保证让选择过程不再是黑箱。
- 选择性查询机制的两个条件（置信区间重叠+不确定性门控）有坚实的理论支撑（Lemma 3.1-3.2），实现了人类认知成本和信息增益的最优权衡。

## 局限与展望

- 人类反馈在实验中由GPT-4o模拟，实际用户研究仅在单个数据集上进行，泛化性有限
- BNN代理模型在特征工程早期轮次数据稀疏时校准质量可能不佳，冷启动问题未充分讨论
- 框架仅考虑单个操作的效用，未建模多操作组合的交互效应

## 相关工作与启发

- **vs CAAFE**: CAAFE让LLM同时提议和选择特征操作，易陷入局部最优；本文解耦后能持续发现高价值操作
- **vs OCTree**: OCTree用决策树反馈指导LLM，但仍依赖LLM的内部启发式选择；本文用BNN提供更校准的效用估计
- **vs 传统贝叶斯优化**: 传统BO用GP做代理模型，在低维空间有效；本文用BNN处理高维语言嵌入空间

## 评分

- 新颖性: ⭐⭐⭐⭐ 解耦提议与选择并引入人类反馈的框架设计新颖，理论分析完整
- 实验充分度: ⭐⭐⭐⭐ 18个数据集+用户研究+计算可扩展性分析，多角度验证
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，理论推导严谨，实验展示全面
- 价值: ⭐⭐⭐ 实际应用场景明确但需要LLM API成本，方法通用性较好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] TabStruct: Measuring Structural Fidelity of Tabular Data](tabstruct_measuring_structural_fidelity_of_tabular_data.md)
- [\[NeurIPS 2025\] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](../../NeurIPS2025/llm_evaluation/climb_class-imbalanced_learning_benchmark_on_tabular_data.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](../../ACL2025/llm_evaluation/elaboration_competitive_programming.md)
- [\[ACL 2026\] HoWToBench: Holistic Evaluation for LLM's Capability in Human-level Writing using Tree of Writing](../../ACL2026/llm_evaluation/howtobench_holistic_evaluation_for_llms_capability_in_human-level_writing_using_.md)
- [\[CVPR 2025\] ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems](../../CVPR2025/llm_evaluation/comfybench_benchmarking_llm-based_agents_in_comfyui_for_autonomously_designing_c.md)

</div>

<!-- RELATED:END -->
