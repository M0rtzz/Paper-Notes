---
title: >-
  [论文解读] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation
description: >-
  [ICLR 2026][人体理解][联邦微调] 提出Fed-PLoRA框架，用多个并行一秩模块(PLoRA)替代多秩LoRA，通过Select-N-Fold策略（选N个训练+折叠其余到冻结权重）实现异构联邦微调的零初始化噪声和最小聚合噪声，在6个LLM/多任务上全面超越现有方法。
tags:
  - ICLR 2026
  - 人体理解
  - 联邦微调
  - LoRA
  - 异构秩
  - 初始化噪声
  - 聚合噪声
---

# Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation

**会议**: ICLR 2026  
**arXiv**: [2602.16936](https://arxiv.org/abs/2602.16936)  
**代码**: [GitHub](https://github.com/TNI-playground/Fed-PLoRA)  
**领域**: 联邦学习/高效微调  
**关键词**: 联邦微调, LoRA, 异构秩, 初始化噪声, 聚合噪声

## 一句话总结
提出Fed-PLoRA框架，用多个并行一秩模块(PLoRA)替代多秩LoRA，通过Select-N-Fold策略（选N个训练+折叠其余到冻结权重）实现异构联邦微调的零初始化噪声和最小聚合噪声，在6个LLM/多任务上全面超越现有方法。

## 研究背景与动机

**领域现状**：联邦微调(FFT)用LoRA跨分布式客户端协作微调LLM，保持数据隐私。但客户端资源异构→不同LoRA秩→初始化和聚合出现维度不匹配问题。

**现有痛点**：(1) FLoRA：每轮随机重新初始化LoRA→巨大初始化噪声；(2) HETLoRA：截断全局LoRA→丢失低秩以外的信息+聚合偏差；(3) FlexLoRA：SVD重构→引入分解误差。所有方法在初始化噪声和聚合噪声之间存在不可调和的矛盾。

**核心矛盾**：全局模型秩R > 客户端秩 $r_i$ → 客户端无法完整继承全局信息（初始化噪声），同时分别训练后的聚合也不完美（聚合噪声）。

**切入角度**：将多秩LoRA分解为多个并行一秩模块→每个模块独立→客户端选择子集训练+折叠其余到冻结权重→零初始化噪声。

## 方法详解

### 整体框架
PLoRA: $\Delta W = \sum_{j=1}^{R} B_{(j)}A_{(j)}$，等价于标准LoRA但模块独立。Select-N-Fold: 客户端 $i$ 选 $r_i$ 个模块训练，剩余折叠到预训练权重冻结。聚合：按秩维度独立平均。

### 关键设计

1. **PLoRA (Parallel One-Rank Adaptation)**:

    - 功能：将秩-R的LoRA分解为R个并行的秩-1模块
    - 核心思路：$\Delta W_{\text{PLoRA}} = \sum_{j=1}^R B_{(j)}A_{(j)} = \sum_{j=1}^R B_{[:,j]}A_{[j,:]} = BA = \Delta W_{\text{LoRA}}$
    - 设计动机：数学等价但模块独立→自然支持子集选择

2. **Select-N-Fold策略**:

    - 功能：客户端随机选 $r_i$ 个PLoRA模块训练，其余折叠到冻结权重
    - 核心思路：$\mathcal{W}_i^t = \mathcal{W}^0 + \sum_{j \notin \mathcal{K}_i^t} B_{(j)}^{t-1}A_{(j)}^{t-1}$，训练在 $\mathcal{W}_i^t$ 上进行
    - 设计动机：折叠保留了未训练模块的信息→零初始化噪声。随机选择确保所有模块在期望下被更新。

3. **噪声分析**:

    - 初始化噪声：$\mathcal{N}_{\text{Init}}^t = 0$（完美保留全局信息）
    - 聚合噪声上界：$\leq \sum_{j=1}^R \frac{1}{|\mathcal{Q}_{(j)}^t|}\sum_i \|B_{i,(j)}^t - \bar{B}_{(j)}^t\|_2 + \|A_{i,(j)}^t - \bar{A}_{(j)}^t\|_2$
    - 余弦相似度分析证明模块间在训练后趋于一致→上界逐渐收紧

### 损失函数 / 训练策略
- 标准联邦微调流程（广播→本地训练→聚合）
- 每轮10%客户端参与
- SGD/AdamW本地优化

## 实验关键数据

### 主实验 (Llama-1B, Natural Instructions)
| 方法 | IID准确率 | non-IID准确率 | 初始化噪声 |
|------|----------|-------------|----------|
| FedIT (同构) | 66.88 | 61.28 | 0 |
| FLoRA | 中 | 中 | 高(随机重初始化) |
| FlexLoRA | 中 | 中 | 中(截断+SVD误差) |
| HETLoRA | 中 | 中 | 中(截断) |
| **Fed-PLoRA** | **最高** | **最高** | **0** |

### 多模型/多任务验证
| 模型 | 任务 | Fed-PLoRA vs 最佳baseline |
|------|------|-------------------------|
| BERT-base | GLUE | 超越 |
| Llama-3.1-8B | 金融NLP | 超越 |
| Qwen3-4B | 指令跟随 | 超越 |
| Mistral-7B | 医学QA | 超越 |

### 关键发现
- 余弦相似度热力图显示：训练后同一秩位的PLoRA模块跨客户端趋于一致（对角线高），不同秩位间保持独立（非对角线低）→每个秩学到了不同的知识但客户端间收敛
- Fed-PLoRA在non-IID设置下优势更大→零初始化噪声对异构数据更关键
- 通信/计算/内存开销与现有方法可比→没有额外代价

## 亮点与洞察
- **零初始化噪声**：通过折叠而非截断/重初始化，完美保留全局信息。这个设计简洁但解决了异构FFT的根本问题。
- **PLoRA的模块独立性**：虽然数学上等价于标准LoRA，但模块独立性使得子集选择+独立聚合自然成立。这是一个architectural trick带来的系统性改进。
- **统一噪声分析框架**：为FLoRA/FlexLoRA/HETLoRA/Fed-PLoRA提供了统一的初始化噪声和聚合噪声分析，清晰展示了各方法的优劣。

## 局限与展望
- 随机选择模块可能不是最优——基于重要性/梯度的选择策略可能更有效
- 折叠操作在每轮增加 $O(dk(R-r_i))$ 计算——虽然比训练小得多但非零
- 下行通信多了 $O((d+k)(R-r_i))$ 比HETLoRA/FlexLoRA
- 仅测试了LoRA应用到self-attention层，应用到FFN层效果未知

## 相关工作与启发
- **vs FLoRA**: FLoRA零聚合噪声但大初始化噪声，Fed-PLoRA零初始化噪声+小聚合噪声→综合更优
- **vs HETLoRA**: HETLoRA截断高秩部分→丢信息，Fed-PLoRA折叠→保留信息
- **vs 标准LoRA/FedIT**: Fed-PLoRA在同构设置下等价于FedIT，在异构设置下优于所有方法

## 评分
- 新颖性: ⭐⭐⭐⭐ PLoRA分解+Select-N-Fold策略设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 6个模型、多领域任务、IID/non-IID、多baseline
- 写作质量: ⭐⭐⭐⭐ 噪声分析框架清晰，对比公平
- 价值: ⭐⭐⭐⭐ 对異构联邦微调有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [Safety Subspaces are Not Linearly Distinct: A Fine-Tuning Case Study](safety_subspaces_are_not_linearly_distinct_a_fine-tuning_case_study.md)
- [Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection](../../AAAI2026/human_understanding/fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)
- [Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence](antibody_strengthening_defense_against_harmful_fine-tuning_for_large_language_mo.md)
- [FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems](../../ICML2025/human_understanding/fedrag_a_framework_for_fine-tuning_retrieval-augmented_generation_systems.md)
- [Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning](../../AAAI2026/human_understanding/small_language_models_for_efficient_agentic_tool_calling_outperforming_large_mod.md)

<!-- RELATED:END -->
