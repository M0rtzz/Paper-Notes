---
title: >-
  [论文解读] CLaC at SemEval-2025 Task 6: A Multi-Architecture Approach for Corporate Environmental Promise Verification
description: >-
  [ACL 2025 (SemEval Workshop)][ESG报告] 本文针对SemEval-2025 Task 6（PromiseEval）的企业ESG报告承诺验证任务，探索了三种递进的模型架构：ESG-BERT基线、语言特征增强版、以及融合注意力池化和多目标学习的联合子任务模型，最终以0.5268的私榜分数略超基线（0.5227），验证了语言特征工程和多任务学习在ESG承诺验证中的有效性。
tags:
  - ACL 2025 (SemEval Workshop)
  - ESG报告
  - 承诺验证
  - 多任务学习
  - 注意力池化
  - 语言特征工程
---

# CLaC at SemEval-2025 Task 6: A Multi-Architecture Approach for Corporate Environmental Promise Verification

**会议**: ACL 2025 (SemEval Workshop)  
**arXiv**: [2505.23538](https://arxiv.org/abs/2505.23538)  
**代码**: https://github.com/CLaC-Lab/SemEval-2025-Task6  
**领域**: Others / ESG Text Analysis  
**关键词**: ESG报告, 承诺验证, 多任务学习, 注意力池化, 语言特征工程

## 一句话总结

本文针对SemEval-2025 Task 6（PromiseEval）的企业ESG报告承诺验证任务，探索了三种递进的模型架构：ESG-BERT基线、语言特征增强版、以及融合注意力池化和多目标学习的联合子任务模型，最终以0.5268的私榜分数略超基线（0.5227），验证了语言特征工程和多任务学习在ESG承诺验证中的有效性。

## 研究背景与动机

企业ESG（环境、社会与治理）报告中的承诺验证是一个日益重要但极具挑战的任务：

**任务背景**：企业发布的ESG报告中包含大量关于环保、社会责任等方面的承诺，这些承诺深刻影响利益相关者的信任和企业声誉。然而，承诺的复杂性和庞大数量使得人工验证极其困难。

**任务分解**：PromiseEval 将承诺验证分解为四个子任务：
   - 承诺识别（是否包含承诺）
   - 支撑证据评估（是否有具体证据）
   - 清晰度评价（清晰/不清晰/误导）
   - 验证时间分类（2年内/2-5年/5年以上/其他）

**核心挑战**：
   - **数据极少**：英语部分仅约400个训练实例
   - **类别不平衡**：四个子任务的类别分布严重不均
   - **领域特异性**：ESG报告的语言模式（如承诺用语、回避措辞）需要专门处理

**研究动机**：在数据稀缺和类别不平衡条件下，单纯依赖预训练模型不足，需要结合领域知识的特征工程和跨任务知识共享的多任务学习来提升性能。

## 方法详解

### 整体框架

三个递进的模型架构，每个都在前一个基础上增加复杂度：
- **Model 1**（基线）：ESG-BERT + 四个独立分类头
- **Model 2**（特征增强）：ESG-BERT + 针对各子任务设计的语言特征标签
- **Model 3**（联合模型）：DeBERTa-v3-large + 注意力池化 + 多目标学习（仅子任务1和2）

### 关键设计

1. **ESG-BERT基线模型（Model 1）**:

    - 使用预训练的ESG-BERT模型，利用其ESG领域专有知识
    - 四个子任务各训练一个独立模型
    - **冻结策略**：仅微调最后2个transformer层和分类头，其余参数冻结。在仅400个训练样本的情况下，这能有效防止过拟合
    - 使用4折分层交叉验证+Optuna进行超参搜索

2. **语言特征增强（Model 2）**:

    - **核心思路**：将任务特定的语言特征以标签形式前置（prepend）到输入文本前，帮助模型快速捕捉关键模式
    - **子任务1（承诺识别）**：检测承诺相关词（"commit"、"pledge"、"goal"等词干匹配）+ 情感极性分析（承诺通常正面表述）
        - 例：`POSITIVE Sentiment. Contains Promise Word. We commit to...`
    - **子任务2（证据检测）**：统计量化指标词（"percentage"、"dollars"）和证据支撑词（"document"）的数量 + NER检测数字和日期
        - 例：`Proof_Count_2. Has_Numbers. Our carbon emissions decreased by 15%...`
    - **子任务3（清晰度评估）**：分别统计模糊用语和明确用语的出现次数
        - 例：`Vague_Terms_2. Specific_Terms_0. We might consider...`
    - **子任务4（时间分类）**：提取日期+识别时间线指示词

3. **联合子任务模型（Model 3）**:

    - **基础模型**：DeBERTa-v3-large（比ESG-BERT更强大）
    - **注意力池化**：替代标准的[CLS] token表示，动态加权所有token的表示
        - $\alpha_i = \text{softmax}(W_{\text{attn}} h_i)$，$r = \sum_{i=1}^{n} \alpha_i h_i$
        - 使模型能聚焦于文档中语义相关的关键token
    - **双任务分类头**：子任务1（承诺）和子任务2（证据）共享encoder，各自有独立的分类通路（Linear → Dropout → LayerNorm → GELU → Linear）
    - **上下文富化**：在文本前加入页码和文档类型标记
        - $x_{\text{enriched}} = \text{"[PAGE } p \text{] [ESG REPORT] "} + x_{\text{raw}}$
    - **多目标加权**：$\mathcal{L} = 0.6 \cdot \mathcal{L}_{\text{promise}} + 0.4 \cdot \mathcal{L}_{\text{evidence}}$，承诺检测权重更高因为它更基础
    - **测试时增强（TTA）**：3次前向传播使用随机词dropout(10%)和元数据变化，集成平均后用校准阈值分类

### 损失函数 / 训练策略

- Model 1&2：标准交叉熵损失，4折分层交叉验证，Optuna超参优化
- Model 3：多目标加权损失（0.6:0.4），余弦学习率调度+10%预热，梯度累积（16步），序列长度256
- 最终提交：子任务1和2使用Model 3预测，子任务3和4使用Model 2预测

## 实验关键数据

### 主实验

| 模型 | 公榜分数 | 私榜分数 | 说明 |
|------|---------|---------|------|
| Kaggle基线 | 0.5523 | 0.5227 | 官方baseline |
| Model 1（Base）| 0.5082 | 0.4994 | ESG-BERT基线 |
| Model 2（Feature-Enhanced）| 0.5137 | 0.5094 | +语言特征 |
| **Model 3（Combined）** | 0.5255 | **0.5268** | +注意力池化+多任务 |

### 消融实验 — 各组件贡献分析

| 改进 | 私榜提升 | 说明 |
|------|---------|------|
| Base → Feature-Enhanced | +0.010 | 语言特征有限帮助 |
| Feature-Enhanced → Combined | +0.017 | 多任务+注意力池化贡献更大 |
| Combined vs Kaggle Baseline | +0.004 | 略超基线 |

### 关键发现

- **语言特征增强的效果有限**（仅+0.010）：可能因为ESG-BERT在预训练中已隐式学习了许多这些模式，造成冗余
- **前置标签方法的局限**：特征标签与相关文本span之间可能存在结构性断裂
- **多任务学习有效但提升温和**：共享表示、注意力池化和测试时增强三个因素共同贡献
- **数据规模是根本瓶颈**：仅400个训练实例严重限制了复杂模型（如Model 3）的学习潜力
- **可能存在负迁移**：承诺和证据任务在某些实例上需要关注矛盾的特征

## 亮点与洞察

- **递进式实验设计值得借鉴**：从简单基线到特征增强再到联合模型，每步都有明确的改进动机和对比，清楚展示了各组件的边际贡献
- **对ESG承诺语言的特征工程有实用价值**：承诺词列表、模糊用语检测、时间指示词等语言资源可直接复用
- **诚实报告了有限的提升**：仅超基线0.004，但通过分析解释了原因（数据少、ESG-BERT冗余、负迁移），这种诚实的分析比吹嘘偏小的改进更有学术价值
- **代码开源**

## 局限与展望

- **数据量是最大制约**：400个训练样本对于任何深度学习方法都极其有限。数据增强、少样本学习或利用外部ESG报告预训练可能有帮助
- **Model 3仅覆盖子任务1和2**：子任务3和4未能纳入联合模型，四个子任务的全联合建模可能更好地捕捉任务间依赖
- **缺乏系统性的消融研究**：未量化各组件（注意力池化、元数据、TTA、多任务学习）的独立贡献
- **仅使用英语数据**：原任务是多语言的，跨语言迁移学习或多语言transformer可能带来额外提升
- **未探索更新更强的PLM**：如GPT系列的in-context learning或更大的开源模型
- **对类别不平衡的处理可以更深入**：仅使用分层采样和focal loss，可以尝试过采样、对比学习等方法

## 相关工作与启发

- ESG-BERT (Mukherjee & Pothireddi, 2021) 提供了ESG领域的预训练基础
- DeBERTa-v3-large (He et al., 2021) 的disentangled attention在NLU任务上表现优异
- ML-Promise数据集 (Seki et al., 2024) 是首个多语言企业承诺验证数据集
- 对ESG文本分析社区的启示：**在小数据场景下，一定要语言特征工程和预训练模型并重**，同时要注意预训练模型已内化的知识可能与手工特征冗余

## 评分

- 新颖性: ⭐⭐ 方法组合较为常规，无显著新技术提出
- 实验充分度: ⭐⭐⭐ 三个模型递进对比清晰，但缺乏组件级消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，每个模型的设计动机和实现细节完整
- 价值: ⭐⭐⭐ 作为SemEval系统描述论文，提供了实用的ESG文本分析方案和经验教训

<!-- RELATED:START -->

## 相关论文

- [MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)
- [Optimizing Decomposition for Optimal Claim Verification](optimizing_decomposition_for_optimal_claim_verification.md)
- [AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)
- [Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation](bone_soups_multi_objective_gen.md)
- [Statistical Deficiency for Task Inclusion Estimation](statistical_deficiency_task_inclusion.md)

<!-- RELATED:END -->
