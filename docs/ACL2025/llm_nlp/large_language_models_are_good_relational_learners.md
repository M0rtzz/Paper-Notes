---
title: >-
  [论文解读] Large Language Models are Good Relational Learners
description: >-
  [ACL 2025][LLM/NLP][关系深度学习] 提出 Rel-LLM 框架，利用 GNN 编码器从关系数据库中提取结构化子图表示，将其作为软提示注入冻结的 LLM，在 RelBench 基准上实现了关系深度学习（RDL）任务的 SOTA 性能，并支持零样本预测。
tags:
  - ACL 2025
  - LLM/NLP
  - 关系深度学习
  - 图神经网络
  - RAG
  - 图提示微调
  - 关系数据库
---

# Large Language Models are Good Relational Learners

**会议**: ACL 2025  
**arXiv**: [2506.05725](https://arxiv.org/abs/2506.05725)  
**代码**: [GitHub](https://github.com/smiles724/Rel-LLM)  
**领域**: 关系型数据学习 / LLM与结构化数据  
**关键词**: 关系深度学习, 图神经网络, RAG, 图提示微调, 关系数据库

## 一句话总结

提出 Rel-LLM 框架，利用 GNN 编码器从关系数据库中提取结构化子图表示，将其作为软提示注入冻结的 LLM，在 RelBench 基准上实现了关系深度学习（RDL）任务的 SOTA 性能，并支持零样本预测。

## 研究背景与动机

**领域现状**: LLM 在 NLP、CV、信息检索等领域表现出色，但在关系数据库（RDB）的处理和推理上仍有不足。全球约 73% 的数据存储在关系数据库中，表间通过主键-外键相互关联，形成复杂的网络结构。

**现有痛点**: 现有方法将关系数据库"平铺"为文本文档输入 LLM，存在三大问题：(1) 丧失表间关系结构；(2) 嵌套连接导致实体重复冗余；(3) 大型数据库序列化后往往超出 LLM 上下文长度限制。

**核心矛盾**: LLM 擅长文本推理但不擅长处理显式关系结构，GNN 擅长图结构建模但缺乏语义理解与泛化能力。

**本文目标**: 如何让 LLM 有效利用关系数据库中的结构化信息，同时保留表间关系语义。

**切入角度**: 将关系数据库建模为异构图，用 GNN 编码局部子图，通过投影层将图嵌入映射到 LLM 的潜在空间作为软提示。

**核心idea**: 用 GNN 捕获关系结构 + RAG 框架注入 LLM，实现结构感知的关系推理。

## 方法详解

### 整体框架

Rel-LLM 由四部分组成：(1) 时间感知子图采样，保证因果一致性；(2) 异构 GNN 编码器，提取实体的结构特征表示；(3) 投影层 + 反规范化提示构建，将图嵌入组织为 LLM 可处理的结构化提示；(4) 冻结的 LLM 接收图提示与文本嵌入进行联合推理。

### 关键设计

1. **关系实体图 (REG)**: 将关系数据库转化为异构图 $G = (\mathcal{V}, \mathcal{E}, \phi, \psi)$，每行数据为节点，主键-外键关系为边，节点/边类型由表名和关系决定。初始节点嵌入由多模态列编码器生成。
2. **时间感知子图采样**: 以目标实体为中心、以预测时间 $t^*$ 为截止点，仅采样时间戳早于 $t^*$ 的邻居节点，避免时间信息泄露。
3. **异构 GraphSAGE 编码器**: 采用 sum 聚合的异构 GraphSAGE 进行 $L$ 层消息传递，得到节点嵌入 $\mathbf{h}_i^{(L)}$，并通过均值池化得到子图级表示 $\mathbf{h}_g^{(L)}$。
4. **MLP 投影层**: 将图嵌入从 GNN 空间 $\mathbb{R}^{d_g}$ 投影到 LLM 的隐层空间 $\mathbb{R}^{d_l}$，实现模态对齐。
5. **反规范化提示构建**: 以目标实体为根，沿主键-外键链接递归展开（广度优先，深度 $\zeta$，每层最多 $n_{\text{nest}}$ 个实体），将关联实体的图嵌入组织为嵌套 JSON 结构，减少多跳推理需求。
6. **三种答案生成策略**: (1) 纯文本生成——直接输出可读文本；(2) Token 分布——输出概率分布用于概率性任务；(3) MLP 变换——用轻量网络将 LLM 隐表示投影到任务空间。不同任务适合不同策略。

### 预训练目标

- **Masked Table Modeling**: 随机选择一部分节点进行遮蔽，用可学习的 mask token 替换原始特征，然后让 LLM 重建被遮蔽实体的属性（列名-值对）。对列顺序进行随机排列以增强鲁棒性。
- 预训练损失为标准的自回归 NLL：$\mathcal{L}_{\text{pretrain}} = -\frac{1}{|\mathcal{V}_{\text{mask}}|} \sum_{v_i} \sum_t \log p_\theta(y_i^{(t)} | y_i^{(<t)}, \hat{\mathbf{h}}_{\text{mask}})$
- 仅优化 GNN 编码器 $\phi_1$、投影层 $\phi_2$ 和 mask token，LLM 参数 $\theta$ 冻结。

## 实验与关键数据

### 实验设置

- **基准**: RelBench——包含 7 个数据集、30 个预测任务（实体分类 + 实体回归）
- **骨干 LLM**: Llama 3.2-1B（128K 上下文）
- **对比基线**: LightGBM、RDL（GNN+深度表格模型）、ICL（LLM 上下文学习）、ICL+MLP

### 主实验结果

**实体分类 (AUROC ↑)**:
| 数据集 | LightGBM | RDL | ICL+MLP | Rel-Zero | Rel-LLM |
|--------|----------|-----|---------|----------|---------|
| rel-amazon user-churn (Test) | 52.22 | 70.42 | 66.56 | 60.07 | **71.89** |
| rel-event user-repeat (Test) | 68.04 | 76.89 | 76.72 | 68.12 | **79.26** |
| rel-stack user-engagement (Test) | 63.39 | 90.59 | 87.09 | 69.46 | **91.21** |
| 全部平均 (Test) | 63.66 | 75.83 | 76.83 | 63.42 | **77.82** |

- Rel-LLM 在所有数据集上均优于或接近 SOTA，平均 AUROC 达到 77.82
- 零样本 Rel-Zero 性能虽低于微调版本，但显著优于 LightGBM 基线

**实体回归 (MAE ↓)**:
- 在 rel-hm item-sales 等任务上，Rel-LLM 均取得最低 MAE
- 相比 ICL+MLP，Rel-LLM 在多数任务上有 5-15% 的改进

### 关键发现

1. GNN 编码器有效保留了关系结构信息，避免了文本序列化带来的信息丢失
2. 图提示微调方式无需改变 LLM 参数，训练成本远低于全量微调
3. 预训练阶段的 masked table modeling 使模型具备零样本迁移能力
4. 不同任务适合不同的答案生成策略，分类任务适合 token 分布，回归任务适合 MLP 变换

## 亮点与洞察

1. **结构保留的 RAG**: 与传统 RAG 不同，Rel-LLM 不是检索文本片段而是检索图结构子图，保留了关系语义
2. **高效微调**: 冻结 LLM，仅训练 GNN 和投影层，参数效率极高
3. **反规范化为嵌套 JSON**: 巧妙地将图结构"翻译"为 LLM 可理解的格式，JSON 格式被证明对表格数据编码效果最好
4. **时间一致性**: 通过时间感知采样严格避免信息泄露，适用于时序预测场景
5. **零样本能力**: 预训练后无需微调即可在新任务上获得合理预测

## 局限性

1. 依赖 Llama 3.2-1B 这样的小模型，在更大模型上效果是否更好尚未验证
2. 反规范化深度和嵌套数量是超参数，需根据数据库结构手动调整
3. 对于缺乏清晰主键-外键关系的非结构化数据无法直接应用
4. 仅在 RelBench 上验证，缺乏更多实际应用场景的测试

## 相关工作

- **关系表格学习**: CTU、SJTUTable、RelBench 等基准推动了关系数据的深度学习研究
- **LLM 处理表格数据**: 现有方法将表格序列化为文本，但面临上下文长度和结构信息丢失的挑战
- **图提示学习**: 将 GNN 嵌入作为 LLM 的软提示，是近年来图-语言多模态的新趋势

## 评分

⭐⭐⭐⭐ — 方法设计巧妙、实验全面，在关系数据库+LLM 这一重要但被忽视的方向上做出了有意义的探索。GNN+RAG+冻结LLM 的组合具有很好的可扩展性。

## 补充细节

- 预训练中随机排列列顺序是一个有效的数据增强手段，防止模型只学会特定列顺序下的重建
- JSON 格式优于 Markdown 和 CSV 进行关系数据编码的早期实验发现来自 Singha et al., 2023
- 实验中 ICL+MLP 在部分任务上接近 Rel-LLM，说明 LLM 的表示能力即使通过简单文本序列化也可被部分激发

<!-- RELATED:START -->

## 相关论文

- [AfroBench: How Good are Large Language Models on African Languages?](afrobench_how_good_are_large_language_models_on_african_languages.md)
- [What Makes a Good Natural Language Prompt?](good_natural_language_prompt.md)
- [Mixtures of In-Context Learners](mixtures_of_in-context_learners.md)
- [Open-Set Living Need Prediction with Large Language Models](open-set_living_need_prediction_with_large_language_models.md)
- [Investigating Context-Faithfulness in Large Language Models: The Roles of Memory Strength and Evidence Style](investigating_context-faithfulness_in_large_language_models_the_roles_of_memory_.md)

<!-- RELATED:END -->
