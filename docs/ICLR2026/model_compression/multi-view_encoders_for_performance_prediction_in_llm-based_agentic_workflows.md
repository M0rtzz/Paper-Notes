---
title: >-
  [论文解读] Multi-View Encoders for Performance Prediction in LLM-Based Agentic Workflows
description: >-
  [ICLR 2026][模型压缩][性能预测] 提出 Agentic Predictor，一种多视图工作流编码框架，通过联合建模图结构、代码语义和提示信息来预测 LLM Agent 工作流的性能，显著减少昂贵的试错评估。
tags:
  - ICLR 2026
  - 模型压缩
  - 性能预测
  - 多视图编码
  - Agent工作流
  - 图神经网络
  - 无监督预训练
---

# Multi-View Encoders for Performance Prediction in LLM-Based Agentic Workflows

**会议**: ICLR 2026  
**arXiv**: [2505.19764](https://arxiv.org/abs/2505.19764)  
**代码**: [GitHub](https://github.com/deepauto-ai/agentic-predictor)  
**领域**: 模型压缩  
**关键词**: 性能预测, 多视图编码, Agent工作流, 图神经网络, 无监督预训练

## 一句话总结

提出 Agentic Predictor，一种多视图工作流编码框架，通过联合建模图结构、代码语义和提示信息来预测 LLM Agent 工作流的性能，显著减少昂贵的试错评估。

## 研究背景与动机

LLM Agent 系统近年发展迅速，但优化其工作流配置面临巨大搜索空间挑战。现有的自动化设计方法（如 ADAS、AFlow）依赖大量 LLM API 调用进行评估，计算代价极高。本文提出用**性能预测器**替代完整执行评估，类似于神经架构搜索（NAS）中的预测器方法。

核心挑战有两个：

**工作流异构性**：不同工作流在通信结构、提示策略、工具调用模式上差异巨大，难以用统一模型建模

**标注数据稀缺**：通过完整执行获取性能标签代价高昂，监督学习数据不足

## 方法详解

### 整体框架

Agentic Predictor 包含三个阶段：(a) 多视图工作流编码器将 Agent 工作流编码为统一表示；(b) 跨域无监督预训练阶段学习通用表示；(c) 预测器引导搜索阶段用少量标注数据训练预测器。

### 关键设计

1. **多视图工作流编码 (Multi-View Encoding)**：

    - **图视图 $\mathcal{G}$**：将工作流建模为 DAG，通过 GNN 编码 Agent 间的通信依赖
    - **代码视图 $\mathcal{C}$**：用 MLP 编码工作流完整代码，捕获逻辑结构和工具使用模式
    - **提示视图 $\mathcal{P}$**：用 MLP 编码系统提示中的角色描述和行为规范
    - 三个视图通过聚合层融合：$\mathbf{Z} = \text{MLP}([\mathbf{Z}_\mathcal{G}, \mathbf{Z}_\mathcal{C}, \mathbf{Z}_\mathcal{P}])$

2. **多图注意力机制 (Cross-Graph Attention)**：

    - 构建三种图：提示图 $\mathcal{G}_\text{prompt}$、代码图 $\mathcal{G}_\text{code}$、算子图 $\mathcal{G}_\text{operator}$
    - 通过跨视图自注意力进行节点级信息交互
    - ViewAttnPool 自适应学习各视图的重要性权重

3. **跨域无监督预训练 (Cross-Domain Unsupervised Pretraining)**：

    - 重建损失：$\mathcal{L}_{rec} = \frac{1}{M}\sum_{i=1}^{M}\|\mathcal{G}_i - \hat{\mathcal{G}}_i\|^2 + \|\mathcal{C}_i - \hat{\mathcal{C}}_i\|^2 + \|\mathcal{P}_i - \hat{\mathcal{P}}_i\|^2$
    - 跨模态对比损失：在 $(\mathcal{G}, \mathcal{C})$、$(\mathcal{G}, \mathcal{P})$、$(\mathcal{C}, \mathcal{P})$ 三对视图间施加 InfoNCE 损失
    - 预训练不使用任何性能标签，避免标签泄漏

### 损失函数 / 训练策略

- 预训练阶段：$\mathcal{L}_{enc} = \mathcal{L}_{rec} + \mathcal{L}_{con}$（重建 + 对比）
- 预测器阶段：二分类用交叉熵损失，回归用 MSE 损失
- 文本编码器用 all-MiniLM-L6-v2（384维），代码编码器用 CodeRankEmbed（768维），统一映射到 512 维空间

## 实验关键数据

### 主实验

| 领域 | 指标 | Agentic Predictor | 最强基线 | 提升 |
|------|------|-------------------|----------|------|
| Code Generation (GD) | Accuracy | 85.33% | 85.24% (Graph Trans.) | +0.09% |
| Code Generation (AF) | Accuracy | 85.62% | 84.71% (Graph Trans.) | +0.91% |
| Math (GD) | Accuracy | 66.20% | 64.84% (GAT) | +1.36% |
| Math (AF) | Accuracy | 79.56% | 76.44% (GAT) | +3.12% |
| Average | Accuracy | **79.97%** | 78.36% (GAT) | +2.05% |
| Average | Utility | **76.33%** | 73.54% (Dir-GNN) | +3.79% |

### 消融实验

| 配置 | 平均 Accuracy | 平均 Utility | 说明 |
|------|-------------|-------------|------|
| Code + Graph + Text (Full) | **84.38%** | **81.88%** | 完整模型 |
| 去除 Code | 降低 | 降低 | 代码视图对逻辑理解重要 |
| 去除 Graph | 降低 | 降低 | 图结构对交互建模重要 |
| 去除 Text | 降低 | 降低 | 提示语义不可或缺 |

### 关键发现

- 三视图编码互补性强，任何单一视图的移除都导致性能下降
- 跨域无监督预训练在标签稀少时尤其有效（预训练版 Agentic Predictor+ 在少标签场景提升更大）
- 方法对搜索算法无关（search-agnostic），可与任意搜索策略组合

## 亮点与洞察

- 将 NAS 中的性能预测思想迁移到 Agent 工作流优化领域，方向新颖
- 多视图编码充分利用了 Agent 工作流的异构信息（结构、代码、语义）
- 跨域预训练策略有效缓解标签稀缺问题，体现了自监督学习在新领域的潜力

## 局限与展望

- 仅在 FLORA-Bench 上验证，数据集覆盖面有限
- 预测器对工作流变化的泛化能力需进一步验证
- 未探索更大规模 Agent 系统和更复杂的多模态工作流
- 代码和提示编码器使用的是固定的预训练模型，未进行端到端微调

## 相关工作与启发

- 与 FLORA-Bench 对比：本文引入多视图编码和无监督预训练，而非单一图视图
- 与 MAS-GPT 对比：本文使用轻量预测器而非 LLM 微调来生成工作流
- NAS 预测器（如 CAP、FlowerFormer）的思路可直接迁移到 Agent 领域

## 评分

- 新颖性: ⭐⭐⭐⭐ 将性能预测引入 Agent 工作流设计是新方向
- 实验充分度: ⭐⭐⭐ 仅一个 benchmark，但消融详细
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，框架描述完整
- 价值: ⭐⭐⭐⭐ 对减少 Agent 系统开发成本有实际意义
