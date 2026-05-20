---
title: >-
  [论文解读] BugSweeper: Function-Level Detection of Smart Contract Vulnerabilities Using Graph Neural Networks
description: >-
  [AAAI2026][图学习][Smart Contract] 提出 BugSweeper，通过构建函数级抽象语法图 (FLAG) 并设计两阶段 GNN 架构，实现无需专家规则的端到端智能合约漏洞检测，在重入攻击检测上 F1 达 98.57%。
tags:
  - "AAAI2026"
  - "图学习"
  - "Smart Contract"
  - "Vulnerability Detection"
  - "图神经网络"
  - "Abstract Syntax Tree"
  - "Pooling"
---

# BugSweeper: Function-Level Detection of Smart Contract Vulnerabilities Using Graph Neural Networks

**会议**: AAAI2026  
**arXiv**: [2512.09385](https://arxiv.org/abs/2512.09385)  
**代码**: 待确认  
**领域**: 图学习  
**关键词**: Smart Contract, Vulnerability Detection, graph neural network, Abstract Syntax Tree, Pooling

## 一句话总结

提出 BugSweeper，通过构建函数级抽象语法图 (FLAG) 并设计两阶段 GNN 架构，实现无需专家规则的端到端智能合约漏洞检测，在重入攻击检测上 F1 达 98.57%。

## 背景与动机

以太坊智能合约一旦部署就无法修改，若存在安全漏洞（如著名的 DAO 攻击导致 360 万 ETH 被盗），后果极其严重。
现有检测方法分为两大类：

1. **传统方法**（静态分析、符号执行）：依赖专家手工制定的规则，如 Slither、Mythril 等工具，面对新型漏洞变体时泛化能力差。
2. **深度学习方法**（TMP、AME、Peculiar、ReVulDL）：虽然引入了 GNN 或预训练模型，但预处理阶段仍依赖基于规则的代码片段提取，存在以下问题：
    - **检测范围受限**：预定义规则无法覆盖所有漏洞类型
    - **泛化能力差**：针对特定漏洞设计的规则难以迁移
    - **信息丢失**：规则提取可能丢弃关键上下文

作者观察到，智能合约中的安全漏洞（尤其是重入攻击）往往源于函数间的不安全交互，因此提出在**函数级别**进行分析，同时保留函数间的调用与引用关系。

## 核心问题

如何在不依赖任何手工规则的情况下，构建能同时捕获代码结构和语义流的表示，实现多类型智能合约漏洞的端到端自动检测？

## 方法详解

BugSweeper 包含三个核心组件：

### 1. Graph Constructor — 构建 FLAG

- **AST 解析**：使用 Solidity 编译器 solc 将源码解析为抽象语法树 (AST)
- **边增强**：在 AST 基础上添加三类边，形成 Flow-Augmented ASG：
    - **Basic edges**：AST 原有的 Child/Parent 结构边
    - **Data-flow edges**：包括 ReferencedDeclaration（变量/函数引用）、FunctionReturnParameter（返回值链接）、SuperFunction（函数重写）、Assignment（赋值）
    - **Control-flow edges**：包括 IfStatement 的 CondTrue/CondFalse、循环的 WhileExecution/ForExecution、NextStatement 顺序边
- **函数级切分 + Coverage 扩展**：将合约按函数拆分为子图，通过 coverage 超参数控制邻域深度：
    - coverage=1：仅目标函数自身
    - coverage=2：加入直接调用的函数和引用的变量（1-hop）
    - coverage=3：进一步扩展至 2-hop 邻居
    - 实验中默认使用 coverage=4

### 2. Code Graph Neural Network (CGNN) — 第一阶段

- 使用 BPE tokenizer 将节点文本属性编码为向量
- 采用 3 层 GraphSAGE（512→1024→1024→1024）进行消息传递，生成节点 embedding
- **CGPool（核心创新）**：基于代码语法角色的确定性语义池化
    - 将属于同一 FunctionDefinition 或 VariableDeclaration 的节点合并为一个 supernode
    - 按原始控制流/数据流边重连 supernode，生成 Pooled FLAG
    - 相比 TopKPool/SAGPool 等方法，既保留层次结构又避免信息丢失，且计算高效

### 3. Second-Stage GNN — 第二阶段

- 在 Pooled FLAG 上使用 3 层 GAT（含 multi-head attention，首层 4 heads）进行高层推理
- GAT 的注意力机制能自动聚焦于关键的函数间连接
- 经 global readout 后接三层 MLP 分类器（1024→1024→C），输出各漏洞类别的概率

### 训练细节

- 优化器：Adam，lr=1e-4，weight decay=1e-5
- 训练 500 epochs，batch size=64
- Dropout：GNN 层 0.5，分类器 0.3
- 每组实验用 4 个不同随机种子重复，报告均值

## 实验关键数据

### 重入攻击检测（AME 数据集，1224 合约）

| 方法 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| Slither | 94.74% | 34.62% | 50.70% |
| AME | 95.45% | 95.38% | 95.42% |
| ReVulDL | 92.95% | 94.62% | 93.74% |
| **BugSweeper** | **99.87%** | **97.35%** | **98.57%** |

BugSweeper F1 超过最强基线 AME 约 **3.1 个百分点**，同时 Precision 接近 100%。

### 多类漏洞检测（SmartBugs Wild 数据集，47,398 Solidity 文件）

最优配置 SAGE + GAT：

| 漏洞类型 | F1 |
|---------|-----|
| Reentrancy | 91.61% |
| Unchecked Low-Level Calls | 80.15% |
| Time Manipulation | 79.63% |

### 消融实验亮点

- **两阶段 vs 单阶段**：两阶段架构在所有漏洞类型上显著优于单阶段基线
- **CGPool vs 其他池化**：CGPool（F1=87.32%）大幅领先 ASAPool（82.41%）、SAGPool（77.10%）、TopKPool（75.29%）
- **GNN 组合**：SAGE（第一阶段）+ GAT（第二阶段）效果最佳，体现互补性——SAGE 擅长大图聚合，GAT 注意力机制精准定位关键特征

## 亮点

1. **完全端到端**：从源码到漏洞检测不依赖任何手工规则，突破了现有深度学习方法仍需规则预处理的瓶颈
2. **FLAG 表示**：将 AST 与控制流/数据流语义融合，并通过 coverage 机制灵活控制函数间信息量
3. **CGPool 语义池化**：巧妙利用代码结构先验（函数/变量定义边界）进行确定性池化，兼具效率与信息保留
4. **两阶段 GNN 设计合理**：第一阶段降噪 + 第二阶段高层推理，SAGE 和 GAT 各发挥所长
5. **多漏洞泛化**：同一框架统一检测三类漏洞，无需为每种漏洞单独设计规则

## 局限与展望

1. **漏洞类型有限**：仅验证了 3 种漏洞，未覆盖更隐蔽的逻辑漏洞（如访问控制、整数溢出等）
2. **数据不均衡影响明显**：Unchecked Low-Level Calls 和 Time Manipulation 的 F1 明显低于 Reentrancy，说明对少样本类型泛化仍有不足
3. **SmartBugs Wild 标签质量**：依赖 3+ 工具共识作为 ground truth，工具本身可能存在系统性偏差
4. **可扩展性未验证**：未在超大规模合约或复杂 DeFi 协议上测试
5. **缺少与 LLM 方法的对比**：近期基于代码大模型（CodeBERT、Code LLaMA）的漏洞检测方法未纳入比较
6. **coverage 参数敏感**：固定为 4，未深入分析不同 coverage 的精度-效率权衡

## 与相关工作的对比

| 维度 | 传统工具 | 现有 DL 方法 | BugSweeper |
|------|---------|-------------|------------|
| 规则依赖 | 完全依赖 | 预处理依赖 | **无** |
| 漏洞类型 | 按规则覆盖 | 多为单类 | **多类统一** |
| 代码表示 | CFG/PDG | 专家抽取图 | **FLAG（自动构建）** |
| 信息保留 | 丢失多 | 部分丢失 | **语义池化保留** |
| F1（重入） | 50-68% | 85-95% | **98.57%** |

## 启发与关联

- **CGPool 的思想可迁移**：基于代码结构先验的确定性池化策略，可推广到其他代码分析任务（如缺陷预测、代码克隆检测）
- **FLAG 构建流程通用**：AST + flow edges + 函数级切分的表示方式不限于 Solidity，可适配 Python/Java 等语言
- **两阶段 GNN 架构启发**：先降噪再推理的范式适用于其他噪声较大的图学习场景（如分子图、电路图）
- **与 LLM 结合的方向**：可考虑用代码大模型替代 BPE tokenizer 生成初始节点 embedding，可能进一步提升性能

## 评分

- 新颖性: ⭐⭐⭐⭐ — FLAG 表示和 CGPool 设计新颖，两阶段架构有合理动机
- 实验充分度: ⭐⭐⭐⭐ — 消融全面，多数据集验证，统计显著性测试完备；但漏洞类型偏少
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富，动机阐述充分
- 价值: ⭐⭐⭐⭐ — 端到端免规则检测方向有实际意义，精度提升显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](adaptive_riemannian_graph_neural_networks.md)
- [\[AAAI 2026\] Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily](beyond_fixed_depth_adaptive_graph_neural_networks_for_node_classification_under_.md)
- [\[ICLR 2026\] Are We Measuring Oversmoothing in Graph Neural Networks Correctly?](../../ICLR2026/graph_learning/are_we_measuring_oversmoothing_in_graph_neural_networks_correctly.md)
- [\[CVPR 2026\] Adaptive Learned Image Compression with Graph Neural Networks](../../CVPR2026/graph_learning/adaptive_learned_image_compression_with_graph_neural_networks.md)
- [\[ICML 2026\] Quantile-Free Uncertainty Quantification in Graph Neural Networks](../../ICML2026/graph_learning/quantile-free_uncertainty_quantification_in_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
