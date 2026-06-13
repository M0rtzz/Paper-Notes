---
title: >-
  [论文解读] Low-Rank Adaptation in Multilinear Operator Networks for Security-Preserving Incremental Learning
description: >-
  [CVPR 2025][LLM安全][多线性算子网络] 针对全同态加密（Leveled FHE）场景下多线性算子网络的灾难性遗忘问题，提出了一种结合低秩适应（LoRA）和梯度投影记忆（GPM）机制的增量学习方法，在保障数据安全的前提下实现持续学习。 领域现状 领域现状：领域现状：在安全敏感领域（如医疗、金融、国防）…
tags:
  - "CVPR 2025"
  - "LLM安全"
  - "多线性算子网络"
  - "全同态加密"
  - "低秩适应"
  - "灾难性遗忘"
  - "梯度投影记忆"
---

# Low-Rank Adaptation in Multilinear Operator Networks for Security-Preserving Incremental Learning

**会议**: CVPR 2025  
**arXiv**: 无  
**代码**: 无  
**领域**: 安全增量学习  
**关键词**: 多线性算子网络, 全同态加密, 低秩适应, 灾难性遗忘, 梯度投影记忆

## 一句话总结

针对全同态加密（Leveled FHE）场景下多线性算子网络的灾难性遗忘问题，提出了一种结合低秩适应（LoRA）和梯度投影记忆（GPM）机制的增量学习方法，在保障数据安全的前提下实现持续学习。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**：在安全敏感领域（如医疗、金融、国防），数据需要加密以防止未经授权的访问。全同态加密（FHE）允许在加密数据上直接计算，无需解密即可得到正确结果。

**现有痛点**：

### 现有痛点

**现有痛点**：传统网络不兼容 FHE**：ViT 和 CNN 中的非线性操作（如 ReLU、Softmax）在加密域下不可计算。处理加密数据时必须先解密，暴露敏感信息造成安全风险

### 核心矛盾

**核心矛盾**：多项式网络兼容但易遗忘**：Multilinear Operator Networks 只使用多线性运算，对原始数据和加密数据返回相同结果，是目前 SOTA 的 FHE 兼容架构。但这类网络在增量学习中存在严重的灾难性遗忘

### 解决思路

**解决思路**：现有增量学习方法不兼容**：EWC、PackNet 等方法依赖非线性操作，无法在 FHE 框架下使用

**核心矛盾**：安全性要求使用多线性网络，但多线性网络缺乏有效的增量学习机制。现有增量学习方法又不兼容 FHE 的多线性约束。

**本文目标** 设计一种既兼容全同态加密、又能有效缓解灾难性遗忘的增量学习方法。

**切入角度**：将 LoRA 和 GPM 两种技术适配到多线性算子网络，使其在保持 FHE 兼容性的同时具备增量学习能力。

**核心 idea**：通过低秩矩阵分解高效适应新任务，同时利用梯度投影到旧任务子空间的正交补空间来防止遗忘，整个过程保持与 FHE 的兼容性。

## 方法详解

### 整体框架

在 Multilinear Operator Network 的基础上，为每个新任务引入低秩适应模块。训练新任务时，通过梯度投影记忆机制确保梯度更新不会破坏旧任务学到的知识。所有操作保持多线性性质，确保 FHE 兼容。

### 关键设计

1. **多线性算子网络基础架构**:
    - 功能：提供与 Leveled FHE 兼容的神经网络基础
    - 核心思路：网络中所有操作都是多线性的（包括卷积、加法等），不包含非线性激活函数。对原始数据 x 和加密数据 Enc(x) 执行相同运算得到一致结果
    - 设计动机：Leveled FHE 只支持加法和乘法运算，多线性网络自然满足这一约束

2. **低秩适应模块 (Low-Rank Adaptation)**:
    - 功能：为新任务提供参数高效的适应机制
    - 核心思路：在多线性算子的权重矩阵旁添加低秩分解 ΔW = A·B，其中 A ∈ R^(d×r), B ∈ R^(r×d)，r << d。新任务只需训练低秩矩阵，大幅减少可训练参数量
    - 设计动机：全参数微调导致严重遗忘，低秩适应限制了参数更新自由度，天然具有防遗忘效果，且保持多线性运算特性

3. **梯度投影记忆 (Gradient Projection Memory, GPM)**:
    - 功能：训练新任务时保护旧任务知识
    - 核心思路：记录每个旧任务的重要特征子空间（通过 SVD 分解激活值获得），训练新任务时将梯度投影到这些子空间的正交补空间，确保更新不影响旧任务的关键参数方向
    - 设计动机：单纯的低秩适应仍可能在某些方向上破坏旧知识，GPM 提供更强的保护

### 损失函数 / 训练策略

- **分类损失**：标准多类分类损失（兼容多线性运算）
- **梯度投影**：反向传播后将梯度投影到旧任务子空间的正交补空间再更新参数
- **增量式训练**：新任务到来时冻结旧的低秩模块，添加新模块并用 GPM 约束训练

## 实验关键数据

### 主实验

论文在标准增量学习基准上进行评估（pp. 24341-24350，共 10 页）：
- 相比无增量学习策略的基线，多任务场景下准确率有显著提升
- 在保持 FHE 兼容性的前提下，性能逼近甚至超过某些需要非线性操作的增量学习方法
- 在 CIFAR-100 等数据集的增量学习设定下验证了方法的有效性

### 消融实验

- **LoRA vs 全参数微调**：低秩适应在防遗忘方面明显优于全参数微调
- **有/无 GPM**：GPM 对旧任务准确率保持至关重要
- **秩 r 的选择**：存在最优秩值，过大接近全参数微调，过小表达能力不足

### 关键发现

- 多线性算子网络对灾难性遗忘比传统网络更敏感（缺少非线性层的隐式正则化）
- LoRA + GPM 组合对多线性网络特别有效
- 在加密数据上的推理结果与明文数据完全一致，验证了 FHE 兼容性

## 亮点与洞察

1. **问题定义新颖**：首次系统性研究加密场景下的增量学习问题，填补了安全性与可塑性交叉领域的空白
2. **技术适配巧妙**：将 LoRA（源自 LLM 领域）适配到多线性算子网络这一特殊架构
3. **安全性保证**：整个训练和推理过程不需要解密数据，实现真正的安全保护
4. **实用价值高**：对医疗影像分析、金融风控等安全敏感场景有直接应用价值

## 局限与展望

1. **多线性网络表达能力有限**：没有非线性激活函数，在复杂任务上性能可能不足
2. **FHE 计算开销大**：FHE 本身的计算开销是实际部署的瓶颈
3. **任务数量扩展性**：随任务增加，低秩模块和 GPM 子空间信息不断累积
4. **仅限分类任务**：检测、分割等密集预测任务的适配有待探索

## 相关工作与启发

- **Multilinear Operator Networks**：兼容 FHE 的基础架构
- **LoRA**：参数高效微调方法，源自 LLM 领域
- **GPM (Gradient Projection Memory)**：基于梯度方向保护的持续学习方法
- **Leveled FHE**：允许在加密数据上进行有限深度运算的加密方案
- **对后续研究的启发**：安全计算与持续学习的交叉领域值得更多关注

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](../../NeurIPS2025/llm_safety/differentially_private_federated_low_rank_adaptation_beyond_fixed-matrix.md)
- [\[CVPR 2025\] Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning](dual_consolidation_for_pre-trained_model-based_domain-incremental_learning.md)
- [\[NeurIPS 2025\] Demystifying Language Model Forgetting with Low-Rank Example Associations](../../NeurIPS2025/llm_safety/demystifying_language_model_forgetting_with_low-rank_example_associations.md)
- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](../../ICLR2026/llm_safety/heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)
- [\[NeurIPS 2025\] Buffer Layers for Test-Time Adaptation](../../NeurIPS2025/llm_safety/buffer_layers_for_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
