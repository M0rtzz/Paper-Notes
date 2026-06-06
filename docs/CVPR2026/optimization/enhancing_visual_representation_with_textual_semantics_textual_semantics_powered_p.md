---
title: >-
  [论文解读] Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning
description: >-
  [CVPR 2026][优化/理论][联邦学习] 针对联邦原型学习中现有方法破坏类间语义关系的问题，提出FedTSP方法利用预训练语言模型构建保留语义结构的文本原型，在异构联邦学习中显著提升性能并加速收敛。
tags:
  - "CVPR 2026"
  - "优化/理论"
  - "联邦学习"
  - "原型学习"
  - "语义关系"
  - "预训练语言模型"
  - "数据异质性"
---

# Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning

**会议**: CVPR 2026  
**arXiv**: [2503.13543](https://arxiv.org/abs/2503.13543)  
**代码**: [GitHub](https://github.com/XinghaoWu/FedTSP)  
**领域**: 优化  
**关键词**: 联邦学习, 原型学习, 语义关系, 预训练语言模型, 数据异质性

## 一句话总结

针对联邦原型学习中现有方法破坏类间语义关系的问题，提出FedTSP方法利用预训练语言模型构建保留语义结构的文本原型，在异构联邦学习中显著提升性能并加速收敛。

## 研究背景与动机

联邦原型学习（FedPL）是处理联邦学习中数据异质性的有效策略，核心思想是让客户端协同构建全局原型，并让本地特征与之对齐。现有方法（如AlignFed、FedTGP）通常追求最大化原型间的类间距离以增强判别性，但这种做法存在一个被忽视的问题：在增大类间距离的同时，不可避免地破坏了类之间的语义关系。

例如，"马"和"狗"属于语义相近的动物类别，它们的原型距离应当小于"马"和"卡车"之间的距离。但均匀分布在超球面上的原型无法保留这种层次化的语义结构。作者通过Spearman相关系数和语义间隔（semantic gap）两个定量指标验证了这一发现。

直接从有限且异质的客户端数据中学习语义关系是困难的。然而，预训练语言模型（PLM）如BERT在大规模文本语料上已经捕获了丰富的语义关系。这启发了本文的核心idea：能否将文本语义知识注入联邦学习的原型中，使其在异质数据下也能保留类间关系？

## 方法详解

### 整体框架

输入：客户端图像数据 → LLM生成类别描述 → PLM编码为文本原型 → 可训练prompt对齐模态 → 客户端本地特征与文本原型对齐 → 输出：各客户端个性化模型。

### 关键设计

1. **LLM生成多视角文本描述**:
    - 功能：为每个类别生成丰富的语义描述
    - 核心思路：使用ChatGPT等LLM为每个类别生成k=3个不同方面的细粒度描述，模板为"A photo of {CLASS}: {description}"
    - 设计动机：手工提示（如"A photo of a {CLASS}"）只有类名差异，语义上下文极其有限，且存在歧义（如"apple"可能指水果或公司）

2. **可训练Prompt进行模态对齐**:
    - 功能：弥合PLM文本特征与客户端图像特征之间的模态鸿沟
    - 核心思路：在文本嵌入序列中引入可训练的embedding向量，替换前m个位置，通过InfoNCE损失与客户端图像原型对齐
    - 设计动机：PLM（如BERT）在预训练时未接触过图像数据，直接使用会导致模态不匹配

3. **基于对比学习的特征对齐**:
    - 功能：将文本原型的语义结构传递给客户端模型
    - 核心思路：采用对比学习损失（而非L2距离）来对齐本地特征与文本原型，关注类间的相对相似度排序而非绝对距离
    - 设计动机：PLM生成的原型基线相似度较高（即使最不相似的类也有0.73的相似度），L2对齐会误导模型将不相关类视为相似

### 损失函数 / 训练策略

- 服务器端：InfoNCE损失更新可训练prompt，对齐文本原型与聚合的图像原型
- 客户端：交叉熵分类损失 + 对比学习对齐损失（带温度参数τ控制相对相似度的敏感度）
- 隐私保护扩展：通过差分隐私（DP）机制对文本嵌入添加高斯噪声，满足(ε,δ)-DP保证

## 实验关键数据

### 主实验

| 数据集 | 指标 | FedTSP-BERT | 之前SOTA | 提升 |
|--------|------|-------------|----------|------|
| CIFAR-10 (α=0.1) | Acc | 87.52% | 86.80% (FedKD) | +0.72% |
| CIFAR-100 (α=0.1) | Acc | 46.08% | 42.82% (FedMRL) | +3.26% |
| TinyImageNet (α=0.1) | Acc | 34.82% (CLIP) | 32.79% (FedKD) | +2.03% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| 对比学习 vs L2对齐 | +2-3% | 对比学习更适合处理高基线相似度 |
| LLM描述 vs 手工模板 | +1-2% | 细粒度描述提供更丰富的语义上下文 |
| CLIP vs BERT | 接近 | BERT虽无图像预训练，但通过可训练prompt可弥合 |

### 关键发现

- FedTSP在强异质性（α=0.1）下提升更显著，说明文本原型对异质数据更鲁棒
- FedTSP-BERT在Top-5准确率上提升更大，说明语义关系有效：即使分类错误，也倾向于放在语义相近的类中
- 隐私保护版本在ε≥1时性能几乎不受影响

## 亮点与洞察

- 首次将PLM/LLM的语义知识引入联邦原型学习，视角新颖
- 发现并量化了现有方法破坏语义关系的问题
- FedTSP兼容CLIP和BERT等不同PLM，且不依赖CLIP的视觉-语言对齐
- 可同时处理数据异质性和模型异质性

## 局限与展望

- 服务器需要部署PLM，增加了服务器端的计算成本
- LLM生成描述的质量依赖于类别名称的明确性
- 未探索更大规模数据集（如ImageNet）和更多样的PLM架构
- 隐私保护扩展仅考虑了类名隐私，未覆盖更广泛的隐私场景

## 相关工作与启发

- **vs FedProto/FedTGP**: 这些方法从客户端数据聚合原型或最大化类间距离，破坏了语义关系；FedTSP从文本模态构建原型，天然保留语义结构
- **vs CLIP-based FL**: CLIP-based方法旨在增强CLIP本身，FedTSP则将语义知识转移给轻量级客户端模型，不依赖CLIP
- **vs FedETF/FedNH**: 使用固定的ETF/均匀分布分类器作为原型，无法编码语义关系

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将PLM语义知识引入联邦原型学习，视角独特
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多异质性设置、多PLM、消融实验完整
- 写作质量: ⭐⭐⭐⭐ 动机清晰，可视化直观，语义对齐和间隔指标设计精巧
- 价值: ⭐⭐⭐⭐ 为联邦学习提供了利用语言模型语义知识的新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Learning Context-Conditioned Predicate Semantics via Prototype Feedback](../../ICML2026/optimization/learning_context-conditioned_predicate_semantics_via_prototype_feedback.md)
- [\[AAAI 2026\] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](../../AAAI2026/optimization/smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)
- [\[ICLR 2026\] DeepAFL: Deep Analytic Federated Learning](../../ICLR2026/optimization/deepafl_deep_analytic_federated_learning.md)
- [\[ICML 2025\] FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging](../../ICML2025/optimization/fedswa_improving_generalization_in_federated_learning_with_highly_heterogeneous_.md)
- [\[CVPR 2026\] The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers](the_power_of_decaying_steps_enhancing_attack_stability_and_transferability_for_s.md)

</div>

<!-- RELATED:END -->
