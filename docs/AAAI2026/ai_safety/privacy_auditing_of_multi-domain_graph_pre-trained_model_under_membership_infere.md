---
title: >-
  [论文解读] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack
description: >-
  [AI安全] 提出 MGP-MIA 框架，首次针对多域图预训练模型开展成员推理攻击（MIA），通过机器遗忘放大成员信号、增量学习构建影子模型、基于相似度的推理机制，有效揭示多域图预训练的隐私泄漏风险。
tags:
  - AI安全
---

# Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack

- **会议**: AAAI 2026
- **arXiv**: [2511.17989](https://arxiv.org/abs/2511.17989)
- **代码**: [RingBDStack/MGP-MIA](https://github.com/RingBDStack/MGP-MIA)
- **领域**: ai_safety
- **关键词**: 成员推理攻击, 多域图预训练, 隐私审计, 机器遗忘, 图神经网络

## 一句话总结

提出 MGP-MIA 框架，首次针对多域图预训练模型开展成员推理攻击（MIA），通过机器遗忘放大成员信号、增量学习构建影子模型、基于相似度的推理机制，有效揭示多域图预训练的隐私泄漏风险。

## 背景与动机

多域图预训练（Multi-domain Graph Pre-training）是构建图基础模型的关键技术，通过在多个领域的图数据上进行自监督预训练（如链接预测、对比学习），使 GNN 获得跨域可迁移的结构与语义表示。当开发者将预训练模型公开发布以支持下游任务时，攻击者可利用模型推断训练数据中是否包含特定样本，造成严重的隐私泄漏。

然而，对多域图预训练模型执行 MIA 面临三大挑战：

1. **泛化能力增强**：多域预训练减少了过拟合，而过拟合正是传统 MIA 所依赖的核心信号
2. **影子数据集不具代表性**：训练数据覆盖多个领域，攻击者难以获取与所有训练域对齐的影子图
3. **成员信号减弱**：预训练编码器输出的是嵌入向量而非 logits，包含的过拟合信号更弱

作者通过 PCA 可视化和扰动稳定性实验验证了两个关键观察：(1) 成员与非成员嵌入的可分性很弱；(2) 成员嵌入在扰动下并不比非成员更稳定。这说明现有图 MIA 方法无法直接适用于多域图预训练场景。

## 方法详解

MGP-MIA 框架包含三个核心模块：成员信号放大机制、增量影子模型构建机制和基于相似度的推理机制。

### 1. 成员信号放大机制（Membership Signal Amplification）

该模块利用机器遗忘（Machine Unlearning）来增强模型对剩余数据的过拟合程度，从而放大成员信号。

具体流程：从影子图 $\mathcal{G}_{\text{Shadow}}$ 中随机抽取子图 $\mathcal{G}_{\text{Unlearn}}$ 作为遗忘目标。先将目标模型 $\mathcal{F}_{\text{Target}}$ 在 $\mathcal{G}_{\text{Unlearn}}$ 上微调几个 epoch 得到增强模型 $\mathcal{F}_{\text{Augment}}$。然后比较两个模型下每个节点与正/负样本的相似度差异，计算教师相似度分数：

$$\mathbf{s}_{\text{Teacher}}^{i} = \mathbf{s}_{\text{Target}}^{i} - \lambda \cdot (\mathbf{s}_{\text{Target}}^{i} - \mathbf{s}_{\text{Augment}}^{i})$$

其中 $\lambda$ 控制遗忘强度。相似度向量定义为节点 $v_i$ 与其 $P$ 个正样本和 $N$ 个负样本的余弦相似度拼接：

$$\mathbf{s}^{i} = [\text{sim}(\mathbf{h}_i, \mathbf{h}_{i_1^+}), \ldots, \text{sim}(\mathbf{h}_i, \mathbf{h}_{i_P^+}), \text{sim}(\mathbf{h}_i, \mathbf{h}_{i_1^-}), \ldots, \text{sim}(\mathbf{h}_i, \mathbf{h}_{i_N^-})]$$

最终通过最小化遗忘模型输出与教师分数的偏差来完成遗忘：

$$\min_{\mathcal{F}_{\text{Unlearn}}} \sum_{n_i \in \mathcal{V}_{\text{Unlearn}}} \|\mathbf{s}_{\text{Unlearn}}^{i} - \mathbf{s}_{\text{Teacher}}^{i}\|^2$$

**核心思想**：不精确的机器遗忘会释放模型容量，使模型对剩余数据产生更强的记忆（过拟合），从而放大成员与非成员之间的行为差异。

### 2. 增量影子模型构建机制（Incremental Shadow Model Construction）

攻击者通常只有一个与目标节点同域的影子图，无法覆盖目标模型的全部训练域。该模块通过增量学习在有限数据上构建可靠的影子模型。

将影子图分为训练集 $\mathcal{G}_{\text{Shadow}}^{\text{Train}}$ 和测试集 $\mathcal{G}_{\text{Shadow}}^{\text{Test}}$。利用影子数据估计 Fisher 信息矩阵来量化遗忘模型各参数的重要性：

$$\mathbf{I}_{\text{Unlearn}}(\theta) = \mathbb{E}_{v \sim \mathcal{G}_{\text{Shadow}}^{\text{Train}}} \left[\frac{\partial^2 \mathcal{L}_{\text{task}}(\mathcal{F}_{\text{Unlearn}}; v)}{\partial \theta^2}\bigg|\theta\right]$$

然后以参数正则化方式微调遗忘模型，得到影子模型：

$$\min_{\mathbf{\Theta}_{\text{Shadow}}} \sum_{v \in \mathcal{G}_{\text{Shadow}}^{\text{Train}}} \mathcal{L}_{\text{task}}(\mathcal{F}_{\text{Shadow}}; v) + \alpha \sum_i \mathbf{I}_{\text{Unlearn}}^{(i)} (\mathbf{\Theta}_{\text{Shadow}}^{(i)} - \mathbf{\Theta}_{\text{Unlearn}}^{(i)})^2$$

其中 $\alpha$ 控制正则化强度。Fisher 信息矩阵约束重要参数不偏离太远，使影子模型能更好地复制目标模型的成员推理特性。

### 3. 基于相似度的推理机制（Similarity-Based Inference）

为从嵌入中提取成员信号，利用自监督预训练"拉近正样本、推远负样本"的原理，构建攻击特征。对每个目标节点 $v$，随机选取 $m$ 个正样本和 $m$ 个负样本，计算与影子模型输出嵌入的相似度向量 $\mathbf{s}_v$ 作为攻击特征，标注成员/非成员标签后训练两层 MLP 攻击模型。

## 实验结果

### 实验设置

- **数据集**：Cora、CiteSeer、PubMed（引文网络）、Photo、Computers（Amazon 共购图）
- **目标模型**：MDGPT、BRIDGE（链接预测）、GCOPE、SAMGPT（对比学习）
- **基线**：Embed-MIA、Grad-MIA、NLO-MIA、GLO-MIA、GE-MIA、GPIA
- **指标**：Accuracy (ACC)、F1-score
- **设备**：单张 NVIDIA V100 GPU，重复 5 次

### 表1：攻击链接预测类多域图预训练模型（MDGPT）

| 方法 | Cora ACC | Cora F1 | CiteSeer ACC | PubMed ACC | Computers ACC |
|------|----------|---------|-------------|------------|---------------|
| Embed-MIA | 68.89 | 60.31 | 66.53 | 60.60 | 61.54 |
| Grad-MIA | 51.51 | 22.03 | 50.76 | 49.21 | 55.15 |
| GPIA | 72.20 | 76.41 | 68.58 | 65.75 | 68.35 |
| **MGP-MIA** | **81.79** | **83.99** | **77.36** | **74.77** | **80.66** |

### 表2：攻击对比学习类多域图预训练模型（SAMGPT）

| 方法 | Cora ACC | Cora F1 | CiteSeer ACC | PubMed ACC | Computers ACC |
|------|----------|---------|-------------|------------|---------------|
| Grad-MIA | 61.82 | 60.71 | 52.19 | 50.03 | 54.70 |
| GE-MIA | 73.32 | 74.99 | 73.97 | 55.21 | 55.77 |
| GPIA | 58.55 | 59.11 | 55.31 | 54.59 | 73.33 |
| **MGP-MIA** | **99.91** | **99.88** | **98.83** | **91.30** | **91.72** |

MGP-MIA 在 SAMGPT 上的表现尤为突出，在 Cora 上 ACC 达到 99.91%，比最强基线 GE-MIA 提升约 26.6 个百分点。

## 关键发现

1. **多域预训练并不安全**：尽管多域预训练增强了泛化能力，但 MGP-MIA 仍能高精度地识别成员节点，揭示了这类模型的严重隐私风险
2. **对比学习模型更脆弱**：在 SAMGPT（对比学习）上的攻击效果远优于 MDGPT（链接预测），因为对比学习显式地编码正/负样本关系，为相似度推理提供了更强的信号
3. **消融实验**：机器遗忘模块（UL）和增量学习模块（IL）均有贡献，其中 IL 提供了主要的内在增益，UL 进一步放大成员信号
4. **超参数鲁棒性**：框架对正则化强度 $\alpha$ 不敏感，在较大范围内保持稳定性能

## 论文亮点

- **首创性**：首个针对多域图预训练模型的成员推理攻击研究，填补了该方向的空白
- **逆向利用机器遗忘**：创造性地将隐私保护工具（机器遗忘）反向用于增强攻击效果，思路新颖
- **增量学习构建影子模型**：巧妙利用 Fisher 信息矩阵进行参数正则化，在有限数据下构建高质量影子模型
- **基于预训练范式设计攻击特征**：利用自监督学习拉近/推远正负样本的固有机制设计相似度特征，比直接使用嵌入或梯度更有效
- **实验充分**：覆盖四种目标模型（两类预训练范式）、五个数据集、六种基线

## 局限性

1. **白盒假设较强**：攻击者需要完全访问目标模型的架构和参数，在某些实际场景中可能不成立
2. **仅考虑节点级 MIA**：未涉及边级或图级成员推理，攻击粒度有限
3. **依赖同域影子数据**：攻击者仍需获取与目标节点同域的影子图，数据获取成本不可忽略
4. **防御措施缺失**：论文主要关注攻击效果，未深入讨论如何防御此类攻击
5. **可扩展性未验证**：实验使用的数据集规模较小（Cora 仅 2708 节点），对大规模图的适用性有待验证

## 相关工作

- **多域图预训练**：GCOPE（虚拟节点连接域）、SAMGPT（结构 token 统一消息聚合）、MDGPT（域 token 对齐语义）、BRIDGE（域对齐器提取共享表示）
- **图成员推理攻击**：He et al. 和 Olatunji et al. 首次将 MIA 扩展到 GNN；ProIA 引入提示增强攻击模型背景知识；GCL-Leak 针对联邦对比学习场景；Conti et al. 和 Dai & Lu 提出标签-only 黑盒 MIA
- **机器遗忘**：Chen et al. 2022 提出机器遗忘隐私保护策略；Hayes et al. 2025 发现不精确遗忘会导致对剩余样本更强的过拟合

## 评分

⭐⭐⭐⭐ — 首个针对多域图预训练模型的 MIA 研究，方法设计巧妙（尤其是逆向利用机器遗忘），实验全面且效果显著。白盒假设和小规模数据集是主要限制。

<!-- RELATED:START -->

## 相关论文

- [Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures](ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)
- [Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)
- [Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning](learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)

<!-- RELATED:END -->
