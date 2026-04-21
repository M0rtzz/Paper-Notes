---
title: >-
  [论文解读] Do We Really Need Permutations? Impact of Model Width on Linear Mode Connectivity
description: >-
  [ICLR 2026][线性模式连通性] 实证表明无需参数置换，仅靠增加模型宽度即可实现独立训练模型间的线性模式连通性（LMC），并提出"逐层指数加权连通性"（LEWC）解释这一现象的机理。
tags:
  - ICLR 2026
  - 线性模式连通性
  - 模型合并
  - 参数置换对称性
  - 模型宽度
  - 损失景观
---

# Do We Really Need Permutations? Impact of Model Width on Linear Mode Connectivity

**会议**: ICLR 2026  
**arXiv**: [2510.08023](https://arxiv.org/abs/2510.08023)  
**代码**: 无  
**领域**: 其他  
**关键词**: 线性模式连通性, 模型合并, 参数置换对称性, 模型宽度, 损失景观

## 一句话总结

实证表明无需参数置换，仅靠增加模型宽度即可实现独立训练模型间的线性模式连通性（LMC），并提出"逐层指数加权连通性"（LEWC）解释这一现象的机理。

## 研究背景与动机

**线性模式连通性（LMC）**是指两个独立训练的模型参数之间存在一条低损失的线性路径，即参数线性插值不会导致显著的损失增加。LMC对于理解损失景观结构和模型合并（federated learning、model merging）至关重要。

### 已有认知

Entezari et al. (2022) 提出假说：对于充分宽的模型，总存在一个参数置换 $\pi$ 使得 LMC 成立。Ainsworth et al. (2023) 通过 Weight Matching (WM) 实证验证了这一假说，但发现需要非常大的宽度倍数（如ResNet-20需32×，VGG-16需4×）。此前人们相信：
- 宽度的作用是增加候选置换空间，提高找到好置换的概率
- 没有置换，LMC 不成立

### 本文的核心发现

即使**不做任何置换**，只要模型足够宽，简单地平均两个独立训练模型的权重就能达到与原始模型相当的测试精度。这颠覆了"置换是LMC的必要条件"的传统认知。

## 方法详解

### 整体框架

本文是一篇**分析性工作**而非方法论文，核心贡献是提出LEWC概念并通过充分条件分析来解释为何宽模型无需置换即可实现LMC。

### 关键设计

#### 1. 逐层指数加权连通性（LEWC）

**定义（LEWC）**：两个模型参数 $\boldsymbol{\theta}_a$ 和 $\boldsymbol{\theta}_b$ 满足LEWC，当且仅当对任意层 $\ell$ 和任意 $\lambda \in [0,1]$：

$$f_\ell(\mathbf{x}; \lambda\boldsymbol{\theta}_a + (1-\lambda)\boldsymbol{\theta}_b) = \lambda^\ell f_\ell(\mathbf{x}; \boldsymbol{\theta}_a) + (1-\lambda)^\ell f_\ell(\mathbf{x}; \boldsymbol{\theta}_b)$$

**解读**：合并模型的第 $\ell$ 层输出是两个原始模型对应层输出的**指数衰减加权和**。在最后一层，这意味着合并模型的输出等价于两个原始模型的加权集成。因为缩放logits不改变预测标签，LEWC直接蕴含LMC（精度不下降）。

#### 2. LEWC的充分条件

**条件1：ReLU弱可加性（Weak Additivity）**

$$\sigma(\lambda \tilde{\mathbf{z}}_\ell^{(a)} + (1-\lambda)\tilde{\mathbf{z}}_\ell^{(b)}) = \lambda\sigma(\tilde{\mathbf{z}}_\ell^{(a)}) + (1-\lambda)\sigma(\tilde{\mathbf{z}}_\ell^{(b)})$$

即ReLU在两个模型的预激活插值路径上表现为线性。成立原因：
- **维度诅咒效应**：高维下两个高斯向量的ReLU的余弦相似度趋向0.93
- **低秩权重导致激活不重叠**：宽模型的权重矩阵低秩，两个模型的大二阶矩维度不重叠

**条件2：互逆正交性（Reciprocal Orthogonality）**

$$\mathbf{W}_\ell^{(b)} \mathbf{z}_{\ell-1}^{(a)} = 0 \quad \text{且} \quad \mathbf{W}_\ell^{(a)} \mathbf{z}_{\ell-1}^{(b)} = 0$$

即一个模型的权重矩阵作用在另一个模型的激活向量上结果为零——两个模型在特征空间中"互不干扰"。

**定理5.3（核心）**：对于无偏置的模型，若弱可加性和互逆正交性同时成立，则LEWC成立。

#### 3. 与LLFC（逐层线性特征连通性）的本质区别

Zhou et al. (2023) 提出的LLFC需要**交换性**（commutativity），即两个模型的权重足够接近。而LEWC需要**互逆正交性**，即两个模型权重高度不同且正交。两者不兼容：
- LLFC解释了置换后的LMC（WM使权重对齐→接近）
- LEWC解释了无置换的LMC（宽模型→权重正交）

#### 4. 低秩结构的关键角色

宽度增加 → 权重矩阵相对秩降低 → 激活向量的有效维度降低 → 两个模型的激活空间不重叠 → 弱可加性和互逆正交性成立 → LEWC成立 → LMC成立

### 损失函数 / 训练策略

本文使用标准训练（SGD + weight decay 0.003），不提出新的训练方法。关键观察是softmax温度校准：由于LEWC导致logit范数指数衰减，需要用逆温度参数校准softmax以使校准后的损失barrier趋近于零。

## 实验关键数据

### 主实验

**Table 1：有/无置换时的Barrier值（$\lambda=1/2$）**

| 网络 | 数据集 | 无置换 Acc barrier | 无置换 Loss barrier | 有WM置换 Acc barrier | 有WM置换 Loss barrier |
|------|--------|:-:|:-:|:-:|:-:|
| MLP (16×) | MNIST | 0.519% | 0.013 | -0.027% | -0.003 |
| VGG-11 (16×) | CIFAR-10 | 1.308% | 0.066 | 7.000% | 0.177 |
| ResNet-20 (32×) | CIFAR-10 | 2.694% | 0.087 | 5.135% | 0.173 |

充分宽的模型无需置换即可达到很小的barrier。甚至在某些情况下，WM置换后的barrier反而更大（如VGG-11和ResNet-20）。

**随机置换实验**：对充分宽的模型施加随机置换后合并，精度依然保持——说明一旦模型足够宽，置换是无关紧要的。

### 消融实验

**弱权重衰减（$10^{-4}$）的影响**

| 条件 | VGG-11 LEWC | VGG-11 弱可加性 | VGG-11 互逆正交性 |
|------|:-:|:-:|:-:|
| 标准WD (0.003) | ✓ (高余弦相似度) | ✓ | ✓ (低比率) |
| 弱WD ($10^{-4}$) | ✗ (低余弦相似度) | ✗ | ✗ (高比率) |

弱权重衰减→高秩权重→LEWC两个充分条件均不成立→LMC失败。这证实了低秩结构是LEWC的关键驱动因素。

### 关键发现

1. **宽度单调提升合并性能**：增加宽度使合并模型精度单调上升直至匹配原始模型
2. **温度校准是必要的**：LEWC导致logit范数指数衰减，需校准softmax才能使loss barrier趋零
3. **LEWC ≠ 平坦性**：随机扰动实验表明，仅靠损失景观平坦无法解释LMC——LEWC是独立机制
4. 维度约2×以上宽度（如VGG-11 16×，ResNet-20 32×）即可实现无置换LMC

## 亮点与洞察

1. **颠覆性发现**：推翻了"置换是LMC必要条件"的普遍假设，揭示宽度本身比置换空间更重要
2. **LEWC概念**：优雅地将合并模型解释为原始模型的指数加权集成，建立了模型合并与集成学习的桥梁
3. **互逆正交性vs交换性**：清晰地区分了两种本质不同的LMC机制，深化了对神经网络损失景观的理解
4. **低秩—>LMC的因果链**：低秩权重→激活不重叠→弱可加性+互逆正交性→LEWC→LMC

## 局限与展望

1. 实验限于简单数据集（MNIST、CIFAR-10），因为无置换LMC需要较大宽度倍数
2. 仅考虑MLP、VGG-11、ResNet-20等标准架构，未验证Transformer等现代架构
3. 作为分析工作，未提出实用的模型合并或联邦学习方法
4. LEWC需要BN重新校准和温度缩放，增加了实际使用复杂度
5. 理论分析主要是充分条件而非必要条件
6. 在更复杂数据集（如CIFAR-100、ImageNet）上LMC的宽度需求可能过大

## 相关工作与启发

- **Ainsworth et al. (2023)**：Weight Matching方法，本文的主要对比框架
- **Zhou et al. (2023)**：提出LLFC概念，与本文的LEWC形成互补解释
- **Entezari et al. (2022)**：提出置换不变性假说，本文实质上修正了这一假说
- 对**联邦学习**的启发：若客户端模型足够宽，简单FedAvg可能就足够好，无需复杂的对齐策略
- 对**模型合并**的实践启示：训练更宽的模型 + 适当weight decay可能是最简单的合并策略

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ★★★★★ |
| 技术深度 | ★★★★☆ |
| 实验充分性 | ★★★★☆ |
| 写作质量 | ★★★★★ |
| 实用价值 | ★★★☆☆ |

<!-- RELATED:START -->

## 相关论文

- [Do ImageNet-trained Models Learn Shortcuts? The Impact of Frequency Shortcuts on Generalization](../../CVPR2025/llm_evaluation/do_imagenet-trained_models_learn_shortcuts_the_impact_of_frequency_shortcuts_on_.md)
- [Revisiting the Past: Data Unlearning with Model State History](revisiting_the_past_data_unlearning_with_model_state_history.md)
- [Predicting LLM Reasoning Performance with Small Proxy Model](predicting_llm_reasoning_performance_with_small_proxy_model.md)
- [Prompt and Parameter Co-Optimization for Large Language Model Task Adaptation](prompt_and_parameter_co-optimization_for_large_language_model_task_adaptation.md)
- [How Reliable is Language Model Micro-Benchmarking?](how_reliable_is_language_model_micro-benchmarking.md)

<!-- RELATED:END -->
