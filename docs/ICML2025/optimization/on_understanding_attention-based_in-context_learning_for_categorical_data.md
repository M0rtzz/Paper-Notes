---
title: >-
  [论文解读] On Understanding Attention-Based In-Context Learning for Categorical Data
description: >-
  [ICML2025][优化][in-context learning] 将 Transformer 的 in-context learning (ICL) 从实值输出推广到**分类数据**（categorical outcomes），证明一种交替使用 self-attention 和 cross-attention 的架构可以**精确实现**多步函数梯度下降（functional GD），并在理论上证明该 GD 参数构型是注意力模型损失函数的驻点。
tags:
  - ICML2025
  - 优化
  - in-context learning
  - functional gradient descent
  - categorical data
  - 注意力机制
  - softmax
  - reproducing kernel Hilbert space
---

# On Understanding Attention-Based In-Context Learning for Categorical Data

**会议**: ICML2025  
**arXiv**: [2405.17248](https://arxiv.org/abs/2405.17248)  
**代码**: 未开源  
**领域**: 优化 / ICL理论  
**关键词**: in-context learning, functional gradient descent, categorical data, attention mechanism, cross-attention, softmax, reproducing kernel Hilbert space

## 一句话总结

将 Transformer 的 in-context learning (ICL) 从实值输出推广到**分类数据**（categorical outcomes），证明一种交替使用 self-attention 和 cross-attention 的架构可以**精确实现**多步函数梯度下降（functional GD），并在理论上证明该 GD 参数构型是注意力模型损失函数的驻点。

## 研究背景与动机

- 已有一系列工作（von Oswald et al. 2023; Ahn et al. 2023; Cheng et al. 2024）将 Transformer 的 ICL 过程理解为**函数梯度下降**，但这些分析仅适用于**实值输出**（如线性回归、核回归），假设 $p(Y|X) = \mathcal{N}(f(x), \sigma^2 I)$。
- 语言模型的输出是**离散 token（分类变量）**，不满足高斯假设。实值情形下 $\mathbb{E}(Y)|_{f_{i,k}} = f_{i,k}$ 是线性的，可以仅靠 self-attention 层叠实现多步 GD；但分类情形下 $\mathbb{E}(w_e)|_{f_{i,k}}$ 是 $f_{i,k}$ 的**非线性函数**（涉及 softmax），单靠 self-attention 无法完成。
- **核心动机**：弥合 ICL 理论分析与真实语言模型之间的鸿沟——将 functional GD 框架推广到 categorical observations，使理论更贴近 Transformer 在 NLP 中的实际运作方式。

## 方法详解

### 整体框架

模型包含两个视角：
1. **Transformer** $T_\theta(z)$：基于注意力的前向推理模型
2. **Softmax 分类模型** $p_\phi(Y=y|X=x) = \frac{\exp(w_{e,y}^T f_\phi(x))}{\sum_{c=1}^C \exp(w_{e,c}^T f_\phi(x))}$

两者共享 embedding 矩阵 $W_e \in \mathbb{R}^{d' \times C}$，Transformer 的前向传播实质上在推断隐函数 $f_\phi(x)$，再输入 softmax 输出 token 概率。

### 函数梯度下降（Functional GD）

假设隐函数 $f_\phi(x) = A\psi(x) + b$ 住在 RKHS 中，对交叉熵损失做 GD，得到更新规则：

$$f_{j,k+1} = f_{j,k} + \frac{\alpha}{N} \sum_{i=1}^N \left[ w_{e,y_i} - \mathbb{E}(w_e)|_{f_{i,k}} \right] \kappa(x_i, x_j)$$

其中 $\mathbb{E}(w_e)|_{f_{i,k}} = \sum_{c=1}^C w_{e,c} \cdot p_{\phi_k}(Y=c|X=x_i)$ 是 embedding 向量的加权期望，$\kappa(x_i, x_j) = \psi(x_i)^T \psi(x_j)$ 是核函数（对应注意力权重）。

### 关键设计：Self-Attention + Cross-Attention 交替架构

每个 attention block 由**两层**组成：

**Self-Attention 层**（两个头）：

- **Head 1（函数更新）**：Key/Query 提取 $x_i$，Value 提取 $w_{e,y_i} - \mathbb{E}(w_e)|_{f_{i,k}}$，通过核注意力计算 $\Delta f_{i,k}$，更新 $f_{i,k} \to f_{i,k+1}$
- **Head 2（期望擦除）**：用大 $\lambda$ 使注意力退化为 Kronecker delta $\delta_{i,j}$（自身匹配），从"scratch space"中擦除旧的 $\mathbb{E}(w_e)|_{f_{i,k}}$，为下一步计算腾位置

**Cross-Attention 层**（单头，核心创新）：

- Query 来自更新后的 $f_{i,k+1}$，Key 和 Value 均为 embedding 矩阵 $W_e$ 的列向量 $\{w_{e,c}\}_{c=1}^C$
- 利用 softmax attention 精确计算 $\mathbb{E}(w_e)|_{f_{i,k+1}} = \sum_{c=1}^C w_{e,c} \frac{\exp(w_{e,c}^T f_{i,k+1})}{\sum_{c'} \exp(w_{e,c'}^T f_{i,k+1})}$
- 将新期望写入之前擦除的位置

### Token Embedding 的自然性

当参数初始化为零时，$\mathbb{E}(w_e)|_{f_{i,0}} = \frac{1}{C}\sum_c w_{e,c} = \bar{w}_e$。若 $\bar{w}_e = 0$，则第一步 GD 的输入恰好是 token 的 embedding 向量 $w_{e,y_i}$——这说明语言模型中普遍使用的"学习 embedding"编码方式与 GD 视角**天然一致**。

### 单步 GD 简化

若仅做一步 GD，则无需更新期望，可移除擦除头和 cross-attention 层，简化为单层 self-attention，输入编码为 $e_{i,0} = (x_i, w_{e,y_i} - \bar{w}_e, 0_{d'})^T$。

## 实验关键数据

### 合成数据（C=25 类，N=10 上下文样本）

| 模型 | 注意力类型 | Top-1 准确率趋势 |
|------|-----------|-----------------|
| GD (1层) | RBF / Softmax | 少量训练集（L<5000）即收敛 |
| Trained TF (1层) | RBF / Softmax | 需大量训练集（L>10000）才与 GD 匹配 |
| GD (2层) | Softmax | L<5000 即收敛，优于 1 层 |
| Trained TF (2层) | Softmax | 需 L>25000 才与 GD 匹配 |

- 增加注意力层数（2→6 blocks）持续提升 GD 模型的 Top-1 准确率和 NLL
- Trained TF 收敛后的参数矩阵与 GD 理论预测的驻点高度吻合

### ImageNet In-Context 分类（900 训练类 / 100 测试类，N=50，VGG 特征 d=512）

| 模型 | Top-1 准确率 |
|------|-------------|
| Linear Probing | 基准线（需对每个测试 context 重新训练） |
| GD 1层 | 略低于 linear probing |
| GD 2-3层 | **与 linear probing 几乎一致**（无需微调） |

### 语言生成（Tiny Stories + Children Stories，C=50257 token，d'=512，8 头）

| 模型 | 参数量 | GPT-4o 评分（Grammar/Consistency/Plot/Creativity） |
|------|-------|--------------------------------------------------|
| Softmax GD | **8K** 注意力参数 | 与 Transformer 差距不大 |
| Softmax GD + FF | **8K** 注意力参数 | **几乎等于 Transformer** |
| 单层 Transformer | **6M** 注意力参数 | 基准线 |

- GD 模型参数量仅为 Transformer 的约 **0.13%**，但加上 FF 后生成质量几乎追平
- 两者的典型失败模式均为**重复生成**

## 亮点与洞察

1. **理论突破**：首次将 functional GD ↔ Transformer ICL 的对应关系从实值推广到分类数据，证明 GD 参数构型是注意力模型损失函数的**驻点**（Theorem 1/2），适用于 softmax attention
2. **架构洞察**：self-attention + cross-attention 交替结构（原始 Transformer decoder 就有！）的理论解释——self-attention 做函数更新，cross-attention 计算非线性期望
3. **Embedding 解释**：从 GD 视角自然推导出 token embedding 的必要性，而非仅作为经验设计
4. **效率启示**：GD 模型用 8K 参数 + FF 即可匹配 6M 参数 Transformer 的语言生成质量，暗示 Transformer 的大量参数可能存在冗余
5. **FF 的重要性**：实验揭示 feedforward 层对 Transformer 性能的关键贡献——GD 模型本身不含 FF，加上 FF 后性能跳跃性提升

## 局限与展望

1. **语言实验受限**：仅在 Tiny Stories 等简单语料上测试单层模型，未验证在大规模真实语言建模中的可扩展性
2. **GD 模型的结构性限制**：query 只能用位置 embedding（不含 token 信息），与真实 Transformer 存在差异
3. **Cross-attention 需要全 C 个 embedding**：对词表很大（C=50K+）的场景，cross-attention 计算 $\mathbb{E}(w_e)$ 涉及对全词表 softmax，效率受限
4. **理论假设较强**：Theorem 1 需要旋转不变性（$x_i$ 分布满足旋转对称），cross-attention 精确计算期望等假设，实际场景未必满足
5. **FF 层的理论解释缺失**：实验表明 FF 极其重要，但文章未从 GD 视角给出 FF 的理论解释

## 相关工作与启发

- **von Oswald et al. (2023)** / **Ahn et al. (2023)**：ICL ↔ GD 在线性回归上的对应，本文是其在分类设定下的直接推广
- **Cheng et al. (2024)**：RKHS 中的核回归 ICL 理论，本文推广其理论到 softmax attention 和 categorical loss
- **Akyurek et al. (2022)**："scratch space" 概念的首次提出
- **Vaswani et al. (2017)**：原始 Transformer 的 decoder 已包含 self + cross attention 交替结构，本文提供了理论视角
- **启发**：FF 层对 Transformer 的重要性值得深入研究；GD 视角可能指导设计更高效的注意力架构

## 评分

- 新颖性: ⭐⭐⭐⭐ — 从实值到分类的推广虽是自然延伸，但 cross-attention 计算非线性期望的设计及理论证明有实质新意
- 实验充分度: ⭐⭐⭐⭐ — 合成数据 + ImageNet + 语言生成三个层次，GPT-4o 自动评分有说服力，但语言实验规模偏小
- 写作质量: ⭐⭐⭐⭐ — 数学推导严谨清晰，从简单到复杂逐步展开，但符号较多需要仔细跟读
- 价值: ⭐⭐⭐⭐ — 对理解 Transformer ICL 机制有重要理论贡献，FF 层的实验发现有实践指导意义

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Training Dynamics of In-Context Learning in Linear Attention](training_dynamics_of_in-context_learning_in_linear_attention.md)
- [\[ICML 2025\] In-Context Linear Regression Demystified: Training Dynamics and Mechanistic Interpretability of Multi-Head Softmax Attention](in-context_linear_regression_demystified_training_dynamics_and_mechanistic_inter.md)
- [\[NeurIPS 2025\] Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers](../../NeurIPS2025/optimization/generalization_or_hallucination_understanding_out-of-context_reasoning_in_transf.md)
- [\[ICML 2025\] Provable In-Context Vector Arithmetic via Retrieving Task Concepts](provable_in-context_vector_arithmetic_via_retrieving_task_concepts.md)
- [\[ICML 2025\] Understanding the Statistical Accuracy-Communication Trade-off in Personalized Federated Learning with Minimax Guarantees](understanding_the_statistical_accuracy-communication_trade-off_in_personalized_f.md)

<!-- RELATED:END -->
