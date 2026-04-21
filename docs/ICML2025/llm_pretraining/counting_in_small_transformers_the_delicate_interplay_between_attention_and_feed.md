---
title: >-
  [论文解读] Counting in Small Transformers: The Delicate Interplay between Attention and Feed-Forward Layers
description: >-
  [ICML2025][Transformer] 通过直方图计数任务，揭示了小型Transformer中注意力层与前馈层之间的精细分工：注意力擅长关系比较（relation-based counting），前馈层负责字典记忆（inventory-based counting），两种策略的出现由嵌入维度 $d$、隐层大小 $p$ 和词表大小 $T$ 的相对关系决定。
tags:
  - ICML2025
  - Transformer
  - 计数任务
  - 注意力机制
  - 嵌入正交性
  - 机制可解释性
---

# Counting in Small Transformers: The Delicate Interplay between Attention and Feed-Forward Layers

**会议**: ICML2025  
**arXiv**: [2407.11542](https://arxiv.org/abs/2407.11542)  
**代码**: [GitHub](https://github.com/SPOC-group/counting-attention)  
**领域**: Transformer理论  
**关键词**: Transformer机制分析, 计数任务, Attention与FFN交互, 嵌入正交性, 机制可解释性

## 一句话总结

通过直方图计数任务，揭示了小型Transformer中注意力层与前馈层之间的精细分工：注意力擅长关系比较（relation-based counting），前馈层负责字典记忆（inventory-based counting），两种策略的出现由嵌入维度 $d$、隐层大小 $p$ 和词表大小 $T$ 的相对关系决定。

## 研究背景与动机

Transformer架构虽然广泛成功，但其内部组件（attention vs. FFN）各自承担什么功能、如何协作，仍缺乏清晰理解。本文选择一个极简但具有揭示性的任务——**直方图任务（histogram task）**——作为分析工具：给定输入序列 $\mathbf{x} = [A, B, D, D, B, B]$，输出每个位置对应token在序列中的出现次数 $\mathbf{y} = [1, 3, 2, 2, 3, 3]$。

这个任务看似简单，但即使8B参数的现代LLM也无法在权重空间中可靠地in-weights求解（需依赖chain-of-thought）。作者据此研究：不同的token混合机制（线性混合 vs. 点积注意力）和前馈层容量如何影响模型能学到的算法类型。

## 方法详解

### 模型架构

单层Transformer block，包含两个组件：

1. **Token混合层**（mixing）：产生混合token $\bar{x}'_\ell = \bar{x}_\ell + [\mathbf{A}(\bar{\mathbf{x}})\bar{\mathbf{x}}]_\ell$
2. **前馈层**（FFN）：$f(\bar{x}'_\ell) = \text{ReLU}(\bar{x}'_\ell W_1 + b_1) W_2 + b_2$，隐层维度 $p$

最终预测：$F(\bar{\mathbf{x}})_\ell = \arg\max_{c \in \{1,\dots,C\}} f(\bar{x}'_\ell)_c$

### 四种Token混合机制

| 机制 | 公式 | 参数量 |
|------|------|--------|
| **lin**（线性混合） | $\mathbf{A} = A$（可学习矩阵） | $L^2$ |
| **lin+sftm** | $\mathbf{A} = \text{softmax}(A)$ | $L^2$ |
| **dot**（点积混合） | $\mathbf{A} = \frac{1}{\sqrt{d}} \bar{\mathbf{x}} W_Q W_K^T \bar{\mathbf{x}}^T$ | $2d^2$ |
| **dot+sftm** | $\mathbf{A} = \text{softmax}(\mathbf{A}_{\text{dot}})$ | $2d^2$ |

此外还有**bos**变体：在输入前添加BOS token $\tilde{\mathbf{x}} = (\$, x_1, \dots, x_L)$。

### 两种计数策略

**策略一：关系计数（Relation-based Counting, RC）**

- 利用点积注意力的**成对比较**能力：同token的注意力得分高，异token得分低
- 仅需 $p=1$ 个隐层神经元，通过单一方向（如BOS方向）提取计数信息
- BOS模型中：$\langle \bar{x}'_\ell, e_{\text{BOS}} \rangle = T + \text{hist}_{\mathbf{x}}(\ell) + 1$，计数线性编码在BOS方向的投影中
- dot模型中：使用"tagged embedding"——在正交embedding上添加公共方向 $e_{\text{cnt}}$，使得 $\langle e_{\text{cnt}}, \bar{x}'_\ell \rangle \propto 1 + \text{hist}(\ell) \cdot a_= + (L - \text{hist}(\ell)) \cdot a_{\neq}$

**策略二：字典计数（Inventory-based Counting, IC）**

- 前馈层充当**查找表**，用 $p \geq T$ 个隐层神经元记忆整个字母表
- 设 $(W_1)_t = e_t$，则 $\text{hist}_{\mathbf{x}}(\ell) = \frac{1}{a} \sum_{t \in \mathcal{T}} \text{ReLU}(\langle \bar{x}'_\ell, e_t \rangle - 1)$
- 由于偏置项 $-1$，只有残差连接中对应token $x_\ell$ 的神经元被激活
- 适用于lin、lin+sftm、dot+sftm等无法做RC的架构

### 关键理论结果

**dot+sftm为何不能做RC？** softmax归一化使 $\sum_m a_{\ell m} = 1$，任何出现在所有token中的公共方向在混合后权重恒为1，丢失计数信息。因此必须转向IC策略（$p \geq T$）。

**非正交嵌入时（$d < T$）的鲁棒性：** 利用Welch下界约束互相干性（mutual coherence），论文推导出不同架构达到完美计数所需的最小 $d$：

- lin/lin+sftm（$p=T$）：$d \geq \lceil \frac{T(2L-3)^2}{T-1+(2L-3)^2} \rceil$
- dot/bos（$p=1$）：上式 $+1$
- dot/bos（$p=T$）：$d \geq \lceil \frac{T(L-1)}{T-1+(L-1)} \rceil$

**softmax的降噪效果：** 高温softmax可非线性压低异token的注意力得分，使所需嵌入维度降至 $d \geq \lceil \log_2(T+1) \rceil + 2$（对 $T=32$ 仅需 $d=7$）。

## 实验关键数据

实验设置：$T=32$ 种token，序列长度 $L=10$，Adam优化器（lr=1e-3），500 epochs，每epoch 10000新样本。

| 架构 | 最小完美准确率配置 | 对 $p$ 的要求 | RC/IC |
|------|-------------------|--------------|-------|
| **bos+sftm** | $d \geq T, p=1$ | $p=1$ 即可 | RC |
| **bos** | $d \geq T, p=1$ | $p=1$ 即可 | RC |
| **dot** | $d \geq T, p=1$ | $p=1$ 即可 | RC |
| **dot+sftm** | $d \geq T, p \geq T$ | 需 $p \geq T$ | IC |
| **lin+sftm** | $d \geq T, p \geq T$ | 需 $p \geq T$ | IC |
| **lin** | $d \geq T, p \geq T$ | 需 $p \geq T$ | IC |

关键发现：
- bos+sftm在 $d < T$ 时仍能通过softmax降噪达到接近100%准确率（$d=7$ 即可）
- dot模型在 $p$ 从1增到 $T$ 时，准确率反而从100%微降至99%，暗示RC与IC可能存在superposition
- 2层Transformer的表现与1层高度一致，说明深度并未改变根本机制

## 亮点与洞察

1. **极简任务揭示深层机制**：一个看似trivial的计数任务，却精确暴露了attention与FFN的功能分工——attention做比较，FFN做记忆
2. **BOS token的本质解释**：BOS不只是序列标记，它在RC策略中充当"计数器"——其注意力权重编码了每个token的出现频次
3. **softmax的双面性**：softmax在dot+sftm中**破坏**了RC能力（归一化抹去计数信号），但在bos+sftm中**增强**了鲁棒性（压低非匹配token噪声）
4. **理论构造与实验验证的闭环**：每个理论命题都有对应的权重构造和训练模型验证，可解释性分析（注意力矩阵、FFN响应曲线）与理论预测高度一致
5. **对LLM的启示**：现实中LLM的词表 $T$ 远大于模型维度 $d$，本文的非正交嵌入分析直接相关

## 局限与展望

1. **仅限1层非自回归模型**：未考虑因果mask和位置编码的影响，向多层/自回归架构的推广需验证
2. **任务单一**：histogram任务的结论能否迁移到排序、查表等其他基础任务不确定
3. **训练预算固定**：$L=30$ 时性能下降，可能是训练不充分而非架构限制
4. **Welch界的实现困难**：理论推导依赖Welch界可达，但实际中难以构造达到该界的嵌入矩阵
5. **superposition现象未深入**：RC与IC共存时的具体机制（如SVD分析仅为初步探索）需更细致研究

## 相关工作与启发

- **RASP语言**（Weiss et al., 2021）：预测histogram需BOS token，本文证明dot模型无需BOS也可实现RC
- **FFN作为记忆模块**（Geva et al., 2021; Meng et al., 2022）：本文从计数任务视角印证了FFN的look-up table功能
- **算法-架构对齐**（Dziri et al., 2023）：LLM幻觉可能源于计算图与任务的错配，本文展示了微小架构变化如何导致截然不同的算法涌现

## 评分

- 新颖性: ⭐⭐⭐⭐ — 从极简任务出发系统性地揭示attention/FFN分工，RC vs IC二分法有启发性
- 实验充分度: ⭐⭐⭐⭐⭐ — 理论构造+大规模超参扫描+机制验证，闭环完整
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，但公式和符号较密集，需要一定背景
- 价值: ⭐⭐⭐⭐ — 对理解Transformer内部机制有重要参考价值，对架构设计有指导意义

<!-- RELATED:START -->

## 相关论文

- [Benign Overfitting in Token Selection of Attention Mechanism](benign_overfitting_in_token_selection_of_attention_mechanism.md)
- [The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training](the_sharpness_disparity_principle_in_transformers_for_accelerating_language_mode.md)
- [Scaling Embedding Layers in Language Models](../../NeurIPS2025/llm_pretraining/scaling_embedding_layers_in_language_models.md)
- [Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?](../../NeurIPS2025/llm_pretraining/does_object_binding_naturally_emerge_in_large_pretrained_vision_transformers.md)
- [Deconstructing Positional Information: From Attention Logits to Training Biases](../../ICLR2026/llm_pretraining/deconstructing_positional_information_from_attention_logits_to_training_biases.md)

<!-- RELATED:END -->
