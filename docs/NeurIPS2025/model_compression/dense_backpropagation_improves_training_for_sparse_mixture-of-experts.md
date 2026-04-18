---
title: >-
  [论文解读] Dense Backpropagation Improves Training for Sparse Mixture-of-Experts
description: >-
  [NeurIPS 2025][模型压缩][Mixture-of-Experts] 提出 Default MoE 方法，用指数移动平均（EMA）近似非激活 expert 的输出，使 MoE router 获得稠密梯度更新，在不显著增加计算开销的情况下提升稀疏 MoE 的训练性能。
tags:
  - NeurIPS 2025
  - 模型压缩
  - Mixture-of-Experts
  - sparse routing
  - dense gradient
  - EMA
  - TopK routing
---

# Dense Backpropagation Improves Training for Sparse Mixture-of-Experts

**会议**: NeurIPS 2025  
**arXiv**: [2504.12463](https://arxiv.org/abs/2504.12463)  
**代码**: 有（已开源训练代码）  
**领域**: model_compression  
**关键词**: Mixture-of-Experts, sparse routing, dense gradient, EMA, TopK routing

## 一句话总结
提出 Default MoE 方法，用指数移动平均（EMA）近似非激活 expert 的输出，使 MoE router 获得稠密梯度更新，在不显著增加计算开销的情况下提升稀疏 MoE 的训练性能。

## 研究背景与动机

**领域现状**：稀疏 MoE 架构已被 DeepSeek-V2/V3、Mixtral、DBRX 等广泛采用。TopK routing 只激活 K 个 expert 处理每个 token，使得参数量可以大幅扩展而计算量不变。

**现有痛点**：TopK routing 意味着 router 只能接收被激活 expert 的梯度反馈——未激活的 expert 对 router 的梯度贡献为零。这导致 router 无法获得"全局视野"来做最优路由决策，拖慢学习速度并可能导致次优收敛。

**核心矛盾**：要让 router 获得完整（稠密）梯度就需要激活所有 expert（即变成 dense 模型），但这会破坏 MoE 的稀疏计算优势。稠密梯度和稀疏计算之间存在根本矛盾。

**本文目标**：在保持稀疏前向传播效率的同时，让 router 获得近似稠密梯度更新。

**切入角度**：straight-through estimator 可以绕过 TopK 的不可微操作提供稠密梯度，但需要所有 expert 的输出。关键洞察：非激活 expert 的输出可以用其历史输出的均值来近似——这个均值已经在正常前向传播中免费计算了。

**核心 idea**：用 EMA 维护每个 expert 的"默认输出向量"，在反向传播时替代非激活 expert 的输出，实现 $O(1)$ 额外内存的稠密 router 梯度。

## 方法详解

### 整体框架
标准 MoE 层：输入 token $x$ → router 计算 $\pi = \text{Softmax}(Wx)$ → TopK 选择激活集 $\mathcal{A}$ → 输出 $y = \sum_{i \in \mathcal{A}} \pi_i E_i(x)$。Default MoE 在此基础上修改反向传播路径：非激活 expert 用默认向量 $\hat{E}_i$ 代替实际输出参与梯度计算。

### 关键设计

1. **Dense Gradient via Default Vectors**:

    - 功能：让 router 的梯度项 $\frac{\partial y}{\partial \pi_i}$ 对所有 expert（而非仅被激活的）都非零。
    - 标准 TopK 的梯度：$\frac{\partial y}{\partial \pi_i} = E_i(x)$ (若 $i \in \mathcal{A}$) 或 $0$ (若 $i \notin \mathcal{A}$)
    - Default MoE 的梯度：$\frac{\partial y}{\partial \pi_i} = E_i(x)$ (若 $i \in \mathcal{A}$) 或 $\hat{E}_i$ (若 $i \notin \mathcal{A}$)
    - 梯度误差分析：标准 TopK 的误差 $\epsilon_{\text{TopK}} \propto \sum_{i' \notin \mathcal{A}} E_{i'}(x)$；Default MoE 的误差 $\epsilon_{\text{default}} \propto \sum_{i \notin \mathcal{A}} (E_i(x) - \mathbb{E}[E_i(x)])$，期望为零。
    - 设计动机：默认向量是对缺失 expert 输出的无偏估计，期望误差为零意味着平均而言能完美修正稠密梯度。

2. **EMA 更新默认向量**:

    - 功能：用指数移动平均跟踪每个 expert 的历史平均输出。
    - 核心公式：$\hat{E}_i^{(t)} = \beta \hat{E}_i^{(t-1)} + (1 - \beta) \overline{E_i(x)}$
    - 其中 $\overline{E_i(x)}$ 是当前 batch 中所有激活了 expert $i$ 的 token 的输出平均值——这些输出在标准前向传播中已经计算好了，无需额外开销。
    - 前向传播公式：$y = \sum_{i=1}^N \pi_i \cdot \begin{cases} E_i(x) & \text{if } i \in \text{TopK}(\pi) \\ \hat{E}_i^{(t)} & \text{otherwise} \end{cases}$
    - 设计动机：EMA 相比简单平均能更好地跟踪训练过程中 expert 参数的变化；每个 expert 只需额外存储一个 hidden_dim 大小的向量（$O(1)$ 额外内存）。

3. **Router-logit 加权的 EMA 更新**:

    - 功能：用 router 的 softmax 概率对 EMA 更新进行加权，自动适应不同稀疏度配置。
    - 设计动机：不同稀疏度（1/8 vs 1/32）下最优 $\beta$ 不同。加权后，$\beta$ 的选择变得不敏感，所有配置都收敛到相同的良好性能。

### 训练策略
- 所有模型 1.96B 总参数，训练 160B tokens
- 使用 FineWeb-Edu 和 FineWeb 数据集，Llama3 tokenizer
- 采用 globally reduced load balancing loss

## 实验关键数据

### 主实验（1.96B 参数, 160B tokens）

| Benchmark | TopK (8c1) | Default (8c1) | 提升 |
|-----------|-----------|--------------|------|
| BoolQ | 58.5 | **62.0** | +6.1% |
| Lambada | 38.6 | **41.2** | +6.6% |
| SocialIQA | 39.7 | **41.0** | +3.2% |
| ARC | 45.7 | **47.4** | +3.7% |
| HellaSwag | 40.4 | **41.2** | +2.0% |
| **平均** | 46.9 | **47.9** | **+2.1%** |

Improvement Score（相对随机基线的改进百分比）平均提升 5.0%。

### 消融实验

| 实验维度 | 关键发现 |
|---------|---------|
| MoE 配置 (8c1/8c2/32c1/32c2/32c4) | DefaultMoE 在所有 5 种稀疏配置下均优于 TopK |
| 学习率 (5e-4 ~ 9e-4) | DefaultMoE 在所有学习率下均更优，且能耐受更大学习率 (9e-4) |
| 模型规模 (557M ~ 7.33B) | 随规模增大优势保持 |
| 收敛速度 | 达到 perplexity 12.18 的 token 数减少 9% |

### 关键发现
- Default MoE 对训练稳定性有帮助：能容忍更大学习率（9e-4 vs 7e-4），TopK 在大学习率下会出现 loss spike，Default MoE 不会
- 稀疏度越高（如 32c1 = 1/32），Default MoE 预热时间越长但最终仍超越 TopK
- 默认向量在浅层更多样化（router 不确定性高时发挥更大作用），在深层趋向相似（router 已自信）
- 吞吐量几乎无损：7.33B 模型下仅降低 0.18%，额外内存占 MoE 参数的 0.03%
- 已有路由改进方法（SparseMixer、ReMoE、Loss-Free Balancing）在使用 globally reduced auxiliary loss 后均无法超越调好的 TopK baseline，但 Default MoE 可以

## 亮点与洞察
- **免费午餐**：EMA 更新利用的是已有前向传播中计算好的 expert 输出均值，额外开销几乎为零（0.03% 内存 + 0.18% 吞吐下降），但带来一致的性能提升。
- **数学优雅**：默认向量是缺失 expert 输出的无偏估计，期望梯度误差为零的理论保证很漂亮。
- **实验方法论值得学习**：(1) 160B tokens 的过训练确保结果不是虚假收敛差异；(2) 用 globally reduced auxiliary loss 作为 baseline，保证对比公平（很多先前工作的改进在此 baseline 下消失）。

## 局限与展望
- 只在 2B 总参数级别验证，未在 10B+ 规模下实验
- EMA 机制假设 expert 输出的分布变化平滑——在训练极早期或学习率剧变时可能不成立
- 默认向量只是输入无关的常量，更精细的近似（如条件化的默认向量）可能进一步提升效果
- 未测试在 fine-tuning 或 downstream task 上的效果，只评估了 pretraining benchmark
- 未与 DeepSeek 系列的 auxiliary-loss-free 方案在同一代码库下对比

## 相关工作与启发
- **vs SparseMixer**: SparseMixer 通过线性近似估计真实 router 梯度，但引入噪声导致前期落后 TopK，最终也未超越 Default MoE
- **vs ReMoE**: ReMoE 尝试学习连续的路由权重但训练极不稳定，在作者的设置下无法收敛
- **vs Loss-Free Balancing (DeepSeek)**: 在 globally reduced auxiliary loss 下反而不如调好的 TopK baseline
- Default MoE 思路可能对 DeepSeek-V3 的 MoE 训练有参考价值——任何使用 TopK routing 的 MoE 都可以几乎零成本地加入 EMA 默认向量

## 评分
- 新颖性: ⭐⭐⭐⭐ 用 EMA 近似缺失 expert 输出的想法简单优雅，理论分析清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 160B tokens 长训练、全面的消融（5 种 MoE 配置、学习率、模型规模）、详细的效率分析
- 写作质量: ⭐⭐⭐⭐ 从梯度误差分析切入的叙事逻辑清晰，图示直观
- 价值: ⭐⭐⭐⭐ 对 MoE 训练的实际改进，开源代码，zero-cost 特性使其易于被广泛采用
