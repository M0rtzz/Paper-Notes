---
title: >-
  [论文解读] LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models
description: >-
  [ACL 2025][状态空间模型] 提出 LaTIM，一种针对 Mamba-1 和 Mamba-2 的 token 级分解方法，将 SSM 的隐式计算重构为类似 Transformer 注意力的 token-to-token 贡献矩阵，实现对 Mamba 模型的细粒度可解释性分析。
tags:
  - ACL 2025
  - 状态空间模型
  - Mamba
  - 可解释性
  - token交互分解
  - 注意力归因
---

# LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models

**会议**: ACL 2025  
**arXiv**: [2502.15612](https://arxiv.org/abs/2502.15612)  
**代码**: [https://github.com/deep-spin/latim](https://github.com/deep-spin/latim)  
**领域**: 其他  
**关键词**: 状态空间模型, Mamba, 可解释性, token交互分解, 注意力归因

## 一句话总结

提出 LaTIM，一种针对 Mamba-1 和 Mamba-2 的 token 级分解方法，将 SSM 的隐式计算重构为类似 Transformer 注意力的 token-to-token 贡献矩阵，实现对 Mamba 模型的细粒度可解释性分析。

## 研究背景与动机

状态空间模型（SSMs）如 Mamba 已成为 Transformer 的高效替代方案，能以线性复杂度处理长序列。然而，Transformer 拥有注意力矩阵这一天然的可解释性工具，可以直观地展示 token 之间的交互关系，而 Mamba 缺乏类似的显式机制。

现有的 Mamba 可解释性工作存在不足：
- **MambaAttention**（Ali et al., 2024）虽然将 Mamba 的计算重新表述为隐式注意力矩阵，但在 Mamba-1 中通道维度往往很大（如370M模型有D=1024个通道），无法给出每层的单一注意力图
- **MambaLRP**（Jafari et al., 2024）使用层级相关传播分析梯度流，但仅支持 Mamba-1，且不能显式分解各 token 的贡献
- 这些方法都**未能实现类似 Transformer 中那样的细粒度 token 级贡献分解**

本文通过引入 LaTIM，弥合了这一可解释性差距，使研究者能在 Mamba 模型上应用类似 ALTI 等成熟的归因方法。

## 方法详解

### 整体框架

LaTIM 的核心思想是：将 Mamba 的前向计算重新排列，使得输出 $\boldsymbol{y}_i$ 可以表示为所有前序 token 贡献 $T_i(\boldsymbol{x}_j)$ 的求和形式，即 $\boldsymbol{y}_i = \sum_{j=1}^{i} T_i(\boldsymbol{x}_j)$。这与 Transformer 中的注意力分解形式完全对应，从而可以复用现有的归因技术。

### 关键设计

1. **Mamba-1 分解**：

    - 首先展开 SSM 递推，得到隐式注意力张量 $\boldsymbol{M}_{i,j}$，表示 token $j$ 对 token $i$ 的隐式贡献
    - 关键挑战是 SiLU 激活函数的非可加性——无法直接将卷积层的输出按 token 拆分
    - 解决方案：假设存在一个可加函数 $f$ 近似 SiLU，将卷积后的激活分解为各 token 的独立贡献
    - 经过实验验证，直接令 $f := \text{SiLU}$ 反而产生了所有层中最低的近似误差
    - 最终，结合门控机制和输出投影，得到 $(i,j)$ 贡献向量：$T_i(\boldsymbol{x}_j) = \boldsymbol{W}_o^\top (\boldsymbol{Z}_i \odot \boldsymbol{\upsilon}_{i \leftarrow j})$

2. **Mamba-2 分解**：

    - Mamba-2 的 $\boldsymbol{A}$ 矩阵简化为标量乘以单位矩阵，使得分解更为简洁
    - 新增的 GroupNorm 层在推理时可视为关于 $\boldsymbol{u}_i$ 的仿射映射，因此各 token 的贡献可以线性通过
    - 最终分解为：$T_i(\boldsymbol{x}_j) = \boldsymbol{W}_o^\top [\gamma_i(\boldsymbol{u}_i) \boldsymbol{u}_{i \leftarrow j}]$

3. **多种聚合方式**：

    - **LaTIM($\ell_p$)**：使用向量范数衡量贡献大小
    - **LaTIM(ALTI)**：采用上下文混合方法，计算移除某 token 贡献后 $\ell_1$ 范数的变化
    - **LaTIM(ALTI-Logit)**：追踪 token 通过残差流对最终预测的贡献

4. **精确分解策略**：提出去除 SiLU 激活的 Mamba 变体（令 $f$ 为恒等函数），需要重新训练但能实现零近似误差。实验表明该变体在保持任务性能的同时，实现了完全精确的分解。

### 损失函数 / 训练策略

- 精确策略需要重训模型（去除 SiLU），但近似策略（$f := \text{SiLU}$）可以直接应用于预训练模型
- 拷贝任务的模型使用 mimetic 初始化方案从头训练
- 机器翻译模型在 IWSLT17 数据集上微调
- 近似误差实验在 FineWeb-Edu 上进行了持续预训练

## 实验关键数据

### 主实验

**拷贝任务（合成基准）**：

| 方法 | AUC | AP | R@K |
|------|-----|-----|-----|
| Mamba-Attention (M1) | 0.84 | 0.36 | 0.22 |
| MambaLRP (M1) | 0.40 | 0.22 | 0.20 |
| **LaTIM(ALTI) (M1)** | **0.86** | **0.47** | **0.36** |
| Mamba-Attention (M2) | 0.79 | 0.49 | 0.39 |
| **LaTIM($\ell_2$) (M2)** | **0.98** | **0.86** | **0.74** |

**机器翻译 AER（IWSLT17 de→en，GoldAlign）**：

| 方法 | M1-Small | M1-Large | M2-Small | M2-Large |
|------|----------|----------|----------|----------|
| Mamba-Attention | 0.84 | 0.85 | 0.84 | 0.85 |
| LaTIM($\ell_2$) | **0.46** | **0.44** | **0.49** | **0.52** |
| LaTIM(ALTI-Logit) | 0.68 | 0.69 | 0.63 | 0.69 |

### 消融实验

**近似误差分析（不同激活函数）**：

| 激活函数 | 0-16层误差 | 16-32层误差 | AER | COMET |
|---------|-----------|-----------|-----|-------|
| SiLU（默认） | 0.21 | 0.45 | 0.47 | 83.4 |
| SiLU + 持续预训练 | 0.21 | 0.43 | 0.46 | 83.6 |
| ReLU | 0.35 | 0.83 | 0.51 | 82.8 |
| Identity（精确） | **0.00** | **0.00** | **0.46** | 83.3 |

### 关键发现

- LaTIM 在 Mamba-2 拷贝任务上的 R@K 达到 0.74，比 Mamba-Attention 的 0.39 提升了近一倍
- 逐层分析比全局聚合效果更好——翻译对齐用逐层方法时 AER 更低
- 去除 SiLU 的精确策略在不损失性能的情况下实现了零近似误差
- Mamba 在多key检索任务中存在明显缺陷：随着 key 数量增加，准确率急剧下降
- Mamba 对重复单词的关注度会随时间衰减，这解释了其在词频提取任务上的失败

## 亮点与洞察

- **方法的优雅性**：巧妙地将 SSM 的递推计算展开为类似注意力的矩阵形式，使得大量为 Transformer 开发的归因方法可以无缝迁移到 Mamba
- **SiLU 近似的反直觉发现**：直接用 SiLU 作为可加近似函数，竟然比 Taylor 展开等更"正式"的方法误差更低
- **可扩展性强**：LaTIM 不仅适用于 Mamba-1/2，原理上可以推广到 DeltaNet、mLSTM 等其他线性递推架构
- **为 Mamba 的局限性提供了机理解释**：通过可视化揭示了 Mamba 在多 key 检索中的注意力分散问题

## 局限与展望

- 近似分解仍然存在误差，精确版本需要去除 SiLU 并重新训练
- 评估主要集中在拷贝和翻译等 token 交互模式清晰的任务上，在更复杂任务中的可解释性质量有待人类评估验证
- 对于混合架构（attention + SSM），需要额外的适配
- 目前只展示了"看到了什么"，对于"如何改进模型"的实际指导价值还需进一步探索

## 相关工作与启发

- 与 Transformer 的注意力分解方法（Kobayashi et al., 2021; Ferrando et al., 2022, 2023）形成了完美对应
- 补充了 Mamba 理论分析（Vo et al., 2025 关于 token 状态渐近行为；Trockman et al., 2024 的 mimetic 初始化）
- 发现去除 SiLU 的线性 Mamba 变体既可解释又不损失性能，呼应了 Bick et al., 2024 的相关工作

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 SSM 递推重构为 token-to-token 分解是自然但有价值的贡献，精确策略进一步提升了方法的完整性
- 实验充分度: ⭐⭐⭐⭐ 三个不同任务（拷贝、翻译、检索生成），多个模型规模，定量+定性分析丰富
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导清晰，背景介绍循序渐进，图表设计出色
- 价值: ⭐⭐⭐⭐ 为日益流行的 Mamba 架构提供了急需的可解释性工具，具有广泛的实用价值
