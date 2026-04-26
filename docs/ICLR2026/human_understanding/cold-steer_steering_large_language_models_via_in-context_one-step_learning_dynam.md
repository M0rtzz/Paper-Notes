---
title: >-
  [论文解读] COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics
description: >-
  [ICLR 2026][人体理解][激活转向] 提出 COLD-Steer，通过近似梯度下降在上下文示例上产生的表征变化来实现无训练的 LLM 激活转向，在仅用 50 分之一样本量的情况下达到 95% 的转向效果。
tags:
  - ICLR 2026
  - 人体理解
  - 激活转向
  - 学习动力学
  - 无训练推理
  - 样本效率
  - 多元对齐
---

# COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics

**会议**: ICLR 2026  
**arXiv**: [2603.06495](https://arxiv.org/abs/2603.06495)  
**代码**: https://github.com/Ksartik/cold-steer  
**领域**: LLM对齐 / 人体理解  
**关键词**: 激活转向, 学习动力学, 无训练推理, 样本效率, 多元对齐

## 一句话总结
提出 COLD-Steer，通过近似梯度下降在上下文示例上产生的表征变化来实现无训练的 LLM 激活转向，在仅用 50 分之一样本量的情况下达到 95% 的转向效果。

## 研究背景与动机

**领域现状**：激活转向（activation steering）可在推理时控制 LLM 行为而无需重训练，分为两类——对比方法（DiffMean/CAA）用正负对的激活差异构造方向向量，参数调优方法（ReFT/BiPO）端到端训练转向参数。

**现有痛点**：对比方法样本效率高但只利用激活层面的信号（不用损失函数），转向精度有限；参数调优方法（ReFT）需要 250-1000 个示例训练，成本高且需多 epoch 调参。

**核心矛盾**：样本效率与转向精度之间存在根本性 trade-off——如何用少量示例、不训练参数，就获得等同于微调的转向效果？

**本文目标**：设计一个 training-free 框架，仅用 10-50 个示例就能高效转向 LLM 行为。

**切入角度**：作者观察到微调时模型表征的变化遵循可分析的模式（学习动力学）。核心洞察是：可以在推理时**模拟**梯度下降对表征的影响，而无需实际更新参数。

**核心 idea**：将激活转向重新定义为"模拟单步梯度下降的学习动力学"——计算上下文示例的梯度会如何改变目标表征，直接将该变化作为转向向量。

## 方法详解

### 整体框架
给定 $N$ 个示例 $\{(\tilde{\mathbf{x}}_i, \tilde{\mathbf{y}}_i)\}$，COLD-Steer 计算模型在这些示例上做单步梯度下降后表征的变化量 $\Delta\mathbf{Z}^*$，然后将该变化量作为激活加法施加到新输入的第 $l$ 层表征上：$\Delta\mathbf{Z}^*(\mathbf{x}) \approx -\frac{\eta}{N} \nabla_\theta \mathbf{Z}(\mathbf{x};\theta) \sum_i \nabla_\theta \mathcal{L}(\mathcal{M}(\tilde{\mathbf{x}}_i), \tilde{\mathbf{y}}_i)$

### 关键设计

1. **COLD-Kernel-Steer（核近似）**:

    - 功能：用核函数近似 eNTK，避免对新输入做反向传播
    - 核心思路：展开梯度的链式法则，引入核函数 $\kappa$：$\Delta\mathbf{Z}^{(\kappa)}(\mathbf{x}) = -\frac{\eta}{N} \sum_i \kappa(\mathbf{Z}(\mathbf{x}), \mathbf{Z}(\tilde{\mathbf{x}}_i)) \nabla_{\mathbf{Z}} \mathcal{L}|_{\mathbf{Z}(\tilde{\mathbf{x}}_i)}$。使用单位核 $\kappa = 1$ 作为近似，利用线性表征假说——同一概念的梯度由共享方向主导
    - 设计动机：对新输入只需 1 次前向传播 + $O(N \cdot d)$ 核相似度计算，极其高效

2. **COLD-FD-Steer（有限差分近似）**:

    - 功能：用有限差分绕过雅可比计算
    - 核心思路：$\Delta\mathbf{Z}^{(fd)} = -\frac{\eta}{\varepsilon N} [\mathbf{Z}(\mathbf{x}; \theta + \varepsilon \sum_i \nabla_\theta \mathcal{L}_i) - \mathbf{Z}(\mathbf{x}; \theta)]$，$\varepsilon = 10^{-6}$。只需 2 次前向传播（原参数 + 微扰参数），但需存储梯度 $O(|\theta|)$
    - 设计动机：完全避免对新输入反向传播，计算成本固定为 2 次前向

3. **统一视角——对比方法是特例**:

    - DiffMean 等价于 COLD-Kernel 使用特定损失函数 $\mathcal{L} = -\sum_i \|\mathbf{Z}(\tilde{\mathbf{x}}_i \oplus \tilde{\mathbf{y}}_i^+) - \mathbf{Z}(\tilde{\mathbf{x}}_i \oplus \tilde{\mathbf{y}}_i^-)\|^2$ 和单位核
    - RepE/ICV 是 COLD-Kernel 的 PCA 降维近似

### 损失函数 / 训练策略
- 配对设置用 DPO 损失，正样本设置用交叉熵损失
- 超参搜索：$\eta \in \{0.1, 1, 2\}$，$l \in \{10, 15, 20, 30\}$
- 开放生成仅在第一个生成 token 处干预，限制转向的复合效应

## 实验关键数据

### 主实验（CAA 数据集，Llama-2-7b-chat，行为选择准确率）

| 方法 | 协调-AIS | 纠正-HH | 幻觉 | 拒绝 | 谄媚 | 平均排名↓ |
|------|---------|--------|------|------|------|----------|
| Base | 0.28 | 0.62 | 0.70 | 0.62 | 0.80 | 5.14 |
| DiffMean | 0.52 | 0.82 | 0.86 | 0.74 | 0.80 | 4.00 |
| ReFT(vec) | 0.48 | 0.62 | 0.70 | 0.72 | 0.82 | 3.29 |
| **COLD-FD** | **0.90** | **0.86** | **0.96** | **0.98** | 0.86 | **2.00** |
| COLD-Kernel | 0.28 | 0.62 | 0.70 | 0.64 | 0.80 | 4.43 |

### 样本效率对比

| 方法 | 所需样本数 | 平均转向准确率 |
|------|----------|--------------|
| ReFT(mlp) | 250-1000 | ~70-80% |
| DiffMean | 50 | ~65-75% |
| **COLD-FD** | **10-50** | **~85-95%** |
| **COLD-Kernel** | **10-50** | **~75-85%** |

### 关键发现
- **COLD-FD 在 CAA 上平均排名 2.00**（pair 设置），显著优于所有基线
- 使用仅 **50 分之一** 的样本即可达到接近 ReFT 的效果
- 对比方法 DiffMean 被证明是 COLD-Kernel 在特定损失下的特例——统一了对比与梯度方法
- 在 OpinionsQA 多元对齐任务上同样有效，支持少数派观点的适配
- 跨模型家族验证：Qwen-2.5-7B-Instruct 上 COLD-FD 准确率提升最高达 96%；Gemma-2-9B 和 Mistral-7B 上同样有效

### 多元对齐（OpinionsQA，Llama-2-7b-chat）
- COLD-Kernel 在所有人群分组上一致最优，将 Black 群体 KL 散度从 2.43 降至 0.86，Republican 从 2.38 降至 0.97
- TV 距离均降至 0.4 以下，表明核方法更适合保持子群体分布保真度
- COLD-FD 在分布式转向设定下效果不佳，原因仍为开放问题

### 行为生成质量（GPT-5-mini 评判）
- COLD-FD 在 CAA 的 hallucination 任务上从 2.98 提升到 3.32（Llama-2-7b-chat），在 survival-instinct 上从 5.26 提升到 6.20
- COLD-Kernel 偏保守，基本维持 Base 水平，适合不希望大幅改变模型行为的场景

## 亮点与洞察
- **将激活转向重新理解为学习动力学的模拟**极为优雅——不是训练一个转向器，而是直接计算"如果微调了会怎样"
- **理论统一性强**：证明 DiffMean/RepE/ICV 都是 COLD-Kernel 的特例，为现有方法提供了统一的梯度视角
- **COLD-FD 的两次前向传播方案**：完全避免新输入的反向传播，实用性极高

## 局限与展望
- COLD-FD 需存储完整模型梯度 $O(|\theta|)$，对 70B+ 模型内存压力大
- 单位核近似在某些任务上效果不佳（如 Llama-2 上 COLD-Kernel 未提升）
- 仅实验了单层干预，多层协同转向可能更强
- 有限差分的 $\varepsilon$ 选择依赖经验

## 相关工作与启发
- **vs CAA/DiffMean**: COLD-Steer 证明对比方法只用了激活差异信号，未利用损失函数信息
- **vs ReFT**: ReFT 训练 MLP 需几百样本+多 epoch；COLD-Steer 零训练、10 样本即可
- **vs prompt tuning**: COLD-Steer 在激活层面操作，更精细且不受上下文窗口限制

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将激活转向重新定义为学习动力学模拟，理论贡献突出
- 实验充分度: ⭐⭐⭐⭐ 5 个 LLM + 多数据集 + 多元对齐，但消融实验可更深入
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，统一视角有说服力
- 价值: ⭐⭐⭐⭐⭐ 50x 样本效率提升有巨大实用价值，尤其对多元对齐

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Scalable In-Context Q-Learning](scalable_in-context_q-learning.md)
- [\[ICLR 2026\] TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](timeomni-1_incentivizing_complex_reasoning_with_time_series_in_large_language_mo.md)
- [\[AAAI 2026\] Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](../../AAAI2026/human_understanding/anti-adversarial_learning_desensitizing_prompts_for_large_la.md)
- [\[ICLR 2026\] EgoHandICL: Egocentric 3D Hand Reconstruction with In-Context Learning](egohandicl_egocentric_3d_hand_reconstruction_with_in-context_learning.md)
- [\[AAAI 2026\] 10 Open Challenges Steering the Future of Vision-Language-Action Models](../../AAAI2026/human_understanding/10_open_challenges_steering_the_future_of_vision-language-ac.md)

<!-- RELATED:END -->
