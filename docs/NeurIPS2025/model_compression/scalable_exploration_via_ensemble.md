---
title: >-
  [论文解读] Ensemble++: Scalable Exploration via Ensemble
description: >-
  [NeurIPS 2025][模型压缩][Thompson Sampling] 提出 Ensemble++，通过共享因子矩阵的增量更新机制，仅需 $\Theta(d\log T)$ 的集成大小即可实现与精确 Thompson Sampling 相当的遗憾界，并自然扩展到非线性/神经网络场景。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "Thompson Sampling"
  - "集成采样"
  - "线性Bandit"
  - "探索-利用"
  - "近似后验"
---

# Ensemble++: Scalable Exploration via Ensemble

**会议**: NeurIPS 2025  
**arXiv**: [2407.13195](https://arxiv.org/abs/2407.13195)  
**代码**: [https://github.com/szrlee/Ensemble_Plus_Plus](https://github.com/szrlee/Ensemble_Plus_Plus)  
**领域**: 模型压缩  
**关键词**: Thompson Sampling, 集成采样, 线性Bandit, 探索-利用, 近似后验

## 一句话总结
提出 Ensemble++，通过共享因子矩阵的增量更新机制，仅需 $\Theta(d\log T)$ 的集成大小即可实现与精确 Thompson Sampling 相当的遗憾界，并自然扩展到非线性/神经网络场景。

## 研究背景与动机
Thompson Sampling (TS) 是序贯决策中平衡探索与利用的经典贝叶斯方法，但在高维或非共轭（如神经网络）设置中，精确的后验采样计算代价极高。集成采样（Ensemble Sampling）通过维护 $M$ 个模型副本来近似 TS，但理论上要达到最优遗憾界需要 $M = \Omega(T \cdot |\mathcal{X}|)$ 的集成大小（Qin et al., 2022），在长时间尺度或大动作空间下完全不可行。

**核心矛盾**：如何用一个实际可行的小集成尺寸来逼近 TS 的后验采样，同时保持近最优的遗憾界？

**本文切入角度**：设计一种新的"共享因子集成"架构，通过随机线性组合机制将 $M$ 个集成方向压缩为对后验协方差矩阵的近似表示，从根本上降低了集成大小需求。

## 方法详解

### 整体框架
Ensemble++ 维护一个共享矩阵因子 $\mathbf{A}_t \in \mathbb{R}^{d \times M}$，通过增量更新近似后验协方差的平方根 $\Sigma_t^{1/2}$。在动作选择时，通过随机线性组合 $\theta_t(\zeta_t) = \mu_{t-1} + \mathbf{A}_{t-1}\zeta_t$ 生成"伪后验样本"，而非从 $M$ 个独立模型中随机选一个。

### 关键设计

1. **共享因子增量更新**：每步仅需 $O(d^2 M)$ 更新——观测奖励后，更新均值 $\mu_t$ 和集成矩阵：
    $\mathbf{A}_t = \Sigma_t(\Sigma_{t-1}^{-1}\mathbf{A}_{t-1} + X_t \mathbf{z}_t^\top)$
   其中 $\mathbf{z}_t \in \mathbb{R}^M$ 是扰动向量。这避免了从头重训或大规模矩阵分解。

2. **随机线性组合采样**：动作选择时，从分布 $P_\zeta$ 中采样参考向量 $\zeta_t$，通过 $\theta_t = \mu_{t-1} + \mathbf{A}_{t-1}\zeta_t$ 构造近似后验样本。不同于传统集成采样"随机选一个模型"，这里对所有列做线性组合，大幅提升信息利用效率。

3. **对称化回归目标**：将基础参数和集成参数统一到一个对称化回归目标中：
    $L(\theta; D, f) = \sum_{m=1}^M \sum_{s \in D} \sum_{\beta \in \{\pm 1\}} (Y_s + \beta \mathbf{z}_{s,m} - f(X_s, \beta e_m))^2 + \lambda\|\theta\|^2$
   在线性情况下有闭合解，在神经网络情况下通过 SGD 求解，实现了理论到实践的无缝桥接。

4. **神经网络扩展**：用可学习的神经特征提取器 $h(x;w)$ 替代线性特征，保持相同的增量更新原则。通过 FIFO 缓冲区和固定 SGD 步数保证恒定时间更新。

### 损失函数 / 训练策略
- 线性情况：共享因子矩阵 $\mathbf{A}_t$ 的闭合解更新
- 神经情况：对称化损失 + SGD，FIFO 缓冲区容量 $C$，每步 $G$ 次梯度更新

## 实验关键数据

### 主实验

| 场景 | 指标 | Ensemble++ | Ensemble+ | EpiNet | 说明 |
|------|------|-----------|-----------|--------|------|
| Quadratic Bandit | 遗憾 | 次线性收敛 | 线性遗憾 | 线性遗憾 | 非线性奖励下优势显著 |
| Neural Bandit | 准确率 | 最高 | 次优 | 次优 | 2层MLP设置 |
| UCI Shuttle | 准确率 | 最优 | 次优 | 次优 | 真实数据集 |
| Hate Speech (GPT-2) | 准确率 | +5% vs Ensemble+ | 基线 | 不适用 | Transformer规模验证 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 集成大小 $M$ vs 维度 $d$ | $M$ 与 $d$ 呈线性关系 | 验证 $M = \Theta(d\log T)$ 理论 |
| $M$ vs 动作集 $|\mathcal{X}|$ | 基本无关 | 与动作集大小解耦 |
| 高斯参考分布 vs 坐标分布 | 高斯更优 | 连续分布的 $\rho/p$ 比值更小 |
| 缓冲区大小 $C$ | 小缓冲区即可 | 不需要存储全部历史 |

### 关键发现
- Ensemble++ 仅需 $M=8$ 即可匹配精确 TS 的性能，仅需传统集成采样一半的计算量
- 集成大小与维度线性相关，与动作空间大小无关，验证了 $M = \Theta(d\log T)$ 的理论预测
- 在 GPT-2 级别的 Transformer 上也能有效工作，展示了向大模型扩展的可行性

## 亮点与洞察
- **理论突破**：首次在线性 Bandit 中证明增量集成更新可以 $\Theta(d\log T)$ 大小达到 TS 级遗憾界，相比 Qin et al. 的 $\Omega(|\mathcal{X}|T)$ 实现了指数级缩减
- **统一性**：同一算法无需修改即可处理紧致/有限动作集 × 不变/时变上下文四种组合
- **Sequential JL 引理**：提出适应性数据收集下的序列 Johnson-Lindenstrauss 变体，解决了标准 JL 要求独立投影的限制

## 局限与展望
- 神经扩展缺乏严格的理论保证，仅有线性情况的理论分析
- 计算复杂度 $O(d^3\log T)$ 相比精确 TS 的 $O(d^3)$ 有额外 $\log T$ 因子
- 未在真正大规模 LLM agent 场景中验证，仅用了 GPT-2 level

## 相关工作与启发
- **vs Ensemble+ (Osband et al., 2019)**: Ensemble++ 无需大集成或重训，通过共享因子更新实现更优的遗憾-计算权衡
- **vs EpiNet (Osband et al., 2023)**: 两者都用架构修改注入不确定性，但 Ensemble++ 有线性情况的理论保证
- **vs LMC-TS (Xu et al., 2022)**: LMC-TS 每步 $O(d^2T)$，Ensemble++ 为 $O(d^2M)$，在长horizon下优势显著

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 共享因子集成架构 + Sequential JL 引理是全新的技术贡献
- 实验充分度: ⭐⭐⭐⭐ 线性和非线性 Bandit 覆盖全面，但大规模场景偏少
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，从线性到非线性的扩展逻辑自然
- 价值: ⭐⭐⭐⭐⭐ 解决了集成采样的核心瓶颈问题，为 LLM agent 探索提供基础框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning](../../CVPR2025/model_compression/tripartite_weight-space_ensemble_for_few-shot_class-incremental_learning.md)
- [\[ICLR 2026\] FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning](../../ICLR2026/model_compression/flyprompt_brain-inspired_random-expanded_routing.md)
- [\[NeurIPS 2025\] GraSS: Scalable Data Attribution with Gradient Sparsification and Sparse Projection](grass_scalable_data_attribution_with_gradient_sparsification_and_sparse_projecti.md)
- [\[ACL 2025\] Pre-training Distillation for Large Language Models: A Design Space Exploration](../../ACL2025/model_compression/pre-training_distillation_for_large_language_models_a_design_space_exploration.md)
- [\[ACL 2025\] STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning](../../ACL2025/model_compression/stun_moe_pruning.md)

</div>

<!-- RELATED:END -->
