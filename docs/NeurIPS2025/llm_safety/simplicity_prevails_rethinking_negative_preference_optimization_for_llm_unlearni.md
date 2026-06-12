---
title: >-
  [论文解读] Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning
description: >-
  [NeurIPS 2025][LLM安全][LLM unlearning] 发现 NPO（负偏好优化）中的参考模型偏差导致遗忘数据的优化功率分配不均和早期梯度权重平滑失效，提出 SimNPO 通过去除参考模型依赖并采用长度归一化奖励，在 TOFU 上将 FQ 从 0.79 提升至 0.99…
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "LLM unlearning"
  - "negative preference optimization"
  - "SimNPO"
  - "reference model bias"
  - "length normalization"
---

# Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning

**会议**: NeurIPS 2025  
**arXiv**: [2410.07163](https://arxiv.org/abs/2410.07163)  
**代码**: [GitHub](https://github.com/OPTML-Group/Unlearn-Simple)  
**领域**: LLM对齐  
**关键词**: LLM unlearning, negative preference optimization, SimNPO, reference model bias, length normalization

## 一句话总结
发现 NPO（负偏好优化）中的参考模型偏差导致遗忘数据的优化功率分配不均和早期梯度权重平滑失效，提出 SimNPO 通过去除参考模型依赖并采用长度归一化奖励，在 TOFU 上将 FQ 从 0.79 提升至 0.99，在所有基准上一致优于 NPO。

## 研究背景与动机

**LLM 遗忘的需求**：消除 LLM 中版权、隐私或有害内容的影响，避免代价高昂的重训练

**GA（梯度上升）的问题**：缺乏优化发散控制，容易导致模型崩溃

**NPO 的进步与局限**：
   - NPO 将遗忘数据作为 DPO 中的负样本，提供有界的遗忘目标和自适应梯度权重平滑
   - 但作者首次发现 NPO 存在**参考模型偏差**——依赖参考模型评估遗忘效果，可能误导优化

## 方法详解

### NPO 的参考模型偏差分析

**限制 L1：优化功率分配不均**

NPO 梯度权重为 $w_\theta(x,y) = \frac{2\pi_\theta(y|x)^\beta}{\pi_\theta(y|x)^\beta + \pi_{\text{ref}}(y|x)^\beta}$。

对于强记忆化数据（$\pi_{\text{ref}}$ 高），权重反而更小，分配更少优化功率。但强记忆化数据恰恰更难遗忘，应获得更多功率。

弱记忆化数据获得过多功率 → 可能过度遗忘 → 浪费优化预算。

**实例验证**：
- 强记忆化 vs 弱记忆化数据：NPO 在强记忆化数据上 FQ 接近 0
- 短响应 vs 长响应数据：NPO 对短响应遗忘效果差（FQ=0.58），长响应好（FQ=0.81）

**限制 L2：早期梯度权重平滑失效**

初始化时 $\theta \approx \theta_{\text{ref}}$，导致 $w_\theta(x,y) \approx 1$，NPO 在早期等价于 GA，可能造成大幅效用下降。

### SimNPO 方法

用 SimPO（无参考模型偏好优化）的长度归一化奖励替代 NPO 的参考模型比较：

$$\ell_{\text{SimNPO}}(\theta) = \mathbb{E}_{(x,y)\in\mathcal{D}_f}\left[-\frac{2}{\beta}\log\sigma\left(-\frac{\beta}{|y|}\log\pi_\theta(y|x) - \gamma\right)\right]$$

其中 $|y|$ 是响应长度，$\gamma \geq 0$ 是奖励边际参数（默认 $\gamma=0$）。

### SimNPO 梯度分析

$$\nabla_\theta \ell_{\text{SimNPO}} = \mathbb{E}\left[\frac{2(\pi_\theta(y|x))^{\beta/|y|}}{1+(\pi_\theta(y|x))^{\beta/|y|}} \cdot \frac{1}{|y|} \cdot \nabla_\theta \log\pi_\theta(y|x)\right]$$

**优势 (a)**：长度归一化 $1/|y|$ 使得长响应获得更少权重，避免不均分配。极端情况 $\beta \to 0$ 时退化为加权 GA：$\mathbb{E}[1/|y| \cdot \nabla_\theta \log\pi_\theta]$，而非 NPO 的纯 GA。

**优势 (b)**：权重 $w'_\theta(x,y) < 2/|y|$，分布依赖数据特性而非参考模型。不存在 NPO 的早期 $w \approx 1$ 问题。

### 损失函数 / 训练策略

完整 SimNPO 方法：$\min_\theta \ell_{\text{SimNPO}}(\theta) + \lambda \mathbb{E}_{(x,y) \in \mathcal{D}_r}[-\log\pi_\theta(y|x)]$

遗忘损失 + 保留集交叉熵正则化。

## 实验关键数据

### TOFU Forget05（LLaMA2-7B-chat）

| 方法 | FQ↑ | MU↑ | 说明 |
|------|-----|-----|------|
| Original | 0.00 | 0.62 | 未遗忘 |
| Retrain | 1.00 | 0.62 | 金标准 |
| GA | ~0 | 0.00 | 模型崩溃 |
| GradDiff | ~0 | 0.56 | 不充分遗忘 |
| IDK | ~0 | 0.57 | 不充分遗忘 |
| NPO | 0.79 | 0.57 | 基线 |
| **SimNPO** | **0.99** | **0.58** | 最优 |

### MUSE News（LLaMA2-7B）

| 方法 | PrivLeak (→0) | KnowMem $\mathcal{D}_r$↑ |
|------|---------------|--------------------------|
| NPO | 108.91 | 37.58 |
| **SimNPO** | **72.93** | **39.65** |
| Retrain | 0.00 | 53.79 |

SimNPO 在所有指标上更接近 Retrain。

### 强记忆化 vs 弱记忆化数据

| 数据类型 | NPO FQ | SimNPO FQ | Retrain FQ |
|----------|--------|-----------|------------|
| 强记忆化 | ≈0 | 显著提升 | 基准 |
| 弱记忆化 | 过度遗忘 | 适度遗忘 | 基准 |

SimNPO 分布更接近 Retrain，验证了参考模型偏差的存在。

### 梯度权重分析

| 时期 | NPO w | SimNPO w' |
|------|-------|-----------|
| Epoch 1 | ≈1（均匀） | 1/\|y\| 调制 |
| Epoch 2-3 | 开始分化 | 优先短响应 |
| Epoch 10 | 充分分化 | 趋于均匀 |

SimNPO 从一开始就根据数据难度分配不同权重。

### 重学习攻击鲁棒性

- SimNPO 在随机重学习和最短响应重学习攻击下都保持更高 FQ
- NPO 对最短响应重学习特别脆弱
- SimNPO 的 FQ 下降速度更慢

### Markov Chain 合成实验

验证了两个核心优势：
1. SimNPO 在不同长度数据上遗忘更平衡
2. SimNPO 在不同记忆化程度数据上遗忘更平衡
NPO 在弱记忆化数据上过度遗忘，强记忆化数据上不足遗忘。

## 亮点与洞察

1. **参考模型偏差的发现**：首次识别 NPO 中的这个根本问题，通过扰动参考模型实验和分数据类型分析验证
2. **简化即改进**：去除参考模型依赖反而更好，长度归一化提供了更合理的数据感知调制
3. **理论与合成实验支撑**：Markov Chain 合成实验精确控制遗忘难度，清晰验证假说
4. **重学习攻击鲁棒性**：SimNPO 对短响应重学习攻击的鲁棒性解释了长度归一化的价值

## 局限与展望

1. SimNPO 仍依赖促进发散来实现遗忘，不可避免地损失部分效用
2. 在知识遗忘场景（如 WMDP）中平衡遗忘效果与效用保留仍具挑战
3. SimNPO 的理论保证尚待建立
4. $\gamma$ 参数选择影响遗忘条件严格程度，需要任务相关调优
5. 长度归一化对所有场景是否最优仍待验证

## 相关工作与启发

- **NPO**：SimNPO 的直接改进对象，保留其有界损失优势但去除参考模型依赖
- **SimPO**：偏好优化中去除参考模型的方法，SimNPO 将其思想迁移到遗忘场景
- **DPO/GA/GradDiff**：遗忘优化的其他基线，GA 无发散控制，GradDiff 遗忘不足
- **启发**：偏好优化与遗忘优化的联系值得更深入探索

## 评分

- 新颖性: ⭐⭐⭐⭐ SimpleNPO 的设计洞察深刻（参考模型偏差），但方法本身是 SimPO 到遗忘的迁移
- 实验充分度: ⭐⭐⭐⭐⭐ TOFU+MUSE+WMDP 三基准，合成实验，重学习攻击，梯度分析
- 写作质量: ⭐⭐⭐⭐⭐ 问题分析从直觉到数学到实验层层深入
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 遗忘实践有直接指导价值，SimNPO 简单易用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning](dragon_guard_llm_unlearning_in_context_via_negative_detection_and_reasoning.md)
- [\[ACL 2026\] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization](../../ACL2026/llm_safety/cipo_counterfactual_unlearning_for_large_reasoning_models_through_iterative_pref.md)
- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](on_the_sample_complexity_of_differentially_private_policy_optimization.md)
- [\[NeurIPS 2025\] Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM](unlearned_but_not_forgotten_data_extraction_after_exact_unlearning_in_llm.md)
- [\[ICLR 2026\] Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures](../../ICLR2026/llm_safety/rethinking_benign_relearning_syntax_as_the_hidden_driver_of_the_safety_tax.md)

</div>

<!-- RELATED:END -->
