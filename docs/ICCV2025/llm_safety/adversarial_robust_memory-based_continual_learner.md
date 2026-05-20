---
title: >-
  [论文解读] Adversarial Robust Memory-Based Continual Learner
description: >-
  [ICCV 2025][LLM安全][持续学习] 揭示持续学习与对抗训练结合时的双重挑战（加速遗忘 + 梯度混淆），提出抗遗忘 Logit 校准（AFLC）和鲁棒感知经验回放（RAER）两个即插即用模块…
tags:
  - "ICCV 2025"
  - "LLM安全"
  - "持续学习"
  - "对抗鲁棒性"
  - "经验回放"
  - "Logit 校准"
  - "梯度混淆"
---

# Adversarial Robust Memory-Based Continual Learner

**会议**: ICCV 2025  
**arXiv**: [2311.17608](https://arxiv.org/abs/2311.17608)  
**代码**: 无  
**领域**: Continual Learning / Adversarial Robustness  
**关键词**: 持续学习, 对抗鲁棒性, 经验回放, Logit 校准, 梯度混淆

## 一句话总结

揭示持续学习与对抗训练结合时的双重挑战（加速遗忘 + 梯度混淆），提出抗遗忘 Logit 校准（AFLC）和鲁棒感知经验回放（RAER）两个即插即用模块，在 Split-CIFAR10/100 和 Split-Tiny-ImageNet 上有效提升对抗鲁棒性达 8.13%。

## 研究背景与动机

持续学习使模型能够从非独立同分布的数据流中不断学习，但现有方法在面对对抗样本时非常脆弱。将持续学习与对抗训练直接结合面临两个核心问题：

**挑战一：对抗训练加速遗忘**
- 对抗样本会增大当前任务数据对过去类别的负梯度，加剧灾难性遗忘
- 对抗样本 $\tilde{x}_t$ 相比干净样本 $x_t$，在过去类别上的 logit 更高：$h_\theta(\tilde{x_t})_p > h_\theta(x_t)_p$
- 这导致在过去类别方向上产生更大的负梯度，加速知识遗忘

**挑战二：持续学习加剧梯度混淆**
- 有限的回放数据导致模型过拟合记忆中的样本
- 梯度方向与联合训练相比严重偏差（通过余弦相似度验证）
- 这是一种特殊的"碎片梯度"现象，只在持续学习设置下出现

此前 Chen et al. 尝试通过大量无标签数据来缓解，但本文旨在**不使用额外数据**的情况下解决这一问题。

## 方法详解

### 整体框架

在基于记忆的持续学习框架（如 ER）上叠加对抗训练，额外引入两个模块：
1. **AFLC（Anti-Forgettable Logit Calibration）**：在 softmax 层前对 logit 做任务顺序感知的校准
2. **RAER（Robustness-Aware Experience Replay）**：基于对抗鲁棒性难度因子选择回放数据

两个模块均以即插即用方式集成，兼容 ER、DER、DER++、X-DER 等持续学习算法以及 Vanilla AT、TRADES 等对抗训练方法。

### 关键设计

**AFLC（抗遗忘 Logit 校准）**：
- 将分类头的输出 logit $h_\theta(x)$ 分为过去类 (p)、当前类 (c)、未来类 (u) 三组
- 对每个类别 $i$ 施加校准值 $v_i$：$h_\theta^{lc}(\tilde{x})_i = h_\theta(\tilde{x})_i - v_i$
- 关键约束：$v_p > v_c$，即对过去类别的校准值更大
- 效果：减小当前数据对抗样本对过去类别的负梯度，增大过去数据对抗样本对当前类别的负梯度
- 具体实现：$v_i = -\log(\frac{n_i}{\sum n_j}) - \alpha_i$，其中 $n_i$ 是类别 $i$ 在记忆+当前数据中的样本数
- **未来类先验调整 (FP)**：对未来类使用已有类别 $v$ 的均值，避免未来类 logit 被隐式增大
- 推理时不做 logit 校准

**RAER（鲁棒感知经验回放）**：
- 定义鲁棒性难度因子 $k$：PGD-10 攻击过程中成功攻击的步数
- $k$ 越大表示样本越脆弱、越接近决策边界
- 设置阈值 $\rho$，仅选择 $k < \rho$ 的样本存入记忆
- 过滤掉过拟合当前分类边界的困难样本，优先保存鲁棒且有代表性的数据分布
- 与传统持续学习数据选择策略（偏好边界附近样本）相反

### 损失函数 / 训练策略

以 ER+AT 为例的训练目标：

$$\mathcal{L}_t = \text{CE}(f_\theta^{lc}(\tilde{x_t}), y_t) + \text{CE}(f_\theta^{lc}(\tilde{x}_\mathcal{M}), y_\mathcal{M})$$

其中对抗样本通过 PGD 生成，logit 层通过 AFLC 校准，记忆数据通过 RAER 选择。

训练流程：
1. 采样当前任务 batch + 记忆 batch
2. PGD 生成对抗样本并记录鲁棒性因子 $K$
3. AFLC 校准 logit 后计算损失并更新参数
4. RAER 根据 $K$ 筛选当前任务数据存入记忆

超参设置：$\alpha = 3.5$，$\rho = 5$（PGD-10 中成功攻击步数阈值）。

## 实验关键数据

### 主实验

Split-CIFAR10 数据集，Buffer=200，ER 为基础框架的系统性分析：

| 方法 | w/ AT | Class-IL Clean FAA↑ | Class-IL Adv FAA↑ | Task-IL Clean FAA↑ | Task-IL Adv FAA↑ |
|------|:-----:|-------|--------|-------|--------|
| ER | ✗ | 48.80 | 0.27 | 92.89 | 0.01 |
| ER+AT | ✓ | 28.18 | 17.86 | 84.49 | 44.30 |
| ER+AT+Ours | ✓ | **提升** | **+8.13%** | **提升** | **提升** |
| Joint (上界) | ✓ | 79.33 | 50.93 | 94.80 | 74.63 |

关键观察：直接结合 AT，clean FAA 从 48.80 降到 28.18（-20.62），但 adversarial FAA 从 0.27 升到 17.86。

### 消融实验

各模块贡献（ER+AT, Buffer=200, Split-CIFAR10）：

| 配置 | CRD↓ | FRI↓ | RRD↓ |
|------|------|------|------|
| ER+AT (baseline) | 14.51 | 12.91 | 31.70 |
| +AFLC | 降低 | 降低 | 降低 |
| +RAER | 降低 | 降低 | 降低 |
| +AFLC+RAER | **最低** | **最低** | **最低** |

- CRD（Clean Relative Decrease）：加 AT 后 clean 性能下降幅度
- FRI（Forgetting Relative Increase）：clean 数据遗忘增幅
- RRD（Robust Relative Decrease vs. Joint）：鲁棒性与联合模型的差距

四种持续学习方法（ER、DER、DER++、X-DER）与 AT 结合后对比：
- ER+AT 在 class-IL 设置下表现最佳（尤其 buffer=200）
- X-DER 过度抑制新任务学习能力
- 更大 buffer（5120）显著改善所有指标

### 关键发现

1. **对抗训练不适合直接与持续学习结合**：所有方法的 CRD 均为正值，clean 性能下降显著
2. **梯度混淆与 buffer 大小负相关**：buffer 从 200 增至 5120，梯度方向余弦相似度显著增加
3. **RAER 的反直觉设计有效**：与传统做法（保存困难样本）相反，保存鲁棒样本更有利于持续对抗学习
4. AFLC 可替代 X-DER 的 logit masking，获得更好性能——因为 logit masking 过于极端（$v_p = +\infty$）会抑制新知识学习

## 亮点与洞察

1. **问题揭示比解决方案更有价值**：首次系统分析持续学习+对抗训练的双重干扰机制，为后续研究指明方向
2. **梯度混淆的新视角**：不同于传统梯度混淆（来自防御方法），这里是有限回放数据造成的结构性梯度偏差
3. **即插即用设计**：AFLC 和 RAER 可自由组合进不同的持续学习算法和对抗训练方法
4. **不需要额外数据**：与之前工作不同，完全依赖现有训练数据和记忆

## 局限与展望

1. 实验主要基于小规模数据集（CIFAR-10/100、Tiny-ImageNet），大规模场景验证缺失
2. 仅考虑了 $\ell_\infty$ 范数的 PGD 攻击，对其他攻击类型的鲁棒性未评估
3. AFLC 的校准值 $v_i$ 依赖人工设置的超参数 $\alpha$，自适应机制待探索
4. RAER 的阈值 $\rho$ 是固定的，不同任务可能需要不同阈值
5. 未讨论预训练模型（如 ViT、CLIP）上的应用前景

## 相关工作与启发

- **ER（Experience Replay）**：最简洁的记忆回放方法，作为本文的主要载体
- **TRADES**：平衡 clean accuracy 和 robustness 的对抗训练方法，本文扩展了其在持续学习中的使用
- **X-DER**：使用 logit masking 缓解遗忘，本文 AFLC 是其更灵活的推广
- 启发：对抗鲁棒性+持续学习的交叉领域仍有大量未解决问题，尤其是在更复杂的现实场景中

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 3.5 |
| 技术深度 | 4 |
| 实验充分性 | 4 |
| 写作质量 | 3.5 |
| 实用价值 | 3.5 |
| 总评 | 3.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Bias in the Mirror: Are LLMs' Opinions Robust to Their Own Adversarial Attacks](../../ACL2025/llm_safety/bias_in_the_mirror_are_llms_opinions_robust_to_their_own_adversarial_attacks.md)
- [\[CVPR 2025\] TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models](../../CVPR2025/llm_safety/tapt_test-time_adversarial_prompt_tuning_for_robust_inference_in_vision-language.md)
- [\[NeurIPS 2025\] Finding Structure in Continual Learning](../../NeurIPS2025/llm_safety/finding_structure_in_continual_learning.md)
- [\[ACL 2025\] Ewe: Improving Factuality with Explicit Working Memory](../../ACL2025/llm_safety/improving_factuality_with_explicit_working_memory.md)
- [\[ICML 2025\] BECAME: BayEsian Continual Learning with Adaptive Model MErging](../../ICML2025/llm_safety/became_bayesian_continual_learning_with_adaptive_model_merging.md)

</div>

<!-- RELATED:END -->
