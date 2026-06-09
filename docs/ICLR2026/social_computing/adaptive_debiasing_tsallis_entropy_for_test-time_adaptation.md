---
title: >-
  [论文解读] Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation
description: >-
  [ICLR 2026][社会计算][测试时自适应] 提出将 Tsallis 熵（SE 的广义形式）引入 VLM 的 Test-Time Adaptation，并进一步发展为自适应去偏 Tsallis 熵（ADTE），为每个类别定制去偏参数 $q^l$…
tags:
  - "ICLR 2026"
  - "社会计算"
  - "测试时自适应"
  - "Tsallis entropy"
  - "CLIP"
  - "去偏"
  - "不确定性估计"
---

# Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation

**会议**: ICLR 2026  
**arXiv**: [2602.11743](https://arxiv.org/abs/2602.11743)  
**代码**: [https://github.com/Jinx630/ADTE](https://github.com/Jinx630/ADTE)  
**领域**: 社会计算  
**关键词**: 测试时自适应, Tsallis entropy, CLIP, 去偏, 不确定性估计

## 一句话总结
提出将 Tsallis 熵（SE 的广义形式）引入 VLM 的 Test-Time Adaptation，并进一步发展为自适应去偏 Tsallis 熵（ADTE），为每个类别定制去偏参数 $q^l$，在不引入分布特定超参数的情况下比 Shannon 熵选择更可靠的高置信视图，在 ImageNet 及其 5 个变体和 10 个跨域 benchmark 上均超越 SOTA。

## 研究背景与动机
**领域现状**：TTA（Test-Time Adaptation）方法通过选择高置信增强视图来提升 CLIP 等 VLM 在分布外数据上的表现。代表方法如 TPT、Zero 等都使用 Shannon 熵来度量不确定性并筛选低熵视图。

**现有痛点**：CLIP 在不平衡的网络爬取数据上预训练，导致对头部类别过度自信、对尾部类别自信度不足。Shannon 熵对所有类别使用统一公式 $-p\log p$，无法区分不同类别的偏差程度，导致熵估计本身就是有偏的，进而影响高置信视图的选择质量。

**核心矛盾**：SE 假设概率分布是无偏的（广延性假设），但 CLIP 的预测分布存在系统性偏差（非广延性），SE 无法刻画这种偏差结构。

**本文目标** 如何在 TTA 过程中纠正 VLM 预测偏差对熵估计的影响？

**切入角度**：Tsallis 熵是 Shannon 熵的推广，通过非广延参数 $q$ 可以刻画概率分布间的统计依赖性。当 $q<1$ 时，TE 倾向于选择更可靠的高置信视图。

**核心 idea**：用 Tsallis 熵替代 Shannon 熵做高置信视图选择，并为每个类别自适应计算去偏参数 $q^l$。

## 方法详解

### 整体框架
ADTE 要解决的是「TTA 里用来挑高置信视图的熵本身就有偏」这件事。它把自己定位成 Zero/TPT 等方法中 Shannon 熵的即插即用替代品，不改动 pipeline 其余部分：一张测试图先扩成 N 个增强视图，对每个视图用 ADTE 而非 SE 算一个不确定性分数，挑出分数低（置信高）的那批视图，再聚合它们的预测得到最终结果。换熵这一步带来两处关键改动——熵的函数形式从 Shannon 换成 Tsallis，以及参数 $q$ 从一个全局常数变成逐类别自适应的 $q^l$。

### 关键设计

**1. 用 Tsallis 熵替代 Shannon 熵：换掉对尾部类别敏感的那块计算**

SE 用统一公式 $\mathbf{H}_{SE} = -\sum_l P_l \log P_l$ 度量不确定性，但 $p\log p$ 这一项对接近 0 的小概率特别敏感，而 CLIP 恰恰在尾部类别上预测概率小且有系统偏差，于是熵估计被这些类别带偏。ADTE 改用 Tsallis 熵 $\mathbf{H}_{TE} = \frac{\sum_l P_l^q - 1}{1-q}$，核心是把 $p\log p$ 换成 $p^q$，从而改变对小概率的处理方式。理论上这个替换是自洽的：当 $q \to 1$ 时 TE 退化为 SE（SE 是 TE 的一个特例 / 下界）；当 $q < 1$ 时，TE 挑出的高置信视图具有更高的 Top-K 累积可靠性（TcrK）；而在 $0 < q < 1$ 区间，TE 能自然缓解 VLM 偏差对视图选择的影响，无需显式建模偏差。

**2. 自适应去偏 Tsallis 熵（ADTE）：让每个类别自己决定纠偏力度**

固定一个全局 $q$ 有两个问题：最优 $q$ 随测试分布漂移、手动调不可行；而且头部类别和尾部类别受偏差影响的程度本就不同，一个常数管不了所有类。ADTE 因此为每个类别 $l$ 单独算一个 $q^l$。做法分两步：先维护一个 memory bank，用伪标签近似、Jacobi 迭代求解，估计出各类别的先验概率 $\tilde{p}_l$ 作为偏差度量；再把这些估计的偏差经 min-max 归一化映射到 $[\alpha, \beta] = [0.01, 0.9]$ 区间当作 $q^l$。映射方向是「偏差越大、$q^l$ 越小」，因为更小的 $q$ 对应更强的纠正力度，于是受偏差影响重的类别被更狠地校正，轻的则接近原始 SE。整个 $q^l$ 的估计不需要任何分布特定的超参数调优。

**3. 与 Logit Adjustment 集成：从两个层面叠加去偏**

ADTE 在熵估计层面纠偏，可以和 logit adjustment 这类在 logits 层面纠偏的策略无缝叠加：先用估计出的偏差调整 logits，再用 ADTE 去选高置信视图，两道工序方向一致、互不冲突。整个流程仍然不引入额外训练，也不需要分布特定的超参数。

### 损失函数 / 训练策略
无需训练。ADTE 是纯推理时方法，直接把 TTA pipeline 里的 Shannon 熵换成它即可，memory bank 大小设为每类 10 个样本。

## 实验关键数据

### 主实验（ImageNet + 5 变体，CLIP ViT-B/16）

| 方法 | IN | IN-A | IN-R | IN-K | Average | OOD Avg |
|------|-----|------|------|------|---------|---------|
| CLIP | 68.7 | 50.6 | 77.7 | 48.3 | 61.5 | 59.7 |
| Zero | 70.9 | 64.0 | 80.8 | 50.3 | 66.2 | 65.0 |
| BCA | 70.2 | 61.1 | 80.7 | 50.9 | 65.6 | 64.4 |
| **ADTE** | **71.8** | **65.5** | **81.4** | **53.5** | **67.5** | **66.5** |

### 跨域 benchmark（10 个数据集最高平均性能）

| 指标 | 说明 |
|------|------|
| ADTE 平均准确率 | 10 个跨域 benchmark 上最高平均表现 |
| 模型无关 | 在 ViT-B/16 和 ViT-L/14 上都优于 SOTA |
| Prompt 无关 | 使用手工模板或 CuPL 生成的文本都有效 |

### 关键发现
- TE 当 $q < 1$ 时始终优于 SE（SE 是 TE 在 $q=1$ 的特例），但最优 $q$ 因测试分布而异
- ADTE 通过自适应 $q^l$ 消除了手动调参的需求，在所有测试分布上表现稳健
- 在 ImageNet-K 上提升最大（48.3→53.5），这是分布偏移最严重的变体
- ADTE 可以直接替换任何基于 SE 的 TTA 方法中的熵计算，无需其他修改

## 亮点与洞察
- **Shannon 熵的有偏性被首次系统分析**：在 VLM TTA 中，SE 隐含假设的广延性不成立，这个问题被忽视已久
- **Tsallis 熵作为直接替代品**：理论优雅（SE 是下界）且实际有效，且是即插即用的——任何用 SE 的 TTA 方法都可以直接换成 TE/ADTE
- **自适应参数估计的设计**：利用已有的偏差估计方法（来自 Frolic）转化为 $q^l$，复用了现有工具

## 局限与展望
- Memory bank 大小固定为每类 10 个，在类别极多（如 ImageNet 1000 类）时可能不够
- 偏差估计依赖伪标签质量，早期样本的伪标签可能不准
- 归一化区间 $[\alpha, \beta] = [0.01, 0.9]$ 仍是手动设定的超参数
- 仅在分类任务上验证，检测/分割等密集预测任务未覆盖

## 相关工作与启发
- **vs Zero/TPT**: ADTE 是它们的直接升级——仅替换熵计算即可获得提升，无需改动其他组件
- **vs Frolic**: Frolic 使用 logit adjustment 做偏差校正，ADTE 从熵估计层面做校正，两者互补
- **vs 传统 Tsallis 熵在域适应中的应用**: 以往工作在源域适应中优化 TE 做伪标签，ADTE 首次将其应用到 online TTA 的视图选择

## 评分
- 新颖性: ⭐⭐⭐⭐ Tsallis 熵在 VLM TTA 中的应用是新颖的理论视角
- 实验充分度: ⭐⭐⭐⭐⭐ ImageNet+5变体、10跨域benchmark、两个模型、两种prompt
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，但公式密集
- 价值: ⭐⭐⭐⭐ 即插即用的 SE 替代品，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Scalable Multi-Task Low-Rank Model Adaptation](scalable_multi-task_low-rank_model_adaptation.md)
- [\[ICLR 2026\] Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction](human_or_machine_a_preliminary_turing_test_for_speech-to-speech_interaction.md)
- [\[ACL 2025\] FairSteer: Inference Time Debiasing for LLMs with Dynamic Activation Steering](../../ACL2025/social_computing/fairsteer_inference_time_debiasing_for_llms_with_dynamic_activation_steering.md)
- [\[ICLR 2026\] SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition](sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn.md)
- [\[AAAI 2026\] SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation](../../AAAI2026/social_computing/scenejaileval_a_scenario-adaptive_multi-dimensional_framework_for_jailbreak_eval.md)

</div>

<!-- RELATED:END -->
