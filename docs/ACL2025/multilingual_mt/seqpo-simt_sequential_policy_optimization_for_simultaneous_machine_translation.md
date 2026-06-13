---
title: >-
  [论文解读] SeqPO-SiMT: Sequential Policy Optimization for Simultaneous Machine Translation
description: >-
  [ACL 2025][多语言/翻译][同步机器翻译] 将同步机器翻译（SiMT）建模为多步序列决策问题，提出 SeqPO-SiMT 策略优化框架，融合翻译质量和延迟的奖励信号，在 7B LLM 上实现 SiMT 性能媲美离线翻译的强模型。 同步机器翻译（SiMT）在源文本流式输入的同时实时生成翻译，应用于同声传译等场景…
tags:
  - "ACL 2025"
  - "多语言/翻译"
  - "同步机器翻译"
  - "策略优化"
  - "GRPO"
  - "多步决策"
  - "延迟-质量权衡"
---

# SeqPO-SiMT: Sequential Policy Optimization for Simultaneous Machine Translation

**会议**: ACL 2025  
**arXiv**: [2505.20622](https://arxiv.org/abs/2505.20622)  
**代码**: 无  
**领域**: 多语言翻译  
**关键词**: 同步机器翻译, 策略优化, GRPO, 多步决策, 延迟-质量权衡

## 一句话总结

将同步机器翻译（SiMT）建模为多步序列决策问题，提出 SeqPO-SiMT 策略优化框架，融合翻译质量和延迟的奖励信号，在 7B LLM 上实现 SiMT 性能媲美离线翻译的强模型。

## 研究背景与动机

同步机器翻译（SiMT）在源文本流式输入的同时实时生成翻译，应用于同声传译等场景。基于 LLM 的 SiMT 方法通常用部分翻译数据做 SFT，但存在以下问题：

**SFT 数据质量差**：部分翻译数据由启发式或注意力对齐工具生成，噪声大

**RLHF 方法不适配**：PPO、DPO 等主要面向单步任务，而 SiMT 是多步序列决策——每步接收新源文本块，决定翻译或等待，前步翻译影响后续结果

**多步依赖复杂**：源文本逐步到达可能存在歧义（如 "bark" 需要后续 "tree" 来消歧），前步误译会级联影响整体质量

**核心动机**：需要一个能建模多步依赖关系、同时优化质量和延迟的策略优化方法。

## 方法详解

### 整体框架

SeqPO-SiMT 将 SiMT 定义为**序列决策过程**：

1. **环境**：将源句 $\mathbf{x}$ 分成 $T$ 个 chunk（每个 $m$ 个词），逐步释放
2. **策略**：LLM 作为策略模型 $\pi_\theta$，每步 $t$ 基于已有源文本和翻译历史生成翻译
3. **采样**：多步采样 $B$ 条完整翻译轨迹
4. **奖励**：在最后一步基于质量+延迟计算奖励
5. **优化**：使用 GRPO 策略梯度优化

每步生成过程：
$$\hat{y}_t \sim \pi_\theta(\hat{y}_t | x_1, \cdots, x_t, \hat{y}_1, \cdots, \hat{y}_{t-1})$$

关键特点：模型自主决定是否翻译和翻译多少内容（翻译长度不固定，可选择等待后续上下文）。

### 关键设计一：融合奖励

翻译质量和延迟有不同量纲且存在权衡，作者设计了归一化+截断的融合奖励：

**质量归一化**：
$$q^i = \frac{\hat{q}^i - \text{mean}(\{\hat{q}^1, \cdots, \hat{q}^B\})}{\text{std}(\{\hat{q}^1, \cdots, \hat{q}^B\})}$$

**延迟归一化+截断**：
$$L^i = \max\left(m, \frac{\hat{L}^i - \text{mean}(\{\hat{L}^1, \cdots, \hat{L}^B\})}{\text{std}(\{\hat{L}^1, \cdots, \hat{L}^B\})}\right)$$

截断阈值 $m$（chunk 大小）防止模型过度拟合延迟指标。

**最终奖励**：$r_T^i = \lambda q^i - L^i$，其中 $\lambda=2$ 平衡质量与延迟。

### 关键设计二：GRPO 优化

选择 GRPO 而非 PPO 的原因：
1. **资源效率**：GRPO 用组内平均作基线，PPO 需要额外的 critic 模型（SiMT 需要质量和延迟两个 critic，内存不可承受）
2. **准确度**：延迟是规则指标，PPO 的神经奖励模型会引入噪声

### 损失函数 / 训练策略

目标函数融合 reward 和 KL 约束：

$$J(\pi_\theta) = \mathbb{E}\sum_{t=1}^{T}\left[r_T - \beta \log \frac{\pi_\theta(\hat{y}_t | x_{1:t}; \hat{y}_{1:t-1})}{\pi_{\text{ref}}(\hat{y}_t | x_{1:t}; \hat{y}_{1:t-1})}\right]$$

训练流程：
1. 先用 40K 样本做 SFT warm-up（构建部分翻译数据）
2. 再用 SeqPO 优化
3. 质量奖励用 COMET，延迟奖励用 AL
4. 骨干模型为 Qwen-2.5-7B，$B=5$，En→Zh 用 $\beta=0.02$，Zh→En 用 $\beta=0.1$

## 实验关键数据

### 主实验

Zh→En SiMT 详细结果（Table 2 摘选，低延迟设置）：

| 数据集 | 方法 | BLEURT↑ | COMET↑ | GPT-4↑ | AL↓ |
|--------|------|---------|--------|--------|-----|
| REALSI | SFT | 64.14 | 83.49 | 83.24 | 15.10 |
| REALSI | SFT+wait-k | 59.37 | 79.60 | 78.90 | 16.75 |
| REALSI | **SeqPO-SiMT** | **65.93** | **84.23** | **85.49** | **14.14** |
| NEWS | SFT | 65.01 | 84.34 | 86.02 | 10.18 |
| NEWS | **SeqPO-SiMT** | **66.67** | **85.17** | **87.67** | **9.29** |

En→Zh SiMT 结果（Table 3 摘选，低延迟设置）：

| 数据集 | 方法 | BLEURT↑ | COMET↑ | GPT-4↑ | AL↓ |
|--------|------|---------|--------|--------|-----|
| MUSTC | SFT | 65.84 | 86.75 | 91.84 | 5.71 |
| MUSTC | **SeqPO-SiMT** | **66.76** | **87.55** | **92.10** | **5.00** |
| NEWS | SFT | 61.12 | 85.54 | 90.99 | 5.02 |
| NEWS | **SeqPO-SiMT** | **63.37** | **87.41** | **91.63** | **4.43** |

平均 COMET 提升：低延迟 +1.3，高延迟 +1.25（COMET 提升 1 分被认为是显著改进）。NEWS En→Zh 上：COMET +1.13 同时 AL -6.17。

### 与离线翻译的对比

SeqPO-SiMT 的 SiMT 结果 vs 强模型的离线翻译结果（Table 4 摘选）：

| 模型 | 模式 | COMET |
|------|------|-------|
| Qwen2.5-7B-Instruct | 离线 | 86.49 |
| LLaMA3-8B-Instruct | 离线 | 83.78 |
| SFT | 离线 | 86.94 |
| **SeqPO-SiMT** | **SiMT** | **87.55** |

SeqPO-SiMT 的 SiMT（只看到部分源文本）成绩超越了多个模型的离线翻译（看到完整源文本）。

### 消融实验

**训练动态分析**（Figure 4a）：训练过程中 AL 先快速下降（模型先学会减少延迟），然后 COMET 逐步提升并稳定。模型先拟合简单的延迟目标，再优化翻译质量。

**仅优化质量的对比**（Figure 4b）：SeqPO-SiMT 在相同质量下有更低延迟，证明融合奖励确实实现了质量和延迟的同时改善，而非此消彼长。

### 关键发现

1. 传统 RLHF（PPO/DPO）无法有效建模 SiMT 的多步依赖
2. 归一化+截断的奖励融合策略成功平衡了质量和延迟
3. GRPO 比 PPO 更适合 SiMT 场景（资源效率+指标准确性）
4. SiMT 在足够好的策略优化下可以达到离线翻译水平

## 亮点与洞察

- **多步 RL 建模 SiMT**：将 SiMT 从 SFT 数据质量问题转化为策略优化问题，绕开了对齐工具的局限
- **奖励设计精巧**：归一化统一量纲，截断防止过拟合延迟，$\lambda$ 控制权衡
- **GRPO 选择合理**：绕开了 PPO 双 critic 的内存问题，利用了延迟作为规则指标的特性
- **实验结果令人印象深刻**：SiMT（部分上下文）竟然能媲美离线翻译（完整上下文）

## 局限与展望

- 仅在中英双语上验证，未测试更多语言对
- chunk size $m$ 是固定超参数，自适应分块可能更优
- 训练需要 reference 翻译来计算 COMET 奖励，限制了无监督场景的应用
- 推理延迟和吞吐量未报告，多步采样可能影响训练效率
- wait-k 作为基线较弱，缺少与更先进的 SiMT 方法（如 adaptive wait）的对比

## 相关工作与启发

- **GRPO / DeepSeek-R1** (Shao et al., 2024; DeepSeek-AI, 2025)：SeqPO 的优化方法基础
- **SiMT with LLM** (Cheng et al., 2024; Koshkin et al., 2024)：LLM-based SiMT 的 SFT 范式，SeqPO 在此基础上加入策略优化
- **Wait-k** (Ma et al., 2019)：经典 SiMT 策略，本文的基线之一

## 评分

- **新颖性**: 4/5 — 将多步 RL 引入 SiMT 的 LLM 优化，视角新颖
- **技术深度**: 4/5 — 奖励设计、优化选择、训练策略都有深思熟虑
- **实验充分度**: 4/5 — 6 个数据集、多指标、离线对比、训练动态分析
- **实用价值**: 3/5 — 方法有效但训练成本较高，未开源代码

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech](../../ACL2026/multilingual_mt/hierarchical_policy_optimization_for_simultaneous_translation_of_unbounded_speec.md)
- [\[ACL 2025\] LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline](llms_can_achieve_high-quality_simultaneous_machine_translation_as_efficiently_as.md)
- [\[ACL 2025\] Machine Translation Models are Zero-Shot Detectors of Translation Direction](machine_translation_models_are_zero-shot_detectors_of_translation_direction.md)
- [\[ACL 2025\] Exploring In-context Example Generation for Machine Translation](exploring_in-context_example_generation_for_machine_translation.md)
- [\[ACL 2025\] Exploring In-Image Machine Translation with Real-World Background](exploring_in-image_machine_translation_with_real-world_background.md)

</div>

<!-- RELATED:END -->
