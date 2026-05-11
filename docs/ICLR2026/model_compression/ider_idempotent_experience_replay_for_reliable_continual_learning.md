---
title: >-
  [论文解读] IDER: IDempotent Experience Replay for Reliable Continual Learning
description: >-
  [ICLR 2026][模型压缩][持续学习] 将幂等性（idempotence）引入持续学习，通过标准幂等模块和幂等蒸馏模块两个组件强制模型在学习新任务时保持输出自一致性，在提升预测可靠性（降低校准误差）的同时显著减少灾难性遗忘。
tags:
  - ICLR 2026
  - 模型压缩
  - 持续学习
  - 幂等性
  - 经验回放
  - 校准误差
  - 灾难性遗忘
---

# IDER: IDempotent Experience Replay for Reliable Continual Learning

**会议**: ICLR 2026  
**arXiv**: [2603.00624](https://arxiv.org/abs/2603.00624)  
**代码**: [GitHub](https://github.com/YutingLi0606/Idempotent-Continual-Learning)  
**领域**: 模型压缩  
**关键词**: 持续学习, 幂等性, 经验回放, 校准误差, 灾难性遗忘

## 一句话总结
将幂等性（idempotence）引入持续学习，通过标准幂等模块和幂等蒸馏模块两个组件强制模型在学习新任务时保持输出自一致性，在提升预测可靠性（降低校准误差）的同时显著减少灾难性遗忘。

## 研究背景与动机
持续学习面临灾难性遗忘这一核心挑战——模型在学习新任务时迅速丢失旧任务知识。基于回放的方法（如ER、DER++）通过缓冲区存储旧样本来缓解，但这些方法通常过度自信且校准不良（ECE高），尤其对近期任务存在偏见。在医疗、交通等安全关键领域，模型不仅要准确还要"知道自己不知道什么"。

现有不确定性感知的CL方法（如NPCL）使用神经过程但存在：(1)参数增长不可忽视；(2)与基于logits的回放方法不兼容（Monte Carlo采样的随机性）。因此需要一种**轻量且兼容**的原则来构建可靠的CL系统。

核心idea：幂等性——一个函数反复应用产生相同结果（$f(f(x)) = f(x)$）。如果模型在旧数据上保持幂等，说明输出落在学习到的稳定流形上，即模型做出了自一致的可靠预测。

## 方法详解

### 整体框架
IDER 在经验回放框架上添加两个模块：(1) 标准幂等模块——使当前模型在当前任务数据上保持幂等；(2) 幂等蒸馏模块——利用旧模型checkpoint实现跨任务的幂等约束。只需两次前向传播，几乎不引入额外参数。

### 关键设计

1. **修改的网络架构**:

    - 功能：使模型可以接受两个输入——图像和辅助标签信号
    - 核心思路：将ResNet骨干分为 $f_t^1$（前半部分）和 $f_t^2$（后半部分）。第二个输入（one-hot标签 $y$ 或均匀分布的"空"信号 $\mathbf{0}$）经线性层+LeakyReLU变换为与 $f_t^1$ 输出同维度的特征，加到中间特征图后送入 $f_t^2$。模型输出（logits经softmax）可作为第二个输入循环使用
    - 设计动机：无需真实标签时（推理阶段），用 $\mathbf{0}$ 作为第二输入获得预测，然后将预测回馈验证幂等性

2. **标准幂等模块 (Standard Idempotent Module)**:

    - 功能：在当前任务数据上训练模型幂等
    - 核心思路：最小化两次前向传播与ground truth的交叉熵：
    $\mathcal{L}_{ice} = \sum_{(x,y)\in\mathcal{T}_t} [\mathcal{L}_{ce}(f_t(x,y^*),y) + \mathcal{L}_{ce}(f_t(x,f_t(x,y^*)),y)]$
      其中 $y^*$ 以概率 $1-P$ 选为ground truth标签、概率 $P$ 选为空信号 $\mathbf{0}$
    - 设计动机：训练后 $f_t(x,\mathbf{0}) \approx y$ 且 $f_t(x,f_t(x,\mathbf{0})) \approx y$，实现预测的自一致性

3. **幂等蒸馏模块 (Idempotent Distillation Module)**:

    - 功能：利用旧模型checkpoint约束当前模型，防止预测分布漂移
    - 核心思路：不是简单最小化 $\|f_t(x,\mathbf{0}) - f_t(x,f_t(x,\mathbf{0}))\|^2$（因为 $f_t$ 对当前数据有偏，可能强化错误预测），而是用冻结的旧checkpoint $f_{t-1}$ 计算第二次传播：
    $\mathcal{L}_{ide} = \sum_{(x,y)\in\mathcal{T}_t,M} \|f_t(x,\mathbf{0}) - f_{t-1}(x,f_t(x,\mathbf{0}))\|_2^2$
    - 设计动机：$f_{t-1}$ 冻结保持先前知识和稳定分布，只更新 $f_t$ 使 $y_0$ 向正确方向调整，避免错误强化。同时起到知识蒸馏作用

### 损失函数 / 训练策略
总损失为三项加权和：$\mathcal{L}_{IDER} = \mathcal{L}_{ice} + \alpha\mathcal{L}_{ide} + \beta\mathcal{L}_{rep\text{-}ice}$，其中回放损失 $\mathcal{L}_{rep\text{-}ice}$ 在缓冲区数据上同样采用幂等训练loss。IDER可即插即用地集成到ER、BFP、CLS-ER等方法。

## 实验关键数据

### 主实验（Final Average Accuracy）

| 方法 | CIFAR-10 Buf200 | CIFAR-10 Buf500 | CIFAR-100 Buf500 | CIFAR-100 Buf2000 |
|------|----------------|----------------|-----------------|------------------|
| ER | 44.46 | 58.84 | 23.41 | 40.47 |
| DER++ | 62.19 | 70.10 | 37.69 | 51.82 |
| XDER | 64.10 | 67.42 | 48.14 | 57.57 |
| BFP | 68.64 | 73.51 | 46.70 | 57.39 |
| **ER+ID (Ours)** | **71.02** | **74.74** | 44.82 | 56.59 |
| **BFP+ID (Ours)** | **71.99** | **76.65** | **48.53** | **57.74** |

### GCIL消融（GCIL-CIFAR-100 Uniform）

| 方法 | Buf200 | 提升 | Buf500 | 提升 |
|------|--------|------|--------|------|
| ER | 16.34 | — | 28.76 | — |
| ER+ID | 26.66 | +10.32 | 40.54 | +11.78 |
| CLS-ER | 22.37 | — | 36.80 | — |
| CLS-ER+ID | 31.17 | +8.80 | 37.57 | +0.77 |

### 关键发现
- ER+ID 在CIFAR-10 Buf200上提升高达26%（44.46→71.02），是所有方法中提升最大的
- 幂等蒸馏有效缓解近期任务偏见，使不同任务的预测概率更均匀
- ECE（校准误差）在所有基线上一致降低，说明模型预测更"诚实"
- 在GCIL更具挑战性的设置中，提升更加明显（如CLS-ER+ID在Longtail设置中提升8.4%）

## 亮点与洞察
- 将代数中的幂等性直接映射为CL中"预测自一致性"的约束，数学直觉清晰且优雅
- 方法极其轻量——仅需一个额外前向传播和旧checkpoint，参数几乎无增长
- 即插即用特性使其成为增强现有CL方法可靠性的通用工具

## 局限与展望
- 需要保存旧任务checkpoint $f_{t-1}$，虽存储开销不大但增加了系统复杂性
- 实验均在ResNet-18上进行，在Transformer骨干和更大模型上的效果未知
- 幂等性假设的理论基础可进一步加强——为何幂等性必然导致更好的泛化？

## 相关工作与启发
- **vs NPCL**: NPCL使用神经过程带来参数增长且与logits方法不兼容，IDER轻量且通用
- **vs DER++**: DER++ 简单存储并匹配logits，IDER通过幂等约束提供更强的正则化

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 幂等性在CL中的应用是全新的视角
- 实验充分度: ⭐⭐⭐⭐⭐ CIL+GCIL+ECE+即插即用验证，非常全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，动机到方法的推导自然
- 价值: ⭐⭐⭐⭐ 为CL领域提供了新的数学原则和实用工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [\[ICLR 2026\] FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning](flyprompt_brain-inspired_random-expanded_routing.md)
- [\[ICCV 2025\] PLAN: Proactive Low-Rank Allocation for Continual Learning](../../ICCV2025/model_compression/plan_proactive_low-rank_allocation_for_continual_learning.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](../../NeurIPS2025/model_compression/gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)

</div>

<!-- RELATED:END -->
