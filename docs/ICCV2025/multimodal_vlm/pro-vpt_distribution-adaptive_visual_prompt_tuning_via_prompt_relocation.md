---
title: >-
  [论文解读] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation
description: >-
  [ICCV 2025][多模态][视觉提示调优] 提出 PRO-VPT 框架，通过嵌套优化将提示分布优化 (ADO) 与视觉提示调优 (VPT) 协同设计，利用闲置分数剪枝和强化学习分配策略迭代重定位提示，在 VTAB-1k 和 FGVC 上较 VPT 分别提升 1.6pp 和 2.0pp。
tags:
  - ICCV 2025
  - 多模态
  - 视觉提示调优
  - 参数高效微调
  - 自适应分布优化
  - 提示重定位
  - 强化学习
---

# PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation

**会议**: ICCV 2025  
**arXiv**: [2503.06901](https://arxiv.org/abs/2503.06901)  
**代码**: https://github.com/ckshang/PRO-VPT  
**领域**: 多模态VLM  
**关键词**: 视觉提示调优, 参数高效微调, 自适应分布优化, 提示重定位, 强化学习

## 一句话总结

提出 PRO-VPT 框架，通过嵌套优化将提示分布优化 (ADO) 与视觉提示调优 (VPT) 协同设计，利用闲置分数剪枝和强化学习分配策略迭代重定位提示，在 VTAB-1k 和 FGVC 上较 VPT 分别提升 1.6pp 和 2.0pp。

## 研究背景与动机

视觉提示调优 (VPT) 通过在 Transformer 块的输入空间插入轻量可学习提示 token 来适配预训练模型，是参数高效微调 (PEFT) 的主流方法之一。现有 VPT 方法通常使用固定的提示分布——要么浅层（仅第一层），要么深层（均匀分布到所有层）。

然而，最新研究揭示了一个关键现象：**每个预训练块的重要性在不同任务间差异很大**，这意味着不加区分地使用固定提示分布无法充分发挥 VPT 的潜力。

本文系统地探索了两个核心问题：
1. 如何恰当定义自适应分布优化 (ADO)？
2. 如何设计基于该定义的自适应分布策略？

通过实证分析，作者发现三个关键 insights：
- 适当调整提示分布能显著提升性能
- 有效的调整随训练过程中提示更新而变化（非静态）
- 分布调整的效果只有在提示调优后才能准确评估（嵌套关系）

## 方法详解

### 整体框架

PRO-VPT 采用嵌套优化框架，将 ADO 和 VPT 交替进行：
- 外层优化：调整提示分布 D
- 内层优化：在当前分布下调优提示参数 P
- 迭代循环："分布调整 → 提示调优 → 调整评估 → 新调整"

### 关键设计

**1. 提示重定位 (PR) 策略**

将重定位分解为两个顺序步骤，将动作空间从 L² 降至 L：

**(a) 闲置分数剪枝 (Idleness Score-based Pruning)**

定义闲置分数 I_k 衡量提示 p_k 对当前块是否多余。直觉上：如果移除一个提示后性能反而提升，说明该提示在当前块是"闲置"的，适合被迁移。

为高效计算，使用一阶 Taylor 展开近似：I_k ≈ g_k^T · p_k，其中 g_k 是关于提示的梯度。选择 I_k 最大的提示进行剪枝。

**(b) 强化学习分配 (RL-based Allocation)**

使用 PPO 算法确定被剪枝提示应该分配到哪个块：
- **状态**: 各块闲置分数 + 当前分布 + 被剪枝提示位置
- **动作**: 选择分配到第 a 层 (a ∈ [L])
- **奖励**: 分配后经过提示调优的性能提升

**2. 与剪枝方法的本质区别**

现有剪枝方法（如 NOAH）只能从饱和块移除提示，无法为不足块添加提示。PRO-VPT 通过"剪枝+分配"实现真正的重定位，最大化整体提示分布的有效性。

### 损失函数 / 训练策略

- 任务损失：标准交叉熵分类损失
- 剪枝评估：Taylor 展开近似的闲置分数
- 分配优化：PPO 策略梯度，奖励包含闲置分数校正项
- 整体流程：周期性执行 PR（每个 epoch 内多次迭代 VPT 后执行一次 PR）

## 实验关键数据

### 主实验

VTAB-1k 基准上的对比（ViT-B/16, ImageNet-21k 预训练）：

| 方法 | 参数 (M) | Natural | Specialized | Structured | 全局均值 |
|------|----------|---------|-------------|------------|----------|
| Full FT | 85.8 | 78.6 | 86.3 | 57.8 | 74.2 |
| VPT-Deep | 0.60 | 82.5 | 84.6 | 62.1 | 76.4 |
| LoRA | 0.29 | 82.4 | 84.3 | 60.1 | 75.6 |
| Adapter | 0.35 | 83.3 | 86.2 | 63.3 | 77.6 |
| **PRO-VPT** | 0.60 | **83.3** | **86.2** | **64.5** | **78.0** |

PRO-VPT 以与 VPT-Deep 相同的参数量达到了 SOTA，超越了包括 Adapter 在内的所有 PEFT 方法。

### 消融实验

分布调整在不同 epoch 的提示上表现差异：

| 调整时机 | Epoch 25 | Epoch 50 | Epoch 75 |
|----------|----------|----------|----------|
| 最优调整块 | 不同 | 不同 | 不同 |
| 性能提升 | 有 | 有 | 有 |

这验证了 Finding 2：有效的分布调整随提示更新而变化，需要迭代过程。

FGVC 基准：PRO-VPT 达到 91.7% 均值准确率，较 VPT 提升 2.0pp。

### 关键发现

- 提示分布确实重要：合适的分布调整可显著提升性能
- 嵌套关系成立：分布优化需要嵌套在提示调优内部
- 一阶近似有效：Taylor 展开的闲置分数与真实值高度相关
- RL 分配的动作空间缩减（L² → L）大幅加速收敛

## 亮点与洞察

1. **首次系统定义 ADO 问题**：从实证出发，揭示 ADO 与 VPT 的嵌套关系
2. **剪枝+分配的分解策略**：既解决了动作空间爆炸问题，又有直观解释（从饱和块到需求块）
3. **Taylor 展开近似**：将 O(N) 次前向传播的闲置分数计算降至一次梯度计算
4. **统一了现有剪枝方法的理论框架**：证明剪枝方法只是 PRO-VPT 的特例

## 局限与展望

- RL 训练引入额外超参数和计算开销
- 仅在 ViT-B/16 上验证，未在更大规模模型上测试
- 每次只重定位一个提示，可能收敛较慢
- 添加式分布调整（直接增加提示数）被发现不稳定，仅使用重定位

## 相关工作与启发

- 与层级微调中"不同层需要不同微调强度"的发现一致
- 嵌套优化框架具有通用性，可扩展到其他 PEFT 方法的超参数优化
- Taylor 展开剪枝思路来自神经网络剪枝领域，在提示场景下的创新应用

## 评分

- 新颖性: ⭐⭐⭐⭐ （ADO 问题的系统定义和嵌套优化框架有创新）
- 实验充分度: ⭐⭐⭐⭐⭐ （VTAB-1k 全部 19 个数据集 + FGVC + 详尽消融）
- 写作质量: ⭐⭐⭐⭐⭐ （问题驱动，层层递进，分析严谨）
- 价值: ⭐⭐⭐⭐ （对 VPT 领域有实质推进，但绝对提升幅度有限）

<!-- RELATED:START -->

## 相关论文

- [Attention to the Burstiness in Visual Prompt Tuning!](attention_to_the_burstiness_in_visual_prompt_tuning.md)
- [Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-Distribution Detection](adaptive_prompt_learning_via_gaussian_outlier_synthesis_for_out-of-distribution_.md)
- [FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)
- [Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts](../../ICLR2026/multimodal_vlm/revisit_visual_prompt_tuning_the_expressiveness_of_prompt_experts.md)
- [CVPT: Cross Visual Prompt Tuning](cvpt_cross_visual_prompt_tuning.md)

<!-- RELATED:END -->
