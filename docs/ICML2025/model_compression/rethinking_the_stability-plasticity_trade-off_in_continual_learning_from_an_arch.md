---
title: >-
  [论文解读] Rethinking the Stability-Plasticity Trade-off in Continual Learning from an Architectural Perspective
description: >-
  [ICML2025][模型压缩][持续学习] 揭示了持续学习中稳定性与可塑性之间在架构层面：的固有冲突——宽浅网络稳定性好、深窄网络可塑性强——并提出 Dual-Arch 框架，用两个专用轻量架构分别负责稳定性和可塑性，通过知识蒸馏协同，实现参数量减少最高 87% 的同时提升 CL 性能。 持续学习（CL）的核心挑战是灾难性…
tags:
  - "ICML2025"
  - "模型压缩"
  - "持续学习"
  - "稳定性-可塑性权衡"
  - "网络架构设计"
  - "知识蒸馏"
  - "双架构框架"
---

# Rethinking the Stability-Plasticity Trade-off in Continual Learning from an Architectural Perspective

**会议**: ICML2025  
**arXiv**: [2506.03951](https://arxiv.org/abs/2506.03951)  
**代码**: [byyx666/Dual-Arch](https://github.com/byyx666/Dual-Arch)  
**领域**: 持续学习 (Continual Learning)  
**关键词**: 持续学习, 稳定性-可塑性权衡, 网络架构设计, 知识蒸馏, 双架构框架

## 一句话总结

揭示了持续学习中稳定性与可塑性之间在**架构层面**的固有冲突——宽浅网络稳定性好、深窄网络可塑性强——并提出 Dual-Arch 框架，用两个专用轻量架构分别负责稳定性和可塑性，通过知识蒸馏协同，实现参数量减少最高 87% 的同时提升 CL 性能。

## 研究背景与动机

持续学习（CL）的核心挑战是**灾难性遗忘**：在学习新任务时，网络会迅速遗忘旧知识。现有方法（重放、正则化、架构扩展）主要在**参数层面**平衡稳定性与可塑性，却忽略了网络架构本身对这一权衡的影响。

已有先驱工作（Mirzadeh et al., 2022）指出宽浅网络整体 CL 性能更好（主要贡献于稳定性），但深度网络在表示学习能力上更强（利于可塑性）。这引发一个关键问题：**在给定参数量约束下，稳定性与可塑性是否在架构层面也存在固有矛盾？**

作者通过对比 ResNet-18 与其等参宽浅变体验证了这一假设：

- **ResNet-18**（深窄）：新任务准确率更高（可塑性好），但遗忘更严重
- **宽浅变体**（10层, 96通道）：遗忘更低（稳定性好），但新任务性能下降

这表明单一架构无法同时兼顾两个目标，现有方法使用统一架构的做法本身就限制了 CL 上限。

## 方法详解

### 核心思想：Dual-Arch 框架

Dual-Arch 是一个**即插即用**的 CL 框架，核心思路是将稳定性和可塑性分配给两个独立的专用网络：

- **Pla-Net（可塑性网络）**：深而窄的架构，专注学习新知识
- **Sta-Net（稳定性网络）**：宽而浅的架构，负责保留旧知识并整合新知识

### 架构设计（基于 ResNet-18 改造）

| 网络 | 深度 | 宽度 | 特殊设计 | 参数量 |
|------|------|------|----------|--------|
| ResNet-18（原始） | 18层 | 64通道 | GAP → 1×1 | 11.23M |
| Sta-Net | 半数残差块 | 64通道 | AvgPool → 2×2 输出 | ~7M |
| Pla-Net | 18层 | 42通道 | 保持原始结构 | ~7M |

两个网络参数量之和（~15M）仍小于原始 ResNet-18（~22M），实现了压缩。

### 训练流程

对于每个新任务 $k$，分两阶段顺序训练：

**阶段一：训练 Pla-Net**
仅用分类损失学习当前任务数据，不关心遗忘：

$$L_{plastic} = L_{CE}(x, y; \phi_k) = -\log \frac{\exp(o_y)}{\sum_{m=1}^{N^k} \exp(o_m)}$$

**阶段二：训练 Sta-Net**
冻结 Pla-Net 作为教师模型，通过知识蒸馏将新知识转移给 Sta-Net，同时结合 CL 方法保留旧知识：

$$L_{stable} = \alpha \cdot L_{CE} + (1-\alpha) \cdot L_{KD} + L_{CL}$$

其中 $\alpha=0.5$ 平衡硬标签损失和蒸馏损失，$L_{CL}$ 是所用 CL 方法自身的损失项。

知识蒸馏损失 $L_{KD}$ 计算教师（Pla-Net）与学生（Sta-Net）软输出间的 KL 散度：

$$L_{KD} = -\sum_{i=1}^{N^k} P_T^i \log P_S^i, \quad P_T = \text{SoftMax}(O_T / t), \quad P_S = \text{SoftMax}(O_S / t)$$

其中 $t$ 为温度因子，控制软输出的平滑度。

**推理阶段**：仅使用 Sta-Net，Pla-Net 不参与推理，因此推理开销反而更小。

## 实验关键数据

### 主实验：与5种 SOTA CL 方法结合（Tab. 2）

在 CIFAR-100 和 ImageNet-100 上（10/20 任务划分），Dual-Arch 作为插件一致提升所有基线方法：

| 方法 | 参数量减少 | LA 最大提升 | AIA 最大提升 | 遗忘降低 |
|------|-----------|------------|-------------|---------|
| iCaRL + Dual-Arch | ↓33% | +3.10% | +2.17% | ↓7.69% |
| WA + Dual-Arch | ↓33% | +8.24% | +6.09% | ↓7.32% |
| DER + Dual-Arch | ↓52% | +5.69% | +3.67% | ↓5.55% |
| Foster + Dual-Arch | ↓32% | +7.70% | +7.62% | ↓11.28% |
| MEMO + Dual-Arch | ↓41% | +10.29% | +5.09% | ↓11.62% |

MEMO + Dual-Arch 表现最突出：LA 提升 10.29%，FAF 降低 11.62%，参数量减少 41%。

### 消融实验（Tab. 3, CIFAR-100/10, AIA）

| 配置 | 平均 AIA | 与完整方案差距 |
|------|---------|--------------|
| Sta-Net + Pla-Net（完整） | 72.92% | — |
| 仅 Sta-Net（无辅助网络） | 70.29% | -2.63% |
| Pla-Net + Pla-Net（统一架构） | 71.18% | -1.74% |
| Sta-Net + Sta-Net（统一架构） | 72.27% | -0.65% |
| Pla-Net + Sta-Net（角色反转） | 71.24% | -1.68% |

消融结果表明：(1) 双网络协作比单网络好 2.63%；(2) 专用架构比统一架构好 0.65~1.74%；(3) 角色分配不可互换。

### 参数效率

Dual-Arch 在参数减少 **87%** 的极端情况下仍可超越原始基线（DER 上 AIA +0.90%，Foster 上 +1.94%）。

### 计算效率

| 指标 | Sta-Net | Pla-Net | 合计 | ResNet-18 |
|------|---------|---------|------|-----------|
| FLOPs | 255M | 241M | 496M | 558M |

训练总 FLOPs 更少，推理仅用 Sta-Net（255M vs 558M），计算量降低 54%。但训练时间因串行训练增加 1.39×~1.77×。

## 亮点与洞察

1. **新颖视角**：首次系统揭示 CL 中稳定性-可塑性在架构层面的内在冲突，将权衡从参数维度拓展到架构维度
2. **即插即用**：Dual-Arch 可无缝集成到 iCaRL/WA/DER/Foster/MEMO 等主流方法，普适性强
3. **参数高效**：以更少参数实现更好性能，不是简单叠加两个大模型，而是用两个轻量专用网络替代一个大网络
4. **推理无开销**：推理阶段仅用 Sta-Net，FLOPs 仅为原始的 46%
5. **实验设计规范**：公平对比了多方法×多数据集×多任务划分，消融实验彻底验证了各组件贡献

## 局限与展望

1. **训练时间增加**：双网络串行训练导致 1.39×~1.77× 训练开销，无法并行化是硬伤
2. **架构设计依赖经验**：Sta-Net/Pla-Net 的具体设计（层数、通道数）基于 ResNet 手工调整，缺乏自动化搜索
3. **仅验证 ResNet 系列**：未在 ViT、ConvNeXt 等现代架构上验证，"深窄=可塑、宽浅=稳定"的结论泛化性待考
4. **仅 Class-IL 场景**：未验证 Task-IL 和 Domain-IL 场景
5. **蒸馏策略单一**：仅用 logit-level KD，未探索特征蒸馏、关系蒸馏等更丰富的知识转移方式
6. **温度因子 $t$ 和平衡系数 $\alpha$ 均用默认值**（$\alpha=0.5$），未做充分调参分析

## 相关工作与启发

- **ArchCraft**（Lu et al., 2024）：单架构优化方案，Dual-Arch 在多数场景优于它
- **MKD / Hare & Tortoise**：也用双模型但架构相同，本文强调专用架构的重要性
- **DER / MEMO**：架构扩展类方法，Dual-Arch 可在其基础上进一步提升
- **启发**：可将「功能分离+专用架构」思路推广到联邦学习、多任务学习等场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — 架构层面的稳定性-可塑性权衡是全新且有价值的研究视角
- 实验充分度: ⭐⭐⭐⭐ — 5种方法×4个基准×消融+效率分析，覆盖全面
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，从观察→问题→方案→验证贯通
- 价值: ⭐⭐⭐⭐ — 即插即用框架有实用价值，但泛化到现代架构待验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity](../../ICLR2026/model_compression/null-space_filtering_for_data-free_continual_model_merging_preserving_stability_.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[ICML 2025\] BECAME: BayEsian Continual Learning with Adaptive Model MErging](became_bayesian_continual_learning_with_adaptive_model_merging.md)
- [\[NeurIPS 2025\] When Worse is Better: Navigating the Compression-Generation Trade-off in Visual Tokenization](../../NeurIPS2025/model_compression/when_worse_is_better_navigating_the_compression-generation_tradeoff_in_visual_to.md)
- [\[NeurIPS 2025\] Mixed Monotonicity Reachability Analysis of Neural ODE: A Trade-Off Between Tightness and Efficiency](../../NeurIPS2025/model_compression/mixed_monotonicity_reachability_analysis_of_neural_ode_a_trade-off_between_tight.md)

</div>

<!-- RELATED:END -->
