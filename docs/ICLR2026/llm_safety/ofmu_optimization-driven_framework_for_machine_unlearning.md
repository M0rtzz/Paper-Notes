---
title: >-
  [论文解读] OFMU: Optimization-Driven Framework for Machine Unlearning
description: >-
  [ICLR 2026][LLM安全][机器遗忘] 将机器遗忘建模为双层优化问题：内层最大化遗忘损失+梯度去相关防止破坏保留集，外层最小化保留损失+惩罚项强制内层平稳点。在TOFU基准上同时实现高遗忘质量和高模型效用保留，平衡性超越现有GA/GradDiff/NPO/RMU方法。
tags:
  - "ICLR 2026"
  - "LLM安全"
  - "机器遗忘"
  - "双层优化"
  - "梯度去相关"
  - "遗忘-保留权衡"
  - "LLM隐私"
---

# OFMU: Optimization-Driven Framework for Machine Unlearning

**会议**: ICLR 2026  
**arXiv**: [2509.22483](https://arxiv.org/abs/2509.22483)  
**代码**: 无  
**领域**: AI安全 / 机器遗忘  
**关键词**: 机器遗忘, 双层优化, 梯度去相关, 遗忘-保留权衡, LLM隐私

## 一句话总结
将机器遗忘建模为双层优化问题：内层最大化遗忘损失+梯度去相关防止破坏保留集，外层最小化保留损失+惩罚项强制内层平稳点。在TOFU基准上同时实现高遗忘质量和高模型效用保留，平衡性超越现有GA/GradDiff/NPO/RMU方法。

## 研究背景与动机

**领域现状**：LLM需要按需遗忘特定知识（GDPR合规/版权/过时信息），但从头重训不切实际。现有方法分输入级（拒绝策略）、数据级（构造辅助数据）、模型级（修改参数）。

**现有痛点**：
   - 输入级方法脆弱，对抗prompt可绕过拒绝
   - 模型级方法用静态权重平衡遗忘/保留目标，无法动态适应
   - GradAscent/GradDiff在难遗忘样本上破坏性强——样本难度与效用损失强耦合

**核心矛盾**：遗忘梯度和保留梯度相关时，提升遗忘会破坏保留

**核忊idea一句话**：双层优化 + 梯度去相关 = 遗忘时不伤及保留

## 方法详解

### 整体框架
OFMU 把遗忘建模成一个嵌套的双层优化：内层在做梯度上升尽力最大化遗忘损失的同时，把遗忘梯度与保留梯度去相关，避免"抹掉知识"的更新顺手破坏保留集；外层则最小化保留损失，并用一个惩罚项强制内层解停在平稳点上，使整个嵌套结构稳定收敛。

### 关键设计

**1. 双层优化建模：把遗忘-保留的冲突显式拆成内外两层。**

现有模型级方法大多把遗忘和保留写成线性加权的单目标 $\lambda$，权重一旦固定就无法随样本难度动态调整，难遗忘样本上的强遗忘信号会顺着耦合方向直接侵蚀保留集。OFMU 改为分层处理：内层只负责遗忘，目标 $\Phi(\theta) = \mathcal{L}_f(\theta) - \beta \cdot \text{Sim}(\nabla\mathcal{L}_f, \nabla\mathcal{L}_r)$，其中第一项推高遗忘损失，第二项用余弦相似度 $\text{Sim}$ 度量遗忘梯度与保留梯度的方向相关性并对其加惩罚，逼迫遗忘更新转向与保留梯度正交的方向，从几何上把"破坏保留"的分量挤掉。外层则在内层解的基础上负责保留，目标 $F(\theta) = \mathcal{L}_r(\theta) + \rho\|\nabla\Phi(\theta)\|^2$，前一项压低保留损失，后一项 $\rho\|\nabla\Phi\|^2$ 惩罚内层梯度的残余范数，确保外层每次落脚处内层都已逼近平稳点——这样遗忘与保留不再共用一组静态权重相互拉扯，而是各自在自己那层达到最优。

**2. 两循环求解算法：用内外两重迭代逼近双层解并给出收敛保证。**

双层结构无法一步求出，OFMU 用嵌套的两循环来近似。内循环固定外层参数、跑 $T$ 步梯度上升求内层近似解 $\theta'^{(t+1)} = \theta'^{(t)} + \eta_{\text{in}}\nabla\Phi(\theta'^{(t)})$，把遗忘和去相关都吃进去；外循环再据此做一步保留更新 $\theta^{(k+1)} = \theta^{(k)} - \eta_{\text{out}}(\nabla\mathcal{L}_r + 2\rho\nabla^2\Phi\cdot\nabla\Phi)$，其中 $\nabla^2\Phi\cdot\nabla\Phi$ 通过 Hessian-向量积计算，避免显式构造 Hessian。内循环只需 $T=5\sim10$ 步、不必完全收敛，惩罚系数 $\rho_k$ 随外循环进度递增，逐步收紧对内层平稳性的约束。作者给出收敛速率：凸场景下为 $O(1/K)+O(K/T^2)$，非凸场景下为 $O(1/K)+O(1/T)+O(\sigma^2)$，说明只要外循环步数 $K$ 与内循环步数 $T$ 配比合适，算法能稳定逼近双层最优而不像单纯梯度上升那样发散。

## 实验关键数据

### 主实验：TOFU基准(LLaMA-2-7B)

| 方法 | FQ(forget01) | MU | FTR | 说明 |
|------|-------------|----|----|------|
| Retrain | 1.00 | 0.63 | 0.68 | 理想上界 |
| GradAscent | 1.88e-4 | 0.55 | 0.36 | 遗忘弱+保留差 |
| GradDiff | 3.02e-3 | 0.57 | 0.41 | 略好 |
| NPO | 0.40 | 0.58 | 0.65 | 中等 |
| RMU | 0.40 | 0.62 | 0.64 | 中等 |
| **OFMU** | **0.42** | **0.63** | **0.68** | **接近Retrain** |

### 消融实验

| 配置 | 关键发现 |
|------|--------|
| 去掉梯度去相关 | 遗忘效果提升但保留严重受损 |
| 去掉双层结构(用线性加权) | $\lambda$ 权衡不稳定，难细调 |
| Full OFMU | 最佳平衡 |

### 关键发现
- **OFMU接近Retrain上界**：MU=0.63等于Retrain，FTR=0.68等于Retrain
- **GA/GradDiff在forget05/10上崩溃**：FQ降到e-119~e-239，说明在大规模遗忘时完全失效
- **梯度去相关解耦难遗忘样本的耦合问题**

## 亮点与洞察
- **双层优化视角重新定义遗忘问题**：不是简单的多目标线性加权，而是将遗忘作为满足梯度平稳性约束的外层优化——这个建模更符合遗忘的本质
- **梯度去相关的精妙设计**：通过余弦相似度惩罚确保遗忘梯度和保留梯度正交，从几何层面消除冲突——与NSPO的零空间投影思路相似但应用于遗忘而非安全对齐

## 局限与展望
- Hessian-向量积计算开销较大
- 未测试>70B模型和多模态场景
- 未探索持续遗忘场景（多次遗忘请求）

## 相关工作与启发
- **vs GradAscent/GradDiff**：简单梯度上升在大规模遗忘时崩溃，OFMU通过双层结构保持稳定
- **vs NPO/RMU**：这些方法用启发式权重平衡，OFMU用严格的双层优化框架，理论保证更强
- **vs NSPO(同会议)**：两者都用梯度正交/去相关策略，但NSPO用于安全对齐，OFMU用于机器遗忘

## 评分
- 新颖性: ⭐⭐⭐⭐ 双层优化+梯度去相关的组合新颖
- 实验充分度: ⭐⭐⭐⭐ TOFU+CIFAR多场景，但缺少大规模LLM实验
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨
- 价值: ⭐⭐⭐⭐ 为机器遗忘提供了理论严格的优化框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] PURGE: Reinforcement Unlearning via Group Relative Policy Optimization](reinforcement_unlearning_via_group_relative_policy_optimization.md)
- [\[NeurIPS 2025\] A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](../../NeurIPS2025/llm_safety/a_reliable_cryptographic_framework_for_empirical_machine_unl.md)
- [\[ICLR 2026\] Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs](model_collapse_is_not_a_bug_but_a_feature_in_machine_unlearning_for_llms.md)
- [\[CVPR 2026\] SineProject: Machine Unlearning for Stable Vision–Language Alignment](../../CVPR2026/llm_safety/sineproject_machine_unlearning_for_stable_vision_language_alignment.md)
- [\[NeurIPS 2025\] SIMU: Selective Influence Machine Unlearning](../../NeurIPS2025/llm_safety/simu_selective_influence_machine_unlearning.md)

</div>

<!-- RELATED:END -->
