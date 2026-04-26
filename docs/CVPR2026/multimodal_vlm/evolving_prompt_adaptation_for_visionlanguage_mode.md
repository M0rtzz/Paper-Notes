---
title: >-
  [论文解读] Evolving Prompt Adaptation for Vision-Language Models
description: >-
  [CVPR 2026][多模态][提示学习] 提出 EvoPrompt 框架，将提示训练视为从通用语义锚点到任务特征的渐进进化过程，通过模态共享提示投影器（MPP）统一跨层跨模态提示生成、进化轨迹感知策略（方向-幅度解耦冻结历史方向）防止遗忘、特征几何正则化（FGR）防止表示坍缩，在 11 个数据集 base-to-novel 泛化上平均 HM 达 80.73%，超越所有现有提示学习方法。
tags:
  - CVPR 2026
  - 多模态
  - 提示学习
  - 视觉语言模型
  - 灾难性遗忘
  - 低秩适应
  - 特征正则化
---

# Evolving Prompt Adaptation for Vision-Language Models

**会议**: CVPR 2026  
**arXiv**: [2603.09493](https://arxiv.org/abs/2603.09493)  
**代码**: 无  
**领域**: 视觉语言模型 / 参数高效微调  
**关键词**: 提示学习, 视觉语言模型, 灾难性遗忘, 低秩适应, 特征正则化

## 一句话总结

提出 EvoPrompt 框架，将提示训练视为从通用语义锚点到任务特征的渐进进化过程，通过模态共享提示投影器（MPP）统一跨层跨模态提示生成、进化轨迹感知策略（方向-幅度解耦冻结历史方向）防止遗忘、特征几何正则化（FGR）防止表示坍缩，在 11 个数据集 base-to-novel 泛化上平均 HM 达 80.73%，超越所有现有提示学习方法。

## 研究背景与动机

**领域现状**：CLIP 等大规模 VLM 通过对比预训练获得强大 zero-shot 能力。提示学习（CoOp/CoCoOp/MaPLe）以极少可训练参数实现高效下游适配，是当前主流参数高效微调方案。

**现有痛点**：(1) **层间孤立**——MaPLe 等方法在每层独立参数化提示，破坏了编码器深层的语义层次流，无法将底层学到的信息传播到高层；(2) **模态偏置**——现有方案以文本为中心（text-centric），未充分利用视觉-语言互补信息；(3) **灾难性遗忘**——少样本适配时提示快速偏离预训练语义锚点，过拟合下游数据导致 zero-shot 泛化能力丧失。

**核心矛盾**：提示学习需要学习任务特定特征，但自由优化会覆盖预训练知识。二者本质上是"适配强度"与"知识保留"的 trade-off。

**本文目标** 在少样本提示学习中显式引导提示的进化轨迹，使其既学任务特征又保留预训练知识。

**切入角度**：将提示训练视为从通用语义锚点到任务特定特征的渐进进化过程。关键观察——在低秩适配中，方向编码语义知识（更关键），幅度编码适配强度。如果冻结已学到的方向、只调幅度，就能在不覆盖旧知识的前提下持续学习。

**核心 idea**：用方向-幅度解耦冻结历史语义方向、只调幅度系数，配合共享投影器和特征正则化，实现提示的可控进化。

## 方法详解

### 整体框架

冻结 CLIP ViT-B/16 双编码器 → 初始化统一可学习嵌入空间 $E \in \mathbb{R}^{K \times d_r}$（$K=5, d_r=512$）→ MPP 通过共享权重 + 层特定低秩适配器将 $E$ 投影为每层每模态的提示 → 注入编码器第 $J=6$ 到 $L=12$ 层 → 进化训练策略在训练过程中逐 epoch 冻结历史方向仅调幅度 → FGR 约束特征几何结构 → InfoNCE + FGR + 知识恒常损失联合优化。

### 关键设计

1. **模态共享提示投影器（MPP）**

    - **功能**：用统一嵌入空间生成跨层跨模态提示，替代 MaPLe 的逐层独立参数化
    - **核心思路**：对模态 $m$ 的第 $i$ 层，投影权重为 $W_i^m = W_{\text{shared}}^m + A_i B_i$，其中 $W_{\text{shared}}^m$ 跨层共享捕获基础语义，$A_i B_i$ 为低秩（$r \ll \min(d_r, d_m)$）层特定适配。参数复杂度从 $\mathcal{O}((L-J+1) \cdot d_r d_m)$ 降到 $\mathcal{O}(d_r d_m + (L-J+1) \cdot r(d_r + d_m))$，比 MaPLe 减少约 4.6 倍参数（0.764M vs 3.555M）
    - **设计动机**：跨层共享基础语义 + 低秩适配器捕获层特定变化，既保证层间信息流通又保持表达能力。统一嵌入空间天然支持跨模态信息交互

2. **进化轨迹感知学习策略**

    - **功能**：将低秩更新在每个 epoch 解耦为方向和幅度，冻结历史方向只调幅度，实现知识保留式渐进适配
    - **核心思路**：将 $\Delta W_i^t$ 在 epoch $t$ 分解为幅度 $\alpha_i^t$ 和归一化方向 $\overline{A_i^t B_i^t}$（Frobenius 归一化）。训练到 epoch $T$ 时，权重累积为 $W_i^T = W_{\text{shared}} + \sum_{t=1}^{T-1} \alpha_i^t \overline{A_i^t B_i^t} + \alpha_i^T \overline{A_i^T B_i^T}$。所有历史方向 $\{\overline{A_i^t B_i^t}\}_{t=1}^{T-1}$ 冻结保留几何结构，只训练幅度系数和当前新方向。配合自适应秩缩减（在预设节点 $\mu, \nu$ 处降低 rank），减少后期过拟合风险
    - **设计动机**：方向编码语义知识（先验研究已证明方向比幅度更关键），冻结方向即保留知识。逐 epoch 累加方向类似持续学习的知识积累，而秩缩减起结构化正则化作用

3. **特征几何正则化（FGR）**

    - **功能**：防止 InfoNCE 训练中的特征维度冗余和表示坍缩
    - **核心思路**：基于 Soft-HGR 最大相关性框架推导。$\mathcal{L}_{fgr} = \frac{1}{2} \text{tr}(\text{cov}(\mathcal{F}^v) \cdot \text{cov}(\mathcal{F}^t))$，最小化视觉和文本特征协方差矩阵之积，强制特征去相关。总损失 $\mathcal{L} = \mathcal{L}_{InfoNCE} + \gamma \mathcal{L}_{fgr} + \eta \mathcal{L}_{kcl}$，$\mathcal{L}_{kcl}$ 为知识恒常损失（约束 prompted 特征不偏离原始 CLIP 特征方向）
    - **设计动机**：InfoNCE 关注实例级对齐但忽略特征空间几何结构，少样本下冗余维度会导致过拟合

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{InfoNCE} + \gamma \mathcal{L}_{fgr} + \eta \mathcal{L}_{kcl}$（$\gamma=25, \eta=0.5$）；16-shot/class；单卡 NVIDIA A800；3 seeds 取平均。

## 实验关键数据

### 主实验——Base-to-Novel 泛化（11 数据集平均）

| 方法 | Base | Novel | HM |
|------|------|-------|------|
| CLIP | 69.34 | 74.22 | 71.70 |
| CoOp | 82.69 | 63.22 | 71.66 |
| MaPLe | 82.28 | 75.14 | 78.55 |
| PromptSRC | 84.26 | 76.10 | 79.97 |
| TCP | 84.13 | 75.36 | 79.51 |
| MMA | 83.20 | 76.80 | 79.87 |
| **EvoPrompt** | **84.28** | **77.76** | **80.73** |

### 消融实验（ImageNet Base-to-Novel）

| 配置 | Base | Novel | HM |
|------|------|-------|------|
| w/o MPP（逐层独立提示） | 75.32 | 70.15 | 72.64 |
| w/o $W_{\text{shared}}$ | 75.80 | 71.42 | 73.54 |
| w/o AB（全秩投影） | 76.15 | 70.90 | 73.43 |
| w/o 进化策略 | 77.42 | 70.25 | 73.66 |
| w/o $\mathcal{L}_{kcl}$ | 77.24 | 70.55 | 73.74 |
| w/o $\mathcal{L}_{fgr}$ | 76.70 | 70.52 | 73.48 |
| **EvoPrompt (Full)** | **76.98** | **71.80** | **74.29** |

### 关键发现

- MPP 贡献最大（去掉后 HM 从 74.29 降到 72.64，-1.65），统一嵌入空间+共享投影对跨层信息流通至关重要
- 去掉进化策略或 $\mathcal{L}_{kcl}$ 后 Base 反而上升（77.42%/77.24%），但 Novel 大幅下降（70.25%/70.55%），印证其核心作用是防止过拟合 base 类
- 跨数据集迁移（ImageNet→10 目标数据集）平均 66.82%，优于 MMA（66.61%）和 MaPLe（66.30%）
- 域泛化（4 种 ImageNet 变体）结果最优，说明有效保留了 CLIP 原始 OOD 泛化能力

## 亮点与洞察

1. **方向-幅度解耦的遗忘防控**：将低秩更新分解为方向（编码语义知识）和幅度（编码适配强度），冻结历史方向只调幅度——思路简洁但直觉清晰，可推广到任何需要防止遗忘的 LoRA/adapter 场景
2. **FGR 正则化有理论支撑**：从 Soft-HGR 最大相关性框架推导出 FGR 是 InfoNCE 缺失的互补项，非 ad hoc 设计
3. **参数效率极高**：仅 0.764M 可训练参数（MaPLe 的 1/4.6），但 HM 提升 2.18%

## 局限与展望

1. 仅在 ViT-B/16 上实验，未验证更大 backbone（ViT-L/14）的表现
2. 逐 epoch 冻结方向+累加的设计引入额外存储开销，epoch 增大时历史矩阵线性增长
3. 秩缩减节点 $\mu, \nu$ 为手动设定的超参数，缺乏自适应策略
4. FGR 仅在 batch 内计算协方差，batch size 较小时估计不准

## 相关工作与启发

- **vs MaPLe**：MaPLe 逐层独立提示+text-centric，EvoPrompt 共享投影器统一管理+模态对等，参数少 4.6 倍但 HM 高 2.18%
- **vs PromptSRC**：PromptSRC 用自一致性正则化，EvoPrompt 直接控制进化轨迹，EvoPrompt HM 高 0.76%
- **vs DePT/DualPrompt**：持续学习系的特征分离方法，EvoPrompt 方向冻结思路更轻量且不需要任务标识

## 评分

⭐⭐⭐⭐

- **新颖性** ⭐⭐⭐⭐：方向-幅度解耦防遗忘思路新颖，FGR 有理论推导支撑
- **实验充分度** ⭐⭐⭐⭐：4 类评测 + 详细消融，但缺少大 backbone 验证
- **写作质量** ⭐⭐⭐⭐：动机清晰，数学推导规范
- **价值** ⭐⭐⭐⭐：为提示学习防遗忘提供新范式，方向冻结思路可迁移到其他 PEFT 方法

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] EvoPrompt: Evolving Prompt Adaptation for Vision-Language Models](evolving_prompt_adaptation_for_vision-language_models.md)
- [\[CVPR 2026\] Towards Calibrating Prompt Tuning of Vision-Language Models](towards_calibrating_prompt_tuning_of_vision-language_models.md)
- [\[CVPR 2026\] Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation](decoupling_stability_and_plasticity_for_multi-modal_test-time_adaptation.md)
- [\[CVPR 2026\] DeAR: Fine-Grained VLM Adaptation by Decomposing Attention Head Roles](dear_fine-grained_vlm_adaptation_by_decomposing_attention_head_roles.md)
- [\[CVPR 2026\] DSCA: Dynamic Subspace Concept Alignment for Lifelong VLM Editing](dsca_dynamic_subspace_concept_alignment_for_lifelong_vlm_editing.md)

<!-- RELATED:END -->
