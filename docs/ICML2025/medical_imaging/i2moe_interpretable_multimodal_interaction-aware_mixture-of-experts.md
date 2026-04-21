---
title: >-
  [论文解读] I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts
description: >-
  [ICML 2025][医学图像][多模态] I2MoE 提出了一种可解释的多模态交互感知混合专家框架，通过四种交互专家（唯一性×2 + 协同 + 冗余）结合弱监督交互损失显式建模模态间的异质交互，并通过重加权模型提供样本级和数据集级的可解释性，在 ADNI 数据集上提升准确率 5.5%。
tags:
  - ICML 2025
  - 医学图像
  - 多模态
  - mixture-of-experts
  - interpretability
  - partial information decomposition
  - modality interaction
---

# I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts

**会议**: ICML 2025  
**arXiv**: [2505.19190](https://arxiv.org/abs/2505.19190)  
**代码**: https://github.com/Raina-Xin/I2MoE  
**领域**: 医学图像 / 多模态VLM  
**关键词**: multimodal fusion, mixture-of-experts, interpretability, partial information decomposition, modality interaction

## 一句话总结

I2MoE 提出了一种可解释的多模态交互感知混合专家框架，通过四种交互专家（唯一性×2 + 协同 + 冗余）结合弱监督交互损失显式建模模态间的异质交互，并通过重加权模型提供样本级和数据集级的可解释性，在 ADNI 数据集上提升准确率 5.5%。

## 研究背景与动机

**领域现状**：多模态融合是多模态学习的核心任务。现有融合方法（早期融合、晚期融合、Transformer 等）通常用同一套参数处理所有类型的模态交互，不区分不同模态间的关系性质。

**现有痛点**：偏信息分解（PID）理论将多模态信息分解为四种类型：模态1的唯一信息、模态2的唯一信息、协同信息（需组合两个模态才涌现的信息）和冗余信息（两个模态共享的信息）。但 PID 停留在理论分析层面，未被整合到端到端深度学习框架中。现有尝试要么仅处理成对交互（Wörtwein et al., 2022），要么需要单独估计每种交互（MMoE, Yu et al., 2024），要么缺乏可解释性（Dufumier et al., 2024）。

**核心矛盾**：多模态数据中不同样本可能依赖截然不同的交互模式——有的样本主要依赖图像唯一信息，有的依赖协同信息——但现有方法用统一方式处理所有样本，既损失了性能也缺乏可解释性。

**本文目标** (1) 在端到端框架中显式建模四种模态交互；(2) 提供样本级和数据集级的可解释性；(3) 与任意融合骨干网络兼容。

**切入角度**：用混合专家（MoE）架构天然匹配四种交互类型——每个专家负责一种交互，通过弱监督损失让专家特化。

**核心 idea**：用四种交互专家 + 弱监督扰动损失显式建模 PID 四要素，加上重加权模型提供可解释的多粒度决策分析。

## 方法详解

### 整体框架

I2MoE 在标准多模态融合流程中插入一层混合专家结构。输入数据经过模态特定编码器后，四个交互专家各自用独立参数进行融合并输出预测。重加权模型根据输入分配各专家权重，加权组合为最终预测。训练时额外进行模态扰动前向传播以计算交互损失。推理时仅需一次完整前向传播。

### 关键设计

1. **四种交互专家的设计与弱监督训练**:

    - 功能：让四个结构相同但参数独立的融合模型各自特化于捕捉一种交互类型
    - 核心思路：训练时对每个样本执行额外的扰动前向传播——将某个模态的嵌入替换为随机向量，模拟单模态场景。然后用扰动输出作为弱监督信号训练专家：
        - **唯一性专家1**：完整预测应接近仅保留模态1的预测（正样本），应远离仅保留模态2的预测（负样本）
        - **唯一性专家2**：与上述对称
        - **协同专家**：完整预测应远离任何单模态预测——因为协同信息需要两个模态缺一不可
        - **冗余专家**：完整预测应接近任何单模态预测——因为冗余信息在单模态下也保留
    - 分类任务用 Triplet Margin Loss 建模唯一性交互，Cosine Similarity 建模协同和冗余。回归任务统一用 MSE
    - 设计动机：通过模态扰动模拟 PID 的信息操作，没有直接计算互信息而是用预测一致性作为代理

2. **可学习的重加权模型**:

    - 功能：对每个样本自适应分配各交互专家的权重，提供样本级可解释性
    - 核心思路：用 MLP 将拼接的模态嵌入映射到四维权重向量 $[w_{\mathrm{uni1}}, w_{\mathrm{uni2}}, w_{\mathrm{syn}}, w_{\mathrm{red}}]$，最终预测为 $\hat{y} = \sum_i w_i \cdot \hat{y}_i$。**局部解释**：分析单个样本的权重分布，了解该预测主要依赖哪种交互。**全局解释**：统计测试集上所有样本的权重分布，揭示数据集级的交互因式
    - 设计动机：相比 Simple-Weight（全局共享权重），MLP 重加权允许对不同样本分配不同权重，消融实验证明这一设计在 ADNI 上带来 4.93% 的准确率提升

3. **向更多模态的扩展**:

    - 功能：将双模态框架推广到 $n$ 个模态
    - 核心思路：唯一性专家数量线性增长为 $n$ 个（每个模态一个），协同和冗余专家各保留一个，总专家数 $n+2$。交互损失做对应调整——唯一性专家 $i$ 的负样本为遮蔽模态 $i$ 的输出，正样本为遮蔽其他模态的输出
    - 设计动机：避免组合爆炸——若每对模态各设专家，复杂度为 $O(n^2)$ 或更高

### 损失函数

总损失 $L_{\text{total}} = L_{\text{task}} + \frac{\lambda_{\text{int}}}{E} \sum_{i=1}^{E} L_{\text{int}}^i$，其中 $L_{\text{task}}$ 是标准任务损失（重加权预测 vs 真值），$L_{\text{int}}^i$ 是第 $i$ 个专家的交互特化损失。$\lambda_{\text{int}}$ 平衡两项损失。

## 实验关键数据

### 主实验表格

I2MoE-MulT 与 7 种融合方法在 5 个数据集上的对比：

| 方法 | ADNI Acc | ADNI AUROC | MIMIC Acc | IMDB Micro-F1 | MOSI Acc |
|---|---|---|---|---|---|
| Early Fusion | 52.01 | 65.69 | 67.63 | 56.10 | 72.16 |
| MulT | 59.57 | 77.21 | 72.42 | 59.68 | 68.80 |
| MoE++ | 58.08 | 75.18 | 72.51 | 58.15 | 70.85 |
| SwitchGate | 62.28 | 79.70 | 70.98 | 55.92 | 72.35 |
| **I2MoE-MulT** | **65.08** | **81.09** | 69.78 | **61.00** | 71.91 |

### 消融表格

关键组件消融（ADNI 数据集）：

| 消融变体 | 描述 | Accuracy 变化 |
|---|---|---|
| No-Interaction | 移除交互损失 | 显著下降——证明交互损失对专家特化至关重要 |
| Simple-Weight | 全局共享权重替代 MLP | 下降约 4.93%——证明自适应重加权的必要性 |
| Synergy-Redundancy | 仅保留协同和冗余专家 | 下降——证明唯一性专家不可或缺 |
| Latent-Contrastive | 交互损失施加在嵌入层 | 下降——输出空间的交互损失更有效 |

### 关键发现

- I2MoE 在 ADNI 上比 vanilla MulT 提升 5.5% 准确率，MOSI 上提升 3%
- 四个专家的预测高度分化：ADNI 和 MIMIC 数据中 81%/85% 的样本在专家间存在分歧，当分歧存在时 I2MoE 仍能正确预测约 49%/64% 的样本
- 全局解释分析显示：ADNI 权重分布较均匀（四种交互均重要），MIMIC 权重变异性大（患者间差异显著），ENRICO 以截图模态的唯一性为主导
- I2MoE 与三种不同融合骨干（SwitchGate、InterpretCC、MoE++）结合均能带来提升，证明框架的通用性

## 亮点与洞察

- 首次将 PID 信息分解理论融入端到端深度学习框架，理论基础扎实
- 弱监督交互损失的设计巧妙——通过模态扰动模拟 PID 操作，无需显式计算互信息
- 提供了真正有意义的可解释性——不仅告诉"哪个模态重要"，还告诉"何种交互重要"

## 局限性

- 交互损失中用随机向量替换模态嵌入只是 PID 的近似，替换策略的选择可能影响专家特化质量
- 在 MIMIC 数据集上 I2MoE 的准确率不升反降（特别是与 InterpretCC 结合时降 2.49%），尽管 AUROC 上升
- 扩展到更多模态时虽然避免了组合爆炸，但统一的协同/冗余专家能否捕捉模态子集间的交互值得商榷
- 训练时需要 $n+1$ 次前向传播（$n$ 次扰动 + 1 次完整），计算开销非平凡

## 相关工作与启发

- MMoE（Yu et al., 2024）是最直接的前序工作，也用交互专家但作为预处理步骤而非端到端学习
- PID 框架（Liang et al., 2023）提供理论基础，I2MoE 是首个将其端到端化的深度学习实现
- InterpretCC（Swamy et al., 2024a）关注可解释融合但不显式建模交互类型
- 启发：在医疗多模态场景中，可解释性不仅是锦上添花而是刚需——I2MoE 的设计思路可迁移到更多需要决策解释的领域

## 评分

⭐⭐⭐⭐ （7/10）

理论基础（PID）与工程实现（MoE）的结合自然优雅，弱监督交互损失设计巧妙，可解释性分析（局部+全局）有实际价值。实验涵盖医疗和通用数据集共5个，消融充分。不足之处在于部分数据集上效果有限（MIMIC 准确率下降），计算开销未得到足够关注，且对 PID 理论的近似程度缺乏严格分析。整体是一篇扎实的多模态融合方法论文。

从应用角度看，I2MoE 在医疗场景中的价值尤为突出：ADNI 数据集上 5.5% 的准确率提升对阿尔茨海默症诊断具有临床意义，而权重可解释性可帮助医生理解"哪种信息来源"驱动了预测——这在医疗 AI 审批和临床信任建设中至关重要。框架的骨干无关特性也意味着可迅速适配到其他医疗多模态融合系统中。

<!-- RELATED:START -->

## 相关论文

- [MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](../../NeurIPS2025/medical_imaging/more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)
- [Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation](../../NeurIPS2025/medical_imaging/mamba_goes_home_hierarchical_soft_mixture-of-experts_for_3d_medical_image_segmen.md)
- [DFLMoE: Decentralized Federated Learning via Mixture of Experts for Medical Data](../../CVPR2025/medical_imaging/dflmoe_decentralized_federated_learning_via_mixture_of_experts_for_medical_data_.md)
- [Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis](../../NeurIPS2025/medical_imaging/dual_mixture-of-experts_framework_for_discrete-time_survival_analysis.md)
- [Bayesian Inference for Correlated Human Experts and Classifiers](bayesian_inference_for_correlated_human_experts_and_classifiers.md)

<!-- RELATED:END -->
