---
title: >-
  [论文解读] Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning
description: >-
  [CVPR 2025][模型压缩][少样本类增量学习] 本文提出 Tri-WE 方法通过在权重空间插值 base、前一 session 和当前 session 三个分类头来更新整个模型（而非冻结特征提取器），并用 amplified data 知识蒸馏（ADKD）缓解少样本场景下的遗忘问题，在 miniImageNet/CUB200/CIFAR100 上达到 FSCIL SOTA。
tags:
  - CVPR 2025
  - 模型压缩
  - 少样本类增量学习
  - 权重空间集成
  - 知识蒸馏
  - 灾难性遗忘
  - 数据增强
---

# Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning

**会议**: CVPR 2025  
**arXiv**: [2506.15720](https://arxiv.org/abs/2506.15720)  
**代码**: 无  
**领域**: 持续学习 / 少样本增量学习  
**关键词**: 少样本类增量学习, 权重空间集成, 知识蒸馏, 灾难性遗忘, 数据增强

## 一句话总结

本文提出 Tri-WE 方法通过在权重空间插值 base、前一 session 和当前 session 三个分类头来更新整个模型（而非冻结特征提取器），并用 amplified data 知识蒸馏（ADKD）缓解少样本场景下的遗忘问题，在 miniImageNet/CUB200/CIFAR100 上达到 FSCIL SOTA。

## 研究背景与动机

**领域现状**：少样本类增量学习（FSCIL）要求模型在每个增量 session 只有少量新类样本的情况下不断扩展分类能力。当前主流方法分为两派：(1) 固定特征提取器，只用类均值作为新类分类器（CEC、ALICE）；(2) 更新部分参数或集成多个分类器（S3C、SAVC、BiDistill）。

**现有痛点**：(1) 固定特征提取器限制了模型对新类的适应能力；(2) 更新整个模型容易导致灾难性遗忘和过拟合（如图 2 所示，更新模型后 base 类准确率显著下降）；(3) 知识蒸馏（KD）是 CIL 中常用的防遗忘手段，但在少样本场景中 KD 容易过拟合到有限样本上，导致蒸馏出偏斜知识而非泛化知识。

**核心矛盾**：更新模型能提升新类性能但损害旧类性能，不更新模型则新类适应性受限。这本质上是 stability-plasticity dilemma。

**本文目标**：在有效缓解灾难性遗忘和过拟合的前提下，允许整个模型被更新，从而兼顾 stability 和 plasticity。

**切入角度**：受 WiseFT（fine-tuned 和 原始 CLIP 权重插值）启发，将 base 模型的泛化能力、前一 session 模型的累积知识、当前 session 模型的新类适应能力通过权重空间插值融合。同时用数据混合增强来丰富 KD 的数据源。

**核心 idea**：三方权重空间集成（base + previous + current 分类头权重插值）+ 增强数据知识蒸馏（混合少样本数据后做 KD），使模型在持续学习中平滑过渡而非突变。

## 方法详解

### 整体框架

在第 $t$ 个增量 session，模型从第 $t-1$ session 初始化，给定 $N$-way $K$-shot 新类数据 $\mathcal{D}^{(t)}$ 和旧类原型缓存 $\mathcal{M}$。通过 Tri-WE 插值三个分类头的权重形成统一分类头 $h_\phi^{(t)}$，用 $\mathcal{L}_{Cls}$ 和 $\mathcal{L}_{Cls-Old}$ 训练；同时将少样本数据增强后做 ADKD（特征+logit 蒸馏）防遗忘。部署时只需一个分类头。

### 关键设计

1. **三方权重空间集成 (Tri-WE)**:

    - 功能：在权重空间融合三个 session 模型的分类头，实现知识的持续平滑传递
    - 核心思路：维护三个分类头——$h_{\phi_0}$（base session 固定不变）、$h_{\phi_{old}}^{(t)}$（从 $t-1$ session 初始化、专注旧类）、$h_{\phi_{all}}^{(t)}$（覆盖所有类）。按类别分三种情况插值：base 类（三方加权 $\bar\alpha_1 \phi_0^n + \bar\alpha_2 \phi_{old}^n + \bar\alpha_3 \phi_{all}^n$）、中间 session 旧类（双方加权 $\bar\alpha_4 \phi_{old}^n + \bar\alpha_5 \phi_{all}^n$）、新类（直接用 $\phi_{all}^n$）。插值权重通过两个可学习标量 $\alpha_1, \alpha_2$（初始化为 1.0）自动调节
    - 设计动机：base 模型权重空间经过大量数据训练具有良好的泛化性，前一模型承载了所有已学类的知识，当前模型适应了新类。三方插值避免决策边界的突变，同时不增加推理时的计算开销（只部署一个 $h_\phi^{(t)}$）

2. **增强数据知识蒸馏 (ADKD)**:

    - 功能：从前一模型蒸馏泛化知识到当前模型，同时避免对少样本数据的过拟合
    - 核心思路：将 $NK$ 个少样本训练样本通过随机混合（如 CutMix、MixUp）增强到 $16NK$ 个。在增强数据上计算两层蒸馏损失：特征级 $\mathcal{L}_{feat} = \mathbb{E}\|g_\theta^{(t-1)}(x) - g_\theta^{(t)}(x)\|_2$ 和 logit 级 $\mathcal{L}_{logit} = \mathbb{E}[-f_{\theta,\phi}^{(t-1)}(x) \log f_{\theta,\phi}^{(t)}(x)]$。前一模型参数冻结
    - 设计动机：直接在少量原始样本上做 KD 会过拟合到这些样本的特定模式，无法蒸馏出泛化知识。通过数据混合增强产生更丰富的数据分布，使 KD 能提取前一模型的泛化能力

3. **Base Session 训练增强**:

    - 功能：确保 base 模型具有强泛化能力
    - 核心思路：采用 ALICE 的 base 训练技术，额外增加几何分类辅助头——将输入做 $B$ 种几何变换（旋转等），训练辅助分类器预测变换类型。此头仅用于 base 训练，增量 session 不使用
    - 设计动机：几何变换分类有助于模型学习数据的内在结构，提升对新任务的泛化能力

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{Cls} + \gamma_1 \mathcal{L}_{Cls-Old} + \gamma_2 \mathcal{L}_{ADKD}$，其中 $\gamma_1 = 1.2, \gamma_2 = 10.0$。增量 session 中特征提取器学习率为 0.001，分类头 $\phi_{all}$ 学习率为 0.1。backbone 为 ResNet18，SGD 优化器。

## 实验关键数据

### 主实验

**miniImageNet (9 sessions)**：

| 方法 | Session 0 | Session 8 (最终) | Avg |
|------|----------|----------------|-----|
| ALICE (ECCV'22) | 80.60 | 55.70 | 63.99 |
| NC-FSCIL (ICLR'23) | 84.02 | 58.31 | 67.82 |
| SAVC (CVPR'23) | 81.12 | 57.11 | 67.05 |
| OrCo (CVPR'24) | 83.30 | 58.08 | 67.12 |
| **Ours** | **84.13** | **60.13** | **70.62** |

最终 session 相比 NC-FSCIL 提升 +1.82，相比 OrCo +2.05。

### 消融实验

CUB200 和 CIFAR100 上同样取得 SOTA。CUB200 上 base/last/avg 准确率均优于 NC-FSCIL、SAVC 和 YourSelf。

### 关键发现

- 更新整个模型确实能提升新类性能（图 2 上半），关键在于如何控制遗忘
- Tri-WE 的三方插值比双方插值（WiseFT）更适合多 session 增量场景
- 可学习的插值权重 $\alpha_1, \alpha_2$ 能自动适应不同 session 的需求
- 数据增强后的 KD 比原始数据 KD 效果好得多
- 在最终 session 上相比所有先前方法有一致且显著的提升

## 亮点与洞察

- **权重空间集成**思路简洁有效——三个分类头在权重空间而非输出空间融合，不增加推理开销
- 将 WiseFT 从两方扩展为三方（加入 base 模型锚定点）适合增量学习的多 session 特性
- ADKD 用简单的 CutMix/MixUp 就能有效增强 KD 数据源，实现成本极低
- 按类别分层插值（base 类/中间旧类/新类用不同插值组合）的设计体现了对不同类别知识来源的精准判断

## 局限与展望

- 仍需为每个旧类维护一个原型，内存开销随 session 线性增长
- 几何辅助分类的 base 训练策略是否对所有数据集都有效未充分讨论
- 在更长的增量序列（如 20+ sessions）上的表现未验证
- 特征提取器虽然被更新但学习率极小（0.001），其实接近"微调"而非"真正更新"
- $\gamma_1, \gamma_2$ 等超参的敏感性未分析

## 相关工作与启发

- **WiseFT** 的权重插值思想从 CLIP 适配推广到增量学习
- **Model Soup** 的多模型权重平均概念被借鉴为三方加权
- ADKD 与 CutMix/MixUp 的结合提供了在低数据场景下做有效 KD 的简单方案

## 评分

- **新颖性**: 7/10 — Tri-WE 是 WiseFT 的自然扩展，ADKD 也较直观
- **实验充分度**: 8/10 — 三个标准 benchmark，与大量方法对比
- **写作质量**: 8/10 — 动机清晰，方法描述明确
- **价值**: 7/10 — FSCIL SOTA 但方法的可推广性有待验证

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Logits DeConfusion with CLIP for Few-Shot Learning](logits_deconfusion_with_clip_for_few-shot_learning.md)
- [\[CVPR 2025\] CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)
- [\[CVPR 2025\] Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning](adapter_merging_with_centroid_prototype_mapping_for_scalable_class-incremental_l.md)
- [\[CVPR 2025\] Incremental Object Keypoint Learning (KAMP)](incremental_object_keypoint_learning.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](../../ICCV2025/model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)

<!-- RELATED:END -->
