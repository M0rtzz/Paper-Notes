---
title: >-
  [论文解读] Learning by Neighbor-Aware Semantics, Deciding by Open-form Flows: Towards Robust Zero-Shot Skeleton Action Recognition
description: >-
  [CVPR 2026][目标检测][零样本学习] Flora 通过邻居感知语义校准实现稳健的骨架-语义跨模态对齐，并利用无噪声流匹配构建分布感知的开放式分类器，在零样本骨架动作识别上取得 SOTA，尤其在低样本训练场景中表现突出。
tags:
  - CVPR 2026
  - 目标检测
  - 零样本学习
  - 骨架动作识别
  - 流匹配
  - 跨模态对齐
  - 语义校准
---

# Learning by Neighbor-Aware Semantics, Deciding by Open-form Flows: Towards Robust Zero-Shot Skeleton Action Recognition

**会议**: CVPR 2026  
**arXiv**: [2511.09388](https://arxiv.org/abs/2511.09388)  
**代码**: https://github.com/cseeyangchen/Flora  
**领域**: 动作识别  
**关键词**: 零样本学习, 骨架动作识别, 流匹配, 跨模态对齐, 语义校准

## 一句话总结

Flora 通过邻居感知语义校准实现稳健的骨架-语义跨模态对齐，并利用无噪声流匹配构建分布感知的开放式分类器，在零样本骨架动作识别上取得 SOTA，尤其在低样本训练场景中表现突出。

## 研究背景与动机

零样本骨架动作识别旨在让模型识别训练阶段未见过的骨架动作类别，这在现实应用中非常重要，因为收集覆盖所有动作类别的大规模数据集是不现实的。现有方法普遍遵循"对齐-分类"范式，但面临两个根本性问题：

1. **语义锚点不可靠**：无论是 LLM 生成的描述性语义（缺乏骨架显式引导而偏差）还是参数高效微调的语义（容易过拟合到已见类别），都会产生刚性、有缺陷的语义锚点，导致点对点的跨模态对齐不稳定——某些类别被正确对齐，其他类别却连带错位。
2. **分类器僵化**：生成式方法合成未见类特征来训练线性分类器，但决策边界是静态的，无法适应新类别；嵌入式方法通过余弦相似度匹配，但将特征压缩为单一向量导致信息损失和粗粒度分类。

核心矛盾在于：语义偏差和静态分类器共同限制了模型在零样本场景下的泛化能力。作者的核心洞察是：虽然单个语义锚点可能有偏差（如地图上标注有误的地标），但其与周围锚点的邻域关系仍然可靠。基于此，Flora 提出"先用邻域信息主动校准语义，再用流匹配桥接模态分布差异"的新范式。

## 方法详解

### 整体框架

Flora 分为**学习阶段**和**决策阶段**。学习阶段：对每个类别语义，首先利用邻近类别的上下文信息进行邻居感知语义校准，然后通过几何一致性目标实现跨模态 VAE 对齐。决策阶段：利用无噪声流匹配在语义和骨架潜在分布之间建立分布传输桥，通过 token 级别的速度场预测实现细粒度分类。

### 关键设计

1. **邻居感知语义校准 (Neighbor Semantic Attunement)**:
    - 功能：主动修正 LLM 生成的有偏差语义表示，使其更加稳健
    - 核心思路：对每个语义特征 $\mathbf{F}_{a_y}$，通过余弦相似度选取 Top-$k$ 邻居类别语义，利用图聚合方式融合邻域信息得到上下文化的语义 $\mathbf{O}_{a_y} = \mathbf{F}_{a_y} + \frac{\tau}{k} \sum w_i \mathbf{F}_i$，其中 $\tau$ 防止过平滑，$w_i$ 是相似度权重。这使单点语义扩展为具有方向感的区域语义
    - 设计动机：类比导航——即使地图上某个地标位置有误，但通过参考周围稳定地标仍可确定大致方位。利用已知且可靠的邻域拓扑关系来纠正个体偏差

2. **几何一致性对齐 (Geometric Consistency Alignment)**:
    - 功能：在潜在空间中实现骨架和语义的稳健跨模态对齐
    - 核心思路：沿用双 VAE 架构，但用直接对齐两个模态的分布几何结构（均值和方差匹配 $\mathcal{L}_{Geo} = \|\mu_s - \mu_a\|_2^2 + \|\sigma_s^2 - \sigma_a^2\|_2^2$）替代传统的 KL 散度正则化到标准高斯先验。结合保留了内/跨重构的目标
    - 设计动机：传统做法将两个模态都正则化到标准高斯，虽对齐了但丧失了类间可分性。直接约束分布几何一致性既缩小模态鸿沟又保留判别性

3. **无噪声开放式流分类器 (Noise-free Open-form Flow Classifier)**:
    - 功能：基于流匹配的分布传输实现细粒度、可扩展的零样本分类
    - 核心思路：将流匹配从生成式任务推广到判别式场景。源分布 $\mathcal{N}_a$ 直接使用语义潜在分布（不近似为高斯、不注入噪声），目标分布为骨架潜在分布 $\mathcal{N}_s$。通过学习 token 级速度场 $v_\theta(z_t, t)$，并引入对比正则化（拉远非配对类别的速度预测误差），使不同类别的传输路径具有判别性。推理时，对每个候选类别计算速度误差 $\varepsilon_y = \|v_\theta(z_t^y, t) - v_y^*\|_2$，选取误差最小者作为分类结果
    - 设计动机：不同类别的语义-骨架分布传输路径天然不同（因为源和目标分布都按类别有差异），这使速度场本身就具有判别力。相比静态线性分类器，流分类器可即插即用扩展到新类别；相比余弦相似度，保留了 token 级别的细粒度信息

### 损失函数 / 训练策略

训练分两阶段：先独立优化对齐目标 $\mathcal{L}_{Align} = \mathcal{L}_{Re} + \lambda_{Align} \cdot \mathcal{L}_{Geo}$，冻结参数后再训练对比流目标 $\mathcal{L}_{ConFlow}$，后者同时包含流匹配损失和对比正则化项 $\lambda_{Flow}$。GZSL 推理时引入阈值 $\gamma$ 通过速度误差比率判断样本属于已见还是未见类别域。

## 实验关键数据

### 主实验

| 数据集 | 分割 | 指标 | Flora | 之前SOTA | 提升 |
|--------|------|------|-------|----------|------|
| NTU-60 Xsub | 55/5 | ZSL Acc | 86.3% | 86.9% (Neuron/FS-VAE) | 持平 |
| NTU-60 Xsub | 55/5 | GZSL H | 77.4% | 75.7% (FS-VAE) | +1.7 |
| NTU-60 Xsub | 48/12 | ZSL Acc | 65.3% | 62.7% (Neuron) | +2.6 |
| NTU-60 Xsub | 48/12 | GZSL H | 60.5% | 59.1% (Neuron) | +1.4 |
| NTU-120 Xsub | 110/10 | ZSL Acc | 80.7% | 74.8% (InfoCPL) | +5.9 |
| NTU-120 Xsub | 110/10 | GZSL H | 66.1% | 63.3% (FS-VAE/Neuron) | +2.8 |
| NTU-120 Xsub | 96/24 | ZSL Acc | 66.4% | 65.1% (TDSM) | +1.3 |

### 消融实验（低样本训练，ZSL Acc）

| 配置 | NTU-60 55/5 (1%) | NTU-60 55/5 (10%) | NTU-120 110/10 (10%) |
|------|-------------------|--------------------|-----------------------|
| Flora | **84.2%** | **85.6%** | **75.4%** |
| FS-VAE | 83.2% | 84.1% | 62.1% |
| CADA-VAE | 76.6% | 76.9% | 39.1% |
| SynSE | 44.3% | 42.8% | 33.2% |

### 关键发现

- Flora 在 NTU-120 的 110/10 分割上表现特别突出（ZSL 80.7%），大幅领先之前 SOTA（InfoCPL 74.8%），说明在更大类别集、更少未见类的场景中优势明显
- 在仅使用 1% 训练数据时，Flora 仍能达到 84.2%（NTU-60 55/5），显示出极强的低样本泛化能力，这主要归功于流分类器不依赖合成特征训练静态分类器
- GZSL 设置中，Flora 在 H 指标上普遍拉高，表示 seen/unseen 之间的平衡更好

## 亮点与洞察

- **将流匹配从生成式推广到判别式**是本文最核心的创新。传统流匹配都用于从噪声生成样本，Flora 首次提出用流的速度场做分类，且完全无噪声、无条件约束，非常新颖
- **邻居语义校准**思路简单而有效——不是训练更复杂的对齐模型去适应有偏差的语义，而是先在语义端做"预处理"，利用邻域关系纠偏。这个思路具有很强的迁移性，可应用到任何跨模态对齐任务
- 流分类器天然支持开集扩展（新类别只需提供语义即可），且是 token 级别的细粒度判别，这两个特性在未来的开放世界识别中价值很高

## 局限与展望

- 语义仍然依赖 LLM 生成的文本描述，虽然做了校准但根本性的语义-动作 gap 未完全解决
- 流分类器推理时需对每个候选类别计算速度误差，类别数增大时推理开销线性增长
- 两阶段训练（先对齐再训流）未尝试端到端联合优化，可能有进一步提升空间
- 实验主要集中在 NTU/PKU 数据集，更大规模或跨域泛化未充分验证

## 相关工作与启发

- **vs STAR/PURLS**: 它们通过 LLM 直接生成精细语义描述再做对齐，Flora 则在语义校准环节引入邻域上下文，更鲁棒
- **vs FS-VAE**: 同样基于 Cross-VAE 但用频域分析，Flora 替换了 KL 正则为几何一致性并引入流分类器，H 指标更高
- **vs CrossFlow/FlowTok**: 这些跨模态流匹配工作仍使用 KL 散度正则化源分布为近似高斯，Flora 完全去除了噪声和条件约束
- 流分类器的思路可启发多模态检索、跨模态匹配等任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将流匹配推广到判别式零样本分类场景，且无噪声无条件约束的设计非常优雅
- 实验充分度: ⭐⭐⭐⭐ 三个数据集多种分割协议，低样本实验有力，但缺少跨域评估
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，类比（地图导航）生动，公式推导完整
- 价值: ⭐⭐⭐⭐ 在零样本骨架动作识别领域推进明显，流分类器的思路有较好的迁移潜力

<!-- RELATED:START -->

## 相关论文

- [Zero-shot HOI Detection with MLLM-based Detector-agnostic Interaction Recognition](../../ICLR2026/object_detection/zero-shot_hoi_detection_with_mllm-based_detector-agnostic_interaction_recognitio.md)
- [TIACam: Text-Anchored Invariant Feature Learning with Auto-Augmentation for Camera-Robust Zero-Watermarking](tiacam_text-anchored_invariant_feature_learning_with_auto-augmentation_for_camer.md)
- [Evaluating Few-Shot Pill Recognition Under Visual Domain Shift](evaluating_fewshot_pill_recognition_under_visual_d.md)
- [Specificity-aware Reinforcement Learning for Fine-grained Open-world Classification](specificity-aware_reinforcement_learning_for_fine-grained_open-world_classificat.md)
- [ReHARK: Refined Hybrid Adaptive RBF Kernels for Robust One-Shot Vision-Language Adaptation](rehark_refined_hybrid_adaptive_rbf_kernels_for_rob.md)

<!-- RELATED:END -->
