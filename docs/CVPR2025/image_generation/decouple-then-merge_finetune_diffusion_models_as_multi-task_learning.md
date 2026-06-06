---
title: >-
  [论文解读] Decouple-Then-Merge: Finetune Diffusion Models as Multi-Task Learning
description: >-
  [CVPR 2025][图像生成][扩散模型微调] 本文将扩散模型训练视为多任务学习问题，提出Decouple-then-Merge（DeMe）框架——先将时间步分组微调多个专用模型以消除梯度冲突，再通过参数空间合并回单一模型，在不增加推理开销的情况下显著提升生成质量。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "扩散模型微调"
  - "多任务学习"
  - "模型合并"
  - "梯度冲突"
  - "时间步解耦"
---

# Decouple-Then-Merge: Finetune Diffusion Models as Multi-Task Learning

**会议**: CVPR 2025  
**arXiv**: [2410.06664](https://arxiv.org/abs/2410.06664)  
**代码**: [GitHub](https://github.com/qianli-ma/deme)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 扩散模型微调, 多任务学习, 模型合并, 梯度冲突, 时间步解耦

## 一句话总结
本文将扩散模型训练视为多任务学习问题，提出Decouple-then-Merge（DeMe）框架——先将时间步分组微调多个专用模型以消除梯度冲突，再通过参数空间合并回单一模型，在不增加推理开销的情况下显著提升生成质量。

## 研究背景与动机

**领域现状**：扩散模型通过学习多步去噪过程生成图像，标准做法是所有时间步共享模型参数。虽然这促进了知识共享和训练效率，但不同时间步的去噪任务实际存在显著差异——大时间步生成低频基础内容，小时间步生成高频细节。

**现有痛点**：不同时间步之间存在梯度冲突。作者通过实验发现非相邻时间步的梯度余弦相似度很低，说明不同去噪任务在训练时相互干扰（负迁移），降低整体生成质量。现有的损失重加权方法只能缓解不能根治这一问题。

**核心矛盾**：一方面需要跨时间步的知识共享以提高效率；另一方面梯度冲突导致负迁移。时间步级别的模型集成能避免冲突，但带来N倍的存储和显存开销（如6个独立模型），不实用。

**本文目标**：在消除梯度冲突的同时保持知识共享，且推理时不引入额外开销。

**切入角度**：从多任务学习和模型合并两个方向切入——先解耦消除冲突，再合并保留知识。

**核心 idea**：将时间步分为N个不重叠区间分别微调N个模型，用专门设计的训练技巧防止过拟合，最后通过task vector加权合并回一个模型。

## 方法详解

### 整体框架
从预训练扩散模型出发，将总时间步 $[0,T)$ 划分为 $N$ 个不重叠区间。对每个区间从预训练模型初始化一个副本进行微调，微调时引入三种训练技巧。微调完成后，计算每个模型与预训练模型的参数差（task vector），加权合并回单一模型用于推理。

### 关键设计

1. **解耦微调 + 三种训练技巧**:

    - 功能：消除梯度冲突的同时保持跨时间步知识
    - 核心思路：(a) Channel-wise Projection——在中间特征上加可学习的通道映射矩阵 $W \in \mathbb{R}^{C \times C}$（初始化为单位矩阵），因为微调前后的差异主要在通道维度而非空间维度；(b) Consistency Loss——约束微调后模型输出与原始模型输出的差距，防止过度偏离；(c) Probabilistic Sampling——以概率 $1-p$ 从对应区间采样时间步，以概率 $p$ 从全局采样，保留对其他区间的记忆
    - 设计动机：单纯解耦会导致每个模型只会做对应区间的去噪而遗忘其他区间。Channel-wise Projection的设计来自于实验观察：微调前后的激活差异主要集中在通道维度

2. **Task Vector 模型合并**:

    - 功能：将N个微调模型无损合并为单一推理模型
    - 核心思路：计算每个模型的 task vector $\tau_i = \theta_i - \theta$，然后 $\theta_{merged} = \theta + \sum_{i=1}^N w_i \tau_i$。合并权重 $w_i$ 通过网格搜索获得最优组合
    - 设计动机：模型合并技术已被证明能整合不同任务/数据上微调模型的知识，这里首次应用于扩散模型跨时间步的合并。合并后模型大小与原始相同，零额外推理开销

3. **Loss Landscape 分析**:

    - 功能：解释为什么解耦微调能改善已收敛的预训练模型
    - 核心思路：可视化分析显示，预训练模型在全时间步 $[0,1000)$ 上处于临界点（梯度为零、等高线稀疏），但对于各子区间，模型处于等高线密集区域（存在优化方向）。解耦使模型能逃离全局临界点
    - 设计动机：提供理论直觉解释为什么对已收敛模型进一步微调能持续改善

### 损失函数 / 训练策略
总体损失为标准去噪损失 + Consistency Loss：$\mathcal{L} = \|\epsilon - \epsilon_{\theta_i}(x_t, t)\|^2 + \|\epsilon_\theta(x_t, t) - \epsilon_{\theta_i}(x_t, t)\|^2$。每个模型微调20K iterations（N=4时总计80K等效iterations）。

## 实验关键数据

### 主实验（无条件生成，DDPM）

| 数据集 | 指标(FID↓) | DeMe(合并后) | 预训练 | Min-SNR-γ | ANT-UW | 提升 |
|--------|-----------|-------------|--------|-----------|--------|------|
| CIFAR10 | FID | **3.51** | 4.42 | 5.77 | 4.21 | -0.91 |
| LSUN-Church | FID | **7.27** | 10.69 | 10.82 | 10.43 | -3.42 |
| LSUN-Bedroom | FID | **5.84** | 6.46 | 6.41 | 6.48 | -0.62 |

### 消融实验（CIFAR10, DDIM 100步）

| 配置 | FID↓ | 说明 |
|------|------|------|
| N=1, 无技巧（传统） | 4.40 | 基线 |
| N=1 + Channel Projection | 4.45 | 不解耦时CP反而有害 |
| N=8 + Prob. Sampling | 4.32 | 解耦后开始改善 |
| N=8 + PS + CL | 4.27 | 加 Consistency Loss 继续改善 |
| N=8 + PS + CL + CP | **3.87** | 全部技巧组合最优，FID降0.53 |

### 关键发现
- 模型合并方案甚至优于模型集成方案（例如 LSUN-Church 上合并 FID=7.27 vs 集成 FID=9.57），说明合并产生了超越简单集成的效果
- Channel-wise Projection 在不解耦的情况下反而有害，只有配合解耦才能发挥作用
- 所有损失重加权baseline在微调设定下几乎无效甚至有害，说明它们无法真正解决梯度冲突
- Stable Diffusion 上同样有效：MS-COCO FID降0.36，CLIP Score升0.23

## 亮点与洞察
- 从多任务学习角度重新审视扩散模型训练，梯度冲突的发现和可视化很有说服力。这一视角可迁移到任何多时间步共享参数的生成模型
- 合并后模型甚至优于集成模型，这在直觉上出乎意料——可能因为合并在参数空间中找到了比各个微调模型更好的平衡点
- Loss Landscape 分析揭示了"看似收敛但仍有优化空间"的现象，对理解大模型训练动态很有启发

## 局限与展望
- 需要微调N个完整模型的训练成本，虽然总iterations相同但需N倍显存（可通过顺序微调缓解）
- 合并权重通过网格搜索获得，扩展到更多区间（大N）时搜索空间增长
- 未探索不均匀分区——不同时间步区间可能需要不同大小的划分
- 可考虑与LoRA等参数高效微调方法结合，进一步降低训练成本

## 相关工作与启发
- **vs Loss Reweighting (Min-SNR, P2)**: 这些方法试图通过调整不同时间步的损失权重来平衡训练，但实验证明在微调设定下基本无效。DeMe从根本上分离了优化方向
- **vs Timestep Ensemble**: 如DMP使用6个独立模型，存储和显存6倍增长。DeMe通过合并将开销降为零
- **vs ANT**: ANT引入MTL优化方法（NashMTL等）到扩散模型，效果不如DeMe的解耦-合并范式

## 评分
- 新颖性: ⭐⭐⭐⭐ 将扩散训练视为MTL并用解耦合并解决是新思路
- 实验充分度: ⭐⭐⭐⭐⭐ 6个数据集、DDPM和SD、多种baseline、完整消融
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、可视化丰富、分析深入
- 价值: ⭐⭐⭐⭐ 通用微调框架，可广泛应用于扩散模型

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] ToMA: Token Merge with Attention for Diffusion Models](../../ICML2025/image_generation/toma_token_merge_with_attention_for_diffusion_models.md)
- [\[ICML 2026\] EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](../../ICML2026/image_generation/evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)
- [\[CVPR 2025\] Generative Modeling of Class Probability for Multi-Modal Representation Learning](generative_modeling_of_class_probability_for_multi_modal_representation_learning.md)
- [\[ICCV 2025\] MMAIF: Multi-task and Multi-degradation All-in-One for Image Fusion with Language Guidance](../../ICCV2025/image_generation/mmaif_multi-task_and_multi-degradation_all-in-one_for_image_fusion_with_language.md)
- [\[AAAI 2026\] Conditional Diffusion Model for Multi-Agent Dynamic Task Decomposition](../../AAAI2026/image_generation/conditional_diffusion_model_for_multi-agent_dynamic_task_dec.md)

</div>

<!-- RELATED:END -->
