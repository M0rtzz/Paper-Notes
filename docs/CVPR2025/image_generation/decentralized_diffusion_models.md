---
title: >-
  [论文解读] Decentralized Diffusion Models
description: >-
  [CVPR 2025][图像生成][去中心化训练] Decentralized Diffusion Models（DDM）提出了一种将扩散模型训练分布到完全隔离的计算集群上的方法，通过在数据分区上独立训练专家模型并用轻量路由器在推理时集成，证明该集成精确优化与单模型相同的全局 Flow Matching 目标，在 FLOP-for-FLOP 下优于单一大模型。
tags:
  - CVPR 2025
  - 图像生成
  - 去中心化训练
  - Flow Matching
  - 专家混合
  - 扩散模型
  - 大规模训练
---

# Decentralized Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2501.05450](https://arxiv.org/abs/2501.05450)  
**代码**: [https://decentralizeddiffusion.github.io/](https://decentralizeddiffusion.github.io/)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 去中心化训练, Flow Matching, 专家混合, 扩散模型, 大规模训练

## 一句话总结

Decentralized Diffusion Models（DDM）提出了一种将扩散模型训练分布到完全隔离的计算集群上的方法，通过在数据分区上独立训练专家模型并用轻量路由器在推理时集成，证明该集成精确优化与单模型相同的全局 Flow Matching 目标，在 FLOP-for-FLOP 下优于单一大模型。

## 研究背景与动机

**领域现状**：扩散模型在图像生成、视频建模、机器人控制等领域取得了突破性成果，但对训练计算量的需求持续增长。Stable Diffusion 1.5 消耗了超过6,000 A100 GPU 天，Meta 的 Movie Gen 更是在6,114 张 H100 上训练。大规模训练依赖于高带宽互连的中心化集群，推高了基础设施成本。

**现有痛点**：（1）数据并行训练需要在每一步同步梯度，要求高带宽网络互连，只有昂贵的中心化单体集群才能支撑；（2）大型集群面临电力供应、硬件故障和网络瓶颈等系统层面的挑战，形成脆弱的耦合系统；（3）学术研究者几乎无法承担如此规模的计算资源，导致扩散模型训练的门槛越来越高。

**核心矛盾**：训练高质量扩散模型需要海量算力和数据，但集中化训练模式要求所有 GPU 必须在同一高带宽网络中，这既昂贵又脆弱。能否像联邦学习一样将训练分散到多个独立节点上？

**本文目标**：设计一种可扩展的去中心化扩散模型训练框架，使得独立的计算集群/数据中心可以零通信地各自训练，然后在推理时组合。

**切入角度**：关键洞察是 Flow Matching 的边缘流可以自然分解为多个数据子集上的专家流的线性组合。这意味着在不同数据分区上独立训练的扩散模型集成在一起时，理论上等价于在全部数据上训练的单一模型。

**核心 idea**：将数据集聚类为 K 个分区，在每个分区上完全独立训练一个扩散专家模型（零跨模型通信），再训练一个轻量路由器预测每个专家的相关性。推理时通过路由器权重线性组合专家预测，集成还原全局 Flow Matching 目标。

## 方法详解

### 整体框架

DDM 训练分三步：（1）**数据分区**——使用 DINOv2 提取图像特征，通过多阶段聚类将数据集分为 K 个语义相关的子集；（2）**专家训练**——在每个数据子集上独立训练一个标准 Flow Matching 扩散模型（如 DiT），各专家之间完全无通信；（3）**路由器训练**——训练一个小型 DiT 分类器，预测给定噪声样本 $x_t$ 属于哪个数据子集。推理时，路由器为每个专家分配权重，专家预测加权组合即为最终输出。

### 关键设计

1. **Decentralized Flow Matching（DFM）目标**:

    - 功能：从理论上证明去中心化训练等价于全局优化
    - 核心思路：标准 Flow Matching 的边缘流为 $u_t(x_t) = \int_{x_0} u_t(x_t|x_0) p_t(x_t|x_0) q(x_0) / p_t(x_t) dx_0$。将数据分为 K 个不相交子集 $\{S_1, ..., S_K\}$ 后，可以证明边缘流分解为：$u_t(x_t) = \sum_{k=1}^{K} \frac{p_{t,S_k}(x_t)}{p_t(x_t)} \cdot v_{k,t}(x_t)$，其中 $\frac{p_{t,S_k}(x_t)}{p_t(x_t)}$ 是路由器权重（$x_t$ 来自子集 $S_k$ 的后验概率），$v_{k,t}(x_t)$ 是第 $k$ 个专家的流预测。每个专家独立优化标准 Flow Matching 损失 $\|v_{\theta,t}(x_t) - u_t(x_t|x_0)\|^2$
    - 设计动机：这个数学分解是整篇论文的理论基础，保证了去中心化训练不会损失表达能力——集成后的模型在理论上与全局训练的模型优化相同目标

2. **基于分类的路由器训练**:

    - 功能：在推理时预测每个专家对当前输入的相关性
    - 核心思路：路由器是一个小型 DiT-B（158M 参数），接收噪声样本 $x_t$ 和时间步 $t$ 作为输入，输出 K 维概率分布。训练目标是标准交叉熵分类损失——给定训练样本 $x_0$ 和其对应的聚类标签 $k$，在同一噪声调度下加噪得到 $x_t$，训练路由器预测正确的聚类标签。路由器独立于所有专家训练，额外增加约4%的训练 FLOPs 开销。使用 DiT 架构加上一个可学习的 CLS token，通过线性头解码为聚类 logits
    - 设计动机：路由器与专家完全解耦训练，不需要端到端梯度传播（不同于传统 MoE），这使得整个系统可以完全分布式运行。小型路由器的计算开销可以忽略不计

3. **Top-1 专家选择推理策略**:

    - 功能：在推理时高效地选择最相关的专家
    - 核心思路：虽然理论上需要所有专家的加权组合才能精确匹配全局流，但实际中大多数专家与给定输入不相关（路由器给出接近零的权重）。作者系统对比了多种推理策略：Full（所有专家加权和），Top-k（选前 k 个），Sample（按概率采样单个），Nucleus Sampling 等。实验发现 **Top-1 选择**（只用路由器权重最大的单个专家）在 FID 上表现最佳且 FLOPs 最低。这意味着推理时 DDM 的计算量与单一模型几乎相同
    - 设计动机：Top-1 选择类似于 MoE 中的稀疏激活，不仅节省计算还在实践中提升了性能——可能因为每个专家在其擅长的数据子分布上的预测比加权平均更准确。Sample 策略产生了巨大的方差，性能极差

### 蒸馏

对于部署场景，DDM 可以将 K 个专家蒸馏为单个稠密模型。用每个训练样本的聚类标签选择对应的专家作为教师，student 模型学习模仿教师预测。蒸馏在1/4 batch size 下训练，仅用1/3 FLOPs 就能匹配从头训练的单模型性能。

## 实验关键数据

### 主实验（DDM vs 单一模型，DiT XL/2 架构）

| 方法 | 数据集 | FID↓ | CLIP-FID↓ | GFLOPs | 说明 |
|------|--------|------|-----------|--------|------|
| Monolith | ImageNet | 12.81 | 5.58 | 308 | 标准单模型 |
| DDM 8-expert Top-1 | ImageNet | **9.84** | **5.48** | 334 | 8专家Top-1推理 |
| Monolith | LAION | — | — | 308 | 标准单模型 |
| DDM 8-expert Top-1 | LAION | **更低** | **更低** | 334 | 200k步=Monolith 800k步 |

### 消融实验（推理策略对比，8专家 ImageNet）

| 推理策略 | GFLOPs | FID↓ | CLIP-FID↓ | 说明 |
|---------|--------|------|-----------|------|
| Monolith | 308 | 12.81 | 5.58 | 基线 |
| Full (8) | 2490 | 10.52 | 5.83 | 全专家加权和 |
| Top-1 | 334 | **9.84** | **5.48** | 最佳策略 |
| Top-2 | 642 | 10.31 | 5.74 | 更多专家反而不好 |
| Sample-1 | 334 | 157.05 | 51.17 | 随机采样极不稳定 |
| Oracle | 308 | 10.46 | 5.83 | 用GT标签选专家 |

### 关键发现

- **8 专家是最优配置**：在 4/8/16 专家对比中，8 专家一致表现最好。4 专家模型能力不足，16 专家因为每个专家的 batch size 太小导致训练不充分
- **DDM FLOP-for-FLOP 优于单一模型**：800k 步后 ImageNet FID 6.08 vs 单模型 8.49，降低 28%。在 LAION 上，DDM 200k 步的效果超过单模型 800k 步，相当于 **4 倍训练加速**
- **Top-1 选择意外地优于 Full 集成**：可能因为专家在自己专长的数据子分布上比加权平均更精确
- **基于特征的数据聚类远优于随机分区**：DINO 特征聚类使专家能更高效地学习各自子分布，随机分区导致每个专家看到的数据没有语义聚焦
- **蒸馏有效**：蒸馏后的稠密模型在1/3 训练 FLOPs 下达到了从头训练的等效性能（FID 7.76 vs 7.82）

## 亮点与洞察

- **理论与实践的完美结合**：DFM 目标从 Flow Matching 的数学形式自然推导出来，不是启发式设计。这个理论结果优雅且重要——去中心化训练不损失任何表达能力
- **去中心化训练的巨大实际价值**：8 个3B参数专家分别在 16 GPU 节点上训练6.5天即可完成，这使得大规模扩散模型训练不再需要昂贵的集中式超算。学术实验室用分散的云计算资源就能训练高质量模型
- **专家化带来隐式性能增益**：DDM 不仅实现了去中心化，还因为专家在数据子分布上的特化学习而获得了比单一通用模型更好的性能。这类似于 MoE 的增加参数化思想，但无需联合训练
- **极简的工程集成**：DDM 的实践门槛极低——只需数据聚类 + 标准扩散训练框架 + 轻量路由器，几乎所有现有的扩散训练基础设施都可以直接复用

## 局限与展望

- 推理时即使用 Top-1，仍需要在内存中加载所有专家模型的参数，对内存需求较高（K 倍于单模型）
- 蒸馏虽然能解决部署问题，但增加了总训练成本，且蒸馏质量可能不如原始集成
- 当前仅验证了图像生成任务，视频生成、3D生成等更复杂的扩散应用尚未验证
- 数据聚类质量对最终性能影响显著，但如何确定最优的聚类数量和策略仍需经验调优
- 理论上每个专家只看到1/K的数据，对于数据中的稀有模式（如罕见类别），覆盖可能不充分
- 潜在的隐私应用价值很大（专家可以在不同数据源本地训练），但尚未进行专门的隐私分析

## 相关工作与启发

- **vs DiLoCo**: DiLoCo 通过内外优化循环平衡本地训练与周期性全局同步，仍需要少量通信。DDM 实现了完全零通信的独立训练，更适合极端分布式场景
- **vs Branch-Train-Merge**: BTM 训练数据专家并合并参数（在 NLP 中），DDM 不合并参数而是推理时集成，利用路由器做智能选择。DDM 的理论基础更强
- **vs Diffusion Soup**: Diffusion Soup 通过平均微调模型权重来合并，是参数空间的合并。DDM 在预测空间做集成，保留了每个专家的独立性，性能更好
- **vs MoE (Mixtral, DeepSeek-V3)**: MoE 在模型内部路由 token，需要端到端训练。DDM 的路由在模型之间，各部分完全独立训练，系统复杂度大幅降低

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ DFM 目标的理论推导是核心贡献，证明去中心化=全局优化具有里程碑意义
- 实验充分度: ⭐⭐⭐⭐⭐ ImageNet+LAION 双数据集，大量消融（推理策略、专家数量、聚类方式、蒸馏），scaling 到 24B 参数
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，实验系统，blog post 辅助阅读
- 价值: ⭐⭐⭐⭐⭐ 大幅降低了高质量扩散模型训练的门槛，对学术和工业界都有重大实际意义

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Heterogeneous Decentralized Diffusion Models](../../CVPR2026/image_generation/heterogeneous_decentralized_diffusion_models.md)
- [\[CVPR 2025\] Enhancing Creative Generation on Stable Diffusion-based Models](enhancing_creative_generation_on_stable_diffusion-based_models.md)
- [\[CVPR 2025\] Erasing Undesirable Influence in Diffusion Models (EraseDiff)](erasing_undesirable_influence_in_diffusion_models.md)
- [\[CVPR 2025\] DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation](diffsensei_bridging_multi-modal_llms_and_diffusion_models_for_customized_manga_g.md)
- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)

<!-- RELATED:END -->
