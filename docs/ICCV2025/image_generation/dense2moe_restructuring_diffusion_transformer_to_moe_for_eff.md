---
title: >-
  [论文解读] Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation
description: >-
  [ICCV 2025][图像生成][Transformer] 首次将预训练的dense DiT（FLUX.1 [dev] 12B参数）通过三步蒸馏pipeline转换为结构化稀疏的MoE架构——用MoE层替换FFN实现token级稀疏、用Mixture of Blocks（MoB）实现block级动态跳过——激活参数从12B降至5.2B（减少56%+）的同时保持原始性能，全面超越同等压缩比的剪枝方法。
tags:
  - ICCV 2025
  - 图像生成
  - Transformer
  - Mixture of Experts
  - 结构化稀疏
  - FLUX
  - dense-to-sparse转换
---

# Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation

**会议**: ICCV 2025  
**arXiv**: [2510.09094](https://arxiv.org/abs/2510.09094)  
**代码**: 无  
**领域**: 图像生成 / 模型压缩  
**关键词**: 扩散 Transformer, Mixture of Experts, 结构化稀疏, FLUX, dense-to-sparse转换

## 一句话总结

首次将预训练的dense DiT（FLUX.1 [dev] 12B参数）通过三步蒸馏pipeline转换为结构化稀疏的MoE架构——用MoE层替换FFN实现token级稀疏、用Mixture of Blocks（MoB）实现block级动态跳过——激活参数从12B降至5.2B（减少56%+）的同时保持原始性能，全面超越同等压缩比的剪枝方法。

## 研究背景与动机

**领域现状**：DiT架构在text-to-image生成中展示了卓越性能，但模型规模持续膨胀——FLUX.1有12B参数，是SD1.5的13.8倍。庞大的参数量导致内存消耗和推理时间大幅增加，高昂的计算成本成为用户使用的主要障碍。

**现有痛点**：现有模型压缩主要依靠剪枝（pruning），但剪枝永久删除参数会导致模型容量下降——在高压缩比下性能严重退化，即使重训练也难以完全恢复。FLUX.1-Lite删除11个double stream block后8B参数的模型在多个指标上明显落后于原模型，FLUX-Mini更是在大幅减参后性能骤降。

**核心矛盾**：剪枝的根本问题是减少了总参数数量，从而约束了模型的表达容量。理想方案应该是保留全部参数但选择性激活——不同输入只激活相关的参数子集，从而在维持容量的同时降低计算成本。

**本文目标**：将现有的dense DiT高效转换为MoE架构，实现"减少激活参数但不减少模型容量"的目标，建立扩散模型从dense到sparse转换的新范式。

**切入角度**：两个关键观察——(1) DiT中FFN占总参数的近50%，天然适合MoE拆分；(2) 不同block在不同时间步和prompt下的贡献差异显著（MSE分析），block级动态选择有巨大空间。将这两者结合为MoE（token级稀疏）+MoB（block级稀疏）的混合稀疏架构。

**核心 idea**：将dense DiT的FFN拆为MoE层+将block组织为MoB+多步知识蒸馏 = 高压缩比下保持性能的结构化稀疏。

## 方法详解

### 整体框架

Dense2MoE是一个三阶段蒸馏pipeline：(1) Enhanced MoE Initialization——用Taylor度量拆分FFN权重初始化专家，并蒸馏增强shared expert；(2) Dense-to-MoE Distillation——组装完整MoE层，冻结shared expert，训练normal experts和gating network；(3) Group Feature Distillation for MoB——将block分组为MoB，用组特征损失训练MoB router。三个阶段顺序执行，MoE和MoB作为独立模块分别处理。

### 关键设计

1. **Taylor度量专家初始化 + MoE层设计**：

    - 功能：将DiT每个block的FFN替换为包含1个shared expert和12个normal experts的MoE层（配置1S12E2A），每个token只激活shared expert + top-2 normal experts，FFN激活参数压缩62.5%。
    - 核心思路：用一阶Taylor度量 $\mathcal{I}_i = |\frac{\partial\mathcal{L}}{\partial w_i} w_i|$ ，将FFN中间维度方向按重要性排序，最重要的权重分配给shared expert（所有token共享），其余均匀分配给normal experts。先将shared expert视为剪枝模型单独蒸馏增强（KD+block feature loss），建立MoE的性能下界，再激活normal experts进行完整MoE训练。
    - 设计动机：随机拆分权重初始化会导致严重的初始性能损失。Taylor度量将"最重要的计算"集中在shared expert中，保证即使只有shared expert工作时模型也有基本生成能力。这种"先打底再补充"的策略比一步到位的全MoE训练更稳定。

2. **Mixture of Blocks (MoB)**：

    - 功能：将连续的transformer block分组，每组通过router动态选择激活其中的一部分block，直接减少推理时的模型深度。
    - 核心思路：多个连续block构成一个MoB组，router根据输入特征和全局条件嵌入（timestep+text通过AdaLN的y）选择激活哪些block。路由公式为 $\text{TopK}(\alpha W_x[x,c] + (1-\alpha) W_y y, \kappa)$，同时利用图像特征和条件信息做出routing决策。组特征损失（Group Feature Loss）将MoB组的输出与teacher模型对应位置的特征对齐。
    - 设计动机：MSE分析揭示block贡献在不同时间步和prompt下差异显著——某些block在高噪声阶段几乎不起作用，某些在特定prompt类型下贡献极小。现有深度剪枝只能估计平均重要性，忽略了样本特异的变化。MoB通过动态routing让每个样本在每个时间步选择最关键的block，比静态剪枝更精细。

3. **多步蒸馏与归一化特征权重**：

    - 功能：设计归一化的block feature loss权重，平衡不同层特征的尺度差异，确保各层参与有效学习。
    - 核心思路：特征权重 $w_l = \frac{|\mathcal{L}_{distill}|}{|\mathcal{L}_{feature}^{(l)}|} \cdot \frac{\sum \|f_{tea}^{(l)}\|_2}{L \cdot \|f_{tea}^{(l)}\|_2}$，同时基于蒸馏损失和teacher特征范数归一化。MoE阶段冻结shared expert+load balance loss，MoB阶段冻结非MoB block+组特征对齐。
    - 设计动机：直接用原始特征损失会因不同层特征尺度差异大而导致某些层主导梯度。归一化确保每层贡献平衡，load balance确保normal experts被均匀训练避免退化。

### 损失函数 / 训练策略

- 输出蒸馏损失：student和teacher最终输出的MSE
- Block特征损失：逐层特征对齐，使用归一化权重
- Load balance损失：确保normal experts负载均衡（$\lambda_{balance}=10^{-2}$）
- 组特征损失：MoB组输出与teacher对应block输出的特征对齐
- 训练：32×A100 GPU，全局batch size 64，训练数据Laion-5B+Coyo-700M+JourneyDB

## 实验关键数据

### 主实验

| 模型 | 激活参数 | FLOPs(T) | GenEval↑ | DPG↑ | CLIP↑ | IR↑ |
|------|---------|---------|---------|------|-------|-----|
| FLUX.1 [dev] | 11.9B | 66.0 | 0.660 | 83.42 | 32.24 | 0.966 |
| FLUX.1-Lite | 8.16B | 53.2 | 0.523 | 79.00 | 31.79 | 0.838 |
| **FLUX.1-MoE-L** | **5.15B** | **43.4** | **0.570** | **81.63** | 31.39 | 0.801 |
| FLUX-Mini | 3.18B | 17.4 | 0.321 | 69.34 | 29.94 | 0.215 |
| **FLUX.1-MoE-S** | **3.19B** | **26.4** | **0.444** | **75.61** | 30.67 | 0.594 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| MLP Pruning (ra=1.5) | GenEval 0.411, DPG 72.2 | 剪枝在高压缩比下严重退化 |
| **MoE (ra=1.5)** | **GenEval 0.573, DPG 81.2** | 同等压缩比下MoE全面超越 |
| Depth Pruning (Lite, 9D+26S) | GenEval 0.093, DPG 41.6 | 深度剪枝在高压缩下崩溃 |
| **MoB (9D+26S)** | **GenEval 0.496, DPG 76.5** | MoB显著优于静态深度剪枝 |
| Taylor初始化 | 基线 | 显著优于随机拆分 |
| 去掉Taylor | -0.16 IR, -0.17 MPS | 初始化质量影响最终性能 |
| 分步蒸馏(shared→full) | 基线 | 优于一步联合蒸馏 |
| 一步联合蒸馏 | -0.02 IR | 略差 |

### 关键发现

- MoE在相同激活参数下全面超越剪枝——FLUX.1-MoE-L（5.2B）以3B更少的激活参数超过FLUX.1-Lite（8.16B）
- 在75%压缩率下MoE优势更加明显——FLUX.1-MoE-S（3.19B）大幅超越FLUX-Mini（3.18B）
- Taylor度量初始化对MoE性能至关重要——重要权重放入shared expert提供了坚实的性能基础
- MoB在block级压缩上显著优于静态深度剪枝——尤其在高压缩比下优势巨大（GenEval 0.496 vs 0.093）
- 专家分析揭示了有趣的专门化模式——不同专家专注于不同空间区域、不同prompt类型和不同时间步

## 亮点与洞察

- "保留容量、减少激活"的理念是MoE相对于剪枝的根本优势——总参数不变但推理只用部分，在需要时可以激活更多专家（Dynamic TopK展示了这种灵活性）
- Taylor度量+分步蒸馏的初始化策略可推广到其他dense-to-MoE场景——"先确保下界可用、再逐步增加能力"是稳健的工程策略
- MoB将时间步/prompt条件信息融入routing——充分利用了扩散模型多步条件去噪的特殊性质，比通用的MoD更适合扩散模型
- 动态TopK能力意味着推理时可以在质量和速度之间无缝权衡——同一个模型可以根据需求运行在不同的"档位"

## 局限与展望

- MoE的并行推理需要grouped GEMM等工程优化——朴素实现下MoE的实际加速比可能低于理论值
- 蒸馏训练仍需32张A100——虽然远低于从头训练MoE DiT但对个人用户仍有门槛
- 未探索与步骤蒸馏的结合——MoE减每步参数×步骤蒸馏减总步数可能实现乘法级加速
- 仅在FLUX.1上验证——SD3、SANA等其他DiT架构的适用性未测试
- MoB的分组策略（几个block一组、每组激活几个）需要人工设定——自动化搜索最优配置可提升实用性

## 相关工作与启发

- **LLM领域Dense-to-MoE**：MoE Jetpack、MSSC等已在NLP验证可行性——本文将此范式首次引入扩散模型
- **DiT-MoE**：从头训练16B MoE DiT——Dense2MoE则是将已有dense模型转换，成本更低
- **Diff-Pruning**：宽度剪枝的代表方法——Dense2MoE在同等压缩比下全面超越
- **与HyperFLUX正交**：HyperFLUX做步骤蒸馏（28步→8步），Dense2MoE做模型稀疏化——两者可叠加（HyperFLUX-MoE-L在8步5.2B时表现优异）

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次在DiT上做dense-to-MoE，MoB设计+条件routing有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 多级压缩(L/M/S/XS)+多基准(GenEval/DPG/T2I-CompBench)+详尽消融+专家分析
- 写作质量: ⭐⭐⭐⭐ Pipeline清晰、实验与分析充实
- 价值: ⭐⭐⭐⭐⭐ 为大型DiT的高效部署建立了新范式，实用价值极高

<!-- RELATED:START -->

## 相关论文

- [UniCombine: Unified Multi-Conditional Combination with Diffusion Transformer](unicombine_unified_multi-conditional_combination_with_diffusion_transformer.md)
- [FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)
- [LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [ReFlex: Text-Guided Editing of Real Images in Rectified Flow via Mid-Step Feature Extraction and Attention Adaptation](reflex_text-guided_editing_of_real_images_in_rectified_flow_via_mid-step_feature.md)
- [DICE: Staleness-Centric Optimizations for Parallel Diffusion MoE Inference](dice_staleness-centric_optimizations_for_parallel_diffusion_moe_inference.md)

<!-- RELATED:END -->
