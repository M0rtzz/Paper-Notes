---
title: >-
  [论文解读] EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image Generation
description: >-
  [CVPR 2026][图像生成][Transformer] EdgeDiT 提出一种硬件感知的扩散 Transformer 优化框架，通过层级知识蒸馏训练轻量级代理块、多目标贝叶斯优化搜索 Pareto 最优架构，实现了 20-30% 参数缩减、36-46% FLOPs 降低、1.65x 端侧加速，同时保持甚至超越原始 DiT-XL/2 的生成质量。
tags:
  - CVPR 2026
  - 图像生成
  - Transformer
  - 端侧部署
  - 硬件感知优化
  - 知识蒸馏
  - 架构搜索
---

# EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image Generation

**会议**: CVPR 2026  
**arXiv**: [2603.28405](https://arxiv.org/abs/2603.28405)  
**代码**: 无  
**领域**: 扩散模型 / 模型压缩  
**关键词**: 扩散Transformer, 端侧部署, 硬件感知优化, 知识蒸馏, 架构搜索

## 一句话总结
EdgeDiT 提出一种硬件感知的扩散 Transformer 优化框架，通过层级知识蒸馏训练轻量级代理块、多目标贝叶斯优化搜索 Pareto 最优架构，实现了 20-30% 参数缩减、36-46% FLOPs 降低、1.65x 端侧加速，同时保持甚至超越原始 DiT-XL/2 的生成质量。

## 研究背景与动机

1. **领域现状**：Diffusion Transformers (DiT) 已成为高保真图像生成的新范式，将 U-Net 替换为 Vision Transformer 骨干网络，具有更好的可扩展性。后续工作如 MDT（掩码建模）、SiT（插值 Transformer）等进一步提升了性能。

2. **现有痛点**：
    - 现有 DiT 模型计算量和内存需求巨大，无法在资源受限的边缘设备上运行
    - 云端推理虽可行，但带来隐私问题、网络依赖和能耗增加
    - 理论计算量（FLOPs/GMACs）不能可靠预测实际端侧延迟——NPU 对特定操作（如 GEMM）有专门优化，减少算术计算不一定等比例降低延迟

3. **核心矛盾**：DiT 的强大生成能力来自于大规模参数和深层架构，但端侧部署要求低延迟和小内存。如何在压缩模型的同时保持生成质量是核心挑战，而且必须考虑实际硬件特性而非仅优化理论指标。

4. **本文目标**
    - 如何系统性地发现适合移动 NPU 的高效 DiT 架构？
    - 如何避免对搜索空间中的每个候选架构都进行完整训练？

5. **切入角度**：将 DiT 架构分解为可替换的硬件友好代理块，通过层级知识蒸馏快速训练代理，再用多目标贝叶斯优化在质量-延迟空间中找到 Pareto 最优架构。

6. **核心 idea**：分解-蒸馏-搜索三步走：将 DiT 分解为代理块搜索空间，用特征级知识蒸馏高效训练每个代理块，再用贝叶斯优化找到 FID-延迟 Pareto 最优的轻量架构。

## 方法详解

### 整体框架
以 DiT-XL/2（28 层、675M 参数）为教师模型，EdgeDiT 框架分为四步：(1) 构建硬件感知的代理块搜索空间；(2) 用特征级知识蒸馏独立训练每个代理块；(3) 组装候选架构并用多目标贝叶斯优化选择 Pareto 最优模型；(4) 端到端训练最终选定的架构。

### 关键设计

1. **硬件感知的代理块搜索空间**:
    - 功能：定义一组硬件友好的轻量替代模块，构成结构化搜索空间
    - 核心思路：定义三种代理类型——(a) 块移除：每两个连续 DiT 层合并为一层（Stage 1，$2^{14}$ 种组合）；(b) MLP 比例修改：FFN 扩展比从 4 改为 2（Stage 2）；(c) 隐藏维度缩减：投影维度从 1152 降到 512（Stage 2）。Stage 2 每层有 4 种选项（2 个 MLP + 2 个维度），产生 $4^{28}$ 种组合。总搜索空间为 $2^{14} + 4^{28}$
    - 设计动机：针对移动 NPU 的数据流特性，识别计算密集且冗余的操作进行结构化简化，而非随机搜索

2. **特征级知识蒸馏 (FwKD)**:
    - 功能：高效训练代理块，避免从头训练整个网络的巨大开销
    - 核心思路：采用分治策略——对每个代理块独立训练，使其输出 $S_l(x)$ 逼近教师模型对应块的输出 $T_l(x)$，损失为 $\mathcal{L}_{KD}^l = \|T_l(x) - S_l(x)\|_2^2$。Stage 1 训练 14 个代理（两层→一层），Stage 2 训练 56 个代理（28 层 × 2 种变体）。由于各块独立蒸馏，过程高度可并行化
    - 设计动机：如果每个候选架构都从头训练，$2^{14} + 4^{28}$ 的搜索空间完全不可行。层级蒸馏让代理块能快速学会局部行为近似，后续只需少量端到端微调

3. **多目标贝叶斯优化 (MOBO) 架构选择**:
    - 功能：在 FID-延迟 的二维空间中高效找到 Pareto 最优架构
    - 核心思路：将架构配置 $a$ 的选择形式化为双目标优化问题：$\max f(a)$（生成质量/FID）和 $\min g(a)$（端侧延迟）。用高斯过程作为代理模型预测候选架构的目标值，通过 Expected Hyper-volume Improvement (EHVI) 采集函数平衡探索和利用。将离散架构配置松弛为连续表示 $x \in [0,1]^{28}$ 后映射回最近的可行架构
    - 设计动机：穷举评估不可行，贝叶斯优化能用少量评估样本高效逼近 Pareto 前沿

### 损失函数 / 训练策略
- 蒸馏阶段：$\mathcal{L}_{KD}^l = \|T_l(x) - S_l(x)\|_2^2$，每个代理块独立训练
- 端到端训练：标准扩散训练目标 $\mathcal{L}_{diff} = \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t, t)\|_2^2]$
- 选出的 EdgeDiT-1 和 EdgeDiT-6 基于 DiT-XL/2 的 400K iteration checkpoint 进行 100K iteration 的端到端训练

## 实验关键数据

### 主实验 — ImageNet 256×256 类条件生成

| 模型 | 参数量 (M) | FID-50K↓ | SFID↓ | IS↑ | Precision↑ | Recall↑ |
|------|-----------|----------|-------|-----|-----------|---------|
| DiT-XL/2 | 675 | 16.23 | 11.06 | 80.91 | 0.93 | 0.26 |
| **EdgeDiT-1** | **471** | **12.3** | 13.97 | 75.72 | 0.92 | 0.24 |
| **EdgeDiT-6** | **530** | **12.4** | 14.96 | 78 | 0.91 | 0.25 |

### 端侧延迟对比 (256×256)

| 模型 | 参数量 (M) | GFLOPs | iPhone 延迟 (ms) | Samsung 延迟 (ms) |
|------|-----------|--------|-----------------|-------------------|
| DiT-XL/2 | 675 | 237.34 | 118.56 | 129.00 |
| EdgeDiT-1 | 471 | 143.96 | 70.86 | 86.13 |
| EdgeDiT-6 | 530 | 169.97 | 72.53 | 89.22 |

### 消融实验 — 知识蒸馏的必要性

| 配置 | 说明 |
|------|------|
| EdgeDiT + FwKD | 正常质量，接近教师模型 |
| EdgeDiT w/o FwKD（随机初始化） | 图像质量显著退化 |

### 关键发现
- **EdgeDiT 以更少参数超越教师模型**：EdgeDiT-1 仅用 471M 参数（少 30%）就把 FID 从 16.23 降到 12.3，说明 DiT-XL/2 存在大量结构冗余
- **参数减少 30%、FLOPs 减少 36-46%、端侧加速 1.65x**：Samsung Galaxy S25 Ultra 上实测加速明显
- **FwKD 不可或缺**：没有特征级蒸馏的 EdgeDiT 图像质量严重退化，蒸馏为后续端到端训练提供了良好初始化
- **搜索空间设计的敏感性分析**：3 块合并（而非 2 块）导致质量骤降，MLP ratio=1 质量差而 ratio=2,3 接近，hidden dim=512 与 768 质量类似

## 亮点与洞察
- **分解-蒸馏-搜索的流水线设计**：这个框架非常工程化且实用——将不可行的全空间搜索分解为可并行的局部蒸馏，大幅降低了搜索成本。这种方法论可以迁移到其他大模型的压缩场景
- **理论指标与实际延迟的脱节**：论文强调了 FLOPs 不能准确预测 NPU 延迟这一实际问题，因此直接将端侧延迟作为优化目标之一，这在工程部署中非常重要
- **代理块蒸馏的高度并行性**：70 个代理块可以独立并行训练，使得整个搜索过程高效且可扩展

## 局限与展望
- 仅基于 DiT-XL/2 做实验，未扩展到 SiT、MDT 等其他扩散 Transformer
- 受限于计算资源，仅训练了 EdgeDiT-1 和 EdgeDiT-6 两个代表性架构，Pareto 前沿上的更多模型未充分探索
- 端到端训练仅 100K iterations（基于 400K checkpoint），与完整训练的 DiT-XL/2（7M iterations）相比训练不充分，FID 还有提升空间
- 仅评估了类条件 ImageNet 生成，未验证文本到图像等更复杂的生成任务
- 搜索空间中未考虑注意力头数的变化和 token 稀疏化等技术

## 相关工作与启发
- **vs MobileDiffusion**: MobileDiffusion 针对 U-Net 架构做移动端优化，而 EdgeDiT 专注于 Transformer 骨干的优化，覆盖了不同的技术路线
- **vs DiT-S/B/L**: DiT 家族通过缩小隐藏维度和层数来减少参数，但这是均匀缩减；EdgeDiT 通过异构搜索在不同层使用不同程度的压缩，实现了更好的质量-效率权衡
- **vs 标准剪枝/NAS**: EdgeDiT 结合了知识蒸馏和贝叶斯 NAS 的优势，避免了纯剪枝可能导致的性能断崖和纯 NAS 的高训练成本

## 评分
- 新颖性: ⭐⭐⭐ 方法论上是已有技术（KD + NAS + MOBO）的组合，但组合方式合理且面向实际问题
- 实验充分度: ⭐⭐⭐ 有端侧实测数据，但仅两个模型做了完整训练，ImageNet 256 单一任务
- 写作质量: ⭐⭐⭐⭐ 清晰易读，框架图和搜索空间设计讲解清楚
- 价值: ⭐⭐⭐⭐ 为扩散 Transformer 的端侧部署提供了可操作的系统方案

<!-- RELATED:START -->

## 相关论文

- [Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers](circuit_mechanisms_for_spatial_relation_generation_in_diffusion_transformers.md)
- [Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping](memory-efficient_fine-tuning_diffusion_transformers_via_dynamic_patch_sampling_a.md)
- [OPRO: Orthogonal Panel-Relative Operators for Panel-Aware In-Context Image Generation](opro_orthogonal_panel-relative_operators_for_panel-aware_in-context_image_genera.md)
- [RelaCtrl: Relevance-Guided Efficient Control for Diffusion Transformers](../../AAAI2026/image_generation/relactrl_relevance-guided_efficient_control_for_diffusion_transformers.md)
- [RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models](razor_ratio_aware_unlearning_vit_diffusion.md)

<!-- RELATED:END -->
