---
title: >-
  [论文解读] DogFit: Domain-guided Fine-tuning for Efficient Transfer Learning of Diffusion Models
description: >-
  [AAAI 2026][图像生成][扩散模型] 提出 DogFit，将域引导（Domain Guidance）内化到扩散模型的微调损失中，使模型在训练时学会引导方向，推理时无需双重前向传播即可实现可控的保真度-多样性权衡，在 6 个目标域上以一半的采样 TFLOPS 超越 SOTA 引导方法。
tags:
  - AAAI 2026
  - 图像生成
  - 扩散模型
  - 迁移学习
  - 引导机制
  - 域自适应
  - 高效推理
---

# DogFit: Domain-guided Fine-tuning for Efficient Transfer Learning of Diffusion Models

**会议**: AAAI 2026  
**arXiv**: [2508.05685](https://arxiv.org/abs/2508.05685)  
**代码**: [GitHub](https://github.com/yaramohamadi/DogFit)  
**领域**: 图像生成  
**关键词**: 扩散模型, 迁移学习, 引导机制, 域自适应, 高效推理

## 一句话总结
提出 DogFit，将域引导（Domain Guidance）内化到扩散模型的微调损失中，使模型在训练时学会引导方向，推理时无需双重前向传播即可实现可控的保真度-多样性权衡，在 6 个目标域上以一半的采样 TFLOPS 超越 SOTA 引导方法。

## 研究背景与动机

**领域现状**：扩散模型迁移学习到小规模目标域时容易过拟合。CFG 和 DoG 等引导方法可以提升生成质量，但推理时需要双重前向传播（2x 计算开销）。MG 在训练时内化引导，但继承了 CFG 在迁移学习中的局限性。

**现有痛点**：
   - CFG 的无条件噪声估计器在小目标域上欠拟合，导致引导方向不准
   - DoG 用源模型做边际估计更好，但推理开销翻倍
   - MG 虽无推理开销，但硬编码引导强度、无法推理时调控

**核心矛盾**：能否设计一种引导机制，既利用源模型的强边际估计、又无推理开销、还支持推理时可控引导强度？

**切入角度**：将 DoG 的域引导偏移注入训练损失，用源模型（而非目标模型的无条件分支）提供引导方向，同时将引导强度 $w$ 编码为模型额外输入。

**核心 idea**：训练时用源模型提供域引导方向并内化到损失中 + 引导强度作为模型输入实现推理可控 + late-start/cut-off 调度提升稳定性。

## 方法详解

### 整体框架
在标准扩散微调目标中注入域引导偏移：$\epsilon' = \epsilon + (w-1) \cdot \text{sg}(\epsilon_\theta(x_t|c, \mathcal{D}^T) - \epsilon_{\theta_0}(x_t))$。模型学习直接预测引导后的噪声方向，推理时单次前向即可。

### 关键设计

1. **域引导偏移注入**:

    - 功能：将 DoG 的引导信号从推理时迁移到训练目标中
    - 核心思路：训练损失变为 $\mathcal{L} = \|\epsilon_\theta(x_t|c) - \epsilon'\|^2$，其中 $\epsilon'$ 包含微调模型与源模型之间的引导偏移
    - 设计动机：源模型在大规模数据上预训练，提供的边际估计比目标域的无条件模型更可靠

2. **引导强度可控性**:

    - 功能：推理时可动态调整保真度-多样性权衡
    - 核心思路：将 $w$ 编码为模型的额外条件输入,训练时在一定范围内采样 $w$
    - 设计动机：MG 硬编码 $w$ 无法推理调控，DogFit 通过 $w$ 条件化解决此问题，开销仅为一个轻量嵌入层

3. **训练调度策略**:

    - **Late-start**：延迟引导注入直到模型学到足够稳定的目标表示
    - **Cut-off**：仅在后期去噪步骤施加引导（细粒度域特征更多出现在后期）
    - 设计动机：实验发现过早或全程引导会导致训练不稳定

### 损失函数 / 训练策略
DogFit 损失：$\mathcal{L} = \|\epsilon_\theta(x_t|c,w) - [\epsilon + (w-1) \cdot \text{sg}(\epsilon_\theta(x_t|c) - \epsilon_{\theta_0}(x_t))]\|^2$。在 DiT/XL-2 和 SiT/XL-2 上验证。

## 实验关键数据

### 主实验（DiT/XL-2，6个目标域平均）

| 方法 | FID↓ | FD_DINOv2↓ | 采样TFLOPS |
|------|------|-----------|----------|
| Fine-tuning | 19.14 | 461.45 | 366 |
| +CFG | 14.46 | 311.03 | 732 (2x) |
| +DoG | 13.09 | 245.31 | 732 (2x) |
| MG | 14.13 | 312.78 | 366 |
| **DogFit** | **12.34** | **246.01** | **366** |

DogFit 在 FID 上超越 DoG 且 TFLOPS 减半。

### 消融实验
- Late-start 比全程引导 FID 提升约 1-2
- Cut-off 进一步在 SiT 上带来改进
- 引导强度可控版本（DogFit + Control）仅比固定版本微弱下降

### 关键发现
- 源模型确实提供了比目标域无条件模型更强的边际估计
- 训练时内化引导不仅省计算，实际上还能超越推理时引导——因为训练时优化更充分
- 在有标签和无标签两种迁移设置下都有效

## 亮点与洞察
- **"训练时学引导方向，推理时免费用"**的思路很实用——相当于把推理成本转移到训练阶段
- **用源模型而非目标模型做边际估计**的 insight 有理论支撑——小域上目标模型的无条件分支必然欠拟合
- **引导强度作为条件输入**实现了 MG 和 DoG 的优点统一

## 局限与展望
- 需要保留源模型权重用于训练时计算引导偏移——额外显存开销
- 仅在类条件生成上验证，文本条件扩散（如 SD）的扩展有待探索
- late-start 和 cut-off 的时机选择需要超参调优

## 相关工作与启发
- **vs DoG**: 性能相当但推理快2倍
- **vs MG**: MG 用 CFG 方向（弱），DogFit 用 DoG 方向（强）；MG 不可控，DogFit 可控
- **vs CFG distillation**: CFG 蒸馏需要额外训练阶段和架构修改，DogFit 直接集成到微调目标中

## 评分
- 新颖性: ⭐⭐⭐⭐ 将域引导内化到训练的思路新颖实用
- 实验充分度: ⭐⭐⭐⭐⭐ 6个数据集+2个骨干+多个引导基线+充分消融
- 写作质量: ⭐⭐⭐⭐ 图示清晰，方法对比到位
- 价值: ⭐⭐⭐⭐ 实用的扩散模型迁移学习加速方案

<!-- RELATED:START -->

## 相关论文

- [Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning](self-npo_data-free_diffusion_model_enhancement_via_truncated_diffusion_fine-tuni.md)
- [Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models](../../ICML2025/image_generation/zero-shot_adaptation_of_parameter-efficient_fine-tuning_in_diffusion_models.md)
- [DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models](../../NeurIPS2025/image_generation/deft_decompositional_efficient_finetuning_for_texttoimage_mo.md)
- [Memory-Efficient Fine-Tuning for Quantized Diffusion Model](../../ECCV2024/image_generation/memory-efficient_fine-tuning_for_quantized_diffusion_model.md)
- [Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](../../CVPR2025/image_generation/efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)

<!-- RELATED:END -->
