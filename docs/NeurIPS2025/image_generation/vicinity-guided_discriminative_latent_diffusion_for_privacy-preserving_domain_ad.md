---
title: >-
  [论文解读] Vicinity-Guided Discriminative Latent Diffusion for Privacy-Preserving Domain Adaptation
description: >-
  [NeurIPS 2025][图像生成][潜扩散模型] 提出 Discriminative Vicinity Diffusion (DVD)，首次将潜扩散模型用于判别式知识迁移，通过在源域特征的近邻潜空间中训练扩散模型生成源样式线索，实现无需源数据访问的域适应，在标准 SFDA 基准上超越 SOTA。
tags:
  - NeurIPS 2025
  - 图像生成
  - 潜扩散模型
  - source-free域适应
  - 隐私保护
  - 判别式迁移
  - 近邻引导
---

# Vicinity-Guided Discriminative Latent Diffusion for Privacy-Preserving Domain Adaptation

**会议**: NeurIPS 2025  
**arXiv**: [2510.00478](https://arxiv.org/abs/2510.00478)  
**代码**: 无  
**领域**: 域适应 / 图像生成  
**关键词**: 潜扩散模型, source-free域适应, 隐私保护, 判别式迁移, 近邻引导

## 一句话总结

提出 Discriminative Vicinity Diffusion (DVD)，首次将潜扩散模型用于判别式知识迁移，通过在源域特征的近邻潜空间中训练扩散模型生成源样式线索，实现无需源数据访问的域适应，在标准 SFDA 基准上超越 SOTA。

## 研究背景与动机

Source-Free Domain Adaptation (SFDA) 因隐私保护需求而日益重要——源数据不可访问，仅能使用预训练的分类器。现有方法的局限：

**隐式知识迁移**: 多数方法仅通过伪标签或一致性正则化间接利用源知识

**缺乏显式决策边界迁移**: 源域的决策边界信息在适应过程中丢失

**扩散模型的潜力未被利用**: LDM 主要用于生成任务，判别式迁移几乎空白

核心创新：利用扩散模型作为隐私保护的桥梁，显式迁移源域的决策边界到目标域。

## 方法详解

### 整体框架

DVD 框架分为三个步骤：
1. **源域训练**: 在源数据上训练分类器 + 辅助扩散模块
2. **扩散模块发布**: 仅发布预训练的扩散模块（不暴露原始数据）
3. **目标域适应**: 利用冻结的扩散模块生成源样式线索，对齐目标编码器

### 关键设计

1. **近邻潜空间编码 (Vicinity Encoding)**:

    - 对每个源特征，找到其 k-最近邻
    - 在近邻集合上拟合高斯先验 $\mathcal{N}(\mu_k, \Sigma_k)$
    - 扩散网络学习将噪声样本漂移回标签一致的表示

2. **标签引导扩散 (Label-Guided Diffusion)**:

    - 将标签信息编码到每个特征的潜在近邻区域
    - 扩散过程保持标签一致性
    - 关键约束：漂移后的样本应与原标签类别中心接近

3. **目标域适应**:

    - 从目标特征的近邻区域采样
    - 通过冻结的扩散模块生成源样式线索
    - InfoNCE 损失对齐目标编码器和扩散生成的线索
    - 显式迁移决策边界

### 损失函数 / 训练策略

- 源域扩散训练：标准 DDPM 损失 + 标签一致性正则化
- 目标域适应：
$$\mathcal{L}_{\text{adapt}} = \mathcal{L}_{\text{InfoNCE}}(f_T(x_T), g_\theta(z_T)) + \lambda \mathcal{L}_{\text{pseudo}}$$

## 实验关键数据

### 主实验（SFDA 基准）

| 方法 | Office-Home Avg ↑ | VisDA-C ↑ | DomainNet Avg ↑ | 需要源数据 |
|------|------------------|----------|----------------|---------|
| SHOT | 71.8 | 82.9 | 43.5 | 否 |
| NRC | 72.5 | 83.5 | 44.2 | 否 |
| AaD | 73.8 | 84.8 | 45.8 | 否 |
| CoWA | 74.2 | 85.1 | 46.5 | 否 |
| PLUE | 75.1 | 85.8 | 47.2 | 否 |
| **DVD (Ours)** | **77.5** | **87.8** | **49.8** | **否** |
| Oracle (有源数据) | 79.2 | 89.5 | 52.3 | 是 |

### 额外能力验证

| 应用场景 | 基线准确率 ↑ | +DVD 准确率 ↑ | 提升 |
|---------|-----------|------------|-----|
| 源域分类器增强 | 85.2 | 87.5 | +2.3 |
| 有监督分类 | 78.5 | 80.8 | +2.3 |
| 域泛化 | 72.3 | 75.1 | +2.8 |

### 消融实验

| 组件 | Office-Home ↑ | VisDA-C ↑ |
|------|-------------|---------|
| DVD 完整 | 77.5 | 87.8 |
| 去掉近邻编码 | 73.8 | 84.2 |
| 去掉标签引导 | 74.5 | 85.1 |
| 去掉 InfoNCE 对齐 | 72.1 | 83.5 |
| 用 GAN 替代扩散 | 74.2 | 84.8 |
| 随机采样(非近邻) | 71.5 | 82.5 |

### 关键发现

1. DVD 将 SFDA 与有源数据方法的差距缩小到 1.7%（Office-Home）
2. 扩散模型在判别式迁移中优于 GAN（+3.3% on Office-Home）
3. 近邻引导是最关键的设计，移除后性能下降 3.7%
4. DVD 的辅助扩散模块还能增强源域分类器本身的性能

## 亮点与洞察

- **全新视角**: 将 LDM 从生成式应用拓展到判别式知识迁移
- **隐私保护**: 不暴露任何源数据样本，符合 GDPR 等隐私法规
- **多用途**: 同一扩散模块可用于 SFDA、域泛化、源域增强等多个场景
- **接近有监督性能**: 无源数据情况下达到接近 oracle 的效果

## 局限与展望

1. 需要在源域上额外训练扩散模块，增加源域训练成本
2. 扩散模块虽不暴露原始数据，但理论上可能泄露部分分布信息
3. 对于类别极多的场景（如 DomainNet 的 345 类），效率有待优化
4. 与文本引导的扩散模型结合可能进一步提升效果

## 相关工作与启发

- **SHOT (Liang et al., 2020)**: SFDA 的代表性方法
- **NRC**: 基于近邻一致性的 SFDA
- **CoWA**: 协方差加权对齐
- **Diffusion Models for DA**: 本文开辟的新方向

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 5 |
| 理论深度 | 3 |
| 实验充分性 | 5 |
| 写作质量 | 4 |
| 实用价值 | 4 |
| 总体推荐 | 4.5 |

<!-- RELATED:START -->

## 相关论文

- [Diffusion-Driven Progressive Target Manipulation for Source-Free Domain Adaptation](diffusion-driven_progressive_target_manipulation_for_source-free_domain_adaptati.md)
- [PixPerfect: Seamless Latent Diffusion Local Editing with Discriminative Pixel-Space Refinement](pixperfect_seamless_latent_diffusion_local_editing_with_discriminative_pixel-spa.md)
- [What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](../../ICCV2025/image_generation/whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)
- [Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models](perturb_a_model_not_an_image_towards_robust_privacy_protection_via_anti-personal.md)
- [ObCLIP: Oblivious Cloud-Device Hybrid Image Generation with Privacy Preservation](obclip_oblivious_cloud-device_hybrid_image_generation_with_privacy_preservation.md)

<!-- RELATED:END -->
