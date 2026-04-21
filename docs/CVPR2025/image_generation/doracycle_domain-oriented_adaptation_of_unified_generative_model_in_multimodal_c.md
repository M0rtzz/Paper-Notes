---
title: >-
  [论文解读] DoraCycle: Domain-Oriented Adaptation of Unified Generative Model in Multimodal Cycles
description: >-
  [CVPR 2025][图像生成][域适应] 提出 DoraCycle 使用两个多模态循环（文→图→文 和 图→文→图）对统一多模态生成模型做无配对域适应，仅用无配对目标域数据即可接近全配对训练效果（FID 27.44 vs 24.93），10% 配对+90% 无配对时几乎无损（FID 25.37）。
tags:
  - CVPR 2025
  - 图像生成
  - 域适应
  - 统一生成模型
  - 循环一致性
  - 无配对训练
  - 多模态循环
---

# DoraCycle: Domain-Oriented Adaptation of Unified Generative Model in Multimodal Cycles

**会议**: CVPR 2025  
**arXiv**: [2503.03651](https://arxiv.org/abs/2503.03651)  
**代码**: https://github.com/showlab/DoraCycle  
**领域**: 图像生成  
**关键词**: 域适应、统一生成模型、循环一致性、无配对训练、多模态循环

## 一句话总结
提出 DoraCycle 使用两个多模态循环（文→图→文 和 图→文→图）对统一多模态生成模型做无配对域适应，仅用无配对目标域数据即可接近全配对训练效果（FID 27.44 vs 24.93），10% 配对+90% 无配对时几乎无损（FID 25.37）。

## 研究背景与动机

**领域现状**：Show-o 等统一多模态生成模型可以同时完成文生图和图生文，但适配到新目标域（如特定绘画风格、特定人物身份）通常需要配对的图文数据。

**现有痛点**：(1) 目标域的配对图文数据难以获取（尤其特定人物/风格）。(2) DreamBooth 等方法需要全配对数据且对齐质量依赖人工标注。(3) 现有领域适应方法针对单一模态模型设计，无法利用统一模型的双向生成能力。

**核心矛盾**：统一模型天然具备双向生成能力（文↔图），能否利用这种对称性从无配对数据中自监督学习？

**本文目标** 利用统一模型的双向能力，通过循环一致性在无配对目标域数据上实现域适应。

**切入角度**：设计两个跨模态循环——T cycle（文→图→文）和 I cycle（图→文→图），在循环端点的同模态空间做重建损失。EMA 模型稳定伪中间数据生成，梯度手术平衡两个循环的优化方向。

**核心 idea**：利用统一多模态模型的文↔图双向生成能力，通过"文→图→文"和"图→文→图"双循环一致性实现无配对域适应。

## 方法详解

### 整体框架
目标域无配对文本+目标域无配对图像 → T cycle：文本→EMA 模型生成伪图像→模型重建文本（CE 损失）→ I cycle：图像→EMA 模型生成伪文本→模型重建图像（token CE 损失）→ 梯度手术平衡两循环。

### 关键设计

1. **双多模态循环**:

    - 功能：从无配对数据自监督学习域适应
    - 核心思路：T cycle 强制文本→图像→文本的一致性（确保生成的图像保留文本语义），I cycle 强制图像→文本→图像的一致性（确保生成的文字描述可重现原图）。两者协同覆盖双向对齐
    - 设计动机：消融显示单用 I cycle（FID 28.93）或 T cycle（FID 36.63）都不如双循环（27.44），两者互补

2. **EMA 稳定化**:

    - 功能：生成稳定的伪中间数据
    - 核心思路：用 EMA 模型（α=0.999 慢速更新）生成循环中间阶段的伪数据（T cycle 的伪图像、I cycle 的伪文本），避免模型自身生成的噪声伪数据导致训练崩溃
    - 设计动机：直接用当前模型生成伪数据会导致误差累积和模式坍塌

3. **梯度手术**:

    - 功能：防止两个循环的梯度方向冲突
    - 核心思路：计算 T cycle 和 I cycle 的梯度方向。当两者方向冲突时（点积为负），将 T cycle 梯度投影到与 I cycle 梯度正交的方向上
    - 设计动机：两个循环优化不同模态的重建质量，方向可能冲突，梯度手术保证两者不互相抵消

### 损失函数 / 训练策略
$\mathcal{L} = \mathcal{L}_{real} + \mathcal{L}_{syn}$，T cycle 用 token-level CE 重建文本，I cycle 用 image token CE 重建图像。LoRA rank=32 应用于注意力层 7-24。β=0.1 为 T cycle 权重。

## 实验关键数据

### 主实验

| 方法 | 数据类型 | FID-1K↓ | CIDEr↑ | T2I Align↑ |
|------|---------|---------|--------|-----------|
| DreamBooth | 10% 配对 | 33.22 | 32.74 | 3.25 |
| DreamBooth | 100% 配对 | 24.93 | 41.55 | 4.13 |
| **DoraCycle** | **100% 无配对** | **27.44** | **38.17** | **3.84** |
| **DoraCycle** | **10%P+90%U** | **25.37** | **40.90** | **4.12** |

### 消融实验

| 循环 | FID↓ | 说明 |
|------|------|------|
| 仅 I cycle | 28.93 | 文→图方向不够 |
| 仅 T cycle | 36.63 | 图→文方向不够 |
| I+T 双循环 | **27.44** | 互补效果最好 |
| 10%配对+90%无配对 | **25.37** | 接近全配对 24.93 |

### 关键发现
- **100% 无配对即可接近全配对**：FID 27.44 vs 24.93，差距仅 10%
- **极少配对+大量无配对最优**：10% 配对数据就足以弥合差距（25.37 vs 24.93）
- **模型自发学到新概念关联**：训练后模型能标注未见过的"black cat"为领域特定标记，展示了域内概念的自发泛化

## 亮点与洞察
- **利用统一模型的对称性做自监督**非常优雅——双向生成能力天然提供了循环一致性的条件
- **对现实场景意义重大**：很多目标域（如特定艺术风格、特定人物）很难获取配对数据，DoraCycle 使适应变得可行

## 局限与展望
- EMA 模型生成的伪数据质量依赖基础模型的能力——如果基础模型完全不了解目标域则循环无法启动
- 身份特定任务仍需 1-3 张配对图像（纯无配对不够），风格迁移可纯无配对
- 仅在 Show-o 上验证，对其他统一模型（Chameleon、Emu）未测试

## 相关工作与启发
- **vs DreamBooth**：需要全配对数据。DoraCycle 用 10% 配对 + 90% 无配对就接近其效果
- **vs CycleGAN**：经典循环一致性但针对图→图。DoraCycle 将循环一致性扩展到跨模态场景

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 跨模态循环一致性用于统一模型域适应是首创
- 实验充分度: ⭐⭐⭐⭐ 风格+身份两类任务、配对/无配对对比、循环消融
- 写作质量: ⭐⭐⭐⭐ 循环概念讲解清楚
- 价值: ⭐⭐⭐⭐ 对统一生成模型的域适应开辟了新方向

<!-- RELATED:START -->

## 相关论文

- [WeGen: A Unified Model for Interactive Multimodal Generation as We Chat](wegen_a_unified_model_for_interactive_multimodal_generation_as_we_chat.md)
- [Everything to the Synthetic: Diffusion-driven Test-time Adaptation via Synthetic-Domain Alignment](everything_to_the_synthetic_diffusion-driven_test-time_adaptation_via_synthetic-.md)
- [TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [Diffusion-Driven Progressive Target Manipulation for Source-Free Domain Adaptation](../../NeurIPS2025/image_generation/diffusion-driven_progressive_target_manipulation_for_source-free_domain_adaptati.md)
- [JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation](janusflow_harmonizing_autoregression_and_rectified_flow_for_unified_multimodal_u.md)

<!-- RELATED:END -->
