---
title: >-
  [论文解读] Co-Spy: Combining Semantic and Pixel Features to Detect Synthetic Images by AI
description: >-
  [CVPR 2025][目标检测][AI生成图像检测] 提出 Co-Spy 融合 VAE 重建伪影特征和 CLIP 语义特征两条互补检测路径——VAE 伪影跨模型泛化但怕 JPEG 压缩，CLIP 语义抗 JPEG 但泛化差——自适应调节器根据输入动态分配两路权重，在 22 个生成模型上建立新 SOTA。
tags:
  - CVPR 2025
  - 目标检测
  - AI生成图像检测
  - 语义-伪影融合
  - VAE伪影
  - CLIP分类
  - JPEG鲁棒性
---

# Co-Spy: Combining Semantic and Pixel Features to Detect Synthetic Images by AI

**会议**: CVPR 2025  
**arXiv**: [2503.18286](https://arxiv.org/abs/2503.18286)  
**代码**: https://github.com/Megum1/Co-Spy  
**领域**: 目标检测  
**关键词**: AI生成图像检测、语义-伪影融合、VAE伪影、CLIP分类、JPEG鲁棒性

## 一句话总结
提出 Co-Spy 融合 VAE 重建伪影特征和 CLIP 语义特征两条互补检测路径——VAE 伪影跨模型泛化但怕 JPEG 压缩，CLIP 语义抗 JPEG 但泛化差——自适应调节器根据输入动态分配两路权重，在 22 个生成模型上建立新 SOTA。

## 研究背景与动机

**领域现状**：AI 生成图像检测分为伪影检测（捕捉生成器指纹）和语义检测（分析高层语义不自然性）两大类。前者泛化好但对压缩敏感，后者鲁棒但对新模型泛化差。

**现有痛点**：(1) 基于上采样伪影的检测器在 JPEG 压缩后失效。(2) 基于 CLIP 的语义检测器对训练时未见过的生成模型泛化差。(3) 简单拼接两类特征（如 DRCT）效果不佳——因为两路特征需要各自增强后再融合。

**核心矛盾**：两类检测方法有互补优势但各有致命弱点。直接融合不work——需要先各自增强再自适应融合。

**本文目标** 分别增强伪影和语义两条路径，然后设计自适应融合机制利用两者的互补性。

**切入角度**：VAE 重建伪影比上采样伪影更高级别且更鲁棒；CLIP 特征空间中做软标签插值增强泛化；调节器网络动态分配两路权重并随机 dropout 防过拟合。

**核心 idea**：用 VAE 重建差异提取高层伪影 + CLIP 软标签插值增强语义 + 调节器自适应融合，三步组合实现跨模型+抗压缩的双鲁棒检测。

## 方法详解

### 整体框架
输入图像 → 分两路：(1) VAE 编码-解码后计算与原图的绝对差 |x'-x| → 伪影编码器提取伪影特征；(2) 冻结 OpenCLIP 提取语义特征 + 软标签插值增强 → 两路特征被调节器动态加权 → 融合后分类。

### 关键设计

1. **VAE 伪影提取**:

    - 功能：比上采样伪影更鲁棒的生成器指纹检测
    - 核心思路：将图像过一次预训练 VAE（编码→解码）得到重建图 x'，计算 |x'-x| 作为伪影图。真实图像和 AI 生成图像经过同一 VAE 重建后，差异模式不同——生成图像中 VAE 会"修复"生成伪影而真实图像则均匀重建
    - 设计动机：上采样伪影（NPR、LNP）在 JPEG 后丢失，VAE 伪影更高级别且更鲁棒——比 NPR/LNP 高 17-37%

2. **CLIP 软标签插值**:

    - 功能：增强语义检测器对新模型的泛化能力
    - 核心思路：在 CLIP 特征空间中，将真实图 embedding 和生成图 embedding 做线性插值生成虚拟训练样本，软标签也相应插值。这扩展了语义特征空间的覆盖范围
    - 设计动机：比直接增强 CLIP 检测器提升 10% 以上的泛化能力

3. **自适应调节器融合**:

    - 功能：根据输入动态分配伪影/语义路径权重
    - 核心思路：两个 MLP 调节器分别对伪影特征和语义特征产生缩放系数 α 和 β。训练时随机 dropout 某一路（使模型不过度依赖单路）。最终特征 = α·伪影特征 + β·语义特征
    - 设计动机：JPEG 后伪影路失效时调节器自动增大语义权重；面对新模型时增大伪影权重

### 损失函数 / 训练策略
二分类交叉熵。Co-SpyBench：100 万+图像，22 个生成模型（含 FLUX），5 个真实数据集，5 万野外数据。

## 实验关键数据

### 主实验

| 方法 | JPEG后 22 模型平均 AP | 无JPEG平均 AP |
|------|---------------------|-------------|
| NPR | ~65% | ~85% |
| DRCT | ~78% | ~92% |
| **Co-Spy** | **~89%** | **~93%** |

### 消融实验

| 配置 | 效果 |
|------|------|
| VAE 伪影 alone | 跨模型好但 JPEG 后降 |
| CLIP 语义 alone | 抗 JPEG 但新模型差 |
| 两路简单拼接 | 不如各自增强后融合 |
| **先增强再自适应融合** | **最优** |

### 关键发现
- **VAE 伪影 >> 上采样伪影**：在 JPEG 压缩下 VAE 伪影检测比 NPR/LNP 高 17-37%
- **增强+融合 > 简单融合**：先各路独立增强再融合的效果超越了增量之和
- **Co-SpyBench 是最全面的评估**：22 个生成模型（包括最新的 FLUX）+ 5 万野外数据

## 亮点与洞察
- **VAE 重建差异作为伪影**是一个优雅的想法——利用了所有扩散模型都经过 VAE 潜空间的共性
- **"先增强再融合"的三步策略**可推广到其他需要融合互补检测器的场景
- **Co-SpyBench 数据集**为合成图像检测领域提供了急需的全面基准

## 局限与展望
- VAE 伪影假设生成图像经过了 VAE 潜空间——GAN 生成图像不经过 VAE，可能不适用
- 调节器的训练需要同时有伪影和语义路的数据——冷启动可能不方便
- 对视频合成检测未扩展

## 相关工作与启发
- **vs DRCT**：DRCT 也融合多路但不做各路增强。Co-Spy 的先增强再融合策略效果更好
- **vs NPR / LNP**：上采样伪影方法。VAE 伪影更鲁棒且更高层次

## 评分
- 新颖性: ⭐⭐⭐⭐ VAE 伪影提取和软标签插值各有创新
- 实验充分度: ⭐⭐⭐⭐⭐ 22 个生成模型、JPEG 鲁棒性、野外数据、Co-SpyBench
- 写作质量: ⭐⭐⭐⭐ 互补性分析清晰
- 价值: ⭐⭐⭐⭐⭐ 对 AI 生成内容检测有重要实用价值

<!-- RELATED:START -->

## 相关论文

- [Beyond Semantic Features: Pixel-Level Mapping for Generalized AI-Generated Image Detection](../../AAAI2026/object_detection/beyond_semantic_features_pixel-level_mapping_for_generalized_ai-generated_image_.md)
- [A Bias-Free Training Paradigm for More General AI-generated Image Detection](a_bias-free_training_paradigm_for_more_general_ai-generated_image_detection.md)
- [Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection](generalized_diffusion_detector_mining_robust_features_from_diffusion_models_for_.md)
- [Zero-Shot Detection of AI-Generated Images](../../ECCV2024/object_detection/zero-shot_detection_of_ai-generated_images.md)
- [Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images](small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)

<!-- RELATED:END -->
