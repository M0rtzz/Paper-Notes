---
title: >-
  [论文解读] Controllable Human Image Generation with Personalized Multi-Garments
description: >-
  [CVPR 2025][图像生成][多服装人体生成] 提出 BootComp，通过两阶段框架（分解模块从真人图像自举生成合成配对数据 + 组合模块用双 SDXL 网络通过扩展自注意力注入多服装特征），实现给定多件参考服装的可控人体图像生成，MP-LPIPS 比基线提升 30%。
tags:
  - CVPR 2025
  - 图像生成
  - 多服装人体生成
  - 分解-组合
  - 自举数据
  - 扩展自注意力
  - 即插即用
---

# Controllable Human Image Generation with Personalized Multi-Garments

**会议**: CVPR 2025  
**arXiv**: [2411.16801](https://arxiv.org/abs/2411.16801)  
**代码**: https://omnious.github.io/BootComp  
**领域**: 扩散模型 / 图像生成  
**关键词**: 多服装人体生成, 分解-组合, 自举数据, 扩展自注意力, 即插即用

## 一句话总结
提出 BootComp，通过两阶段框架（分解模块从真人图像自举生成合成配对数据 + 组合模块用双 SDXL 网络通过扩展自注意力注入多服装特征），实现给定多件参考服装的可控人体图像生成，MP-LPIPS 比基线提升 30%。

## 研究背景与动机

**领域现状**：虚拟试穿和人体生成需要同时条件化多件服装参考图（上衣+下装+配饰等）。现有方法大多只支持单件服装或需要手动指定服装区域。

**现有痛点**：多服装人体生成缺乏大规模高质量配对数据（商品图-穿着图对）。直接从真实数据获取的服装图像噪声大、分辨率低。现有方法在多服装一致性上不佳——不同服装特征容易混合或丢失。

**核心矛盾**：需要高质量的 {多件商品图, 穿着效果图} 配对数据来训练，但真实世界中这种数据稀缺且难以大规模获取。

**本文目标** 如何在缺乏配对数据的情况下训练多服装条件人体生成模型。

**切入角度**：自举（bootstrapping）——先训练分解模块从真人图像中提取商品视角的服装图像，构建合成配对数据集，再用此数据训练组合模块。

**核心 idea**：分解模块从真人图像自举生成 54K 高质量配对数据，组合模块用冻结 SDXL 生成器 + 可训练编码器通过扩展自注意力注入多服装特征，且生成器冻结使其可即插即用接入 ControlNet/IP-Adapter。

## 方法详解

### 关键设计

1. **分解模块（自举数据生成）**：从 240K 真人图像中用人体解析分割服装区域，训练模型将穿着视角服装映射到商品视角。用 DreamSim 阈值 τ=0.4 过滤低质量结果，得到 54K 高质量配对

2. **组合模块（双网络注入）**：两个并行 SDXL 网络——冻结的生成器和可训练的编码器。编码器处理多件参考服装图像，通过扩展自注意力将服装特征注入生成器。生成器冻结使其可直接接入 ControlNet（姿态控制）、IP-Adapter（风格/人脸控制）等适配器

3. **多服装特征融合**：多件服装图像拼接输入编码器，扩展自注意力中 Key/Value 包含所有服装的 token，Query 来自生成器

## 实验关键数据

### 主实验

| 方法 | MP-LPIPS↓ | DINO↑ | M-DINO↑ | FID↓ |
|------|----------|-------|---------|------|
| MIP-Adapter | 0.276 | 0.308 | 0.025 | 59.99 |
| Parts2Whole | 0.267 | 0.362 | 0.036 | 28.39 |
| **BootComp** | **0.187** | **0.379** | **0.046** | **27.63** |

### 关键发现
- 自举数据的过滤阈值 τ=0.4 得到最优 DINO 分数，太松（τ=0.6）引入噪声数据，太紧数据量不足
- 冻结生成器的即插即用能力是核心优势——一次训练可接入多种控制模块
- 数据量增加单调提升性能（可扩展性良好）

## 局限与展望
- 依赖人体解析模型质量——解析噪声导致分解服装质量差
- 限于时尚领域数据集
- 冻结生成器的能力上限受预训练模型约束

## 评分
- 新颖性: ⭐⭐⭐⭐ 自举数据+冻结生成器即插即用的设计有巧思
- 实验充分度: ⭐⭐⭐⭐ 多基线+数据消融+扩展性验证
- 写作质量: ⭐⭐⭐⭐ 两阶段流程描述清晰
- 价值: ⭐⭐⭐⭐ 对电商虚拟试穿有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [DreamCache: Finetuning-Free Lightweight Personalized Image Generation via Feature Caching](dreamcache_finetuning-free_lightweight_personalized_image_generation_via_feature.md)
- [Learning Flow Fields in Attention for Controllable Person Image Generation](learning_flow_fields_in_attention_for_controllable_person_image_generation.md)
- [PersonaBooth: Personalized Text-to-Motion Generation](personabooth_personalized_text-to-motion_generation.md)
- [SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation](../../NeurIPS2025/image_generation/scenedesigner_controllable_multi-object_image_generation_with_9-dof_pose_manipul.md)
- [ConceptGuard: Continual Personalized Text-to-Image Generation with Forgetting and Confusion Mitigation](conceptguard_continual_personalized_text-to-image_generation_with_forgetting_and.md)

<!-- RELATED:END -->
