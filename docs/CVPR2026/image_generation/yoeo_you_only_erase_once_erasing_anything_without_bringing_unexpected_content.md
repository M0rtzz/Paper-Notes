---
title: >-
  [论文解读] YOEO: You Only Erase Once - Erasing Anything without Bringing Unexpected Content
description: >-
  [CVPR 2026][图像生成][目标擦除] YOEO 提出一个单次擦除框架，通过将多步扩散模型蒸馏为少步模型实现高效推理，并设计杂物抑制损失（基于实体分割检测新生成的不应出现的物体）和实体特征一致性损失（确保擦除区域与周围语义一致），解决扩散模型在目标擦除中的幻觉问题。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "目标擦除"
  - "扩散蒸馏"
  - "幻觉抑制"
  - "实体一致性"
  - "无配对训练"
---

# YOEO: You Only Erase Once - Erasing Anything without Bringing Unexpected Content

**会议**: CVPR 2026  
**arXiv**: [2603.27599](https://arxiv.org/abs/2603.27599)  
**代码**: [https://zyxunh.github.io/YOEO-ProjectPage/](https://zyxunh.github.io/YOEO-ProjectPage/)  
**领域**: 图像生成 / 图像编辑  
**关键词**: 目标擦除, 扩散蒸馏, 幻觉抑制, 实体一致性, 无配对训练

## 一句话总结

YOEO 提出一个单次擦除框架，通过将多步扩散模型蒸馏为少步模型实现高效推理，并设计杂物抑制损失（基于实体分割检测新生成的不应出现的物体）和实体特征一致性损失（确保擦除区域与周围语义一致），解决扩散模型在目标擦除中的幻觉问题。

## 研究背景与动机

扩散模型在图像修复方面表现优异，但用于目标擦除时常"画蛇添足"——移除目标物体后在掩码区域生成不应存在的新物体。现有闭源方案（ChatGPT、Nano Banana）效果好但计算开销大，不适合边缘设备部署。

**两个根本原因**：(1) 缺乏真实擦除数据——合成配对数据（随机遮挡+原图做GT）不能代表真实擦除场景；(2) SFT 只教模型"去噪"而非"擦除"——像素级重建损失不包含"不应生成新物体"的约束。

## 方法详解

### 整体框架

以预训练擦除扩散模型为教师，蒸馏为少步学生模型。训练使用两类数据：配对数据 $\mathcal{D}_1$（随机遮挡背景区域，原图做GT）和无配对数据 $\mathcal{D}_2$（遮挡目标物体，无GT）。在蒸馏基础上加入杂物抑制损失和实体特征一致性损失。

### 关键设计

1. **擦除扩散蒸馏**:

    - 功能：将多步扩散模型压缩为少步模型，同时保持擦除质量
    - 核心思路：使用 DMD2 + GAN 损失的蒸馏框架。少步模型在早期去噪步就能产生清晰结果（而非模糊），这使得后续的杂物检测和一致性评估成为可能——多步模型的早期输出太模糊无法做有意义的评估
    - 设计动机：蒸馏不仅提升推理效率，更关键的是使无配对数据的端到端监督成为可能

2. **杂物抑制损失 (Sundries Suppression Loss)**:

    - 功能：检测并抑制擦除后在掩码区域生成的不应存在的物体
    - 核心思路：用预训练实体分割模型对擦除结果做分割，计算每个检测到的实体与修复掩码的 IoS（Intersection over Segment）。IoS 超过阈值 $\lambda$ 的实体被判定为新生成的"杂物"，以此构建损失函数惩罚模型生成杂物
    - 设计动机：传统像素级损失不知道"什么不该出现"。利用实体分割模型作为自动检测器，等价于引入了"擦除区域不应有独立实体"的先验

3. **实体特征一致性损失 (Entity Feature Coherence Loss)**:

    - 功能：确保擦除区域与周围环境在语义上一致
    - 核心思路：从预训练分割网络提取特征，计算掩码内生成区域与掩码外原始区域的特征余弦相似度。如果生成区域与周围环境语义一致，其特征应围绕相同的表示中心聚集
    - 设计动机：即使不生成杂物，如果填充内容与周围环境风格/语义不一致也是失败的擦除

### 损失函数 / 训练策略

总损失 = LPIPS蒸馏损失 + DMD损失 + GAN损失 + 杂物抑制损失 + 实体特征一致性损失。配对数据和无配对数据交替训练。

## 实验关键数据

### 主实验

| 方法 | 擦除干净度 | 语义一致性 | 推理速度 | 说明 |
|------|-----------|-----------|---------|------|
| SmartEraser | 低 | 低 | 慢 | 容易生成杂物 |
| ASUKA | 中 | 中 | 慢 | MAE+扩散 |
| **YOEO** | **高** | **高** | **快（少步）** | 单次干净擦除 |

YOEO 在定量和定性指标上全面超越现有方法。

### 消融实验

| 配置 | 杂物率↓ | 一致性↑ | 说明 |
|------|--------|---------|------|
| 仅蒸馏 | 高 | 低 | 和教师模型一样有幻觉 |
| + 杂物抑制损失 | 显著降低 | 低 | 有效减少杂物 |
| + 实体一致性损失 | 显著降低 | 高 | 语义一致性增强 |
| 完整 YOEO | 最低 | 最高 | 两个损失互补 |

### 关键发现

- 蒸馏为少步模型是启用无配对监督的前提——多步扩散的中间状态太模糊无法做有意义的评估
- 杂物抑制损失的贡献最大，说明"不生成新物体"是擦除任务最核心的约束
- 实体特征一致性提供了"和谐感"，防止填充区域与周围环境格格不入

## 亮点与洞察

- **从"去噪"到"擦除"的认知转变**：传统像素级重建损失只教模型"修复图像"，YOEO 通过杂物检测和一致性约束显式教模型"什么不该做"
- **蒸馏的意外价值**：蒸馏不仅加速推理，更启用了之前不可能的端到端无配对训练——这一洞察可迁移到其他需要端到端评估的生成任务
- **实体分割作为通用评估器**：用预训练分割模型自动检测"不该出现的东西"，比人工设计规则更鲁棒更通用

## 局限与展望

- 依赖实体分割模型的质量——如果分割模型漏检或误检会影响损失准确性
- 单次擦除对极大面积的擦除区域可能不够
- 少步蒸馏可能损失部分生成细节
- 未来可探索视频中的目标擦除（时序一致性）

## 相关工作与启发

- **vs SmartEraser**: SmartEraser 合成配对数据+目标prompt，YOEO 无需配对数据且不需要显式prompt
- **vs ASUKA**: ASUKA 用MAE+扩散减少幻觉，YOEO 用杂物抑制损失更直接
- **vs TurboFill**: TurboFill 专注于高效扩散修复，但没有擦除专用的约束

## 评分

- 新颖性: ⭐⭐⭐⭐ 杂物抑制损失和蒸馏启用无配对训练的思路有创意
- 实验充分度: ⭐⭐⭐⭐ 对比充分，定性结果有说服力
- 写作质量: ⭐⭐⭐⭐ 问题分析透彻
- 价值: ⭐⭐⭐⭐ 实际编辑应用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] VOSR: A Vision-Only Generative Model for Image Super-Resolution](vosr_a_vision_only_generative_model_for_image_super_resolution.md)
- [\[CVPR 2026\] RewardFlow: Generate Images by Optimizing What You Reward](rewardflow_generate_images_by_optimizing_what_you_reward.md)
- [\[ECCV 2024\] You Only Need One Step: Fast Super-Resolution with Stable Diffusion via Scale Distillation](../../ECCV2024/image_generation/you_only_need_one_step_fast_super-resolution_with_stable_diffusion_via_scale_dis.md)
- [\[CVPR 2026\] Towards Robust Content Watermarking Against Removal and Forgery Attacks](towards_robust_content_watermarking_against_removal_and_forgery_attacks.md)
- [\[CVPR 2026\] Low-Resolution Editing is All You Need for High-Resolution Editing](low-resolution_editing_is_all_you_need_for_high-resolution_editing.md)

</div>

<!-- RELATED:END -->
