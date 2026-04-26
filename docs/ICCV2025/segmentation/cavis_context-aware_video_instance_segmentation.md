---
title: >-
  [论文解读] CAVIS: Context-Aware Video Instance Segmentation
description: >-
  [ICCV 2025][图像分割][视频实例分割] 提出CAVIS，通过引入上下文感知实例追踪器（CAIT）融合物体边界周围的上下文信息来增强实例关联，并设计原型化跨帧对比损失（PCC）保证跨帧特征一致性，在VIS和VPS任务上全面刷新SOTA。
tags:
  - ICCV 2025
  - 图像分割
  - 视频实例分割
  - 上下文感知
  - 对比学习
  - 实例追踪
  - Mask2Former
---

# CAVIS: Context-Aware Video Instance Segmentation

**会议**: ICCV 2025  
**arXiv**: [2407.03010](https://arxiv.org/abs/2407.03010)  
**代码**: https://github.com/seunghunlee918/cavis (有)  
**领域**: 分割  
**关键词**: 视频实例分割, 上下文感知, 对比学习, 实例追踪, Mask2Former

## 一句话总结

提出CAVIS，通过引入上下文感知实例追踪器（CAIT）融合物体边界周围的上下文信息来增强实例关联，并设计原型化跨帧对比损失（PCC）保证跨帧特征一致性，在VIS和VPS任务上全面刷新SOTA。

## 研究背景与动机

视频实例分割（VIS）要求在视频序列中同时分割和识别每个物体实例。现代VIS方法基于query-based架构（如Mask2Former），通过跨帧实例特征的关联来实现追踪。

然而，现有方法在以下情况下追踪容易失败：

**严重遮挡**：物体被遮挡后重新出现时，仅靠核心实例特征难以正确重识别

**相似外观**：多个外观相似的物体同时出现时（如多辆同色汽车），instance center特征无法区分

作者从认知科学和神经科学中获得启发：人类感知在解析复杂场景时会重度依赖上下文线索。例如，判断一辆自行车的身份时，如果看到"有人骑在上面"这个上下文信息，就能大大提高识别准确率。

**核心创新**：将物体边界周围的上下文语义信息融入实例特征，使追踪器不仅"看到"物体本身，还能"看到"物体所处的环境。

## 方法详解

### 整体框架

CAVIS基于Mask2Former分割网络，包含两个核心组件：

1. **上下文感知实例追踪器（CAIT）**：提取并融合物体周围的上下文特征
2. **原型化跨帧对比损失（PCC Loss）**：通过pixel-level的原型匹配增强帧间特征一致性

### 关键设计

**1. 上下文感知特征提取**

从Mask2Former获得实例特征、特征图F和分割mask M后：

- 对特征图F做平均滤波（9x9 kernel）得到模糊化特征，获取区域级上下文
- 对分割mask M做Laplacian滤波，提取物体边界区域
- 在边界区域内对模糊化特征取平均池化，得到实例周围特征
- 将核心特征和周围特征拼接后通过MLP融合，得到上下文感知特征Q

Laplacian边界比膨胀mask更精确，平均滤波自然覆盖了边界外侧的语义信息，两者配合实现了高效的"周围上下文"提取。

**2. 上下文感知跨帧匹配**

将上下文感知特征Q送入改进的transformer-based追踪器：
- 使用context-aware cross-attention机制，让当前帧的上下文感知特征与历史帧的对应特征进行注意力匹配
- 通过Hungarian matching对齐帧间实例特征的顺序，序号一致即为同一物体

**3. 原型化跨帧对比损失（PCC Loss）**

- 为每个实例构建原型：基于mask区域内高级特征图的加权平均
- 将原型与pixel embedding做cosine相似度匹配，生成instance-pixel相关图
- 对比损失确保同一实例在不同帧的原型与其对应区域的pixel embedding保持一致
- 既增强帧内区域一致性（intra-frame），又强化帧间连续性（inter-frame）

### 损失函数 / 训练策略

总训练损失：L = L_VIS + lambda_Emb * L_Emb + lambda_PCC * L_PCC

- L_VIS: VIS标准损失（分类+BCE+Dice）
- L_Emb: 实例嵌入对比损失（跨帧）
- L_PCC: 原型化跨帧对比损失
- 使用Mask2Former(Swin-L)作为分割backbone
- 训练使用标准VIS训练策略

## 实验关键数据

### 主实验

OVIS数据集（最具挑战性的VIS benchmark）：

| 方法 | AP |
|------|----|
| DVIS (ICCV'23) | 37.8 |
| CTVIS (ICCV'23) | 38.7 |
| GenVIS (CVPR'23) | 36.4 |
| **CAVIS** | **41.0** |

YTVIS19数据集：

| 方法 | AP |
|------|----|
| DVIS | 55.1 |
| CTVIS | 55.4 |
| **CAVIS** | **57.2** |

VIPSeg（视频全景分割）：

| 方法 | VPQ |
|------|-----|
| DVIS | 47.2 |
| **CAVIS** | **49.5** |

### 消融实验

OVIS数据集上各组件贡献：

| 配置 | AP |
|------|----|
| Baseline (无上下文) | 37.8 |
| + Context-aware features | 39.2 |
| + Context-aware cross-attention | 39.8 |
| + PCC Loss | 40.5 |
| + Full CAVIS | **41.0** |

上下文提取方式对比：

| 上下文策略 | AP |
|------------|-----|
| 无上下文 | 37.8 |
| 全图上下文 (CAROQ-style) | 38.9 |
| 膨胀mask区域 | 39.0 |
| **边界区域上下文 (Ours)** | **39.2** |

### 关键发现

1. CAVIS在OVIS这种高遮挡、高复杂度的benchmark上优势最为明显（+2.3 AP），说明上下文信息在遮挡场景中尤其重要
2. 边界区域上下文（Laplacian方式）优于全图上下文，后者引入了无关噪声且内存开销大
3. PCC Loss单独即带来显著AP提升，说明pixel-level的原型一致性比instance-level的嵌入对比更有效
4. 方法在VIS和VPS两种任务上都有效，具有较好的泛化性

## 亮点与洞察

- **认知科学驱动的设计**：从人类利用上下文线索识别物体的认知过程出发，是很好的motivation
- **轻量级上下文提取**：Laplacian + 平均滤波的组合简单高效，不需要额外网络
- **PCC Loss的设计智慧**：通过原型化桥接pixel和instance两个尺度，比直接的实例对比loss更细粒度
- 在OVIS上的大幅提升证明了上下文信息对解决遮挡问题的关键作用

## 局限与展望

1. 上下文特征的Laplacian边界宽度是固定的，不同大小物体可能需要自适应边界
2. 平均滤波的9x9 kernel大小是手动设定的，缺乏理论指导
3. 缓存中实验部分不完整，部分定量结果需要从论文原文确认
4. 对极小物体，边界上下文可能被背景噪声淹没
5. PCC Loss增加了训练开销，实际推理速度影响有待评估

## 相关工作与启发

- 继承了DVIS的解耦框架思想（分割-追踪-优化），在追踪环节加入上下文增强
- CAROQ也使用上下文特征但依赖全图memory bank，存在内存瓶颈；CAVIS聚焦边界区域更高效
- PCC Loss借鉴了SimCLR的对比学习思想，但从instance级扩展到pixel-instance级

## 评分

- 新颖性: ⭐⭐⭐⭐ — 上下文感知追踪思路新颖，PCC Loss设计巧妙，但整体增量性较强
- 实验充分度: ⭐⭐⭐⭐ — 覆盖VIS和VPS多个benchmark，消融充分
- 写作质量: ⭐⭐⭐⭐ — Motivation从认知科学引入很吸引人，符号体系完整
- 价值: ⭐⭐⭐⭐ — 在challenging VIS场景（遮挡、相似外观）上的提升有实际意义

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Hierarchical Visual Prompt Learning for Continual Video Instance Segmentation](hierarchical_visual_prompt_learning_for_continual_video_instance_segmentation.md)
- [\[ICCV 2025\] SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation](score_scene_context_matters_in_openvocabulary_remote_sensing.md)
- [\[ECCV 2024\] VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement](../../ECCV2024/segmentation/visage_video_instance_segmentation_with_appearance-guided_enhancement.md)
- [\[CVPR 2025\] RipVIS: Rip Currents Video Instance Segmentation Benchmark for Beach Monitoring](../../CVPR2025/segmentation/ripvis_rip_currents_video_instance_segmentation_benchmark_for_beach_monitoring_a.md)
- [\[ICML 2025\] ConText: Driving In-context Learning for Text Removal and Segmentation](../../ICML2025/segmentation/context_driving_in-context_learning_for_text_removal_and_segmentation.md)

<!-- RELATED:END -->
