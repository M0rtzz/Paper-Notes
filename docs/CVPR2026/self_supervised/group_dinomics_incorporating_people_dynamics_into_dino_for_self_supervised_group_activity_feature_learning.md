---
title: >-
  [论文解读] Group-DINOmics: Incorporating People Dynamics into DINO for Self-supervised Group Activity Feature Learning
description: >-
  [CVPR 2026][自监督学习][group activity] 提出利用 DINOv3 结合两个自监督预训练任务（人物光流估计和群体相关物体定位）来学习群体活动特征（GAF），在无群体活动标注的情况下大幅超越现有方法。
tags:
  - "CVPR 2026"
  - "自监督学习"
  - "group activity"
  - "DINO"
  - "光流"
  - "activity retrieval"
---

# Group-DINOmics: Incorporating People Dynamics into DINO for Self-supervised Group Activity Feature Learning

**会议**: CVPR 2026  
**arXiv**: [2604.04467](https://arxiv.org/abs/2604.04467)  
**代码**: [https://github.com/tezuka0001/Group-DINOmics](https://github.com/tezuka0001/Group-DINOmics)  
**领域**: 自监督  
**关键词**: group activity, self-supervised learning, DINO, optical flow, activity retrieval

## 一句话总结

提出利用 DINOv3 结合两个自监督预训练任务（人物光流估计和群体相关物体定位）来学习群体活动特征（GAF），在无群体活动标注的情况下大幅超越现有方法。

## 研究背景与动机

群体活动分析在体育分析、机器人和监控中有重要应用。自监督 GAF 学习避免了大量人工标注和预定义活动类别的需求。但现有方法（如 HRN、GAFL）仅通过局部图像重建作为预训练任务，只能嵌入局部外观线索（如服装纹理），缺乏两个关键能力：(1) 每个人的运动动态特征（局部动态）；(2) 人群空间配置等全局特征。

即使 DINOv3 能平衡局部和全局特征，直接应用仍不足，因为其在静态图像上训练的通用特征包含大量与群体活动无关的局部特征（如地板纹理），且缺乏时序运动特征。

## 方法详解

### 整体框架

这篇论文想在没有群体活动标注的前提下学到好的群体活动特征（GAF）。整体流程是：DINOv3 先抽每帧图像特征，过 Transformer 编码器 + MLP，再做时序池化得到 GAF；训练时不直接监督活动类别，而是让模型从 GAF 去完成两个自监督代理任务——估计每个人的光流、定位群体相关物体，借这两个任务把"局部运动动态"和"全局空间配置"逼进 GAF 里。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["视频帧"] -->|"训练时先抹除<br/>群体相关物体 (inpainting)"| B["DINOv3 抽帧特征 I"]
    B --> C["Transformer 编码器 + MLP<br/>→ 视频特征 V"]
    C --> D["时序池化<br/>→ 群体活动特征 GAF"]
    D --> E["人物光流估计<br/>从 GAF + 辅助分支从单帧特征估 xy 光流"]
    D --> F["群体相关物体定位<br/>从 GAF 推断被抹物体位置"]
    E --> G["两阶段训练<br/>先光流 50 epoch，再物体 30 epoch"]
    F --> G
    G --> H["输出 GAF<br/>用于群体活动检索 / 识别"]
```

### 关键设计

**1. 人物光流估计：把局部运动动态逼进 GAF**

DINOv3 是在静态图像上训练的，特征里没有时序运动信息，而群体活动恰恰离不开"每个人怎么动"。本文让模型从 GAF 估计每个人的 xy 光流值，并额外设一个辅助分支直接从单帧特征估光流——后者缩短了反向传播路径，使 DINOv3 backbone 也能被运动信号有效更新，从而把局部动态特征学进表示里。

**2. 群体相关物体定位与修复：用 inpainting 逼出全局推理**

光有局部动态还不够，群体活动还依赖人群的整体空间配置。本文训练时把球这类群体相关物体在图像上抹掉做修复（inpainting），并从 GAF 估计被抹物体的位置。因为局部外观线索被遮住了，模型只能从全局上下文（人群怎么站位）来推断物体在哪，这就强迫 GAF 编码进全局特征，而非只记住服装纹理这类局部外观。

**3. 两阶段训练：先光流后物体，避开任务冲突**

两个代理任务的优化方向并不完全一致，同时训容易互相干扰。本文分两阶段：第一阶段用光流损失训 50 epochs 先把局部动态学好，第二阶段再用物体定位损失训 30 epochs 补全局特征，分阶段优化避免两个目标打架。

### 损失函数 / 训练策略

光流损失 $L_F = L_{F,G} + L_{F,I}$（分别从 GAF 和单帧特征估计），物体定位损失 $L_O = L_{O,G} + L_{O,I}$。伪标签来自 RAFT（光流）和 YOLOX（检测），仅训练时使用。

## 实验关键数据

### 主实验

| 数据集 | 方法 | Hit@1 | Hit@3 |
|--------|------|-------|-------|
| VBD | GAFL (之前SOTA) | 61.1 | 82.4 |
| VBD | **Ours** | **82.7** | **93.0** |
| NBA | GAFL | 24.7 | 50.4 |
| NBA | **Ours** | **43.9** | **72.0** |

### 消融实验

| 配置 | VBD Hit@1 | NBA Hit@1 |
|------|----------|----------|
| 无任何预训练任务 | 43.0 | 22.6 |
| 仅光流损失 | 75.4 | 34.1 |
| 仅物体定位损失 | 74.0 | 40.3 |
| 两阶段训练（完整） | **82.7** | **43.9** |

### 关键发现

- Hit@1 相比 GAFL 提升 21.6 (VBD) 和 19.2 (NBA)，是巨大的性能跃升
- 光流估计和物体定位任务各自独立贡献，两阶段组合效果最优
- 辅助分支（从单帧估计）对 DINOv3 学习动态特征至关重要

## 亮点与洞察

- 两个预训练任务设计精巧且各有明确动机
- Inpainting 迫使全局推理的思路巧妙
- 性能提升幅度巨大，证明动态+全局特征对群体活动理解至关重要

## 局限与展望

- 目前仅在团队球类运动场景验证
- 群体相关物体需要预定义（如球），通用性受限

## 评分

- 新颖性：⭐⭐⭐⭐ — 预训练任务设计新颖
- 技术深度：⭐⭐⭐⭐ — 双分支+两阶段设计合理
- 实验充分度：⭐⭐⭐⭐ — 充分消融验证
- 实用价值：⭐⭐⭐ — 场景相对受限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] FedGRPO: Privately Optimizing Foundation Models with Group-Relative Rewards from Domain Clients](../../AAAI2026/self_supervised/fedgrpo_privately_optimizing_foundation_models_with_group-relative_rewards_from_.md)
- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [\[CVPR 2026\] A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking](a_stitch_in_time_learning_procedural_workflow_via_self_supervised_plackett_luce_r.md)
- [\[CVPR 2026\] Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers](zero_ablation_overstates_register_content_dependence_in_dino_vision_transformers.md)
- [\[CVPR 2026\] Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting](redepth_anything_test-time_depth_refinement_via_self-supervised_re-lighting.md)

</div>

<!-- RELATED:END -->
