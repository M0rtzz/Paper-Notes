---
title: >-
  [论文解读] UniBlendNet: Unified Global, Multi-Scale, and Region-Adaptive Modeling for Ambient Lighting Normalization
description: >-
  [CVPR 2026][图像恢复][ambient lighting normalization] 提出 UniBlendNet，在 IFBlend 基础上统一融合全局上下文建模、多尺度特征聚合和区域自适应残差精修三个模块，用于复杂空间变化光照条件下的环境光归一化任务。
tags:
  - CVPR 2026
  - 图像恢复
  - ambient lighting normalization
  - shadow removal
  - multi-scale aggregation
  - mask-guided refinement
  - frequency-spatial restoration
---

# UniBlendNet: Unified Global, Multi-Scale, and Region-Adaptive Modeling for Ambient Lighting Normalization

**会议**: CVPR 2026  
**arXiv**: [2604.13383](https://arxiv.org/abs/2604.13383)  
**代码**: 无  
**领域**: 图像复原  
**关键词**: ambient lighting normalization, shadow removal, multi-scale aggregation, mask-guided refinement, frequency-spatial restoration

## 一句话总结

提出 UniBlendNet，在 IFBlend 基础上统一融合全局上下文建模、多尺度特征聚合和区域自适应残差精修三个模块，用于复杂空间变化光照条件下的环境光归一化任务。

## 研究背景与动机

环境光归一化 (ALN) 旨在恢复因多光源、物体几何和材质属性交互导致的复杂空间变化光照退化图像。现有方法如 IFBlend 利用频域先验建模光照变化，但仍存在三个关键不足：(1) 全局上下文建模能力有限，无法捕获场景级长距离光照依赖；(2) 残差校正在空间上均匀施加，导致亮区过增强、暗区欠校正；(3) 缺乏自适应的多尺度特征聚合策略来处理不同空间尺度的阴影和光照不一致。这三个局限性促使作者设计统一框架同时解决这三个方面。

## 方法详解

### 整体框架

UniBlendNet 建立在 IFBlend 的编码器-解码器频率-空间联合恢复骨干之上，引入三个互补组件：(1) 基于 UniConvNet 的全局上下文分支，与主骨干并行提取全局特征；(2) 在瓶颈层插入尺度感知聚合模块 (SAAM)；(3) 掩码引导的残差精修机制。最终恢复公式为 $\mathbf{I}_r = \mathbf{I}_{inp} + \mathbf{M} \odot \mathbf{R}$，其中 $\mathbf{M}$ 为软引导掩码，$\mathbf{R}$ 为预测残差。

### 关键设计

1. **UniConvNet 全局上下文建模**: 利用 UniConvNet 通过逐步增大卷积核的聚合来扩展有效感受野，直接从输入图像提取全局上下文特征 $\mathbf{F}_g$，与解码器最终特征融合后用于残差预测。这补充了频率-空间骨干在长距离依赖建模上的不足。

2. **尺度感知聚合模块 (SAAM)**: 在瓶颈层构建三级金字塔（原尺度、2× 降采样、4× 降采样），共享权重卷积分支处理后上采样回原尺度。通过全局平均池化计算全局描述符，轻量级 MLP 预测动态尺度权重，残差融合得到最终瓶颈特征。这使网络能动态强调最有用的尺度。

3. **掩码引导残差精修**: 设计两个独立预测头：掩码预测头通过 sigmoid 输出连续软引导掩码 $\mathbf{M} \in [0,1]$，控制每个位置的残差校正强度；残差预测头融合解码器特征和全局上下文特征生成残差 $\mathbf{R}$。伪二值掩码通过退化与干净图像的相对灰度差构建用于监督。

### 损失函数 / 训练策略

采用多目标损失函数联合训练：$\mathcal{L} = \mathcal{L}_{rec} + \alpha_1 \mathcal{L}_{ssim} + \alpha_2 \mathcal{L}_{grad} + \alpha_3 \mathcal{L}_{perc} + \lambda \mathcal{L}_{mask}$，包括 L1 重建损失、SSIM 结构相似性损失、梯度一致性损失、感知损失和掩码 L1 监督损失。伪掩码通过正向相对灰度差阈值化构建。

## 实验关键数据

### 主实验

在 NTIRE Ambient Lighting Normalization 基准上评估。UniBlendNet 在 PSNR 和 SSIM 上一致优于 IFBlend 基线，视觉效果更自然稳定。

| 指标 | IFBlend (基线) | UniBlendNet | 提升 |
|------|---------------|-------------|------|
| PSNR | 基线值 | 更高 | 持续提升 |
| SSIM | 基线值 | 更高 | 持续提升 |

### 消融实验

消融研究确认三个关键组件的各自贡献：全局光照建模、尺度感知特征聚合和区域自适应残差精修各自带来增量改进。

### 关键发现

- 三个模块以互补方式协作，共同提升光照一致性和结构保真度
- 掩码引导机制有效实现了空间自适应恢复，避免了全局均匀校正的问题
- SAAM 的动态权重学习使网络能根据输入自适应选择最重要的尺度

## 亮点与洞察

- 将 ALN 问题分解为全局-多尺度-局部三个层次的统一框架设计思路清晰
- 掩码引导残差精修的"在哪修复、修复多少"思想具有通用性
- SAAM 的动态尺度权重避免了手工设计尺度融合策略

## 局限与展望

- 实验仅在单一 ALN 基准上验证，泛化能力有待考察
- 模型复杂度增加带来的推理速度损失未讨论
- 多光源场景中光源估计和分离仍是开放问题

## 相关工作与启发

- IFBlend 的频域先验为光照归一化提供了有效基础
- UniConvNet 的大核聚合策略可扩展到其他需要大感受野的底层视觉任务
- 掩码引导的区域自适应思想可借鉴到其他空间非均匀退化恢复任务

## 评分

6/10 — 方法设计合理、三模块互补，但贡献偏增量且仅在单一基准上验证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization](unirain_unified_image_deraining_rag_dataset_distillation.md)
- [\[NeurIPS 2025\] MS-BART: Unified Modeling of Mass Spectra and Molecules for Structure Elucidation](../../NeurIPS2025/image_restoration/ms-bart_unified_modeling_of_mass_spectra_and_molecules_for_structure_elucidation.md)
- [\[CVPR 2026\] IA-CLAHE: Image-Adaptive Clip Limit Estimation for CLAHE](ia_clahe_image_adaptive_clip_limit.md)
- [\[CVPR 2026\] RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration](rar_restore_assess_repeat_a_unified_framework_for_iterative_image_restoration.md)
- [\[CVPR 2026\] UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution](ucan_unified_convolutional_attention_lightweight_sr.md)

</div>

<!-- RELATED:END -->
