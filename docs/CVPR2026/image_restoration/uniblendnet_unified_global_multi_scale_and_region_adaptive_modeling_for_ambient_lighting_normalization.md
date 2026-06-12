---
title: >-
  [论文解读] UniBlendNet: Unified Global, Multi-Scale, and Region-Adaptive Modeling for Ambient Lighting Normalization
description: >-
  [CVPR 2026][图像恢复][ambient lighting normalization] 提出 UniBlendNet，在 IFBlend 基础上统一融合全局上下文建模、多尺度特征聚合和区域自适应残差精修三个模块，用于复杂空间变化光照条件下的环境光归一化任务。
tags:
  - "CVPR 2026"
  - "图像恢复"
  - "ambient lighting normalization"
  - "shadow removal"
  - "multi-scale aggregation"
  - "mask-guided refinement"
  - "frequency-spatial restoration"
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

UniBlendNet 要解决的是环境光归一化（ALN）：多光源、几何与材质交互在画面上留下空间高度不均的光照退化，亮区、暗区、不同尺度的阴影混在一起，需要一次性还原成正常照明。它沿用 IFBlend 的频率-空间联合恢复编码器-解码器作骨干，但在三处动手——旁挂一条全局上下文分支补长距离依赖，在瓶颈层插入多尺度自适应聚合，最后用一张软掩码控制"在哪修、修多重"。最终还原写成 $\mathbf{I}_r = \mathbf{I}_{inp} + \mathbf{M} \odot \mathbf{R}$，掩码 $\mathbf{M}$ 决定残差 $\mathbf{R}$ 在每个位置的施加强度。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["退化输入 I_inp"] --> B["IFBlend 频率-空间编码器"]
    A --> G["全局上下文分支 UniConvNet<br/>逐步增大卷积核扩感受野 → 全局特征 F_g"]
    B --> S["尺度感知聚合 SAAM<br/>瓶颈层三级金字塔 + 动态尺度权重"]
    S --> D["IFBlend 解码器末端特征"]
    D --> F["特征融合<br/>解码器特征 + 全局特征 F_g"]
    G --> F
    F --> M["掩码头<br/>sigmoid → 软掩码 M∈[0,1]"]
    F --> R["残差头 → 残差 R"]
    M --> C["掩码引导残差精修<br/>合成 I_r = I_inp + M⊙R"]
    R --> C
    C --> O["归一化输出图"]
```

### 关键设计

**1. UniConvNet 全局上下文分支：补回频率-空间骨干看不到的场景级光照依赖**

IFBlend 的频率-空间骨干擅长局部纹理与频域先验，但感受野有限，捕不到"这盏灯照亮了半个房间"这种长距离光照关系。UniBlendNet 并行挂一条 UniConvNet 分支，靠逐步增大卷积核把有效感受野撑开，直接从输入图像抽全局特征 $\mathbf{F}_g$，再与解码器末端特征融合一起喂给残差预测头。这样全局光照趋势交给专门的大感受野分支负责，骨干只需管好局部细节，分工更干净。

**2. 尺度感知聚合模块 SAAM：让网络自己挑该信哪个尺度**

阴影和光照不一致出现在不同空间尺度上——大片暗角是粗尺度问题，边缘渗光是细尺度问题，固定的多尺度融合权重无法兼顾。SAAM 在瓶颈层搭三级金字塔（原尺度、2× 与 4× 降采样），用共享权重卷积分别处理后上采样回原尺度；再用全局平均池化得到全局描述符，过一个轻量 MLP 预测动态尺度权重，残差融合成最终瓶颈特征。权重随输入自适应，网络就能针对当前图像强调最有用的那个尺度，而不是套一组手工系数。

**3. 掩码引导残差精修：把"哪里修、修多重"和"修成什么"解耦**

均匀施加残差会让亮区被过度增强、暗区又欠校正。UniBlendNet 拆出两个独立预测头：掩码头经 sigmoid 输出连续软掩码 $\mathbf{M} \in [0,1]$，逐位置控制校正强度；残差头融合解码器特征与全局上下文 $\mathbf{F}_g$ 生成残差 $\mathbf{R}$。两者按 $\mathbf{I}_r = \mathbf{I}_{inp} + \mathbf{M} \odot \mathbf{R}$ 合成，于是"修复力度"成了空间自适应的可学习量；监督上用退化图与干净图的相对灰度差构造伪二值掩码来引导 $\mathbf{M}$。

### 损失函数 / 训练策略

多目标联合训练：$\mathcal{L} = \mathcal{L}_{rec} + \alpha_1 \mathcal{L}_{ssim} + \alpha_2 \mathcal{L}_{grad} + \alpha_3 \mathcal{L}_{perc} + \lambda \mathcal{L}_{mask}$，依次是 L1 重建、SSIM 结构相似、梯度一致性、感知损失，以及对软掩码的 L1 监督；伪掩码由正向相对灰度差阈值化得到。

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
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](../../ICCV2025/image_restoration/learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)

</div>

<!-- RELATED:END -->
