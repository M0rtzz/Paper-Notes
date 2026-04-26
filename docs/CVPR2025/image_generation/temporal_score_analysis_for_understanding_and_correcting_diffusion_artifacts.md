---
title: >-
  [论文解读] Temporal Score Analysis for Understanding and Correcting Diffusion Artifacts
description: >-
  [CVPR 2025][图像生成][扩散模型伪影] 发现扩散生成过程中的三阶段（Profiling-Mutation-Refinement）及伪影形成的"分数陷阱"机制，提出 ASCED 通过监控异常分数动力学实时检测和校正伪影，无需训练即可匹配或超越有监督方法。
tags:
  - CVPR 2025
  - 图像生成
  - 扩散模型伪影
  - 分数动力学
  - 无监督检测
  - 在线校正
  - 生成质量
---

# Temporal Score Analysis for Understanding and Correcting Diffusion Artifacts

**会议**: CVPR 2025  
**arXiv**: [2503.16218](https://arxiv.org/abs/2503.16218)  
**代码**: [项目主页](https://YuCao16.github.io/ASCED)  
**领域**: 图像生成 / 其他  
**关键词**: 扩散模型伪影, 分数动力学, 无监督检测, 在线校正, 生成质量

## 一句话总结

发现扩散生成过程中的三阶段（Profiling-Mutation-Refinement）及伪影形成的"分数陷阱"机制，提出 ASCED 通过监控异常分数动力学实时检测和校正伪影，无需训练即可匹配或超越有监督方法。

## 研究背景与动机

### 领域现状

**领域现状**：即使在大规模数据集上训练，扩散模型生成的图像仍会出现视觉伪影（局部纹理/结构异常）

### 核心矛盾

**核心矛盾**：现有方法主要依赖有监督分类器（在标注数据集上训练或利用大型多模态模型），但未解释伪影为何产生

### 现有痛点

**现有痛点**：基于不确定性的方法仅分析最终输出的空间方差 $Var(x_0)$，忽略了生成过程中关键的时序动力学

### 解决思路

**解决思路**：后处理校正方法（如生成后再加噪-去噪）在已完成的生成结果上操作，效率低且效果有限

### 补充说明

**补充说明**：缺乏对伪影形成机制的根本理解——为什么这些区域变成了伪影？何时形成的？

## 方法详解

### 整体框架

ASCED 分为检测和校正两步，嵌入在标准扩散推理过程中。从检测起始步 $T_d$ 开始，在分数银行中记录各步的分数值。到校正步 $T_c$ 时，分析分数银行中的时序动力学识别异常区域 $\Omega^a$。对伪影区域施加轨迹感知的定向扰动，在不中断推理流程的情况下实时校正。

### 关键设计

**1. 扩散生成三阶段发现 (Profiling-Mutation-Refinement)**
- **功能**: 揭示扩散生成过程的内在机制，为伪影检测提供理论基础
- **核心思路**: 发现扩散生成实质上经历三个阶段：(1) Profiling——恢复全局均值模板和基础语义布局；(2) Mutation——引入局部像素级变化以创建局部结构，此阶段伪影区域出现异常分数动力学；(3) Refinement——将局部变化整合为上下文连贯的视觉细节
- **设计动机**: 伪影在 Mutation 阶段形成：某些区域经历剧烈的分数变化后被"锁住"（分数陷阱），在 Refinement 阶段无法被正常修复。仅分析最终输出无法捕获这一时序过程

**2. 异常分数动力学检测**
- **功能**: 通过监控分数的时序演化实时定位潜在伪影区域
- **核心思路**: 定义分数动力学为相邻步分数差 $\Delta s_\theta(x_t^{i,j}, t)$。维护分数银行 $\mathcal{S}$ 记录历史分数值。使用时序权重函数 $w(t) = \frac{1-\bar{\alpha_t}}{\sqrt{\bar{\alpha_t}}}$ 补偿分数幅度的自然衰减。当加权分数变化超过自适应阈值 $\tau = \max\{\text{MAD}(\Delta(\cdot)), \text{mean}(\mathcal{S})\}$ 时标记为伪影区域
- **设计动机**: 伪影区域在分数加速度曲线上表现为特征性的"快速加速后突然减速"模式，而正常区域保持稳定演化。时序权重 $w(t)$ 有理论分析支撑

**3. 轨迹感知定向校正 (Trajectory-aware Targeted Correction, TTC)**
- **功能**: 在不中断推理过程的情况下校正伪影区域
- **核心思路**: 仅对检测到的伪影区域 $\Omega^a$ 施加受控扰动：$\hat{x}_{T_c} = x_{T_c} \cdot \mathbb{1}_{\bar{\Omega}^a} + (\sqrt{\bar{\alpha}_{T_c}} x_0' + \sqrt{1-\bar{\alpha}_{T_c}} \epsilon) \cdot \gamma \xi \cdot \mathbb{1}_{\Omega^a}$，其中 $x_0'$ 为从当前步预测的清洁图像。非伪影区域完全保持不变
- **设计动机**: 状态替换和分数裁剪虽能修复伪影但会破坏生成多样性。TTC 通过注入随机扰动打破分数陷阱的固定模式，让这些区域重新与周围区域耦合演化，同时保持非伪影区域的原始轨迹

### 损失函数

ASCED 完全无监督，无需训练。检测基于分数动力学的统计分析（MAD 阈值），校正基于控制论的扰动注入。

## 实验关键数据

### 主实验：五数据集定量对比

| 方法 | 类型 | FID↓ | Artifact Rate↓ |
|------|------|------|----------------|
| BayesDiff | UnS | — | 较高 |
| SARGD | Sup | — | 中等 |
| State Replacement | UnS | — | 低但多样性差 |
| Score Clipping | UnS | — | 低但多样性差 |
| **ASCED (TTC)** | **UnS** | **最低** | **最低** |

*ASCED 在五个数据集上均匹配或超越有监督方法（SARGD），同时保持生成多样性*

### 消融实验

| 校正策略 | FID↓ | 多样性 | 伪影率↓ |
|----------|------|--------|---------|
| 无校正 | baseline | 高 | 高 |
| 状态替换 | 改善 | 显著降低 | 低 |
| 分数裁剪 | 略改善 | 降低 | 中等 |
| **TTC** | **最优** | **保持** | **最低** |

### 关键发现

- 扩散模型在生成过程中无法自主识别伪影——将伪影与正常 Mutation 混淆
- SDEdit 的加噪-去噪可以修复伪影区域，间接证明伪影在加噪后的特征空间中与正常状态无法区分
- 时序权重 $w(t)$ 的理论推导与实验观察吻合
- TTC 在保持多样性的同时有效删除伪影，优于简单的状态替换和分数裁剪

## 亮点与洞察

1. **机理性分析**: 首次从分数动力学角度揭示了扩散伪影的形成机制（分数陷阱），超越了现象级的检测
2. **"在线"干预**: 将伪影校正嵌入生成过程本身而非后处理，既高效又保持多样性
3. **无监督超越有监督**: 完全无需标注数据即可达到或超过有监督方法的伪影消除效果

## 局限与展望

- 检测需要存储分数银行 $\mathcal{S}$，增加内存开销
- $T_d$ 和 $T_c$ 的选择需要经验设定
- 对语义幻觉（如多余肢体）的处理不在范围内
- 未来可探索将三阶段分析应用于可控生成

## 相关工作与启发

- 与 BayesDiff 的空间不确定性分析相比，ASCED 的时序分析更能精确定位伪影
- 分数陷阱的概念可推广到理解其他生成模型的失败模式
- TTC 的扰动注入策略对其他需要局部修复的生成任务有参考价值

## 评分

⭐⭐⭐⭐⭐ — 既有深刻的理论洞察（三阶段+分数陷阱），又有实用的无监督解决方案，论文质量极高。将伪影的检测和校正优雅地嵌入推理过程，无需额外训练即超越有监督方法。对扩散模型内部机制的理解具有开创性意义。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_unified_generation_understanding.md)
- [\[CVPR 2025\] CLIP Under the Microscope: A Fine-Grained Analysis of Multi-Object Representation](clip_under_the_microscope_a_fine-grained_analysis_of_multi-object_representation.md)
- [\[CVPR 2025\] Hiding Images in Diffusion Models by Editing Learned Score Functions](hiding_images_in_diffusion_models_by_editing_learned_score_functions.md)
- [\[CVPR 2025\] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction](pursuing_temporal-consistent_video_virtual_try-on_via_dynamic_pose_interaction.md)
- [\[NeurIPS 2025\] GeoRemover: Removing Objects and Their Causal Visual Artifacts](../../NeurIPS2025/image_generation/georemover_removing_objects_and_their_causal_visual_artifacts.md)

<!-- RELATED:END -->
