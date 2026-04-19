---
title: >-
  [论文解读] Aether: Geometric-Aware Unified World Modeling
description: >-
  [ICCV 2025][图像生成][世界模型] Aether 提出一个几何感知的统一世界模型框架，通过在合成 4D 数据上联合训练重建、预测和规划三大能力，基于 CogVideoX 后训练实现零样本泛化到真实场景。
tags:
  - ICCV 2025
  - 图像生成
  - 世界模型
  - 4D重建
  - 动作条件视频预测
  - 视觉规划
  - 扩散模型
---

# Aether: Geometric-Aware Unified World Modeling

**会议**: ICCV 2025  
**arXiv**: [2503.18945](https://arxiv.org/abs/2503.18945)  
**代码**: https://github.com/AetherWorld (有)  
**领域**: 世界模型/视频生成  
**关键词**: 世界模型, 4D重建, 动作条件视频预测, 视觉规划, 扩散模型

## 一句话总结

Aether 提出一个几何感知的统一世界模型框架，通过在合成 4D 数据上联合训练重建、预测和规划三大能力，基于 CogVideoX 后训练实现零样本泛化到真实场景。

## 研究背景与动机

世界模型是构建自主系统的基础范式，核心需要三大能力：**感知**（4D 动态重建）、**预测**（动作条件视频生成）和**规划**（目标导向路径规划）。然而现有方法通常只关注其中一个方面：

**重建方法**（DUSt3R、MonST3R 等）只恢复几何结构，不具备预测和规划能力

**视频生成模型**（CogVideoX、SVD）能生成视觉真实的视频，但缺乏几何一致性

**规划系统**通常依赖显式状态表示，难以直接从视觉输入进行规划

作者的核心洞察：这三种能力可以通过共享的几何先验实现协同学习——重建教会模型理解几何，预测让模型学会动态演化，规划将二者结合用于决策。另一关键动机是 **4D 数据稀缺**，作者选择合成数据训练并利用视频生成模型的预训练先验实现零样本真实世界迁移。

## 方法详解

### 整体框架

Aether 基于 CogVideoX-5b-I2V 进行后训练（post-training）。模型的目标潜变量 z₀ 包含三种模态：

- **彩色视频潜变量 z_c**：RGB 视频编码
- **深度视频潜变量 z_d**：深度视频的归一化视差编码
- **动作潜变量 z_a**：相机轨迹的 raymap 编码

通过随机组合不同的输入条件 c = c_c ⊗ c_a，同一模型可完成不同任务：

- **4D 重建**：输入完整视频 → 输出深度 + 相机参数
- **视频预测**：输入观测图像（+ 可选动作条件）→ 生成未来帧
- **视觉规划**：输入起始 + 目标图像 → 生成中间路径视频

### 关键设计

**1. 自动化 4D 合成数据标注流水线**

包含四个阶段：

- **动态掩码**：用 Grounded SAM 2 分割动态物体（行人/车辆），保证相机估计准确性
- **视频切片**：通过 SIFT 关键点检测和 RAFT 光流估计过滤不适合重建的帧
- **粗略相机估计**：用 DroidCalib 初步估计相机内外参
- **相机精化**：用 CoTracker3 获取长期对应点，通过 Bundle Adjustment + Ceres Solver 精化

**2. 深度视频处理 — 尺度不变归一化视差**

将深度转换为视差表示以适配 VAE：depth → clip → sqrt → reciprocal → normalize to [-1,1] → 复制为 3 通道 → VAE 编码。

**3. 相机轨迹 Raymap 表示**

将相机参数转为 6 通道 raymap 视频（3 射线方向 + 3 射线原点），平移分量经尺度归一化和 signed-log 变换，空间下采样 8×、时间每 4 帧拼接，对齐 DiT 的时空维度。

**4. 多任务条件随机 Masking**

训练时随机 mask 条件输入以适配不同任务：30% 视觉规划（首尾帧）、40% 视频预测（仅首帧）、28% 重建（完整视频）、2% 无条件；动作条件 50% 概率保留/mask。

### 损失函数 / 训练策略

**两阶段训练**：

- **阶段一**：标准潜空间 MSE 去噪损失
- **阶段二**（1/4 步数）：解码到图像空间，增加 MS-SSIM loss（彩色视频）、scale-shift invariant loss（深度）、pointmap loss（深度 + raymap 投影一致性）

使用 FSDP + DDP 混合并行策略，80 张 A100-80GB GPU，有效 batch size 320，训练约两周。AdamW + OneCycle 学习率调度。

## 实验关键数据

### 主实验

**视频深度估计**（零样本，per-sequence scale 对齐）：

| 方法 | Sintel AbsRel↓ | Sintel δ<1.25↑ | KITTI AbsRel↓ | KITTI δ<1.25↑ |
|------|:-:|:-:|:-:|:-:|
| MonST3R-GA | 0.378 | 55.8 | 0.168 | 74.4 |
| CUT3R | 0.421 | 47.9 | 0.118 | 88.1 |
| DepthCrafter | 0.590 | 55.5 | 0.124 | 86.5 |
| **Aether** | **0.324** | 50.2 | **0.056** | **97.8** |

**动作条件视频预测 VBench 指标（overall）**：

| 方法 | 主体一致性 | 背景一致性 | 运动平滑度 | 动态度 | 加权均值 |
|------|:-:|:-:|:-:|:-:|:-:|
| CogVideoX | 90.51 | 92.77 | 98.24 | 86.76 | 79.92 |
| **Aether** | **91.54** | **94.06** | **98.56** | **94.85** | **80.71** |

### 消融实验

**重建目标对导航能力的影响**：

| 方法 | PSNR↑ | SSIM↑ | MS-SSIM↑ | LPIPS↓ |
|------|:-:|:-:|:-:|:-:|
| Aether-no-depth | 18.97 | 0.5353 | 0.5376 | 0.3074 |
| **Aether (full)** | **19.70** | **0.5545** | **0.5760** | **0.2659** |

加入深度重建目标后所有导航指标均有显著提升，验证多任务协同学习的有效性。

### 关键发现

1. **零样本泛化**：完全在合成数据训练，KITTI 真实数据深度估计 AbsRel 仅 0.056，超过所有专用模型
2. **重建增强生成**：深度重建目标同时提升视频预测和规划质量
3. **Raymap 动作表示**：相比文本描述，相机轨迹 raymap 提供更精确的动作控制
4. **高效推理**：重建任务仅需 4 步去噪

## 亮点与洞察

1. **统一框架极其精巧**：通过条件 masking 策略，一个模型同时支持重建/预测/规划，无需任何任务特定模块
2. **合成到真实零样本迁移成功**：利用视频生成模型预训练先验跨越 domain gap
3. **Raymap 表示**：将相机轨迹转为与视频帧对齐的 6 通道表示，优雅解决异构模态融合问题
4. **4D 标注流水线**具有独立的工程应用价值

## 局限与展望

1. 仅支持相机轨迹作为动作空间，不直接支持机器人关节动作等模态
2. 生成的相机轨迹有噪声，需 Kalman 滤波后处理
3. 训练成本高（80 × A100 两周）
4. BONN 数据集深度估计表现一般（AbsRel 0.273），室内小范围场景有提升空间
5. 规划限于视觉路径规划，缺乏物理引擎交互

## 相关工作与启发

- **DUSt3R / MASt3R / MonST3R / CUT3R**：端到端 3D/4D 重建方法，重建部分的主要对比基线
- **CogVideoX**：基础视频生成模型，通过后训练注入几何感知
- **Genie 2 / Cat3D / Cat4D**：其他世界模型方法，不具备统一重建能力
- **DepthCrafter / DA-V**：基于扩散的深度估计方法
- **启发**：统一范式（重建 + 生成 + 规划）是通向 embodied AI 的重要路径；合成数据 + 预训练先验可有效缓解数据稀缺

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [SummDiff: Generative Modeling of Video Summarization with Diffusion](summdiff_generative_modeling_of_video_summarization_with_diffusion.md)
- [Long-Context State-Space Video World Models](long-context_state-space_video_world_models.md)
- [A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation](a0_an_affordance-aware_hierarchical_model_for_general_robotic_manipulation.md)
- [Generative Modeling of Shape-Dependent Self-Contact Human Poses](generative_modeling_of_shape-dependent_self-contact_human_poses.md)
- [Domain Generalizable Portrait Style Transfer](domain_generalizable_portrait_style_transfer.md)

<!-- RELATED:END -->
