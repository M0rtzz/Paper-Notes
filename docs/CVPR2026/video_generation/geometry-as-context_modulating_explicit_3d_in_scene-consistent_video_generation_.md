---
title: >-
  [论文解读] Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context
description: >-
  [CVPR 2026][场景一致性视频生成] 提出 Geometry-as-Context (GaC) 框架，将基于重建的场景视频生成中的不可微算子（3D重建+渲染）替换为统一的自回归视频生成模型，通过将几何信息（深度图）作为交错上下文嵌入生成序列，实现端到端训练并缓解累积误差。
tags:
  - CVPR 2026
  - 场景一致性视频生成
  - 几何上下文
  - 自回归生成
  - 相机控制
  - 3D重建
---

# Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context

**会议**: CVPR 2026  
**arXiv**: [2602.21929](https://arxiv.org/abs/2602.21929)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 场景一致性视频生成, 几何上下文, 自回归生成, 相机控制, 3D重建

## 一句话总结

提出 Geometry-as-Context (GaC) 框架，将基于重建的场景视频生成中的不可微算子（3D重建+渲染）替换为统一的自回归视频生成模型，通过将几何信息（深度图）作为交错上下文嵌入生成序列，实现端到端训练并缓解累积误差。

## 研究背景与动机

场景一致性视频生成旨在沿相机轨迹探索 3D 场景，需保持高度的 3D 一致性。现有方法分为两类：
- **视频方法**（CameraCtrl、VMem 等）：仅靠视频模型维持一致性，记忆检索难以应对复杂场景和大相机运动
- **重建方法**（SceneScape、ViewCrafter、GEN3C 等）：迭代执行「几何估计→3D重建→渲染→补全」，但存在两个根本问题：
  1. **不可微算子**：逆渲染中的反投影和渲染操作不可微，梯度无法传播
  2. **非端到端训练**：几何预测和图像补全使用独立模型，累积误差无法通过学习缓解

与长程视频生成中可通过自回归训练缓解的累积误差不同，重建方法中的累积误差因不可微操作和模型分离而难以消除，这是本文要解决的核心问题。

## 方法详解

### 整体框架

GaC 将重建方法的迭代过程「展平」为一个自回归视频生成框架：单一 DiT 模型同时处理几何估计、视角变换模拟和图像补全。输入序列交错排列 RGB 帧和几何帧：$\{I_i, \text{<Geometry>}, G_i, \text{<Image>}, I_{i+1}, \cdots\}$，其中文本标记指示模型下一步生成几何还是 RGB。

### 关键设计

1. **几何作为上下文 (Variant #1)**：将原始四步迭代（几何估计→反投影→渲染→补全）简化为：$\{G_i, I_{i+1}\} = \varrho(\{I_i, G_i\}, P_{i+1})$。模型先估计当前帧的几何 $G_i$，再基于 $G_i$ 和目标位姿 $P_{i+1}$ 生成下一帧 RGB 图像。几何上下文的加入：(a) 缩短序列长度提升效率；(b) 让模型获得 3D 感知能力增强场景一致性；(c) RGB 和几何模态差异大，模型易区分任务。

2. **Camera Gated Attention (CGA)**：增强模型对相机位姿的利用能力。将 Plücker 射线编码的相机位姿 $r_i$ 加到 self-attention 的 query 上，并生成门控矩阵调制注意力输出：

    - $\{Q_{res}, Gate\} = \text{Linear}_2(Q + r_i)$
    - $O = \text{SDPA}(Q + Q_{res}, K, V)$
    - $O = \text{Linear}_3(O * \sigma(Gate))$
   
   这种设计帮助模型区分相机位姿在几何预测 vs. 视角合成中的不同作用。

3. **Geometry Dropout**：训练时以比率 $r$ 随机丢弃交错序列中的几何上下文，被丢弃的帧退化为纯图像到图像生成 (Variant #3)。好处：(a) 减少序列长度提升训练效率；(b) 推理时可仅生成 RGB 输出而不需要几何预测；(c) 模型在有/无几何上下文的情况下都保持场景一致性。训练时间从 24 s/step 减半到 11 s/step，推理从 4.6 s/img 减半到 2.2 s/img，性能几乎不降。

### 损失函数 / 训练策略

- 基座模型：Bagel-7B（支持文本-图像交错建模）
- 训练数据：RealEstate10K（66033 视频片段）
- 8 帧序列训练，前 1-4 帧为上下文视图，后续为目标视图
- 每 4 个连续视图拼成一个 grid 帧增强一致性（$640 \times 352$ 分辨率）
- 图像用 FLUX-VAE 编码
- 8 张 H100 训练 40000 步，约 2 天
- 推理时用 context-as-memory 选择上下文视图，不使用 classifier-free guidance

## 实验关键数据

### 主实验

| 数据集 | 指标 | GaC(本文) | Voyager | GEN3C | ViewCrafter |
|--------|------|-----------|---------|-------|-------------|
| RE10K | PSNR↑ | **19.01** | 18.70 | 18.12 | 16.72 |
| RE10K | SSIM↑ | **0.656** | 0.616 | 0.624 | 0.585 |
| RE10K | LPIPS↓ | **0.354** | 0.395 | 0.402 | 0.417 |
| RE10K | FID↓ | **55.76** | 65.12 | 66.20 | 80.47 |
| RE10K | $R_{err}$↓ | **0.024** | 0.035 | 0.027 | 0.022 |
| RE10K | $T_{err}$↓ | **0.270** | 0.596 | 0.344 | 0.327 |
| T&T | PSNR↑ | **15.77** | 15.24 | 15.32 | 12.59 |
| RE10K(来回) | PSNR↑ | **16.34** | 15.80 | 15.28 | 15.77 |
| RE10K(来回) | FID↓ | **64.31** | 79.81 | 80.03 | 72.14 |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | $T_{err}$↓ | 说明 |
|------|-------|-------|--------|------|-----------|------|
| None (Variant #3) | 16.34 | 0.551 | 0.412 | 89.03 | 0.351 | 无几何上下文 |
| Warped img (V#2) | 18.33 | 0.671 | 0.383 | 59.12 | 0.299 | 渲染图作上下文 |
| Geometry (V#1) | **19.01** | 0.656 | **0.354** | **55.76** | **0.270** | 几何作上下文 |
| w/o CGA | 18.57 | 0.581 | 0.461 | 68.42 | 0.469 | 去掉 CGA |
| w/ CGA | **19.01** | **0.656** | **0.354** | **55.76** | **0.270** | 完整方法 |
| w/o Geo Dropout | 19.23 | 0.660 | 0.342 | 57.18 | 0.248 | 不丢弃(略好但2x慢) |
| w/ Geo Dropout | 19.01 | 0.656 | 0.354 | 55.76 | 0.270 | ~2x 加速 |

### 关键发现

- 几何作为上下文 vs. 无上下文：PSNR 提升 2.67，FID 降低 33.27，证明显式 3D 信息的关键作用
- CGA 使平移误差 $T_{err}$ 从 0.469 降至 0.270（42% 降幅），大幅提升相机控制精度
- Geometry Dropout 将训练速度提升 ~2 倍，推理速度提升 ~2 倍，性能损失可忽略
- 深度图 vs. 点图作为几何：性能相近，但深度图略优（与自然图像模态差距更小，VAE 更容易编码）
- 来回轨迹测试中，GaC 能忠实恢复返回时的物体（如消失的电脑），体现长程 3D 记忆能力

## 亮点与洞察

- **统一框架优雅**：将四步迭代的重建方法展平为一个自回归 DiT 模型，从根本上解决不可微操作和非端到端训练的问题
- **Geometry Dropout 一举两得**：不仅减少计算成本，还让模型在推理时可灵活选择是否输出几何信息
- **CGA 设计精巧**：通过 query 调制 + 门控输出，让同一个模型能区分相机位姿在不同子任务中的角色
- **来回轨迹鲁棒性**：GaC 在前往-返回轨迹上显示出良好的场景记忆和一致性

## 局限与展望

- 所有方法在来回轨迹上性能显著下降，长程上下文记忆策略仍需改进
- 仅在 RealEstate10K 上训练，泛化到更多样化的场景（室外、野外）需更多数据
- 分辨率 $640 \times 352$ 较低，高分辨率场景生成待探索
- 来回轨迹中 Tanks-and-Temples 的 FID 指标不如 Voyager，说明在大运动场景中仍有改进空间
- 基座模型 Bagel-7B 较大，推理成本仍然不低（2.2 s/img）

## 相关工作与启发

- **ViewCrafter**：点云 + 视频扩散的迭代方法，本文的统一框架更优雅且误差更小
- **GEN3C/Voyager**：引入点云/3DGS 作为 3D 表示，但仍受不可微渲染限制
- **ReCamMaster**：帧维度拼接的相机控制方法，GaC 继承其思路但加入几何上下文
- **启发**：「将不可微操作内化为生成模型能力」的思路可推广到更多 3D 视觉任务；文本引导的多任务调度（几何 vs. RGB 生成）是交错多模态模型的有效设计范式

## 评分

- 新颖性: ⭐⭐⭐⭐ 将重建方法的迭代过程展平为自回归生成是一个优雅的创新
- 实验充分度: ⭐⭐⭐⭐ 多个基准、来回轨迹、充分消融，但训练数据较单一
- 写作质量: ⭐⭐⭐⭐ 动机分析透彻，Variant 分析清晰，算法描述完整
- 价值: ⭐⭐⭐⭐ 对场景视频生成领域提供了新范式，端到端思想有广泛价值
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)
- [\[CVPR 2026\] Rethinking Position Embedding as a Context Controller for Multi-Reference and Multi-Shot Video Generation](rethinking_position_embedding_as_a_context_controller_for_multi-reference_and_mu.md)
- [\[CVPR 2026\] Towards Realistic and Consistent Orbital Video Generation via 3D Foundation Priors](orbital_video_3d_foundation_priors.md)
- [\[ICLR 2026\] Geometry-aware 4D Video Generation for Robot Manipulation](../../ICLR2026/video_generation/geometry-aware_4d_video_generation_for_robot_manipulation.md)
- [\[CVPR 2025\] World-Consistent Video Diffusion with Explicit 3D Modeling](../../CVPR2025/video_generation/world-consistent_video_diffusion_with_explicit_3d_modeling.md)

<!-- RELATED:END -->
