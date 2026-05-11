---
title: >-
  [论文解读] Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning
description: >-
  [CVPR 2026][图像生成][空间理解] 本文提出Spatial-SSRL，一种自监督强化学习范式，通过从普通RGB/RGB-D图像自动构造五种pretext任务（patch重排、翻转识别、裁剪修补、深度排序、相对3D位置预测），利用GRPO优化LVLM的空间理解能力，在七个空间benchmark上平均提升3.89%-4.63%，且无需人工标注或外部工具。
tags:
  - CVPR 2026
  - 图像生成
  - 空间理解
  - 自监督学习
  - 强化学习RLVR
  - 大视觉语言模型
  - 深度感知
---

# Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning

**会议**: CVPR 2026  
**arXiv**: [2510.27606](https://arxiv.org/abs/2510.27606)  
**代码**: [GitHub](https://github.com/InternLM/Spatial-SSRL)  
**领域**: 图像生成  
**关键词**: 空间理解, 自监督学习, 强化学习RLVR, 大视觉语言模型, 深度感知

## 一句话总结
本文提出Spatial-SSRL，一种自监督强化学习范式，通过从普通RGB/RGB-D图像自动构造五种pretext任务（patch重排、翻转识别、裁剪修补、深度排序、相对3D位置预测），利用GRPO优化LVLM的空间理解能力，在七个空间benchmark上平均提升3.89%-4.63%，且无需人工标注或外部工具。

## 研究背景与动机

**领域现状**：大视觉语言模型（LVLM）在VQA、图像描述等任务上接近饱和，但空间理解能力远低于人类水平。现有提升方法分为两类：数据驱动的SFT（构造空间QA对微调）和RLVR（用可验证奖励做强化学习）。

**现有痛点**：SFT方法依赖昂贵的人工标注或GPT-4生成QA对，且容易过拟合到数据集特定模式。工具型方法（如使用深度估计器、目标检测器等）pipeline复杂、计算成本高。模拟环境方法（rendering 3D场景）与真实世界存在domain gap。RLVR方法受限于特定环境（如3D扫描），数据规模和覆盖范围有限。

**核心矛盾**：空间理解需要大规模可验证的监督信号，但现有方法获取这类信号的代价太高——要么需要昂贵的人工标注，要么需要复杂的工具链，要么受限于特定3D数据集。

**本文目标** 设计一种零标注、无工具、可扩展的自监督方案来生成可验证的空间理解训练信号，并与RLVR训练范式自然结合。

**切入角度**：图像内部固有的结构一致性（相对深度、几何一致性、视角不变性）本身就提供了确定性可验证的监督信号。将视觉自监督学习（SSL）的pretext任务重新定位为RLVR的奖励函数，而非传统的特征预训练目标。

**核心 idea**：把经典的SSL任务（jigsaw、翻转检测等）改造成LVLM的QA prompt + 确定性验证函数，直接用GRPO做后训练。

## 方法详解

### 整体框架
Spatial-SSRL分为两个阶段：（1）自监督任务构造——从RGB/RGB-D图像自动生成五种pretext任务的QA对（Spatial-SSRL-81k数据集，100%标注准确率）；（2）RL训练——先SFT冷启动让模型熟悉任务格式，再用GRPO优化。五种任务覆盖2D布局理解（depth-free）和3D空间推理（depth-based）。

### 关键设计

1. **Shuffled Patch Reordering（图像块重排）**:

    - 功能：将图像分成 $M \times N$ 的grid，随机打乱，要求模型预测恢复原图的排列 $\pi^{-1}$
    - 核心思路：对图像 $I$ 划分patch网格 $\mathcal{X} = \{x_{i,j}\}$，应用随机排列 $\pi$ 得到打乱图像，ground-truth答案为逆排列 $\pi^{-1} = [\pi^{-1}(0), \pi^{-1}(1), \ldots, \pi^{-1}(M \times N - 1)]$。可选地mask一个随机patch为白色以增加难度，防止模型靠边缘匹配取巧
    - 设计动机：恢复打乱的patch排列本质上需要理解全局2D布局一致性和相对位置关系，这些能力直接迁移到理解真实场景中的物体排列

2. **Flipped Patch Recognition（翻转识别）**:

    - 功能：随机选一个patch做水平或垂直翻转，要求模型识别翻转的patch索引和翻转方向
    - 核心思路：对选中patch $\hat{x}_t$ 以等概率做 $x_{\text{vert}}(r,c) = x(P_H - 1 - r, c)$ 或 $x_{\text{horz}}(r,c) = x(r, P_W - 1 - c)$，答案为 $[t, d]$
    - 设计动机：检测细微的方向违反需要对局部几何、镜像对称性和方向线索（文字/人脸/阴影）的敏感度

3. **Cropped Patch Inpainting（裁剪修补）**:

    - 功能：随机裁剪一个区域mask为黑色，给出4个候选patch（含3个干扰项），要求模型选出正确的填充patch
    - 核心思路：干扰项设计巧妙——90°旋转版、内部子区域、外部扩展区域，都与正确答案视觉上相似，迫使模型关注精细的纹理连续性和语义一致性
    - 设计动机：测试纹理-上下文匹配和细粒度结构推理能力

4. **Regional Depth Ordering（区域深度排序）**:

    - 功能：选3个深度明确分离的区域，标记数字标签后打乱展示，要求模型按从近到远排序
    - 核心思路：从depth map $D$ 中选择满足约束的3个区域：区域内深度范围 $r(R_i) < r_{\max} = 0.15$（区域内一致），区域间间隔 $d(R_i, R_{i+1}) > d_{\min} = 0.05$（区域间可分）
    - 设计动机：排序任务需要整合深度线索、透视理解和序数推理，是3D场景理解的基础能力

5. **Relative 3D Position Prediction（相对3D位置预测）**:

    - 功能：给定物体朝向，预测另一个点在该物体坐标系中的相对位置（左/右/前/后的组合）
    - 核心思路：通过2D刚体变换将相机坐标系的 $(x_2, z_2)$ 转换到物体坐标系 $(\tilde{x}_2, \tilde{z}_2)$，根据阈值判断方向标签 $(\tilde{p}_x, \tilde{p}_z)$。物体朝向 $\theta$ 从四个基本方向均匀采样
    - 设计动机：需要心象旋转、自我中心坐标变换和深度集成能力，是空间理解中最高阶的任务

### 损失函数 / 训练策略
- **冷启动SFT**：先在~3600样本上做5 epoch的SFT（lr=$1 \times 10^{-5}$），让模型熟悉任务格式
- **GRPO优化**：KL正则权重0.01，每样本rollout 5次，temperature 1.0，batch size 128，lr=$1 \times 10^{-6}$，360步
- **奖励设计**：$r = 0.9 \cdot r_{\text{acc}} + 0.1 \cdot r_{\text{fmt}}$，准确率权重远高于格式权重
- 使用think标签引导推理链输出

## 实验关键数据

### 主实验（7个空间理解benchmark）

| 模型 | Spatial457 | 3DSRBench | SpatialEval | QSpatial+ | What'sUp | ViewSpatial | VSI-Bench | Avg |
|------|-----------|-----------|-------------|-----------|----------|-------------|-----------|-----|
| Qwen2.5-VL-3B | 33.70 | 50.30 | 54.65 | 33.66 | 85.85 | 35.38 | 27.84 | 45.91 |
| Spatial-SSRL-3B | 46.07 | 51.72 | 59.59 | 39.60 | 86.71 | 36.62 | 33.49 | 50.54 |
| Δ | **+12.37** | +1.42 | +4.94 | +5.95 | +0.86 | +1.24 | +5.65 | **+4.63** |
| Qwen2.5-VL-7B | 44.67 | 53.39 | 62.37 | 46.53 | 86.95 | 36.83 | 38.08 | 52.69 |
| Spatial-SSRL-7B | 53.34 | 56.53 | 64.03 | 54.46 | 90.61 | 37.81 | 39.29 | 56.58 |
| Δ | **+8.67** | +3.14 | +1.66 | +7.93 | +3.66 | +0.98 | +1.21 | **+3.89** |

所有7个benchmark上均有提升，Spatial457上最大提升+12.37%。

### 推理能力验证

| 配置 | Avg准确率 | 说明 |
|------|----------|------|
| Qwen2.5-VL-7B (无推理) | 52.69 | baseline |
| Qwen2.5-VL-7B (有推理) | 49.58 | 推理反而降低！(-3.11) |
| Spatial-SSRL-7B (有推理) | **56.58** | 推理链真正有效(+3.89) |

baseline开启推理反而掉点（What'sUp: 86.95→70.61），说明基础模型缺乏有效的空间推理能力。Spatial-SSRL通过RL训练成功教会了模型生成有效推理链。

### 关键发现
- 3D推理类benchmark获益最多（Spatial457 +12.37%, QSpatial+ +7.93%），验证depth-based任务的贡献
- 基础模型开启CoT推理反而掉点是重要发现——说明空间推理能力需要专门训练而非简单的prompt engineering
- Qwen3-VL-4B上也有+1.29%空间提升且通用VQA也涨+1.18%，说明方法不损害通用能力
- Spatial-SSRL-81k数据集实现100%标注准确率（因为所有答案来自确定性变换），这是依赖noisy检测器的方法无法达到的
- 冷启动SFT很必要——直接RL训练导致生成正确格式的成功率<5%

## 亮点与洞察
- **SSL + RLVR的结合范式**是最大创新点。SSL pretext任务天然提供确定性可验证答案，与RLVR要求的verifiable reward完美契合——这个insight可能催生大量follow-up工作将其他SSL任务引入LVLM后训练
- **五种任务的互补设计**覆盖了从2D布局到3D空间关系的完整层次：patch reordering（全局布局）→ flip recognition（局部方向）→ inpainting（纹理一致性）→ depth ordering（3D深度）→ 3D position（自我中心坐标变换）
- **干扰项设计**的巧妙之处：inpainting任务中用旋转版、内部子区域、外部扩展区域作为干扰项，防止模型靠低级特征取巧

## 局限与展望
- 依赖depth map的两个任务需要RGB-D数据，限制了数据源（虽然depth可以用单目估计但会引入噪声）
- 五种任务的相对权重未做细致调优，当前是等比例混合
- 仅在Qwen2.5-VL和Qwen3-VL上测试，对其他LVLM架构（如LLaVA、InternVL）的泛化性未知
- 81k的数据规模相比SFT方法已经较小但RL训练效率还有提升空间
- 可以探索更多SSL任务——如颜色通道重排、频域变换预测、多视角一致性验证

## 相关工作与启发
- **vs SpatialLadder/SpaceR**: 这些方法依赖3D扫描数据集+复杂pipeline，Spatial-SSRL仅需普通RGB/RGB-D图像，pipeline极简。性能上Spatial-SSRL-7B（56.58）超越SpaceR-7B（54.54）
- **vs Jigsaw-R1/Visual Jigsaw**: 类似的SSL+RL思路但只覆盖jigsaw一种任务，Spatial-SSRL设计了五种互补任务，更全面地提升空间理解
- **vs SSL4RL**: 仅关注2D任务，Spatial-SSRL同时覆盖2D和3D，depth-based任务贡献了显著的性能提升

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ SSL pretext任务作为RLVR reward的范式具有开创性，五种任务设计全面
- 实验充分度: ⭐⭐⭐⭐ 7个benchmark + 3个base model + 通用能力验证，但消融可以更详细
- 写作质量: ⭐⭐⭐⭐⭐ 方法动机清晰，任务设计的数学形式化严谨，图示优秀
- 价值: ⭐⭐⭐⭐⭐ 零标注、可扩展、与RLVR天然兼容，为LVLM空间理解提升开辟了新路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Enhancing Spatial Understanding in Image Generation via Reward Modeling](enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)
- [\[ICCV 2025\] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](../../ICCV2025/image_generation/compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers](circuit_mechanisms_for_spatial_relation_generation_in_diffusion_models.md)
- [\[CVPR 2026\] Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods](training-free_detection_of_generated_videos_via_spatial-temporal_likelihoods.md)
- [\[CVPR 2026\] Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models](learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)

</div>

<!-- RELATED:END -->
