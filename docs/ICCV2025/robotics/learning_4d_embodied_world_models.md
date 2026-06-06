---
title: >-
  [论文解读] TesserAct: Learning 4D Embodied World Models
description: >-
  [ICCV 2025][机器人][4D世界模型] 提出 TesserAct——一种 4D 具身世界模型，通过训练视频生成模型联合预测 RGB、深度和法线视频，再转换为高质量 4D 场景，实现空间-时间一致的 3D 世界动态模拟和机器人动作规划。
tags:
  - "ICCV 2025"
  - "机器人"
  - "4D世界模型"
  - "具身智能"
  - "视频扩散模型"
  - "RGB-DN"
  - "深度法线联合预测"
  - "机器人规划"
---

# TesserAct: Learning 4D Embodied World Models

**会议**: ICCV 2025  
**arXiv**: [2504.20995](https://arxiv.org/abs/2504.20995)  
**领域**: 机器人  
**关键词**: 4D世界模型, 具身智能, 视频扩散模型, RGB-DN, 深度法线联合预测, 机器人规划

## 一句话总结

提出 TesserAct——一种 4D 具身世界模型，通过训练视频生成模型联合预测 RGB、深度和法线视频，再转换为高质量 4D 场景，实现空间-时间一致的 3D 世界动态模拟和机器人动作规划。

## 研究背景与动机

世界模型是具身智能的核心组件。然而现有世界模型存在根本性局限：

- **2D 像素空间操作**：UniPi、SuSIE 等在 2D 建模，无法提供精确深度和位姿信息
- **不真实的预测**：2D 模型可能产生跨时间步的物体形状不一致
- **4D 建模高成本**：直接在 3D+时间维度生成输出计算开销极大
- **缺乏 4D 标注数据**：大规模机器人数据集普遍缺少深度和法线标注

核心思路：用 RGB-DN（RGB+深度+法线）视频作为 4D 世界的轻量中间表示，利用预训练视频模型高效构建 4D 世界模型。

## 方法详解

### 整体框架

四个核心组件：4D 数据集构建 → RGB-DN 生成模型 → 4D 场景重建 → 动作规划

### 关键设计

**1. 4D 具身视频数据集（~285k 视频）**

- RLBench 合成（80k）：20个任务×1000实例×4视角，精确深度+DSINE法线，Colosseum随机化
- RT1 Fractal 真实（80k）：RollingDepth 深度+Marigold 法线
- Bridge（25k）：同上标注
- SomethingSomethingV2（100k）：人手交互，丰富指令多样性

**2. 模型架构（基于 CogVideoX 微调）**

- 3D VAE 分别编码 RGB/深度/法线（不微调 VAE）
- 三个独立 InputProj 提取模态嵌入，求和输入 DiT 骨干
- RGB 保留原始路径；深度法线通过 Conv3D+DNProj 额外解码
- **零初始化关键设计**：新模块全部零初始化，训练起点=CogVideoX
- 文本条件："[动作指令] + [机器人臂名称]"
- 多分辨率训练，49帧预测

**3. 4D 场景重建**

- 法线积分优化深度：透视相机模型约束对数深度，迭代求解二次损失
- RAFT 光流分割动态/静态/背景区域
- 时间一致性损失：光流引导帧间深度一致（动态/背景分别加权）
- 正则化损失：约束优化深度接近生成深度
- 总损失 = 空间一致性 + 时间一致性 + 正则化

**4. 动作规划**

PointNet 编码 4D 点云 → 结合文本嵌入 → MLP 输出 7-DoF 动作

### 损失函数

- 视频生成：标准去噪损失，联合 RGB+深度+法线
- 4D 重建：L_s(法线积分) + L_c(光流时间一致) + L_r(深度正则)

## 实验关键数据

### 4D 场景生成

**真实域（RT1+Bridge）：**

| 方法 | AbsRel↓ | 法线Mean↓ | Chamfer↓ |
|------|---------|---------|---------|
| OpenSora | 31.41 | 41.82 | 0.3013 |
| CogVideoX | 26.17 | 19.53 | 0.2191 |
| **TesserAct** | **22.07** | **15.74** | **0.2030** |

**合成域（RLBench）：**

| 方法 | AbsRel↓ | 法线Mean↓ | Chamfer↓ |
|------|---------|---------|---------|
| CogVideoX | 19.81 | 20.36 | 0.2884 |
| **TesserAct** | **16.02** | **14.75** | **0.0811** |

### 机器人动作规划（RLBench 成功率%）

| 方法 | close box | open drawer | open jar | open micro | put knife |
|------|-----------|-------------|----------|------------|-----------|
| Image-BC | 53 | 4 | 0 | 5 | 0 |
| UniPi | 81 | 67 | 38 | 72 | 66 |
| **TesserAct** | **88** | **80** | **44** | **70** | **70** |

### 新视角合成

| 方法 | PSNR | SSIM | 时间 |
|------|------|------|------|
| SoM | 10.94 | 24.02 | ~2h |
| **TesserAct** | **12.99** | **42.62** | **~1min** |

### 关键发现

- 联合预测 RGB-DN 显著优于先生成 RGB 再后处理
- 法线积分使平面重建不再倾斜
- 一致性损失和正则化对 4D 重建都至关重要
- 模型能泛化到未见场景和新物体

## 亮点与洞察

1. **RGB-DN 中间表示**：既保留 3D 几何信息又兼容视频模型，高效训练
2. **零初始化策略**：精心保留预训练模型先验知识
3. **法线辅助深度优化**：补偿仿射不变深度的不足
4. **跨机器人平台泛化**：文本指定机器人臂，单模型适配多平台
5. **自动 4D 标注**：现成估计器将视频数据集转化为 4D 训练数据
6. **光流引导动静分离**：在 4D 重建中自然分离，分别约束

## 局限性

- RGB 质量略低于纯 RGB 微调（SSIM 下降约 3.5%）
- 深度法线来自估计器而非传感器，存在噪声
- 仅固定视角预测，非真正多视角 4D 生成
- 仿射不变深度缺乏绝对尺度
- 训练数据量（~285k）与大规模视频模型有差距

## 相关工作

- **具身基础模型**：RT-2、Octo 等 VLA 直接输出动作
- **视频世界模型**：UniPi、Genie 等在 2D 操作
- **4D 生成**：DreamGaussian4D 等基于 SDS 优化慢
- **深度优化**：GeoWizard、DSINE 等法线积分方法

## 评分

- **新颖性**: ★★★★☆ — 首次提出 4D 具身世界模型，RGB-DN 表示新颖
- **技术深度**: ★★★★☆ — 数据→模型→重建→规划完整闭环
- **实验充分度**: ★★★★☆ — 真实+合成域评估，下游任务验证
- **实用性**: ★★★★☆ — 对机器人规划有实际促进
- **总分**: 8.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making](../../ICML2025/robotics/founder_grounding_foundation_models_in_world_models_for_open-ended_embodied_deci.md)
- [\[ICLR 2026\] Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments](../../ICLR2026/robotics/test-time_mixture_of_world_models_for_embodied_agents_in_dynamic_environments.md)
- [\[CVPR 2025\] UniAct: Universal Actions for Enhanced Embodied Foundation Models](../../CVPR2025/robotics/universal_actions_for_enhanced_embodied_foundation_models.md)
- [\[NeurIPS 2025\] LLM World Models Are Mental: Output Layer Evidence of Brittle World Model Use in LLM Mechanical Reasoning](../../NeurIPS2025/robotics/llm_world_models_are_mental_output_layer_evidence_of_brittle_world_model_use_in_.md)
- [\[ICCV 2025\] Embodied Representation Alignment with Mirror Neurons](embodied_representation_alignment_with_mirror_neurons.md)

</div>

<!-- RELATED:END -->
