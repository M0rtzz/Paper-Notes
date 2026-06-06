---
title: >-
  [论文解读] EgoM2P: Egocentric Multimodal Multitask Pretraining
description: >-
  [ICCV 2025][3D视觉][自我中心视觉] EgoM2P 是首个面向自我中心(egocentric)4D理解的多模态多任务大模型，通过时序感知的掩码建模框架统一处理 RGB 视频、深度、注视和相机轨迹四种模态，在多个下游任务上匹配或超越专用模型且快一个数量级。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "自我中心视觉"
  - "多模态预训练"
  - "多任务学习"
  - "掩码建模"
  - "注视预测"
  - "相机追踪"
  - "深度估计"
  - "视频生成"
---

# EgoM2P: Egocentric Multimodal Multitask Pretraining

**会议**: ICCV 2025  
**arXiv**: [2506.07886](https://arxiv.org/abs/2506.07886)  
**代码**: [项目页](https://egom2p.github.io/)  
**领域**: 3D视觉  
**关键词**: 自我中心视觉, 多模态预训练, 多任务学习, 掩码建模, 注视预测, 相机追踪, 深度估计, 视频生成

## 一句话总结

EgoM2P 是首个面向自我中心(egocentric)4D理解的多模态多任务大模型，通过时序感知的掩码建模框架统一处理 RGB 视频、深度、注视和相机轨迹四种模态，在多个下游任务上匹配或超越专用模型且快一个数量级。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：自我中心视觉中的多模态信号（RGB、深度、相机位姿、注视等）对增强现实、机器人和人机交互至关重要，但构建大规模自我中心多模态多任务模型面临独特挑战：

**数据异构性**：不同设备和场景下的模态覆盖差异巨大，有些数据集有注视但无深度，有些有深度但无注视，导致模态缺失问题严重

**伪标签不可行**：对于注视、头戴相机轨迹等模态，生成伪标签往往不现实，标准监督学习难以扩展

**时空复杂性**：自我中心视频的动态相机运动和复杂时空结构给现有多模态基础模型带来额外挑战

**现有模型局限**：第三人称视频模型不适合自我中心视角；现有自我中心基础模型仅关注视频问答，忽略人体相关模态和3D/4D重建能力

**图像级模型局限**：4M 等多模态基础模型仅处理单图像，缺乏时序一致性保障

### 解决思路

**本文目标**：### 整体框架

EgoM2P 基于 T5-Base 架构，采用编码器-解码器设计。


## 方法详解

### 整体框架

EgoM2P 基于 T5-Base 架构，采用编码器-解码器设计。pipeline包含三个阶段：数据整理 → 模态分词 → 掩码预训练。

### 关键设计一：数据整理管道（Data Curation）

整合8个异构数据集（EgoExo4D, HoloAssist, HOT3D, ARCTIC, TACO, H2O, EgoGen等）：

- **切分**：统一切为 T 帧片段，编码为高质量 mp4
- **标注**：RollingDepth 生成深度伪标签；EgoGen 生成~30小时合成数据；缺失注视不伪标签
- **标准化**：统一30 FPS；深度用逆深度按序列归一化；注视投影到2D；相机轨迹统一为 OpenCV camera-to-world，首帧为参考

### 关键设计二：时序感知分词器（Temporal Tokenizers）

- **视频模态**（RGB/Depth）：Cosmos Tokenizer，时间4×、空间8×压缩
- **注视/相机轨迹**：基于 Transformer 的 VQ-VAE
    - N维卷积 2×时间下采样 + N维位置编码 + 12层 Transformer
    - 余弦相似度向量量化，模态特定码本
    - 训练损失：重建 + 码本 + 承诺损失

### 关键设计三：多模态掩码预训练

相比 4M 的图像级掩码建模，做了四项关键适配：

1. **token 数量扩展**：4M 的256 → 2048个可见 token，适应视频5000+ token
2. **数据集平衡**：~40亿 vs 1300万注视 token → 按数据集大小比例采样 + 模态均匀采样
3. **缺失模态处理**：占位符+掩码，而非伪标签
4. **时序嵌入**：注视/相机1D + 视频3D正弦余弦编码 + 可学习模态类别嵌入

### 推理

预训练后通过采样不同 token 组合实现任意到任意模态预测，支持并行推理。

## 实验关键数据

### 支持的下游任务

1. **注视估计**：预测2D注视点
2. **自我中心相机追踪**：6DoF 轨迹估计
3. **单目深度估计**：从 RGB 视频估计深度
4. **条件视频生成**：基于多模态条件生成视频

### 主要结果

- 在所有4个任务上**匹配或超越**专用模型
- 推理速度快**一个数量级**
- 单一模型覆盖理解+生成

### 训练规模


### 主实验

| 项目 | 数据 |
|------|------|
| 数据集数量 | 8个（真实+合成） |
| 总训练 token | ~40亿 |
| 注视 token | ~1300万 |
| 视频 token/样本 | >5000 |
| 可见 token 数 | 2048 |

### 关键发现

1. EgoGen 合成数据对深度估计有明显提升
2. 数据集按大小比例采样 + 模态均匀采样最稳定
3. 缺失模态不需伪标签也能被有效预测
4. 时序 token 化设计显著优于图像级处理

## 亮点与洞察

1. **模态缺失的优雅处理**：通过掩码占位符自然处理缺失模态，是面对自我中心数据异构性的务实方案
2. **统一分词器架构**：新增模态只需训练新分词器和码本，扩展性强
3. **理解与生成统一**：单模型同时支持感知和生成任务
4. **速度优势**：并行推理比多个专用模型快一个数量级，对实时AR/VR重要
5. **数据工程**：整合8个异构数据集的完整管道是实用的工程贡献

## 局限与展望

1. 注视数据量远少于其他模态（1300万 vs 40亿 token），可能限制注视预测精度
2. 仅覆盖4种模态，未包含音频、触觉、手部姿态等
3. 视频分辨率受限于 Cosmos Tokenizer 压缩能力
4. 各任务具体数值与专用模型对比表格在补充材料中
5. 数据集偏向室内/操作场景，户外泛化未充分验证

## 相关工作

- **图像级多模态基础模型**：4M（掩码建模）→ EgoM2P 扩展到时序视频
- **视频基础模型**：VideoMAE, Cosmos → 主要第三人称
- **自我中心基础模型**：EgoVLP/v2（视频-语言QA）→ 缺乏人体模态和3D能力
- **多模态绑定**：ImageBind → 仅对齐嵌入空间

## 评分

- **新颖性**：8/10 — 首个面向自我中心4D的多模态多任务模型
- **技术深度**：7/10 — 基于4M的扩展设计合理但非革命性
- **实验充分性**：6/10 — 缓存缺少详细数字
- **实用性**：9/10 — 单模型多任务 + 快速推理对 AR/VR 有直接价值
- **总评**：7.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Look and Tell: A Dataset for Multimodal Grounding Across Egocentric and Exocentric Views](../../NeurIPS2025/3d_vision/look_and_tell_a_dataset_for_multimodal_grounding_across_egocentric_and_exocentri.md)
- [\[ICCV 2025\] TokenUnify: Scaling Up Autoregressive Pretraining for Neuron Segmentation](tokenunify_scaling_up_autoregressive_pretraining_for_neuron_segmentation.md)
- [\[ICCV 2025\] HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding](his-gpt_towards_3d_human-in-scene_multimodal_understanding.md)
- [\[ICCV 2025\] Benchmarking Egocentric Visual-Inertial SLAM at City Scale](benchmarking_egocentric_visualinertial_slam_at_city_scale.md)
- [\[ICCV 2025\] Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision](fish2mesh_transformer_3d_human_mesh_recovery_from_egocentric_vision.md)

</div>

<!-- RELATED:END -->
