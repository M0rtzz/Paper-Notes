---
title: >-
  [论文解读] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities
description: >-
  [图像生成] > 提出 VLN-PE，首个物理真实的视觉-语言导航平台，支持人形、四足和轮式机器人，系统评估现有 VLN 方法在真实物理约束下的性能，揭示了仿真到物理部署中 34% 的成功率下降。
tags:
  - 图像生成
---

# Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities

| 信息 | 内容 |
|------|------|
| 会议 | ICCV 2025 |
| arXiv | [2507.13019](https://arxiv.org/abs/2507.13019) |
| 代码 | [项目页面](https://crystalsixone.github.io/vln_pe.github.io) |
| 领域 | 具身导航 · 视觉-语言导航 · 跨具身 |
| 关键词 | VLN, cross-embodiment, physical simulation, diffusion policy, benchmark |

## 一句话总结

> 提出 VLN-PE，首个物理真实的视觉-语言导航平台，支持人形、四足和轮式机器人，系统评估现有 VLN 方法在真实物理约束下的性能，揭示了仿真到物理部署中 34% 的成功率下降。

## 研究背景与动机

### 仿真与物理部署的鸿沟

视觉-语言导航 (VLN) 从离散图跳转（R2R）发展到连续导航（VLN-CE），但现有方法在物理部署时面临严重问题：

**理想化假设**：大多数 VLN 基准只支持理想化的轮式或点状 agent，忽略了真实机器人的物理具身约束

**测试条件过于理想**：忽略了视角偏移、摔倒、死锁、运动误差等关键物理问题

**缺乏跨具身评估**：没有系统分析不同类型机器人（人形、四足、轮式）对 VLN 方法的影响

**运动控制缺失**：现有平台使用导航网格的伪运动，不反映真实物理动力学

### 核心问题

> 物理具身约束和视觉环境变化在多大程度上影响现有 VLN 方法的性能？

## 方法详解

### 整体框架：VLN-PE 平台

基于 GRUTopia 物理仿真器构建，支持三种机器人：
- **人形**：Unitree H1、G1
- **四足**：Unitree Aliengo
- **轮式**：Jetbot

#### 场景支持
- 90 个 Matterport3D 场景（修复了可能导致腿部机器人卡住的地板缝隙）
- 10 个高质量合成家庭场景（GRScenes）
- 3DGS 渲染的实验室环境

#### 新增指标
- **Fall Rate (FR)**：意外摔倒频率
- **Stuck Rate (StR)**：agent 卡住不动的频率

### 评估的三类方法

#### 1. 端到端分类方法（单步预测）

**Seq2Seq**：LSTM 编码指令 + ResNet50 编码 RGB/Depth + GRU 预测动作

$$h_t = \text{GRU}([V_t, D_t, I], h_{t-1}), \quad a_t = \arg\min_a \text{softmax}(W_a h_t + b_a)$$

**CMA**：在 Seq2Seq 基础上增加交叉模态注意力，通过两个 GRU 分别处理视觉观测和指令引导决策。

**NaVid**：7B 参数的视频多模态大语言模型，基于 LLaMa-VID 构建，RGB-only 导航。

#### 2. 多步扩散策略（RDP）——本文新提出的基线

首次将扩散策略应用于 VLN。使用 LongCLIP 编码 RGB+指令，ResNet50 编码深度，跨模态注意力融合后用 Transformer 作为扩散解码器：

$$a_t^{k-1} = \alpha \cdot (a_t^k - \gamma \epsilon_\theta(c_t, a_t^k, k) + \mathcal{N}(0, \mu^2 I))$$

创新点：引入 GRU 维护历史观测 + 额外 MLP 预测停止进度（0→1）。

$$\mathcal{L}_{\text{RDP}} = \text{MSE}(\epsilon^k, \epsilon_\theta(c_t, a_t^0 + \epsilon^k, k)) + \lambda \cdot \text{MSE}(\mathcal{S}_{\text{stop}}(c_t), \hat{p}_{\text{stop}})$$

#### 3. 无训练地图方法（改进 VLMaps）

使用 LLM 解析指令为子目标代码，LSeg 在语义地图上定位目标。新增 VLFM 前沿探索策略应对目标不可见情况。

## 实验

### 主实验：VLN-CE 到 VLN-PE 的迁移（R2R 数据集，Humanoid H1）

| 方法 | 设置 | Val-Seen SR↑ | Val-Seen SPL↑ | Val-Unseen SR↑ | Val-Unseen SPL↑ |
|------|------|:---:|:---:|:---:|:---:|
| Seq2Seq-Full | 零样本迁移 | 13.83 | 11.17 | 15.00 | 11.99 |
| CMA-Full | 零样本迁移 | 15.50 | 14.00 | 16.04 | 14.63 |
| NaVid | 零样本迁移 | 21.58 | 17.45 | 22.42 | 18.58 |
| CMA+ | VLN-PE 微调 | **28.72** | **24.24** | 23.31 | 19.66 |
| RDP | VLN-PE 训练 | 23.86 | 17.35 | 21.98 | 16.44 |

关键发现：
- 从 VLN-CE 直接迁移到 VLN-PE，SR 下降约 **34%**
- NaVid (7B) 零样本表现最好，StR 和 FR 远低于小模型（世界知识帮助避障）
- 仅用 VLN-PE 441 条数据微调的 CMA-CLIP 即可超越 NaVid 零样本

### 跨场景泛化（GRU-VLN10 + 3DGS-Lab）

| 方法 | GRU-VLN10 Unseen SR↑ | 3DGS-Lab SR↑ |
|------|:---:|:---:|
| NaVid (零样本) | 18.64 | 5.81 |
| CMA-CLIP (微调) | 22.46 | 24.88 |
| RDP (微调) | **28.52** | **30.63** |

NaVid 在 3DGS 场景中**完全失败**（SR 仅 5.81%），可能因 3DGS 渲染噪声干扰了大模型的 RGB 感知。

### 光照条件影响

| 光照条件 | CMA (RGB-D) SR | CMA (RGB-only) SR |
|----------|:---:|:---:|
| 高亮度 (50k) | 21.74 | 14.72 |
| 极低亮度 (1k) | 19.84 | 3.36 |

RGB-only 模型在低光照下性能骤降，而 RGB+Depth 模型更鲁棒。

## 亮点与洞察

1. **首个系统性跨具身 VLN 评估**：揭示了人形/四足/轮式机器人在 VLN 中的独特挑战（视角高度差异、腿部运动约束）
2. **扩散策略首次用于 VLN**：提出 RDP 基线，密集轨迹预测在从头训练时优于 CMA/Seq2Seq
3. **大模型悖论**：NaVid (7B) 虽有更好的零样本能力和避障能力，但在新场景（3DGS）中完全失效，且存在目标附近反复旋转的问题
4. **小数据微调的威力**：仅 441 条域内数据微调即可显著超越 7B 大模型的零样本性能

## 局限性

- RL 运动控制器尚不能可靠处理楼梯导航，需要过滤含楼梯的 episode
- 腿部机器人（人形、四足）在复杂环境中的摔倒率仍然较高
- 评估场景规模有限（3DGS 仅 1 个实验室环境）
- RDP 扩散策略的停止决策仍需额外 MLP 辅助

## 相关工作

- **离散 VLN**：R2R、REVERIE 等图导航
- **连续 VLN**：VLN-CE、NaVid、CMA 等 Habitat 平台方法
- **扩散策略**：Diffusion Policy 在机器人操作中的成功
- **跨具身基准**：GRUTopia、BEHAVIOR-100、ARIO 等

## 评分

| 维度 | 分数 |
|------|:----:|
| 创新性 | ⭐⭐⭐⭐ |
| 有效性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐⭐ |
| 综合推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing](eedit_rethinking_the_spatial_and_temporal_redundancy_for_efficient_image_editing.md)
- [Rethinking Layered Graphic Design Generation with a Top-Down Approach](rethinking_layered_graphic_design_generation_with_a_top-down_approach.md)
- [DEXTER: Diffusion-Guided EXplanations with TExtual Reasoning for Vision Models](../../NeurIPS2025/image_generation/dexter_diffusion-guided_explanations_with_textual_reasoning_for_vision_models.md)
- [Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior](../../NeurIPS2025/image_generation/diff-icmh_harmonizing_machine_and_human_vision_in_image_compression_with_generat.md)
- [InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](../../NeurIPS2025/image_generation/infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)

<!-- RELATED:END -->
