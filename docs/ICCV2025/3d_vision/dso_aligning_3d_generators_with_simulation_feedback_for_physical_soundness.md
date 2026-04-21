---
title: >-
  [论文解读] DSO: Aligning 3D Generators with Simulation Feedback for Physical Soundness
description: >-
  [3D视觉] 提出 Direct Simulation Optimization (DSO) 框架，利用物理仿真器的（非可微）稳定性反馈作为奖励信号，通过 DPO 或新提出的 DRO 目标函数微调 3D 生成器，使其前馈式地直接输出物理上自支撑的 3D 物体，无需测试时优化。
tags:
  - 3D视觉
---

# DSO: Aligning 3D Generators with Simulation Feedback for Physical Soundness

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2503.22677](https://arxiv.org/abs/2503.22677)
- **代码**: [ruiningli.com/dso](https://ruiningli.com/dso)
- **领域**: 3D Vision / 物理稳定性
- **关键词**: 3D生成, 物理仿真反馈, DPO/DRO对齐, 扩散模型微调, 自支撑
- **作者**: Ruining Li, Chuanxia Zheng, Christian Rupprecht, Andrea Vedaldi (Oxford VGG)

## 一句话总结

提出 Direct Simulation Optimization (DSO) 框架，利用物理仿真器的（非可微）稳定性反馈作为奖励信号，通过 DPO 或新提出的 DRO 目标函数微调 3D 生成器，使其前馈式地直接输出物理上自支撑的 3D 物体，无需测试时优化。

## 研究背景与动机

当前最先进的 3D 生成器（如 TRELLIS、Hunyuan3D 2.0）重视几何和外观质量，但忽略了物理约束——特别是在重力下的自支撑稳定性。实验显示即使输入的是稳定物体图像，TRELLIS 仍有约 30% 的概率生成不稳定的 3D 模型。

已有方法（如 Atlas3D、PhysComp）依赖可微物理仿真器在测试时优化几何，但可微仿真器速度慢、数值不稳定、容易陷入局部最优。本文的核心动机是：能否让生成器在训练阶段就学会生成物理稳定的物体，从而在推理时无需额外优化？

稳定性的本质是离散的（要么稳定要么倒塌），不适合梯度下降直接优化，但物理仿真器可以轻松判定稳定性——这天然适合基于奖励的学习范式。

## 方法详解

### 整体框架

DSO 框架分三步：
1. **生成数据**：用基础模型 $p_{\text{ref}}$ 从图像生成大量 3D 物体
2. **仿真标注**：用非可微物理仿真器（MuJoCo）对每个生成物体进行稳定性二值标注 $o(\mathbf{x}_0) \in \{0, 1\}$
3. **对齐训练**：用 DPO 或 DRO 目标函数微调生成器，强化稳定样本、抑制不稳定样本

### 核心优化目标

原始目标函数要求最大化生成物体的稳定性期望，同时用 KL 散度约束新模型不偏离基础模型：

$$\max_\theta \mathbb{E}[o(\mathbf{x}_0)] - \beta \mathbb{D}_{\text{KL}}[p_\theta \| p_{\text{ref}}]$$

但 $o$ 不可微，直接 RL 优化又因解码 3D 表征开销过大而不可行。

### Direct Reward Optimization (DRO) — 本文新提出

通过重参数化技巧，将奖励信号 $o(\mathbf{x}_0)$ 用最优逆扩散过程表示，最终推导出不需要成对偏好数据的训练目标：

$$\mathcal{L}_{\text{DRO}} = -T \mathbb{E}\left[ w(t)(1 - 2o(\mathbf{x}_0)) \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \|^2_2 \right]$$

直观含义：对稳定样本（$o=1$），系数为 $-1$，鼓励模型更好地去噪；对不稳定样本（$o=0$），系数为 $+1$，让模型"忘记"去噪这些样本。

**DRO 相比 DPO 的优势**：
- 不需要成对偏好数据
- 训练时不需要查询基础模型 $\epsilon_{\text{ref}}$
- 收敛更快，对齐效果更好

### Direct Preference Optimization (DPO)

作为对比，也可以用 Diffusion-DPO 的目标函数训练，需要对同一张图的稳定/不稳定 3D 模型配对：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[ \log \sigma\left( -\beta T w(t) \left( \Delta_\theta^w - \Delta_\theta^l \right) \right) \right]$$

### 自改进数据构建

关键创新：训练数据完全来自生成器自身的输出，无需真实 3D 物体。
- 用基础模型从 Objaverse 渲染图生成 3D 模型（13k 物体 × 6 图 × 4 生成 = 312k 模型）
- 用 MuJoCo 仿真，倾斜角 <20° 判为稳定
- 甚至可以完全用合成 2D 图像（GPT-4 生成描述 → FLUX 生成图像）替代渲染图

### 训练细节

- 基础模型：TRELLIS（rectified flow transformer）
- 仅微调第一个 coarse geometry transformer 的线性层（LoRA rank 64）
- AdamW 优化器，batch size 48，4× A100
- DRO 训练 4,000 步，DPO 训练 8,000 步

## 实验关键数据

### 主实验结果（Table 1：PhysComp 评估集）

| 方法 | % Stable ↑ | Rot. ↓ | CD ↓ | F-Score ↑ |
|------|-----------|--------|------|-----------|
| TRELLIS (基线) | 85.1 | 14.14° | 0.0485 | 73.12 |
| Atlas3D | 69.4 | 32.86° | — | — |
| DSO + DPO | 95.1 | 5.42° | 0.0480 | 73.62 |
| **DSO + DRO** | **99.0** | **1.88°** | **0.0440** | **76.17** |

在困难子集（11个不稳定物体）上：

| 方法 | % Stable ↑ | % Output ↑ | Rot. ↓ |
|------|-----------|-----------|--------|
| TRELLIS | 54.5 | 100 | 39.18° |
| PhysComp | 80.3 | 46.2 | 18.14° |
| DSO + DPO | 82.6 | 100 | 16.83° |
| **DSO + DRO** | **95.5** | **100** | **5.58°** |

### 消融实验（Table 2 & 3）

| 方法 | % Stable ↑ | Rot. ↓ | CD ↓ | F-Score ↑ |
|------|-----------|--------|------|-----------|
| TRELLIS | 85.1 | 14.14° | 0.0485 | 73.12 |
| TRELLIS + SFT | 89.5 | 10.22° | 0.0440 | 76.17 |
| DSO + DPO | 95.1 | 5.42° | 0.0480 | 73.62 |
| DSO + DRO | 99.0 | 1.88° | 0.0440 | 76.17 |

合成数据消融（Table 3）：

| 方法 | 纯合成? | 损失 | % Stable ↑ | Rot. ↓ |
|------|---------|------|-----------|--------|
| DSO | ✓ | DPO | 93.5 | 6.92° |
| DSO | ✗ | DPO | 95.1 | 5.42° |
| DSO | ✓ | DRO | 97.6 | 3.17° |
| DSO | ✗ | DRO | 99.0 | 1.88° |

### 关键发现

1. **DRO 全面优于 DPO**：稳定率 99.0% vs 95.1%，且收敛更快
2. **物理稳定性 vs 几何质量无冲突**：DSO 微调后几何质量甚至略有提升（CD 从 0.0485 降至 0.0440）
3. **SFT 不如 DSO**：仅在稳定数据上 SFT 不够有效，需要同时暴露稳定/不稳定样本
4. **数据效率高**：仅用 1/16 数据（~19.2k 模型）即可达到接近全量数据的效果
5. **纯合成可行**：完全不用真实 3D 数据，仅用合成图像即可训练出有效模型

## 亮点与洞察

- **DRO 是一个通用创新**：不需要成对偏好数据、不需要训练时查询基础模型，可推广到其他扩散模型对齐场景
- **自改进闭环**：生成器生成 → 仿真器评估 → 反馈微调，形成无需人工标注的自动改进管线
- **非可微仿真器的巧妙利用**：绕开可微仿真的局限，将物理属性优化重新表述为奖励对齐问题
- 3D 打印实验验证了在真实物理世界中的有效性

## 局限性

- 仅关注重力下的自支撑稳定性，未涉及其他物理属性（可变形、关节等）
- 过长训练会"作弊"——生成底座平板来防止倾倒
- 仅在 TRELLIS 上验证，其他 3D 生成器的适用性待检验
- 稳定性判定阈值（20°）的选择缺乏充分讨论

## 相关工作与启发

- **RLHF → 仿真反馈**：将 LLM 对齐中的 DPO 范式推广到 3D 生成的物理属性对齐
- **可推广到更多物理属性**：如场景分解、零件交互、可抓取性等
- **对 3D 生成质量评估的启示**：几何质量和实用性（物理稳定性）应并重评估

## 评分 ⭐⭐⭐⭐

一篇思路清晰、方法优雅的工作。DRO 作为不需要成对偏好数据的扩散模型对齐新目标，具有较强的通用性。自改进管线和纯合成数据方案展示了强大的实用价值。实验全面，物理3D打印验证令人信服。

<!-- RELATED:START -->

## 相关论文

- [GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs](gaussianproperty_integrating_physical_properties_to_3d_gaussians_with_lmms.md)
- [SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes](placeit3d_language-guided_object_placement_in_real_3d_scenes.md)
- [Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)

<!-- RELATED:END -->
