---
title: >-
  [论文解读] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation
description: >-
  [ECCV 2024][图像生成] 提出 GuidedMotion，以局部动作作为细粒度控制信号引导全局运动扩散生成，通过语义图解析和图注意力网络估计引导权重，支持连续可调的运动控制，在生成复杂多动作运动时优势显著。
tags:
  - ECCV 2024
  - 图像生成
---

# Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation

**会议**: ECCV 2024  
**arXiv**: [2407.10528](https://arxiv.org/abs/2407.10528)  
**领域**: 图像生成

## 一句话总结

提出 GuidedMotion，以局部动作作为细粒度控制信号引导全局运动扩散生成，通过语义图解析和图注意力网络估计引导权重，支持连续可调的运动控制，在生成复杂多动作运动时优势显著。

## 研究背景与动机

文本到运动生成需要将语言中的局部动作（如"走路"、"举手"）落地并无缝融合为多样且真实的全局运动。但现有方法主要关注直接合成全局运动，忽视了对局部动作的生成和控制。用文本精确表达复杂轨迹、姿态和包含多个动作的长运动序列非常困难，通常需要反复迭代编辑 prompt。

本文提出 **局部动作引导** 的新范式：将参考局部动作作为控制信号，在全局运动扩散过程中提供条件引导，实现从局部到全局的运动生成。

## 方法详解

### 整体框架

GuidedMotion 包含以下核心模块：
1. **自动局部动作采样**：语义图解析 → 局部动作描述 → 文本到运动模型（MLD）生成局部动作
2. **局部动作扩散引导**：利用能量函数计算局部动作梯度，在扩散反向过程中提供条件引导
3. **层级运动扩散模型**：将扩散过程分为运动级、动作级和细节级三个语义层次

### 关键设计

**语义图解析**：将运动描述解析为层级图结构，包含三类节点（运动/动作/细节）和十二类边。例如 "a person jogs and looks around" 解析为全局运动节点、"jogs" 和 "looks" 两个动作节点，以及各自的属性节点。

**局部动作引导**基于能量函数和得分理论。反向扩散过程修正为：

$$\mathbf{z}_{t-1} = \tilde{\mathbf{z}}_{t-1} - \sum_{k=1}^K \lambda_t^k \nabla_{\mathbf{z}_t} \mathcal{E}(\mathbf{c}^k, \mathbf{z}_t)$$

其中 $\mathcal{E}$ 为能量函数（使用潜空间 L2 距离），$\lambda_t^k$ 为引导权重。

**引导权重估计**：利用 GAT（图注意力网络）建模语义图中动作节点与全局运动节点之间的注意力系数作为引导权重，用户也可手动调节 $\rho$ 参数放大或缩小引导强度。

**层级扩散**：将扩散分为三级——运动级提供粗略初始值，动作级施加局部动作引导，细节级进一步精调以符合原始描述。

### 损失函数

三级独立训练，总损失为：$\mathcal{L} = \mathcal{L}_M + \mathcal{L}_A + \mathcal{L}_S$

每级均为标准扩散去噪损失（MSE），使用 DDIM 加速（50 步）。

## 实验关键数据

### 主实验

HumanML3D 数据集上与 SOTA 方法对比：

| 方法 | R-Top3 ↑ | FID ↓ | MM-Dist ↓ | Diversity → | MModality ↑ |
|------|:-:|:-:|:-:|:-:|:-:|
| MDM | 0.611 | 0.544 | 5.566 | 9.559 | **2.799** |
| MLD | 0.772 | 0.473 | 3.196 | 9.724 | 2.413 |
| T2M-GPT | 0.775 | 0.116 | 3.118 | 9.761 | 1.856 |
| ReMoDiffuse | **0.795** | 0.103 | **2.974** | 9.018 | 1.795 |
| **GuidedMotion** | 0.788 | **0.057** | 3.040 | 9.864 | 2.473 |

GuidedMotion 在 FID 指标上以 0.057 显著领先（次优 0.103），且多模态性优于大部分方法。

### 消融实验

HumanML3D 各模块消融：

| 运动级 | 动作级 | 细节级 | 局部动作引导 | R-Top3 ↑ | FID ↓ |
|:-:|:-:|:-:|:-:|:-:|:-:|
| ✓ | | | | 0.760 | 0.186 |
| ✓ | ✓ | | | 0.771 | 0.133 |
| ✓ | ✓ | | ✓ | 0.778 | 0.119 |
| ✓ | ✓ | ✓ | | 0.769 | 0.107 |
| ✓ | ✓ | ✓ | ✓ | **0.788** | **0.057** |

局部动作引导在有/无细节级的情况下均带来显著提升。

复杂运动子集（≥3 个局部动作且 ≥150 帧）上的对比：

| 方法 | R-Top3 ↑ | FID ↓ |
|------|:-:|:-:|
| MLD | 0.710 | 0.783 |
| T2M-GPT | 0.712 | 0.314 |
| **GuidedMotion** | **0.732** | **0.144** |

在复杂运动上优势更加明显。

### 关键发现

- 局部到全局范式降低了直接生成复杂全局运动的难度
- 通过采样不同的局部动作组合，可生成满足不同用户偏好的多样运动
- 引导权重支持连续调节，实现对运动轨迹和姿态的细粒度控制
- 在 KIT 数据集上同样达到 SOTA（FID 0.213，MModality 4.138）

## 亮点与洞察

- **从局部到全局**的范式比直接全局生成更可控，对复杂运动优势尤为突出
- 语义图解析 + GAT 估计引导权重的设计使得方法可解释性强
- 用户可灵活组合偏好的局部动作，无需反复调整文本 prompt
- 层级扩散的三阶段策略有效平衡了生成稳定性和细节质量

## 局限性

- 语义图解析依赖语义角色标注工具，对不规范或歧义描述可能失效
- 局部动作采样使用 MLD 单独生成，其质量上界受限于基础模型
- 引导权重调节需要一定的用户理解，自动化程度有待提升

## 评分

⭐⭐⭐⭐ 范式创新，局部到全局思路优雅，复杂运动场景优势明显，实验扎实

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)

<!-- RELATED:END -->
