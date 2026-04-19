---
title: >-
  [论文解读] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model
description: >-
  [ECCV 2024][图像生成] 提出 MotionLCM，首次将一致性蒸馏引入人体运动生成领域，在运动潜在空间中实现单步/少步推理的实时运动生成（~30ms/序列），并通过 Motion ControlNet 实现潜在空间中的实时可控运动生成。
tags:
  - ECCV 2024
  - 图像生成
---

# MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model

**会议**: ECCV 2024  
**arXiv**: [2404.19759](https://arxiv.org/abs/2404.19759)  
**领域**: 图像生成

## 一句话总结

提出 MotionLCM，首次将一致性蒸馏引入人体运动生成领域，在运动潜在空间中实现单步/少步推理的实时运动生成（~30ms/序列），并通过 Motion ControlNet 实现潜在空间中的实时可控运动生成。

## 研究背景与动机

- **扩散模型的效率瓶颈**：现有运动扩散模型（MDM ~24s，MLD ~0.2s）推理速度慢，无法满足实时应用需求
- **时空控制的高延迟**：OmniControl 等可控运动生成方法推理时间约 81s/序列，距离实时应用差距巨大
- **潜在空间控制的困难**：在潜在扩散模型中，潜在表示缺乏显式运动语义，无法直接操作控制信号
- **一致性模型的机遇**：一致性模型（CM）通过学习 PF-ODE 轨迹上的一致性函数，实现高效的单步/少步生成，与加速运动生成的目标完美契合
- **核心问题**：如何在保证生成质量和控制能力的前提下，将运动生成加速到实时级别？

## 方法详解

### 整体框架

分两阶段训练：
1. **运动潜在一致性蒸馏**：从预训练的 MLD（运动潜在扩散模型）蒸馏一致性模型，实现 1-4 步推理
2. **潜在空间运动控制**：在 MotionLCM 的潜在空间引入 Motion ControlNet，并利用 VAE 解码器提供运动空间的显式控制监督

### 关键设计

**运动潜在一致性蒸馏**：
- 以 MLD 为教师模型，在运动潜在空间学习一致性函数 $\textbf{f}_\Theta : (\mathbf{z}_t, t, w, \mathbf{c}) \mapsto \mathbf{z}_0$
- 采用 k-step 跳跃一致性蒸馏（LCM 方案），而非逐步一致性，大幅减少收敛时间
- 将 classifier-free guidance（CFG）集成到蒸馏中，$w$ 在训练时从 $[5, 15]$ 均匀采样
- 使用 DDIM 求解器（跳跃间隔 $k=20$）+ Huber 损失作为距离度量

**Motion ControlNet**：
- 以 MotionLCM 的可训练副本初始化，每层附加零初始化线性层
- 控制任务定义：给定前 $\tau$ 帧初始姿态（6 个关键关节的全局 3D 位置）和文本描述，生成后续运动
- Trajectory Encoder：堆叠 Transformer 层编码轨迹信号，[CLS] token 的输出特征加到噪声潜在上

**运动空间显式监督**（核心创新）：
- 仅在潜在空间做重建损失不足以提供详细的控制约束
- 利用冻结的 VAE 解码器将预测潜在 $\hat{\mathbf{z}}_0$ 解码到运动空间，计算控制关节位置误差
- 得益于 MotionLCM 的单步推理能力，这一解码过程相比 MLD 更高效

### 损失函数

**第一阶段 — 一致性蒸馏损失**：

$$\mathcal{L}_{LCD} = \mathbb{E}[d(\textbf{f}_\Theta(\mathbf{z}_{n+k}, t_{n+k}, w, \mathbf{c}), \textbf{f}_{\Theta^-}(\hat{\mathbf{z}}_n, t_n, w, \mathbf{c}))]$$

**第二阶段 — 控制训练的总损失**：

$$\Theta^a, \Theta^b = \arg\min_{\Theta^a, \Theta^b} (\mathcal{L}_{recon} + \lambda \mathcal{L}_{control})$$

其中 $\mathcal{L}_{recon}$ 为潜在空间重建损失，$\mathcal{L}_{control}$ 为运动空间的控制关节位置误差，$\lambda=1.0$。

## 实验关键数据

### 主实验

**文本到运动生成对比（HumanML3D）**：

| 方法 | AITS(s)↓ | R-Precision Top3↑ | FID↓ | MM Dist↓ | Diversity→ | MModality↑ |
|------|----------|-------------------|------|----------|------------|------------|
| MDM | 24.74 | 0.611 | 0.544 | 5.566 | 9.559 | 2.799 |
| MotionDiffuse | 14.74 | 0.782 | 0.630 | 3.113 | 9.410 | 1.553 |
| MLD | 0.217 | 0.772 | 0.473 | 3.196 | 9.724 | 2.413 |
| MLD* (复现) | 0.225 | 0.796 | 0.450 | 3.052 | 9.634 | 2.267 |
| **MotionLCM (1步)** | **0.030** | 0.803 | 0.467 | 3.022 | 9.631 | 2.172 |
| **MotionLCM (2步)** | **0.035** | **0.805** | 0.368 | **2.986** | 9.640 | 2.187 |
| **MotionLCM (4步)** | **0.043** | 0.798 | **0.304** | 3.012 | 9.607 | 2.259 |

**可控运动生成对比**：

| 方法 | AITS(s)↓ | FID↓ | R-Precision Top3↑ | Traj. err.↓ | Loc. err.↓ | Avg. err.↓ |
|------|----------|------|-------------------|-------------|------------|------------|
| OmniControl | 81.00 | 2.328 | 0.557 | 0.3362 | 0.0322 | 0.0977 |
| MLD (LC&MC) | 0.552 | 0.555 | 0.754 | 0.2722 | 0.0215 | 0.1265 |
| **MotionLCM 1步 (LC&MC)** | **0.042** | **0.419** | **0.756** | **0.1988** | **0.0147** | **0.1127** |
| **MotionLCM 2步 (LC&MC)** | **0.047** | **0.397** | **0.759** | **0.1960** | **0.0143** | **0.1092** |

### 消融实验

**训练 guidance scale 范围和 EMA 率的影响**：

| 设置 | R-Precision Top1↑ | FID↓ | MM Dist↓ | Diversity→ |
|------|-------------------|------|----------|------------|
| $w \in [5,15]$ (默认) | **0.502** | 0.467 | **3.022** | 9.631 |
| $w \in [2,18]$ | 0.497 | — | — | — |
| Huber 损失 (默认) | **0.502** | **0.467** | — | — |
| L2 损失 | — | 0.592 | — | — |

### 关键发现

1. **速度**：MotionLCM 单步推理仅需 ~30ms，比 OmniControl 快 **1929 倍**，比 MLD 快 **13 倍**
2. **质量不降反升**：单步推理即超越 MLD 50 步 DDIM 在 R-Precision 上的表现，4 步推理达到最佳 FID（0.304）
3. **控制能力**：MotionLCM 生成的潜在表示比 MLD 更适合训练 Motion ControlNet，在相同设置下控制精度显著更高
4. **运动空间监督的关键性**：加入运动空间的显式控制损失后，Traj. err. 从 0.2986 降至 0.1988（降低 33%）

## 亮点与洞察

- **首次将一致性蒸馏引入运动生成**：证明了在运动潜在空间中 LCM 方案的可行性和有效性
- **实时可控生成**：MotionLCM + ControlNet 实现了自回归式实时运动生成（用上一段运动的末尾帧作为下一段的初始控制信号）
- **潜在-运动双空间监督**：利用 VAE 解码器将潜在空间预测解码到运动空间做显式控制监督，巧妙解决了潜在空间缺乏运动语义的问题
- **效率与质量的帕累托最优**：在推理时间-质量的散点图上，MotionLCM 最接近原点，实现了两者的最优平衡

## 局限性

- 依赖预训练的 MLD 模型质量，蒸馏的上界受教师模型限制
- 控制任务目前仅定义为初始姿态控制（前 25% 帧），尚未探索更灵活的时空控制模式
- 单步推理时 FID 略逊于 4 步推理，说明极端加速仍有微小质量损失

## 评分

⭐⭐⭐⭐⭐ (5/5) — 将运动生成推进到实时级别，一致性蒸馏+ControlNet 的双阶段设计简洁高效，运动空间显式监督解决了潜在空间控制的核心难题，对实际应用意义重大

<!-- RELATED:START -->

## 相关论文

- [Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)

<!-- RELATED:END -->
