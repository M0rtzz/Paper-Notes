---
title: >-
  [论文解读] EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation
description: >-
  [ECCV 2024][图像生成][人体动作生成] 提出 EMDM，通过条件去噪扩散 GAN 捕获大步长下的复杂去噪分布，实现仅需不超过 10 步采样即可实时生成高质量人体动作，推理速度较 MDM 提升约 200 倍。
tags:
  - ECCV 2024
  - 图像生成
  - 人体动作生成
  - 扩散模型
  - GAN
  - 高效采样
  - 文本驱动运动
---

# EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation

**会议**: ECCV 2024  
**arXiv**: [2312.02256](https://arxiv.org/abs/2312.02256)  
**代码**: [GitHub](https://github.com/Frank-ZY-Dou/EMDM) (有)  
**领域**: 图像生成  
**关键词**: 人体动作生成, 扩散模型, GAN, 高效采样, 文本驱动运动

## 一句话总结

提出 EMDM，通过条件去噪扩散 GAN 捕获大步长下的复杂去噪分布，实现仅需不超过 10 步采样即可实时生成高质量人体动作，推理速度较 MDM 提升约 200 倍。

## 研究背景与动机

当前基于扩散模型的人体动作生成方法（如 MDM、MotionDiffuse）虽然在生成质量上表现出色，但推理速度极慢：MDM 生成一段文本描述对应的动作序列需要约 12 秒，严重限制了实际应用（如在线动作合成、游戏开发）。

现有加速方案存在明显缺陷：

**潜空间扩散（MLD）**：先学习运动的潜在空间再进行潜在扩散，但两阶段方法中潜在空间的质量直接限制了下游生成效果，且无法端到端训练

**DDIM 加速采样**：简单增大采样步长时，去噪分布从高斯分布变为复杂的多模态分布，高斯假设不再成立，导致生成质量显著下降

核心问题在于：当减少采样步数时，每步的去噪分布变得非高斯且复杂，现有方法无法有效建模这种复杂分布。EMDM 的动机就是用条件 GAN 来捕获这种复杂分布，从而在少步采样下维持高质量生成。

## 方法详解

### 整体框架

EMDM 的核心思路是用**条件去噪扩散 GAN** 替代标准扩散模型中的高斯去噪假设。整个框架包含两个核心组件：

1. **条件生成器** $G_\theta(\mathbf{x}_t, \mathbf{z}, \mathbf{c}, t)$：输入含噪运动 $\mathbf{x}_t$、随机变量 $\mathbf{z} \sim \mathcal{N}(0, I)$（64维）、控制信号 $\mathbf{c}$（文本或动作标签）和时间步 $t$，输出预测的干净运动 $\hat{\mathbf{x}}_0$
2. **条件判别器** $D_\phi(\mathbf{x}_{t-1}, \mathbf{x}_t, \mathbf{c}, t)$：判断 $\mathbf{x}_{t-1}$ 是否为 $\mathbf{x}_t$ 的合理去噪结果

在推理时，模型只需要不超过 10 步即可从噪声生成高质量运动序列。

### 关键设计

#### 条件生成器的分布建模

核心公式是通过生成器和后验采样构建去噪分布：

$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \int p(\mathbf{z}) q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0 = G_\theta(\mathbf{x}_t, \mathbf{z}, \mathbf{c}, t)) d\mathbf{z}$$

这里的关键在于：生成器通过额外的随机变量 $\mathbf{z}$ 来建模多模态分布的多样性。不同于标准扩散模型假设每个去噪步骤的分布为高斯分布，EMDM 通过 GAN 的对抗训练来隐式地捕获任意复杂的分布形状。

工作流程为：
1. 生成器先预测 $\hat{\mathbf{x}}_0 = G_\theta(\mathbf{x}_t, \mathbf{z}, \mathbf{c}, t)$
2. 然后通过后验分布 $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ 采样得到 $\hat{\mathbf{x}}_{t-1}$
3. 判别器区分真实的 $({\mathbf{x}_{t-1}}, {\mathbf{x}_t})$ 对和生成的 $(\hat{\mathbf{x}}_{t-1}, {\mathbf{x}_t})$ 对

#### 条件判别器

判别器不仅依赖时间步 $t$，还以控制信号 $\mathbf{c}$ 为条件。训练目标为：

$$\min_\phi \sum_{t \geq 1} \mathbb{E}_{q(\mathbf{x}_t)} [\mathbb{E}_{q(\mathbf{x}_{t-1}|\mathbf{x}_t)} [F(-D_\phi)] + \mathbb{E}_{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)} [F(D_\phi)]]$$

其中 $F(\cdot) = \text{softplus}(\cdot)$。判别器采用 7 层 MLP 架构。

#### 几何损失函数

作者发现仅靠 GAN 的对抗损失不足以生成高质量的人体运动，需要引入运动领域特定的几何约束。几何损失包含四部分：

| 损失名称 | 公式含义 | 作用 |
|---------|---------|------|
| $\mathcal{L}_{\text{recon}}$ | 预测运动与真实运动的 L2 距离 | 整体重建质量 |
| $\mathcal{L}_{\text{pos}}$ | 通过前向运动学转换后的关节位置误差 | 关节位置准确性 |
| $\mathcal{L}_{\text{foot}}$ | 脚部接触时位移约束 | 减少脚部滑动伪影 |
| $\mathcal{L}_{\text{vel}}$ | 关节速度的一致性约束 | 运动平滑性 |

总几何损失为 $\mathcal{L}_{\text{geo}} = \mathcal{L}_{\text{recon}} + \lambda(\mathcal{L}_{\text{pos}} + \mathcal{L}_{\text{vel}} + \mathcal{L}_{\text{foot}})$，其中 $\lambda$ 在 action-to-motion 任务中为 1，text-to-motion 中为 0。

#### Classifier-free Guidance

EMDM 采用 classifier-free guidance：训练时随机以 10% 概率将条件设为空 $\mathbf{c} = \emptyset$，推理时通过插值两个生成结果来平衡多样性和保真度：

$$G_s = G(\mathbf{x}_t, \mathbf{z}, \emptyset, t) + s \cdot (G(\mathbf{x}_t, \mathbf{z}, \mathbf{c}, t) - G(\mathbf{x}_t, \mathbf{z}, \emptyset, t))$$

### 损失函数 / 训练策略

最终训练目标为：$\min_\theta (\mathcal{L}_{\text{disc}} + R \cdot \mathcal{L}_{\text{geo}})$

- 生成器：12 层 Transformer，32 个 attention head，带 skip connections
- 文本编码器：冻结的 CLIP-ViT-L-14
- 优化器：AdamW，学习率 $2 \times 10^{-5}$（text-to-motion）/ $3 \times 10^{-5}$（action-to-motion）
- 使用 EMA decay
- 批大小：64
- 端到端训练，无需分阶段

## 实验关键数据

### 主实验

**Text-to-motion (HumanML3D)**

| 方法 | FID↓ | R-Prec Top1↑ | MM Dist↓ | 每帧时间(ms)↓ | 是否端到端 |
|------|------|-------------|----------|-------------|----------|
| MDM | 0.508 | 0.418 | 3.630 | 62.505 | ✓ |
| MLD | 0.473 | 0.481 | 3.196 | 0.598 | ✗ |
| MotionDiffuse | 0.630 | 0.491 | 3.113 | 38.235 | ✓ |
| ReMoDiffuse | 0.103 | 0.510 | 2.974 | 0.959 | ✗ |
| **EMDM** | **0.112** | **0.498** | **3.110** | **0.280** | **✓** |

**Action-to-motion (HumanAct12)**

| 方法 | FID↓ | ACC↑ | 每帧时间(ms)↓ | 端到端 |
|------|------|------|-------------|--------|
| MDM | 0.100 | 0.990 | 41.154 | ✓ |
| MLD | 0.077 | 0.964 | 1.998 | ✗ |
| ACTOR | 0.120 | 0.955 | 0.523 | ✓ |
| **EMDM** | **0.084** | **0.991** | **0.337** | **✓** |

### 消融实验

**采样步数影响（HumanML3D）**：

| 采样步数 | FID↓ | R-Prec Top1↑ | 每帧时间(ms) |
|---------|------|-------------|------------|
| 1 (纯GAN) | 5.640 | 0.345 | 0.004 |
| 5 | 1.306 | 0.368 | 0.152 |
| 10 (默认) | 0.112 | 0.498 | 0.280 |
| 20 | - | - | - |
| 50 | - | - | - |

1 步采样退化为纯 GAN，质量大幅下降；10 步为最佳平衡点。

### 关键发现

1. **速度优势显著**：EMDM 在 HumanML3D 上每帧仅需 0.280ms，比 MDM (62.5ms) 快约 200 倍
2. **质量不妥协**：FID 仅 0.112，优于 MLD (0.473)，接近使用额外检索数据库的 ReMoDiffuse (0.103)
3. **几何损失至关重要**：没有几何损失时 GAN 训练不稳定，运动质量明显下降
4. **条件信号增强 GAN 表现**：条件控制信号提供了强先验，使得对复杂分布的建模更高效

## 亮点与洞察

1. **理论洞察深刻**：清楚揭示了大步长去噪分布变为非高斯的本质问题，并用 GAN 的隐式分布建模能力作为解决方案
2. **端到端训练**：相比 MLD 的两阶段方案，EMDM 的端到端训练流程大幅简化了实践应用
3. **条件 GAN + 扩散模型**的融合范式：不是简单套用 DDGAN，而是引入运动特定的条件信号和几何约束，使其适配运动生成任务
4. **实用性强**：实时生成能力使其可直接用于游戏、VR 等实时应用场景

## 局限与展望

1. 仅在相对较小的动作数据集上验证，未在大规模数据上测试泛化能力
2. 10 步采样虽已很快，但能否进一步压缩到 1-2 步而不损失质量值得探索
3. 判别器使用简单的 MLP，可能限制了对复杂运动分布的判别能力
4. text-to-motion 中未使用位置和足部接触损失（$\lambda=0$），可能存在进一步提升空间
5. 缺少对长序列运动生成的讨论

## 相关工作与启发

- **DDGAN**：本文的核心灵感来源，将扩散 GAN 范式从图像迁移到运动生成领域
- **MDM**：运动扩散模型的开创性工作，但 1000 步采样过慢
- **MLD**：通过潜空间扩散提速，但两阶段训练是瓶颈
- 启发：将 GAN 与扩散模型结合的思路可推广到其他需要快速采样的序列生成任务（如舞蹈生成、手势生成）

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 理论深度 | 4 |
| 实验充分性 | 4 |
| 实用价值 | 5 |
| 写作质量 | 4 |
| 总体评分 | 4.2 |

<!-- RELATED:START -->

## 相关论文

- [Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [Prompting Future Driven Diffusion Model for Hand Motion Prediction](prompting_future_driven_diffusion_model_for_hand_motion_prediction.md)

<!-- RELATED:END -->
