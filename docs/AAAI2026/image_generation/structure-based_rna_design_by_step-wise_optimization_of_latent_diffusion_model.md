---
title: >-
  [论文解读] Structure-based RNA Design by Step-wise Optimization of Latent Diffusion Model
description: >-
  [AAAI 2026][图像生成][潜在扩散模型] 提出SOLD框架，将潜在扩散模型（LDM）与强化学习（RL）结合，通过步进式单步采样优化策略，直接优化RNA逆折叠中不可微的结构指标（二级结构相似度SS、最小自由能MFE、LDDT），在多个指标上全面超越现有方法。
tags:
  - "AAAI 2026"
  - "图像生成"
  - "潜在扩散模型"
  - "RNA逆折叠"
  - "强化学习"
  - "PPO"
  - "步进优化"
  - "RNA-FM"
---

# Structure-based RNA Design by Step-wise Optimization of Latent Diffusion Model

**会议**: AAAI 2026  
**arXiv**: [2601.19232](https://arxiv.org/abs/2601.19232)  
**代码**: [有](https://github.com/darkflash03/SOLD)  
**领域**: 扩散模型 / RNA逆折叠 / 强化学习  
**关键词**: 潜在扩散模型, RNA逆折叠, 强化学习, PPO, 步进优化, RNA-FM  

## 一句话总结

提出SOLD框架，将潜在扩散模型（LDM）与强化学习（RL）结合，通过步进式单步采样优化策略，直接优化RNA逆折叠中不可微的结构指标（二级结构相似度SS、最小自由能MFE、LDDT），在多个指标上全面超越现有方法。

## 研究背景与动机

RNA逆折叠任务旨在设计能折叠成特定3D结构的RNA序列，在RNA疗法、基因调控和合成生物学中有重要应用。现有方法存在以下局限：

**物理方法（如Rosetta）**：基于蒙特卡洛优化，计算开销大且难以处理多态构象

**基于2D结构的方法（如ViennaRNA）**：忽视了关键的3D几何信息

**深度学习方法（如RhoDesign, RDesign）**：基于VAE的方法难以建模复杂分布和长程依赖

**已有扩散方法（如RiboDiffusion）**：在序列空间直接做SDE扩散，未能充分利用RNA的共进化信息

**核心痛点**：现有方法只优化序列恢复率，无法直接优化SS、MFE、LDDT等不可微的结构目标

## 方法详解

### 整体框架

SOLD分两阶段：(1) 预训练LDM学习序列生成能力；(2) 用RL微调LDM直接优化结构指标。整体架构包含MLP编码器、GVP-GNN + DiT去噪网络、MLP解码器三部分。

### 关键设计一：基于RNA-FM的潜在扩散模型

不同于RiboDiffusion直接在one-hot序列空间$(L,4)$做扩散，SOLD利用预训练RNA-FM模型提取嵌入$(L, 640)$，通过MLP编码器压缩到潜在空间$(L, 32)$后进行扩散。RNA-FM嵌入编码了RNA的共进化模式和结构信息，显著提升序列恢复率。消融实验表明$D=32$是生成质量和效率的最优平衡点（$D=32$时生成性能最佳，$D>32$反而下降）。

去噪网络由4层GVP-GNN（处理骨架几何特征）和8层DiT Transformer（捕捉序列依赖）组成，预测清洁潜在嵌入$\hat{z}_0$。

### 关键设计二：步进式RL优化

SOLD的核心创新在于不需要像DDPO/DPOK那样采样完整去噪轨迹，而是：
- 随机采样一个时间步$t$，从噪声$z_t$通过DDIM单步预测直接得到$z_0'$
- 将$z_0'$解码为序列$s_0'$，计算**长期奖励** $r_0(t) = R_i(s_0')$
- 同时用DDPM单步采样得到$z_{t-1}'$，解码计算**短期奖励** $r_t(t) = R_i(s_{t-1}')$
- 通过分段奖励函数融合两者：大$t$（早期去噪）用短期奖励，小$t$（后期去噪）用长期奖励

这种设计带来了巨大的速度优势，单轮训练时间仅为DDPO的4%。

### 关键设计三：分段奖励策略

$$r_{\text{total}}(t) = w(t) \cdot r_t(t) + u(t) \cdot r_0(t)$$

其中阈值$\tau$控制切换点。消融实验表明不同目标的最优$\tau$不同：MFE和LDDT适合$\tau=90$（长期奖励主导90步），SS适合$\tau=60$。这种分段策略的直觉是：早期去噪步骤噪声大，直接预测$z_0'$不准确，用短期奖励更可靠；后期噪声小，长期奖励更有效。

### 损失函数

LDM预训练阶段使用MSE + 交叉熵联合损失：

$$\mathcal{L} = \mathbb{E}_{z_0,t}[\|z_0 - \hat{z}_0\|^2] - \mathbb{E}_s[\sum_{i=1}^L \log p(s_i | \text{Dec}(\hat{z}_0)_i)]$$

RL阶段使用PPO + KL正则化优化，clip范围$\epsilon=0.0001$，KL权重$\lambda_{\text{ref}}$控制与参考策略的偏离程度。

## 实验

### 主实验：多目标优化性能（Table 4）

| 方法 | Seq Recovery↑ | MFE↓ | SS↑ | RMSD↓ | LDDT↑ |
|------|-------------|------|-----|-------|-------|
| RhoDesign | 0.2734 | -11.92 | 0.650 | 16.36 | 0.503 |
| RDesign | 0.4457 | -10.70 | 0.614 | 16.13 | 0.524 |
| gRNAde | 0.5108 | -10.54 | 0.562 | 18.00 | 0.485 |
| RiboDiffusion | 0.5125 | -15.21 | 0.763 | 12.32 | 0.610 |
| DRAKES | 0.4400 | -14.24 | 0.769 | 11.91 | 0.619 |
| LDM (baseline) | 0.5728 | -13.33 | 0.727 | 12.57 | 0.618 |
| **SOLD** | **0.5732** | **-16.86** | **0.760** | **11.86** | **0.636** |

SOLD在保持序列恢复率的同时，MFE降低了26.5%，LDDT提升了2.9%，RMSD降低了5.7%。

### 消融实验：训练效率对比（Table 3）

| 方法 | MFE训练时间(s) | SS训练时间(s) | LDDT训练时间(s) |
|------|-------------|-------------|--------------|
| DDPO | 5953 | 6190 | 14000 |
| DPOK | 7677 | 7330 | 14200 |
| **SOLD** | **256** | **263** | **6900** |

SOLD在MFE和SS优化上比DDPO快**23倍**，比DPOK快**30倍**。

### 关键发现

1. **RNA-FM嵌入的优势**：LDM在潜在空间做扩散比RiboDiffusion在序列空间做扩散，序列恢复率从0.5125提升到0.5728（+11.8%）
2. **单步采样的效率**：步进优化避免了完整轨迹采样，训练速度提升超20倍
3. **分段奖励的必要性**：纯长期或纯短期奖励都不如混合策略（MFE: -19.74 vs -17.24/-17.73）
4. **实际案例验证**：在TPP核糖开关设计（PDB: 3D2V）上，仅SOLD成功设计出能折叠到目标结构的序列，其他方法均失败

## 亮点

1. **首次将RL引入潜在扩散模型进行RNA逆折叠**：填补了扩散模型在RNA设计中无法优化不可微结构指标的空白
2. **步进优化策略极具创新性**：独立优化每个去噪步骤而非采样完整轨迹，理论上与DDPO目标等价但效率提升20倍以上
3. **多目标同时优化**：无需可微奖励模型，直接调用ViennaRNA评估，避免了DRAKES所需的额外奖励模型训练
4. **强实验设计**：全面的消融实验（潜在维度、分段奖励策略、不同序列长度）和统计显著性检验

## 局限性

1. **数据稀缺**：高质量RNA 3D结构数据有限（仅7067条训练数据），制约了模型泛化能力
2. **多指标协同未充分探索**：1D/2D/3D指标间的交互关系和最优权重配比尚不清楚
3. **奖励工具的近似误差**：ViennaRNA和RhoFold本身预测存在误差，可能影响优化精度
4. **长序列性能下降**：在长序列（128-512nt）上各指标均有显著下降，复杂RNA折叠仍是挑战
5. **生成采样仍需完整轨迹**：训练时用单步采样加速，但推理时仍需100步完整去噪

## 相关工作

| 方法 | 扩散空间 | RL微调 | 奖励模型 | 优化目标 |
|------|---------|--------|---------|---------|
| RiboDiffusion | 序列空间(SDE) | ✗ | - | 序列恢复 |
| DRAKES | 离散扩散 | ✓(单目标) | 需训练 | MFE |
| RNAdiffusion | 潜在空间 | ✗ | 需训练 | 翻译效率 |
| DDPO/DPOK | 连续空间 | ✓(完整轨迹) | 无 | 图像质量 |
| **SOLD** | **潜在空间(LDM)** | **✓(步进单步)** | **无(直接评估)** | **SS+MFE+LDDT** |

## 评分

- **新颖性**: ⭐⭐⭐⭐ (步进RL优化+潜在扩散的结合原创性强)
- **技术贡献**: ⭐⭐⭐⭐⭐ (完整的框架设计+理论推导+充分消融)
- **实验充分度**: ⭐⭐⭐⭐⭐ (SOLD TEST + CASP15双数据集、多指标、统计检验)
- **写作质量**: ⭐⭐⭐⭐ (结构清晰，数学推导完整)
- **实际影响力**: ⭐⭐⭐⭐ (RNA药物设计领域有实际应用潜力)
- **综合推荐**: ⭐⭐⭐⭐ (4.5/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion](../../ICLR2026/image_generation/rider_3d_rna_inverse_design_with_reinforcement_learning-guided_diffusion.md)
- [\[AAAI 2026\] Steering One-Step Diffusion Model with Fidelity-Rich Decoder for Fast Image Compression](steering_one-step_diffusion_model_with_fidelity-rich_decoder_for_fast_image_comp.md)
- [\[ICML 2025\] PPO-MI: Efficient Black-Box Model Inversion via Proximal Policy Optimization](../../ICML2025/image_generation/ppo-mi_efficient_black-box_model_inversion_via_proximal_policy_optimization.md)
- [\[ICLR 2026\] Latent Diffusion Model without Variational Autoencoder](../../ICLR2026/image_generation/latent_diffusion_model_without_variational_autoencoder.md)
- [\[AAAI 2026\] Realism Control One-step Diffusion for Real-World Image Super-Resolution](realism_control_one-step_diffusion_for_real-world_image_super-resolution.md)

</div>

<!-- RELATED:END -->
