---
title: >-
  [论文解读] Rethinking Direct Preference Optimization in Diffusion Models
description: >-
  [SPIGM@NeurIPS 2025 / AAAI 2026 (Oral)][LLM对齐][扩散模型] 针对扩散模型中 DPO 的两个核心问题——有限探索和奖励尺度不平衡,提出稳定参考模型更新策略和时间步感知训练策略,可集成到各种偏好优化算法中。
tags:
  - SPIGM@NeurIPS 2025 / AAAI 2026 (Oral)
  - LLM对齐
  - 扩散模型
  - DPO
  - 偏好优化
  - 参考模型更新
  - 时间步感知
---

# Rethinking Direct Preference Optimization in Diffusion Models

**会议**: SPIGM@NeurIPS 2025 / AAAI 2026 (Oral)

**arXiv**: [2505.18736](https://arxiv.org/abs/2505.18736)

**代码**: [GitHub](https://github.com/kaist-cvml/RethinkingDPO_Diffusion_Models)

**领域**: LLM对齐 / 扩散模型

**关键词**: 扩散模型, DPO, 偏好优化, 参考模型更新, 时间步感知

## 一句话总结

针对扩散模型中 DPO 的两个核心问题——有限探索和奖励尺度不平衡,提出稳定参考模型更新策略和时间步感知训练策略,可集成到各种偏好优化算法中。

## 研究背景与动机

将文本到图像（T2I）扩散模型与人类偏好对齐是当前的热门研究方向。虽然 DPO 等偏好优化技术已从 LLM 扩展到扩散模型,但存在特有的挑战：

**有限探索**: 冻结的参考模型限制了策略的探索空间,导致生成多样性不足

**奖励尺度不平衡**: 不同去噪时间步的奖励信号量级差异巨大,影响训练稳定性

**扩散特有困难**: 与 LLM 不同,扩散模型的生成过程涉及多步去噪,每步的优化目标不同

## 方法详解

### 整体框架

提出两个正交的改进策略,可与现有的扩散 DPO/DPOK/Diffusion-DPO 等方法结合使用。

### 关键设计

**1. 稳定参考模型更新策略 (Stable Reference Model Update)**

标准 DPO 中参考模型 $\pi_{\text{ref}}$ 固定不变,限制了探索：
- 提出**渐进更新**参考模型: $\pi_{\text{ref}}^{(t+1)} = (1-\alpha) \pi_{\text{ref}}^{(t)} + \alpha \pi_\theta^{(t)}$
- 参考模型正则化: 约束更新后的参考模型不偏离初始模型太远
- 平衡探索与稳定性: $\alpha$ 小时保守,$\alpha$ 大时激进

$$\mathcal{L}_{\text{reg}} = \text{KL}(\pi_{\text{ref}}^{(t)} \| \pi_{\text{ref}}^{(0)})$$

**2. 时间步感知训练策略 (Timestep-Aware Training)**

扩散模型在不同时间步的信噪比差异巨大:
- 发现: 大时间步（高噪声）的奖励信号幅度远大于小时间步（低噪声）
- 这导致训练被大时间步主导,小时间步学习不足
- 解决方案: 对不同时间步的损失进行**归一化加权**

$$\mathcal{L}_t = \frac{w(t)}{\sum_t w(t)} \mathcal{L}_{\text{DPO}}^{(t)}$$

其中 $w(t)$ 是基于时间步 $t$ 处奖励信号方差的归一化权重。

### 损失函数 / 训练策略

完整损失:
$$\mathcal{L} = \mathcal{L}_{\text{DPO}} + \lambda_1 \mathcal{L}_{\text{reg}} + \lambda_2 \mathcal{L}_{\text{timestep-norm}}$$

训练流程: 在标准 DPO 训练循环中添加参考模型更新和时间步归一化。

## 实验关键数据

### 主实验

SDXL 上的人类偏好评估 (Pick-a-Pic, HPSv2):

| 方法 | HPSv2 ↑ | Pick Score ↑ | Aesthetic ↑ | CLIP Score |
|------|---------|-------------|------------|-----------|
| SDXL (基线) | 27.5 | 21.8 | 5.82 | 0.315 |
| Diffusion-DPO | 28.2 | 22.3 | 5.95 | 0.318 |
| Diffusion-DPO + Ours | **28.8** | **22.9** | **6.12** | **0.322** |
| D3PO | 28.0 | 22.1 | 5.90 | 0.316 |
| D3PO + Ours | **28.5** | **22.6** | **6.05** | **0.320** |

SD1.5 上的结果:

| 方法 | HPSv2 ↑ | Aesthetic ↑ | 多样性 (FID) |
|------|---------|------------|-------------|
| SD1.5 | 25.8 | 5.45 | 15.2 |
| Diffusion-DPO | 26.5 | 5.68 | 18.5 |
| Diffusion-DPO + Ours | **27.1** | **5.85** | **16.8** |

### 消融实验

两个策略的独立贡献 (SDXL, HPSv2):

| 配置 | HPSv2 | 改善 |
|------|-------|------|
| Diffusion-DPO (基线) | 28.2 | - |
| + 参考模型更新 | 28.5 | +0.3 |
| + 时间步感知 | 28.5 | +0.3 |
| + 两者结合 | **28.8** | **+0.6** |

### 关键发现

1. 两个策略贡献互补,各自提供约 +0.3 的 HPSv2 改善
2. 参考模型更新在训练后期效果更明显,随着策略偏离初始模型
3. 时间步感知在低噪声步改善最大，因为这些步决定了细节质量
4. 方法可无缝集成到 Diffusion-DPO、D3PO 等多种算法中

## 亮点与洞察

- **正交改进**: 两个策略相互独立,可分别或联合使用
- **通用性**: 可作为即插即用模块集成到任何扩散偏好优化方法中
- **实际洞察**: 时间步奖励不平衡是扩散 DPO 中被忽视但重要的问题

## 局限与展望

1. $\alpha$ 的最优值依赖于具体任务和模型
2. 参考模型更新增加了内存需求（需要存储额外模型参数）
3. 仅在图像生成上验证,视频生成场景未探索
4. 时间步归一化权重的设计缺乏自适应性

## 相关工作与启发

- **Diffusion-DPO** (Wallace et al.): 将 DPO 扩展到扩散模型的工作
- **D3PO**: 另一种扩散偏好优化方法
- **Online DPO**: LLM 中参考模型更新的相关工作

## 评分

- ⭐ 创新性: 7/10 — 两个改进虽有效但思路相对直接
- ⭐ 实用性: 8/10 — 即插即用,开源代码,实用价值高
- ⭐ 写作质量: 8/10 — 消融分析清晰,实验设计合理

<!-- RELATED:START -->

## 相关论文

- [Curriculum Direct Preference Optimization for Diffusion and Consistency Models](../../CVPR2025/llm_alignment/curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [ADHMR: Aligning Diffusion-based Human Mesh Recovery via Direct Preference Optimization](../../ICML2025/llm_alignment/adhmr_aligning_diffusion-based_human_mesh_recovery_via_direct_preference_optimiz.md)
- [DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution](dp2o-sr_direct_perceptual_preference_optimization_for_real-world_image_super-res.md)
- [DiffPO: Diffusion Alignment with Direct Preference Optimization](../../ACL2025/llm_alignment/diffpo_diffusion_alignment.md)
- [DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models](densedpo_finegrained_temporal_preference_optimization_for_vi.md)

<!-- RELATED:END -->
