---
title: >-
  [论文解读] RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion
description: >-
  [ICLR 2026][图像生成][RNA 逆向设计] 提出 RIDER 框架，首次将强化学习引入 RNA 3D 逆向设计，先预训练条件扩散模型 RIDE 学习序列-结构关系，再用 RL 微调以直接优化 3D 结构相似性而非序列恢复率，在所有 3D 自一致性指标上实现超过 100% 的提升。
tags:
  - ICLR 2026
  - 图像生成
  - RNA 逆向设计
  - 3D 结构相似性
  - 扩散模型
  - 强化学习微调
  - DDPO
---

# RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion

**会议**: ICLR 2026  
**arXiv**: [2602.16548](https://arxiv.org/abs/2602.16548)  
**代码**: —  
**领域**: 生物分子设计 / 扩散模型 / 强化学习  
**关键词**: RNA 逆向设计, 3D 结构相似性, 扩散模型, 强化学习微调, DDPO

## 一句话总结

提出 RIDER 框架，首次将强化学习引入 RNA 3D 逆向设计，先预训练条件扩散模型 RIDE 学习序列-结构关系，再用 RL 微调以直接优化 3D 结构相似性而非序列恢复率，在所有 3D 自一致性指标上实现超过 100% 的提升。

## 研究背景与动机

RNA 逆向设计（给定目标 3D 结构，找到能折叠为该结构的核苷酸序列）是治疗药物和合成生物学的关键问题。

**现有方法的根本问题**：几乎所有 SOTA 方法（gRNAde、RiboDiffusion、RDesign 等）都优化**天然序列恢复率 (NSR)**作为代理目标。但 RNA 存在高度简并性——多个不同序列可折叠为相似结构，且相似序列不一定产生相似结构。因此：

1. NSR 与结构相似性无明显相关（在 NSR≈50% 时，GDT_TS 可从 0 变到 0.9）
2. 过度优化 NSR 限制了对非天然序列的探索

## 方法详解

### 整体框架

RIDER = RIDE（预训练扩散模型）+ RL 微调

### 阶段一：条件扩散模型 RIDE

**结构表示**：将 RNA 3D 骨架结构表示为几何图，节点为核苷酸，边编码空间邻近关系。用 GVP-GNN 编码器处理得到等变的节点嵌入 $\mathbf{h}_c$。

**扩散模型**：学习条件分布 $p(\mathbf{x}_0 | \mathbf{h}_c)$，其中 $\mathbf{x}_0 \in \{0,1\}^{N \times 4}$ 为独热编码序列。

前向过程：$\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \varepsilon$

训练目标：

$$\mathcal{L}_{\text{pretrain}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \varepsilon, \mathbf{h}_c}\left[\|\varepsilon - \epsilon_\theta(\alpha_t \mathbf{x}_0 + \sigma_t \varepsilon, t, \mathbf{h}_c)\|^2\right]$$

噪声预测网络由 5 层 GVP-GNN 组成，推理时使用 DDIM 采样器（50 步）。

### 阶段二：RL 微调

将去噪采样过程建模为 MDP：
- **状态** $s_t = (\mathbf{x}_t, t, \mathbf{h}_c)$
- **动作** $a_t$：从 $\mathbf{x}_t$ 到 $\mathbf{x}_{t-\Delta t}$ 的转移
- **策略** $\pi_\theta(a_t|s_t)$：由扩散模型参数化
- **奖励**：仅在轨迹末尾获得

**优势估计改进**：
1. 批量均值基线：$b = \mathbb{E}_\tau[R_{\text{traj}}]$
2. **滑动平均策略**稳定训练：$b^{(i)} = \beta_{\text{baseline}} \cdot b^{(i-1)} + (1-\beta_{\text{baseline}}) \cdot \bar{R}^{(i)}_{\text{batch}}$

策略梯度（带 PPO 裁剪）：

$$\mathcal{L}^{RL}(\theta) = \mathbb{E}\left[\sum_{k=0}^{N_{\text{steps}}-1}\min(r_k(\theta)A, \text{clip}(r_k(\theta), 1-\epsilon_{\text{clip}}, 1+\epsilon_{\text{clip}})A)\right]$$

### 奖励函数

基于三种 3D 结构相似性指标设计四种奖励函数：
- $R^{\text{gdt}} = (\text{GDT\_TS} \times w)^2$
- $R^{\text{tm}} = (\text{TM-score} \times w)^2$  
- $R^{\text{rmsd}} = -(\text{RMSD} \times w)^2$
- $R^{\text{gdt\_rmsd}}$：组合奖励（效果最好）

额外设计 $R_{\text{bonus}}$：当 GDT_TS > 0.5 或 RMSD < 2.0Å 时给予额外奖励。

## 实验

### 预训练结果

| 方法 | NSR ↑ |
|------|------|
| gRNAde | 50% |
| RiboDiffusion | 52% |
| **RIDE (Ours)** | **61%** |

### RL 微调结果

| 方法 | GDT_TS ↑ | RMSD ↓ | TM-score ↑ |
|------|----------|--------|-----------|
| gRNAde | 0.28 (27%) | 10.89 (3%) | 0.30 (28%) |
| RIDE (预训练) | 0.33 (31%) | 10.36 (8%) | 0.33 (36%) |
| **RIDER** ($R^{\text{tm}}$) | **0.62 (72%)** | 4.31 (31%) | **0.61 (72%)** |
| **RIDER** ($R^{\text{gdt\_rmsd}}$) | **0.62 (72%)** | **3.35 (33%)** | 0.56 (68%) |

百分比表示超过设计阈值的比例。RIDER 在所有指标上实现 100%+ 提升。

### 跨预测器验证

使用 AlphaFold3 替代 RhoFold 验证泛化性：RIDER 的 GDT_TS = 0.57，比 gRNAde (0.26) 提升 119%，证明框架捕获了可泛化的 RNA 设计原则。

### 关键发现

- NSR 确实与 3D 结构相似性无明显相关
- RL 微调后 NSR 通常降低，但 GDT_TS 提升，说明模型发现了不同于天然序列但折叠正确的新序列
- GDT_TS 和 TM-score 相关性高（Pearson 0.885），但各有侧重
- 组合奖励 $R^{\text{gdt\_rmsd}}$ 效果最均衡

## 亮点

- 首个面向 RNA 3D 逆向设计的 RL 框架，直接优化结构相似性
- 从数据和理论两方面证明了 NSR 作为代理目标的不足
- RL 微调策略（滑动平均基线 + PPO 裁剪）稳定有效
- 轻量模型（仅 10.2M 参数）即可取得显著效果

## 局限性

- 依赖 RhoFold 等结构预测模型作为折叠 oracle，其预测误差会传播
- RL 训练需要大量采样（每 epoch 60 条轨迹 × 80 epochs）
- 仅在 12,011 个 RNA 结构上训练和评估，数据规模有限
- 尚未进行实验验证（设计序列的湿实验验证）

## 相关工作

- **RNA 逆向设计**：gRNAde、RiboDiffusion、RDesign 等基于监督学习
- **RNA 结构预测**：RhoFold、AlphaFold3 等预测工具
- **RL 微调生成模型**：DDPO、RLHF、Constitutional AI 等

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 首个 RL 驱动的 RNA 3D 逆向设计
- 动机：⭐⭐⭐⭐⭐ — NSR 缺陷的分析清晰有力
- 实验：⭐⭐⭐⭐ — 多种奖励函数 + 跨 oracle 验证
- 影响力：⭐⭐⭐⭐ — 对 RNA 药物设计有重要意义

<!-- RELATED:START -->

## 相关论文

- [Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion](hierarchical_entity-centric_reinforcement_learning_with_factored_subgoal_diffusi.md)
- [Refining Few-Step Text-to-Multiview Diffusion via Reinforcement Learning](../../CVPR2026/image_generation/refining_few-step_text-to-multiview_diffusion_via_reinforcement_learning.md)
- [Structure-based RNA Design by Step-wise Optimization of Latent Diffusion Model](../../AAAI2026/image_generation/structure-based_rna_design_by_step-wise_optimization_of_latent_diffusion_model.md)
- [Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [Hierarchical Reinforcement Learning with Uncertainty-Guided Diffusional Subgoals](../../ICML2025/image_generation/hierarchical_reinforcement_learning_with_uncertainty-guided_diffusional_subgoals.md)

<!-- RELATED:END -->
