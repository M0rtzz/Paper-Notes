---
title: >-
  [论文解读] Piloting Structure-Based Drug Design via Modality-Specific Optimal Schedule
description: >-
  [ICML 2025][图像生成][基于结构的药物设计] 提出 VLB-Optimal Scheduling (VOS) 策略，通过理论分析多模态（连续3D位置+离散2D拓扑）联合噪声调度的路径依赖VLB特性，用动态规划搜索最优噪声调度路径，在 CrossDock 上以 95.9% PoseBusters 通过率达到 SBDD SOTA。
tags:
  - ICML 2025
  - 图像生成
  - 基于结构的药物设计
  - Bayesian Flow Network
  - 多模态噪声调度
  - VLB优化
  - 分子几何
---

# Piloting Structure-Based Drug Design via Modality-Specific Optimal Schedule

**会议**: ICML 2025  
**arXiv**: [2505.07286](https://arxiv.org/abs/2505.07286)  
**代码**: https://github.com/AlgoMole/MolCRAFT  
**领域**: 扩散模型 / 分子生成  
**关键词**: 基于结构的药物设计, Bayesian Flow Network, 多模态噪声调度, VLB优化, 分子几何

## 一句话总结
提出 VLB-Optimal Scheduling (VOS) 策略，通过理论分析多模态（连续3D位置+离散2D拓扑）联合噪声调度的路径依赖VLB特性，用动态规划搜索最优噪声调度路径，在 CrossDock 上以 95.9% PoseBusters 通过率达到 SBDD SOTA。

## 研究背景与动机
**领域现状**：基于结构的药物设计（SBDD）使用深度生成模型（如扩散模型、BFN）条件于蛋白质结构生成配体分子。

**现有痛点**：现有模型在 PoseBusters 严格测试下分子内有效性（键长/键角分布）显著下降，说明生成的 3D 几何结构与 2D 分子拓扑不一致。

**核心矛盾**：默认噪声调度导致 3D 模态（原子坐标）先被去噪，模型无法有效利用 2D 拓扑信息来约束 3D 结构——"3D 主导"的概率路径是次优的。

**本文目标** 如何设计多模态的最优噪声调度，使 2D 和 3D 模态能互相引导？

**切入角度**：证明在多模态情况下 VLB 是路径依赖的（与单模态不同），然后搜索最优路径。

**核心 idea**：多模态 VLB 不再是调度形状不变的，可通过解耦时间步训练+动态规划搜索找到最优的模态间调度路径。

## 方法详解

### 整体框架
基于 BFN 框架：(1) 训练阶段用解耦的 $(t_c, t_d)$ 独立采样训练广义损失，使一个模型覆盖所有可能的噪声组合；(2) 估计平面代价矩阵 $C(t_c, t_d)$；(3) 动态规划搜索最小累积代价路径，得到最优调度 $\boldsymbol{\beta}^*$。

### 关键设计

1. **路径依赖 VLB (Eq.8-10)**:

    - 功能：证明多模态 VLB 与噪声调度的耦合路径有关
    - 核心思路：单模态时 $\mathcal{L}^\infty$ 只依赖 $\beta(0), \beta(1)$（端点不变性）；多模态时改为路径积分 $\int_{\beta_c,\beta_d} \mathbb{E}\|\mathbf{x} - \tilde{\mathbf{x}}_\phi\|^2 d\boldsymbol{\beta}$，不同路径给出不同 VLB
    - 设计动机：这是设计最优调度的理论基础

2. **广义损失 + 解耦时间步训练 (Eq.14)**:

    - 功能：训练一个模型支持任意 $(t_c, t_d)$ 组合
    - 核心思路：$\dot{\mathcal{L}}^\infty = \frac{1}{2}\int_0^1\int_0^1 \mathbb{E}\|\mathbf{x} - \tilde{\mathbf{x}}_\phi(\boldsymbol{\theta}, \boldsymbol{t})\|^2 dt_c dt_d$，训练时独立采样 $t_c, t_d \sim U(0,1)$
    - 设计动机：一次训练即可在整个 $[0,1]^2$ 平面上评估损失，避免每个调度都重新训练

3. **动态规划搜索最优路径 (Algorithm 1)**:

    - 功能：在离散化的 $(t_c, t_d)$ 网格上找最小累积代价路径
    - 核心思路：$J(t_c,t_d) = \min_{(\epsilon_c,\epsilon_d)}(J(t_c-\epsilon_c, t_d-\epsilon_d) + \boldsymbol{\alpha}C(t_c,t_d))$，从 $[0,0]$ 到 $[1,1]$
    - 设计动机：穷举所有路径不可行，DP 保证在多项式时间内找到全局最优

### 最优调度的直觉
搜索到的最优路径呈两阶段特征：(1) 形状驱动的描画阶段 ($t<0.3$)：先生成大致的 3D 原子位置；(2) 拓扑驱动的对接阶段 ($t>0.8$)：根据 2D 分子拓扑调整 3D 构型。

## 实验关键数据

### 主实验
| 方法 | PB-Valid ↑ | Vina Dock ↓ | scRMSD<2Å ↑ |
|------|-----------|-------------|-------------|
| TargetDiff | 50.5% | -7.80 | 37.1% |
| DecompDiff | 71.7% | -7.03 | 24.2% |
| MolCRAFT (默认调度) | 84.3% | -7.90 | 38.7% |
| **MolPilot (VOS)** | **95.9%** | **-8.08** | **34.0%** |

### 消融实验
| 调度类型 | PB-Valid ↑ | 说明 |
|----------|-----------|------|
| 默认调度 | 84.3% | 3D 主导路径 |
| 线性插值 | ~90% | 均匀耦合 |
| **VOS 最优** | **95.9%** | 两阶段路径 |

### 关键发现
- VOS 相比默认调度提升 >10% PoseBusters 通过率
- 键长/键角分布图显示 MolPilot 最接近真实分布
- OOD (PoseBusters) 测试中性能保持鲁棒

## 亮点与洞察
- **理论驱动的调度设计**：从 VLB 路径依赖性出发，而非启发式试错
- **一次训练，多调度评估**：解耦时间步的广义损失设计非常优雅
- **可迁移思路**：VOS 方法可推广到任何多模态生成任务的噪声调度设计

## 相关工作与启发
- **vs TargetDiff/DecompDiff (扩散)**: 这些方法使用固定噪声调度，未考虑多模态间的调度耦合；MolPilot 通过 VOS 找到最优耦合路径
- **vs MolCRAFT (BFN)**: MolPilot 直接基于 MolCRAFT 框架，VOS 可视为对 BFN 的噪声调度优化
- **vs EquiFM**: EquiFM 用信息论启发式设计混合路径，缺乏理论最优性保证；VOS 有 VLB 最优性的理论基础
- VOS 的思路可直接迁移到其他多模态生成问题（如文本+图像联合生成中文本和图像的噪声调度设计）

## 局限与展望
- DP 搜索在更多模态（>2）时复杂度指数增长，需要近似算法
- 最优调度的可解释性仍需更多理论分析（为什么两阶段路径最优？）
- 分子对接任务上 RMSD<2Å 仅 44%，仍有很大提升空间
- 训练广义损失（解耦时间步）可能比标准训练收敛更慢
- 未验证 VOS 在蛋白质设计等其他分子生成任务上的效果

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 多模态 VLB 路径依赖性是重要理论发现
- 实验充分度: ⭐⭐⭐⭐ CrossDock + PoseBusters OOD 验证
- 写作质量: ⭐⭐⭐⭐ 理论+实验结合紧密
- 价值: ⭐⭐⭐⭐⭐ 对 SBDD 和多模态生成有广泛意义

<!-- RELATED:START -->

## 相关论文

- [Structure-based RNA Design by Step-wise Optimization of Latent Diffusion Model](../../AAAI2026/image_generation/structure-based_rna_design_by_step-wise_optimization_of_latent_diffusion_model.md)
- [Improved Noise Schedule for Diffusion Training](../../ICCV2025/image_generation/improved_noise_schedule_for_diffusion_training.md)
- [ETTA: Elucidating the Design Space of Text-to-Audio Models](etta_elucidating_the_design_space_of_text-to-audio_models.md)
- [Compositional Flows for 3D Molecule and Synthesis Pathway Co-design](compositional_flows_for_3d_molecule_and_synthesis_pathway_co-design.md)
- [From Elements to Design: A Layered Approach for Automatic Graphic Design Composition](../../CVPR2025/image_generation/from_elements_to_design_a_layered_approach_for_automatic_graphic_design_composit.md)

<!-- RELATED:END -->
