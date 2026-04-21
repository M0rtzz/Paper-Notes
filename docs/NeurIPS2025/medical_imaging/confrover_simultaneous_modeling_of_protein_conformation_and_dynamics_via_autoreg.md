---
title: >-
  [论文解读] ConfRover: Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression
description: >-
  [NeurIPS 2025][医学图像][protein dynamics] ConfRover 提出自回归框架将蛋白质 MD 轨迹分解为逐帧条件生成 $p(\mathbf{x}^{1:L}) = \prod_l p(\mathbf{x}^l | \mathbf{x}^{<l})$，通过编码器 + 因果 Transformer + SE(3) 扩散解码器的模块化架构，首次在单一模型中统一轨迹模拟、时间无关构象采样和构象插值三大任务，在 ATLAS 数据集上全面超越 MDGen。
tags:
  - NeurIPS 2025
  - 医学图像
  - protein dynamics
  - autoregressive generation
  - 扩散模型
  - conformational sampling
  - molecular dynamics
---

# ConfRover: Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression

**会议**: NeurIPS 2025  
**arXiv**: [2505.17478](https://arxiv.org/abs/2505.17478)  
**项目**: https://bytedance-seed.github.io/ConfRover  
**机构**: ByteDance Seed, 同济大学, 清华大学  
**领域**: 蛋白质构象建模 / 分子动力学  
**关键词**: protein dynamics, autoregressive generation, SE(3) diffusion, conformational sampling, molecular dynamics

## 一句话总结

ConfRover 提出自回归框架将蛋白质 MD 轨迹分解为逐帧条件生成 $p(\mathbf{x}^{1:L}) = \prod_l p(\mathbf{x}^l | \mathbf{x}^{<l})$，通过编码器 + 因果 Transformer + SE(3) 扩散解码器的模块化架构，首次在单一模型中统一轨迹模拟、时间无关构象采样和构象插值三大任务，在 ATLAS 数据集上全面超越 MDGen。

## 研究背景与动机

**领域现状**：蛋白质构象动力学对理解其生物功能至关重要。分子动力学 (MD) 模拟是研究构象变化的"金标准"，但计算成本极高且容易陷入局部能量极小值。大规模 MD 数据集（如 ATLAS）的出现催生了深度生成模型替代方案。

**现有方法三条割裂路线**：(a) 轨迹模拟方法（MDGen、Timewarp 等）捕获时序依赖但无法直接采样时间无关构象；(b) 构象分布学习方法（AlphaFlow、ConfDiff、BioEmu）能采样独立构象但完全忽视时序信息；(c) 构象插值方法需要训练独立模型且仅在小蛋白质上验证过。

**核心矛盾**：三类任务本质上都源于同一物理原理——蛋白质构象空间采样，但现有方法各自为政。MDGen 非自回归设计限制了灵活性（固定长度、任务需独立模型），GST 虽采用自回归但为确定性预测、无法捕获轨迹分布。

**关键观察**：对 MD 轨迹 $\mathbf{x}^{1:L}$ 采用自回归分解后，不同条件设置自然对应不同任务——全条件生成 = 轨迹模拟，无条件 ($L=1$) = 独立采样，前后端点条件 = 插值。这一统一视角来源于序列模型在文本 infilling 中的类似思路。

## 方法详解

### 整体框架

**输入**：蛋白质序列信息 $\mathcal{P}$ + MD 轨迹帧序列 $\mathbf{x}^{1:L}$。  
**输出**：新的构象帧（轨迹延续 / 独立样本 / 插值中间帧）。  
**三阶段 Pipeline**：(1) 编码层将每帧构象映射为潜表示 $\mathbf{h}^l = [\mathbf{s}^l, \mathbf{z}^l]$；(2) 轨迹模块通过因果 Transformer 更新潜表示以编码时序依赖；(3) SE(3) 扩散解码器从更新后的潜表示生成 3D 构象。

### 核心设计 1：自回归建模统一多任务

将 MD 轨迹联合分布分解为：

$$p(\mathbf{x}^{1:L}|\mathcal{P}) = \prod_{l=1}^{L} p(\mathbf{x}^l | \mathbf{x}^{<l}, \mathcal{P})$$

- **轨迹模拟**：以前序帧为条件，逐帧生成后续构象，天然支持非马尔可夫动力学
- **独立采样**：当 $L=1$ 时退化为 $p(\mathbf{x}|\mathcal{P})$，无前序帧条件
- **构象插值**：将终帧前置到起始帧前重定义依赖结构，类似文本 infilling 的序列重排技巧
- **优势**：与 MDGen 固定长度+非自回归相比，自回归天然支持可变长度和灵活条件化

### 核心设计 2：潜空间因果建模

编码器 $f_\eta^{\text{enc}}$ 将每帧构象编码为潜表示，因果注意力模块 $f_\xi^{\text{temp}}$ 仅让帧 $l$ 关注前序帧：

$$\mathbf{h}^i = f_\eta^{\text{enc}}(\mathbf{x}^i, \mathcal{P}), \quad \mathbf{h}^l_{\text{updated}} = f_\xi^{\text{temp}}(\mathbf{h}^1, \dots, \mathbf{h}^{l-1})$$

由于编码器和时序模块均为确定性映射，$p(\mathbf{x}^l | \mathbf{x}^{<l})$ 简化为对更新后潜表示的条件生成 $p_\theta^{\text{dec}}(\mathbf{x}^l | \mathbf{h}^l_{\text{updated}})$。禁用帧间注意力（identity attention）即切换为独立采样模式，思路类似视频生成中的图像-视频联合训练。

### 核心设计 3：SE(3) 扩散解码器

避免将蛋白质结构离散化为 token（VQ-VAE 方式的离散化误差），直接在 SE(3) 连续流形上建模。以 ConfDiff 为基础，分别对平移和旋转定义前向扩散核，训练 DSM 损失：

$$\mathcal{L}_{\text{DSM}}^{\text{SE(3)}} = \mathbb{E}\left[\lambda(t)\|s_\theta(\mathbf{T}_t^l, \mathbf{h}^l, t) - \nabla_{\mathbf{T}_t^l}\log p_{t|0}(\mathbf{T}_t^l|\mathbf{T}_0^l)\|^2\right] + \text{旋转项}$$

梯度通过 $\mathbf{h}^l$ 回传更新时序模块和编码器权重。推理时从 SE(3) 先验噪声逆扩散生成干净构象。

### 架构细节

- **编码层**：冻结 OpenFold（3轮 recycle）提取共享蛋白质表示 $\mathcal{P} = [\mathbf{s}, \mathbf{z}]$。FrameEncoder 从伪-Cβ 原子成对距离出发，通过三角更新编码帧级 pair representation。掩码帧 "[M]" 通过将 Cβ 距离置零实现
- **轨迹模块**：由 StructuralUpdate（Pairformer 三角运算更新 single/pair embedding）和 TemporalUpdate（Llama 因果 Transformer + RoPE，channel-wise 独立注意力）交替组成
- **结构解码器**：ConfDiff 架构（IPA + Transformer 层），预测残基 SE(3) rigid；额外用 AngleResNet 预测 7 个扭转角恢复全原子坐标

### 训练策略

- **混合训练**：基础版轨迹+单帧 1:1 比例；ConfRover-interp 版加入插值目标 1:1:1
- **多时间尺度**：$L=8$ 帧轨迹，步幅 1~1024 MD 快照（间隔 10 ps），覆盖多种动力学尺度
- **初始化**：DiffusionDecoder 从 ConfDiff 预训练权重初始化，FoldingModule 冻结

## 实验与结果

### 数据集

ATLAS：约 1300 种蛋白质，每种 3 条 100 ns 模拟轨迹，按蛋白质 identity 划分训练/测试集。

### 轨迹模拟 — Multi-Start 基准（Pearson 相关性）

| 指标空间 | 指标 | MDGen | ConfRover |
|----------|------|-------|-----------|
| Cα 坐标 | Trajectory | 0.56±0.03 | **0.75±0.01** |
| Cα 坐标 | Frame | 0.47±0.03 | **0.63±0.01** |
| Cα 坐标 | ΔFrame | 0.41±0.02 | **0.53±0.01** |
| PCA 2D | Trajectory | 0.18±0.01 | **0.73±0.01** |
| PCA 2D | Frame | 0.15±0.01 | **0.50±0.01** |

PCA 空间提升尤为显著（+306%），表明自回归更好捕获了与主要结构变异方向一致的构象变化。

### 100ns 长轨迹 — 构象状态恢复

| 方法 | JSD ↓ | Recall ↑ | F1 ↑ |
|------|-------|----------|------|
| MD 100ns (oracle) | 0.31 | 0.67 | 0.79 |
| MDGen | 0.56±0.01 | 0.29±0.01 | 0.42±0.01 |
| **ConfRover** | **0.51±0.01** | **0.42±0.00** | **0.58±0.00** |

ConfRover 在 tICA 主动力学模式恢复上表现与 MD oracle 可比，显著优于 MDGen。部分蛋白质（如 7NMQ-A）中 MD 能跨越能量壁垒达到更远构象状态，ConfRover 尚未完全做到。

### 时间无关构象采样

| 方法 | Pairwise RMSD r↑ | RMSF r↑ | RMWD↓ | MD PCA $\mathcal{W}_2$↓ |
|------|-------------------|---------|-------|-------------|
| AlphaFlow | **0.56** | **0.85** | 2.62 | 1.52 |
| ConfDiff | 0.54 | **0.85** | 2.70 | **1.44** |
| ConfRover | 0.51 | **0.85** | **2.66** | 1.47 |
| ConfRover-traj | 0.48 | 0.84 | 2.85 | 1.43 |

作为通用模型，ConfRover 在独立采样上与专用模型 AlphaFlow/ConfDiff 可比，8 项指标中 5 项优于至少一个 SOTA。

### 构象插值

ConfRover-interp 生成的中间帧与起始帧距离单调递增、与终帧距离单调递减，形成平滑过渡路径。而未经插值训练的基础 ConfRover 无法趋向终态。可视化显示中间构象与 MD 参考的过渡路径高度一致。

### 构象质量（MolProbity + MadraX 能量）

| 方法 | Rama outliers%↓ | Rotamer outliers%↓ | MolProbity↓ | MadraX 能量↓ |
|------|-----------------|---------------------|-------------|-------------|
| MD Reference | 0.38 | 1.02 | 0.72 | -519.3 |
| MDGen | 0.93 | 2.86 | 2.24 | -314.7 |
| **ConfRover** | **0.58** | **1.98** | **1.72** | **-522.2** |

ConfRover 生成构象的几何质量和能量水平均接近 MD 参考，远优于 MDGen。

### 消融：混合训练的重要性

去掉单帧训练目标后（ConfRover-traj），RMWD 从 2.66 升至 2.85，Pairwise RMSD 相关性从 0.51 降至 0.48，证实混合训练对平衡多任务学习至关重要。

## 亮点与洞察

- **统一框架设计优雅**：通过改变条件设置和序列顺序，一个模型服务三种任务。插值通过序列重排实现（终帧前置），无需修改架构，灵感源于文本 infilling
- **连续空间自回归**：用 SE(3) 扩散替代离散 tokenization 作为自回归模型的输出头，类似图像领域 MAR 的思路，但扩展到 SE(3) 流形
- **Llama 架构迁移**：将大语言模型的因果 Transformer 引入蛋白质动力学，channel-wise attention 处理 single/pair embedding，计算效率高
- **可迁移性**：混合训练策略和序列重排实现插值的技巧可迁移到 RNA 动力学、小分子构象等时序结构生成任务

## 局限性

- 轨迹模拟仅与 MDGen 对比（AlphaFolding 等权重不公开），baseline 有限
- ATLAS 仅 100ns 单链蛋白质模拟，无法覆盖大构象变化或蛋白质复合物
- Pairformer 三角更新计算昂贵，限制大蛋白质和长轨迹的可扩展性
- 与 MD oracle 仍有差距，尤其在跨越能量壁垒探索远端构象状态方面

## 相关工作对比

| 方法 | 轨迹模拟 | 独立采样 | 插值 | 可变长度 | 概率生成 | 跨蛋白质泛化 |
|------|---------|---------|------|---------|---------|------------|
| MDGen | ✓ | ✗ | 需独立模型 | ✗ | ✓ | ✓ |
| AlphaFlow/ConfDiff | ✗ | ✓ | ✗ | - | ✓ | ✓ |
| GST | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| **ConfRover** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

## 评分

- 新颖性: ⭐⭐⭐⭐ 自回归统一多任务的 framing 优雅，但各模块为已有技术组合
- 实验充分度: ⭐⭐⭐⭐ 三种任务+消融+质量评估全面，但 baseline 数量有限
- 写作质量: ⭐⭐⭐⭐⭐ 叙事清晰，从观察到方法推导自然，图表设计优秀
- 综合价值: ⭐⭐⭐⭐ 首个统一蛋白质构象+动力学的生成框架，对计算生物学领域有重要推动

<!-- RELATED:START -->

## 相关论文

- [Aligning Protein Conformation Ensemble Generation with Physical Feedback](../../ICML2025/medical_imaging/aligning_protein_conformation_ensemble_generation_with_physical_feedback.md)
- [Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)
- [Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)
- [Modeling Spatiotemporal Neural Frames for High Resolution Brain Dynamics](../../CVPR2026/medical_imaging/modeling_spatiotemporal_neural_frames_for_high_resolution_brain_dynamic.md)

<!-- RELATED:END -->
