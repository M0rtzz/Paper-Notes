---
title: >-
  [论文解读] Towards a Foundation Model for Partial Differential Equations Across Physics Domains
description: >-
  [AAAI 2026][科学计算][偏微分方程] 提出 PDE-FM，一个结合空间-频谱 tokenization、物理感知 FiLM 调制和 Mamba 状态空间 backbone 的模块化 PDE foundation model，在 The Well 基准的 12 个跨物理域数据集上平均降低 VRMSE 46%。
tags:
  - AAAI 2026
  - 科学计算
  - 偏微分方程
  - foundation model
  - 神经算子
  - Mamba
  - FNO
  - multi-physics
  - The Well benchmark
---

# Towards a Foundation Model for Partial Differential Equations Across Physics Domains

**会议**: AAAI 2026  
**arXiv**: [2511.21861](https://arxiv.org/abs/2511.21861)  
**代码**: 待确认  
**领域**: scientific_computing  
**关键词**: PDE, foundation model, neural operator, Mamba, FNO, multi-physics, The Well benchmark

## 一句话总结

提出 PDE-FM，一个结合空间-频谱 tokenization、物理感知 FiLM 调制和 Mamba 状态空间 backbone 的模块化 PDE foundation model，在 The Well 基准的 12 个跨物理域数据集上平均降低 VRMSE 46%。

## 背景与动机

现有 neural operator（FNO、Transformer-based 等）是领域特定的——在单一数据集上训练，仅适用于窄类 PDE，边界条件或物理规律变化时性能骤降。这与 NLP/Vision 中 foundation model 的"一次预训练，多任务迁移"范式形成鲜明对比。物理系统的独特挑战：多分辨率多尺度、守恒定律约束、连续时空演化、非线性算子耦合。

## 核心问题

如何设计一个统一的 foundation model 架构，在异构 PDE 系统（流体、辐射、弹性、天体物理）上预训练一次后，无需架构或数据特定修改即可迁移到新物理域？

## 方法详解

### 整体框架

输入 $u \in \mathbb{R}^{C \times H \times W}$ → 空间+频谱 Tokenization → FiLM 物理调制 → Cross-Attention 融合 → Mamba Backbone → FNO Decoder → 输出

### 关键设计

**双模态 Tokenization**：
- 空间 token：$T_{spatial} = \text{PatchConv}(u) \in \mathbb{R}^{N_p \times d}$
- 频谱 token：$T_{spectral} = \text{Linear}(\text{FFT}_m(u)) \in \mathbb{R}^{1 \times d}$，保留低频模态的全局结构

**FiLM 物理条件调制**：利用物理元数据 $c$（边界条件、本构参数、时间网格）：
$$\tilde{T}_{spatial} = T_{spatial} \odot (1 + \gamma(c)) + \beta(c)$$

**Cross-Attention 融合**：空间与频谱 token 双向交叉注意力，单个频谱 token 控制全局上下文。

**Mamba State-Space Backbone**：$T^{(l+1)} = T^{(l)} + \text{MambaLayer}(T^{(l)})$，$\mathcal{O}(N_p d)$ 线性复杂度 vs Transformer 的 $\mathcal{O}(N_p^2)$。

**FNO Spectral Decoder**：浅层 2D FNO 解码，保留频谱平滑先验：
$$\hat{u}(x) = \sum_{|k| \leq m} W_k \cdot \mathcal{F}[z](k) e^{2\pi i k \cdot x}$$

**双目标损失**：$\mathcal{L} = \text{VRMSE} + \lambda \sum_k w(k) \|\hat{U}(k) - U(k)\|^2$，高频加权。可选守恒量约束 $\mathcal{L}_{cons}$ 和 PDE 残差约束 $\mathcal{L}_{PDE}$。

**多数据集预训练**：数据集特定 1×1 适配器统一通道数；采样概率 $p(i) \propto (\epsilon + \bar{\mathcal{L}}_i)^\alpha \cdot |\mathcal{D}_i|^\tau$ 结合难度感知和温度缩放。

## 实验关键数据

| 数据集 | FNO | CNextU-net | PhysiX | **PDE-FM** |
|--------|------|-----------|--------|----------|
| rayleigh_benard | 0.8395 | 0.6699 | 0.1470 | **0.0415** |
| shear_flow | 1.189 | 0.808 | 0.070 | **0.0345** |
| gray_scott_RD | 0.1365 | 0.1761 | 0.0210 | **0.0183** |
| post_neutron_star | 0.3866 | - | - | **0.2995** |
| turbulence_gravity | 0.2429 | 0.2096 | - | **0.0796** |
| active_matter | 0.3691 | 0.1034 | 0.0904 | 0.1974 |

- 12 个数据集中 6 个 SOTA，5 个第二
- 平均 VRMSE 降低 46%（相对先前 baseline）
- Rayleigh-Bénard 和 shear_flow 改进最为显著（>80% VRMSE 降低）

**Ablation**：Mamba+FNO+SpecTok+X-Attn+FiLM+LayerNorm = 最优配置（mean VRMSE 0.2581）

## 亮点

- 真正的跨物理域 foundation model：从流体湍流到中子星合并、超新星用同一模型
- Mamba backbone 提供线性复杂度，支持大网格和长上下文
- 空间-频谱双模态 tokenization + FiLM 物理调制的设计空间值得探索
- 难度感知的多数据集采样策略有效缓解负迁移

## 局限性

- Ablation 仅在短训练（8 epochs, 600 steps）上进行，最终结果仅由最优配置跑 30 epochs
- 在 active_matter 和 helmholtz_staircase 等数据集上不如 U-net 变体
- 模型复杂度高（Tokenizer+CrossAttn+Mamba+FNO），训练成本未报告
- 3D 数据集的效果报告不如 2D 充分

## 对比

| 方法 | 跨物理域 | 预训练 | Backbone | 复杂度 |
|------|---------|--------|----------|--------|
| FNO | ✗ | ✗ | 频谱 | $\mathcal{O}(N\log N)$ |
| GNOT | ✗ | ✗ | Transformer | $\mathcal{O}(N^2)$ |
| PhysiX | 部分 | ✗ | - | - |
| **PDE-FM** | **✓** | **✓** | **Mamba** | **$\mathcal{O}(Nd)$** |

## 启发

- "算子作为物理分布上的学习"而非孤立映射——foundation model 思维应用于科学计算
- FiLM 调制是引入物理元数据的轻量有效方式
- 困难度感知采样对多数据集训练至关重要
- Mamba 在时空序列建模上有望替代 Transformer

## 评分

⭐⭐⭐⭐⭐ — 问题定义重要，架构设计全面，实验覆盖 12 个异构物理域，结果出色

<!-- RELATED:START -->

## 相关论文

- [OmniArch: Building Foundation Model For Scientific Computing](../../ICML2025/scientific_computing/omniarch_building_foundation_model_for_scientific_computing.md)
- [Closed-form Symbolic Solutions: A New Perspective on Solving Partial Differential Equations](../../ICML2025/scientific_computing/closed-form_solutions_a_new_perspective_on_solving_differential_equations.md)
- [SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs](saot_an_enhanced_locality-aware_spectral_transformer_for_solving_pdes.md)
- [PIMRL: Physics-Informed Multi-Scale Recurrent Learning for Burst-Sampled Spatiotemporal Dynamics](pimrl_physics-informed_multi-scale_recurrent_learning_for_burst-sampled_spatiote.md)
- [PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations](physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)

<!-- RELATED:END -->
