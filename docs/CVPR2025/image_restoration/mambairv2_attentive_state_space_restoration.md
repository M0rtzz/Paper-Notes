---
title: >-
  [论文解读] MambaIRv2: Attentive State Space Restoration
description: >-
  [CVPR 2025][image restoration] 提出Attentive State-space Equation(ASE)将ViT的非因果查询能力注入Mamba，加上语义引导邻居重排(SGN)缓解长距离衰减，单方向扫描即超越多方向扫描方法。
tags:
  - CVPR 2025
  - image restoration
  - Mamba
  - state space model
  - attention mechanism
  - super-resolution
---

# MambaIRv2: Attentive State Space Restoration

**会议**: CVPR 2025  
**arXiv**: [2411.15269](https://arxiv.org/abs/2411.15269)  
**代码**: [GitHub](https://github.com/csguoh/MambaIR)  
**领域**: image_restoration  
**关键词**: Mamba, state space model, attentive state-space equation, semantic guided neighboring, non-causal modeling, super-resolution, denoising

## 一句话总结

提出 MambaIRv2，通过 Attentive State-space Equation（ASE）在 Mamba 的输出矩阵 $\mathbf{C}$ 中注入可学习 prompt 实现类似注意力的非因果全局查询，并用 Semantic Guided Neighboring（SGN）按语义标签重排序列缓解长距离衰减，仅需单方向扫描即超越多方向方法，轻量 SR 上以 9.3% 更少参数超 SRFormer 0.35dB。

## 研究背景与动机

**领域现状**: Mamba 因线性复杂度 + 全局感受野被引入图像恢复并取得不错结果（MambaIR 等），但其因果建模本质是图像恢复这一非因果任务的固有瓶颈。

**现有痛点**:
1. **因果局限**: Mamba 中第 $i$ 个 token 只能看到前 $i-1$ 个 token，图像后续像素无法被利用
2. **多方向扫描冗余**: 为弥补因果缺陷，现有方法（MambaIR）用 4 个方向扫描，但实验表明不同方向序列的余弦相似度 > 0.7，存在大量冗余且增加计算
3. **长距离衰减**: 控制矩阵 $\bar{\mathbf{A}}$ 统计上 < 1，使得 $\bar{\mathbf{A}}^k$ 随距离 $k$ 指数衰减，远距离像素交互极弱

**核心矛盾**: Mamba 的因果性和图像恢复的非因果性之间的根本不匹配。

**本文切入角度**: 从注意力和状态空间的数学联系出发，发现输出矩阵 $\mathbf{C}$ 对应注意力中的 Query，通过在 $\mathbf{C}$ 中注入全局语义 prompt 实现非因果查询。

## 方法详解

### 整体框架

输入低质量图像 → 3×3 conv 提取浅层特征 → 多个 Attentive State Space Group（ASSG），每组含多个 ASSB → 任务特定重建（pixel-shuffle 超分 / conv 去噪）。每个 ASSB 采用 local-to-global 渐进建模：窗口 MHSA（局部）+ ASSM（全局）。

### 关键设计

**1. 注意力与状态空间的桥接分析**
- **功能**: 将因果线性注意力和状态空间方程统一为通用形式进行对比
- **核心发现**:
    - 隐藏状态 $h_i \sim \mathbf{S}_i$（注意力的累积 KV）
    - 输入矩阵 $\mathbf{B} \sim \mathbf{K}^\top$（类似 Key）
    - 输出矩阵 $\mathbf{C} \sim \mathbf{Q}$（类似 Query）
    - 控制矩阵 $\bar{\mathbf{A}} \sim \mathbf{I}$（注意力是恒等，SSM 带衰减）
- **意义**: 既然 $\mathbf{C}$ 扮演 Query 角色，就可以让它"查询"未扫描的像素信息

**2. Attentive State-space Equation（ASE）**
- **功能**: 在原始状态空间方程的输出矩阵 $\mathbf{C}$ 中加入可学习 prompt $\mathbf{P}$
- **核心思路**:
    - 构建 prompt pool $\mathcal{P} \in \mathbb{R}^{T \times d}$，用低秩分解 $\mathcal{P} = \mathbf{M}\mathbf{N}$（$\mathbf{N}$ 跨 block 共享，$\mathbf{M}$ block 特定）
    - 路由策略：线性投影 + LogSoftmax 预测概率 → Gumbel-Softmax 可微选择 → 得到 one-hot 路由矩阵 $\mathbf{R}$  → $\mathbf{P} = \mathbf{R}\mathcal{P}$
    - 修改状态空间方程：$y_i = (\mathbf{C} + \mathbf{P})h_i + \mathbf{D}x_i$
- **设计动机**: prompt 代表全图中语义相似的像素集合，注入后 $\mathbf{C}$ 能"看到"未扫描的像素 → 单方向扫描即可获得全局信息，消除多方向扫描的冗余和开销

**3. Semantic Guided Neighboring（SGN）**
- **功能**: 在送入 ASE 之前，将图像按语义标签重新排列，使语义相似的像素在 1D 序列中空间相邻
- **核心思路**:
    - 复用 ASE 中的路由矩阵 $\mathbf{R}$（已为每个像素分配了语义标签）
    - SGN-unfold：将相同 prompt 类别的像素聚合为一组，各组按类别值顺序拼接形成语义邻域序列
    - 经 ASE 处理后，SGN-fold 执行逆变换恢复空间排列
- **设计动机**: 缓解 Mamba 的长距离衰减——原本空间远但语义近的像素在重排后变为序列邻居，$\bar{\mathbf{A}}^k$ 不再需要跨越长距离

### 损失函数 / 训练策略

- 超分辨率：$L_1$ loss
- 去噪/JPEG CAR：Charbonnier loss
- 初始学习率 $2 \times 10^{-4}$，milestone 衰减
- Adam 优化器，$\beta_1=0.9, \beta_2=0.999$
- 训练 patch：SR 64×64，去噪 128×128；batch size SR=32，去噪=8
- 2× 预训练权重初始化 3×/4× 模型，减半学习率和迭代轮数
- 三个变体：MambaIRv2-S/B/L（小/基础/大）

## 实验关键数据

### 主实验 — 轻量超分辨率（×2）

| 方法 | #Param | Urban100 PSNR | Manga109 PSNR |
|---|---|---|---|
| SwinIR-light | 910K | 32.76 | 39.12 |
| MambaIR-light | 905K | 32.85 | 39.20 |
| SRFormer-light | 853K | 32.91 | 39.28 |
| **MambaIRv2-light** | **774K** | **33.26** | **39.35** |

Urban100 上超 SRFormer 0.35dB，参数少 9.3%。

### 主实验 — 轻量超分辨率（×4）

| 方法 | #Param | Urban100 PSNR | Manga109 PSNR |
|---|---|---|---|
| SwinIR-light | 930K | 26.47 | 30.92 |
| SRFormer-light | 873K | 26.67 | 31.17 |
| MambaIR-light | 925K | 26.75 | 31.26 |
| **MambaIRv2-light** | **794K** | **26.92** | **31.37** |

### 消融实验

**组件有效性（轻量 2× SR，250K iter）**:

| MHSA | ASE | SGN | Urban100 PSNR | Manga109 PSNR |
|---|---|---|---|---|
| ✔ | | | 32.89 | 39.11 |
| ✔ | ✔ | | 32.94 | 39.20 |
| ✔ | ✔ | ✔ | **32.97** | **39.24** |

**Prompt 注入位置消融**:

| 位置 | Urban100 PSNR | Manga109 PSNR |
|---|---|---|
| $\mathbf{B}$（输入矩阵） | 32.96 | 39.23 |
| $\Delta$ | 32.93 | 39.19 |
| $y$（输出） | 32.94 | 39.21 |
| **$\mathbf{C}$（输出矩阵）** | **32.97** | **39.24** |

### 关键发现

1. **因果建模是 Mamba 图像恢复的核心瓶颈**: 4 方向扫描间余弦相似度 > 0.7，冗余极大
2. **$\mathbf{C}$ 是最佳 prompt 注入位置**: 与理论分析一致——$\mathbf{C}$ 对应 Query，语义 prompt 在此注入最有效
3. **单方向扫描可超越多方向**: ASE 的非因果能力使得不再需要 4 方向扫描，效率提升且性能更优
4. **SGN 语义重排有效缓解长距离衰减**: 但参数开销几乎为零（复用路由矩阵）
5. **HAT 这个强 Transformer 基线也被超越**: 经典 SR ×2 Manga109 上超 HAT 0.29dB

## 亮点与洞察

- 从注意力-状态空间的数学等价性出发推导设计，理论基础扎实
- Prompt pool + Gumbel-Softmax 路由的设计优雅，且 SGN 零开销复用路由信息
- 单方向扫描消除冗余是一个有吸引力的效率优势
- "让 Mamba 变得像 Attention 一样非因果"是一个清晰的叙事和有价值的研究方向

## 局限与展望

- Prompt pool 大小 $T$ 和内在秩 $r$ 需要超参调优
- SGN 的语义分组基于简单的 prompt 路由，分组粒度可能不够精细
- 论文仅在 SR、去噪、JPEG CAR 三个任务上验证，未涉及去模糊、去雾等
- Gumbel-Softmax 的温度参数可能影响训练稳定性
- 与 MaIR 等同期 Mamba 恢复方法缺乏直接对比

## 相关工作与启发

- MambaIR 开创了 Mamba 用于图像恢复的先河，但未解决因果性问题；本文是正式的"v2"升级
- ATD（Adaptive Token Dictionary）类似地在注意力中引入外部知识，但方式不同
- 线性注意力与 SSM 的数学联系（Mamba-2 等工作）为本文提供了理论基础
- 启发：架构设计不是"用什么模块"而是"改造模块解什么问题"——将非因果能力注入因果模型是一种通用思路

## 评分

⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration](mair_a_locality-_and_continuity-preserving_mamba_for_image_restoration.md)
- [Efficient Visual State Space Model for Image Deblurring](efficient_visual_state_space_model_for_image_deblurring.md)
- [EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](../../ICCV2025/image_restoration/eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)
- [MambaIR: A Simple Baseline for Image Restoration with State-Space Model](../../ECCV2024/image_restoration/mambair_a_simple_baseline_for_image_restoration_with_state-space_model.md)
- [MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model](../../AAAI2026/image_restoration/mfmamba_a_multi-function_network_for_panchromatic_image_resolution_restoration_b.md)

<!-- RELATED:END -->
