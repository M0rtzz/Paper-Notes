---
title: >-
  [论文解读] SE(3)-Equivariant Diffusion Policy in Spherical Fourier Space
description: >-
  [ICML2025][3D视觉][SE(3)等变] 提出 Spherical Diffusion Policy（SDP），通过将状态、动作和去噪过程嵌入球面傅里叶空间实现端到端 SE(3) 等变的闭环操作策略，在 20 个仿真和 5 个真机任务上大幅超越基线。
tags:
  - ICML2025
  - 3D视觉
  - SE(3)等变
  - 扩散策略
  - 球面傅里叶
  - 闭环操作
  - 机器人学习
---

# SE(3)-Equivariant Diffusion Policy in Spherical Fourier Space

**会议**: ICML2025  
**arXiv**: [2507.01723](https://arxiv.org/abs/2507.01723)  
**代码**: [SDP](https://github.com/amazon-science/Spherical_Diffusion_Policy)  
**领域**: 3d_vision  
**关键词**: SE(3)等变, 扩散策略, 球面傅里叶, 闭环操作, 机器人学习

## 一句话总结

提出 Spherical Diffusion Policy（SDP），通过将状态、动作和去噪过程嵌入球面傅里叶空间实现端到端 SE(3) 等变的闭环操作策略，在 20 个仿真和 5 个真机任务上大幅超越基线。

## 研究背景与动机

- **Diffusion Policy**：从人类示教学习闭环策略的有效方法，但对 3D 姿态变化泛化差
- **现有等变方法局限**：EquiDiff 仅 SO(2)；EquiBot 仅 degree-1 表示丢信息；ET-SEED 计算重
- **目标**：实现对整个 3D 场景（多物体）的连续 SE(3) 等变策略

## 方法详解

### 整体架构

$$\pi(gS) = gA, \quad g \in SO(3), \quad \pi(tS) = A, \quad t \in \mathbb{T}(3)$$

1. **球面编码器**：EquiformerV2 将点云编码为多通道球面特征 $C$
2. **球面去噪时序 U-net (SDTU)**：在球面傅里叶空间中去噪动作轨迹
3. **球面 FiLM 层 (SFiLM)**：等变条件化

### 状态/动作球面表示

$$e, a_t, \epsilon \in \rho_{ee} = \rho_1^4 \oplus \rho_0$$

- 位置向量→degree-1，旋转矩阵→3个 degree-1 向量，夹爪→标量 degree-0

### SDTU（球面去噪时序 U-net）

- 混合通道时序卷积：在每个 degree $l$ 上独立进行，保持 SO(3) 等变性（Proposition 4.1）
- 步进/转置卷积用于 U-net 上下采样
- 球面激活函数保证表达力

### SFiLM

$$\text{SFiLM}(h_l|\gamma_l,\beta_l) = \gamma_l^T h_l \frac{h_l}{\|h_l\|} + \beta_l$$

支持高阶傅里叶系数（Proposition 4.2 证明 SO(3) 等变性）

### 平移不变性

通过相对动作表示（gripper frame canonicalization）实现 $\mathbb{T}(3)$ 不变性

## 实验关键数据

### MimicGen SE(3) 初始化（4任务，3级旋转）

| 方法 | 等变性 | 0° | 15° | 30° |
|---|---|---|---|---|
| DiffPo | 无 | 18 | 9 | 7 |
| EquiDiff | C8⊂SO(2) | 45 | 20 | 13 |
| EquiBot | SO(3) | 2 | 2 | 1 |
| **SDP** | **SE(3)** | **63** | **49** | **35** |

### MimicGen SE(2) 初始化（12任务）：SDP 平均成功率 76% vs EquiDiff 64%

### 真机实验：5 任务（含双臂），SDP 显著优于基线

### 消融

- 移除球面表示→性能剧降
- Degree=0（标量）vs degree≥1：高阶表示关键
- SFiLM vs 无条件化：差距明显

## 亮点与洞察

1. **首个端到端连续 SE(3) 等变闭环扩散策略**
2. 球面傅里叶空间紧凑且具表达力，优于 SO(3) irreps
3. SFiLM 设计精巧，支持高阶等变条件化
4. 支持单臂和双臂操作

## 局限与展望

- EquiformerV2 编码器计算成本仍较大
- 球面表示对非刚体变形的适用性有限
- 真机实验数量有限（5 个任务）

## 相关工作与启发

- Chi et al. (2023) Diffusion Policy
- Liao et al. (2024) EquiformerV2
- Wang et al. (2024b) EquiDiff

## 评分

⭐⭐⭐⭐⭐ — 理论优雅实验充分，SE(3) 等变策略的里程碑工作，球面傅里叶空间设计具有开创性


## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评
