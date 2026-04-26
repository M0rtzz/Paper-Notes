---
title: >-
  [论文解读] Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group
description: >-
  [CVPR 2026][LLM对齐][可操纵卷积核] 提出一种绕过 Clebsch-Gordan 系数计算、直接从群表示矩阵元素构造可操纵核显式基的方法，通过"稳定子约束 + Schur 引理 + Steering"三步策略统一覆盖 SO(2)、O(2)、SO(3)、O(3) 和非紧致 Lorentz 群，大幅简化等变 CNN 的核设计流程。
tags:
  - CVPR 2026
  - LLM对齐
  - 可操纵卷积核
  - 等变CNN
  - 对称群
  - Lorentz群
  - Clebsch-Gordan系数
---

# Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group

**会议**: CVPR 2026  
**arXiv**: [2603.12459](https://arxiv.org/abs/2603.12459)  
**代码**: 无  
**领域**: 等变神经网络 / 群论  
**关键词**: 可操纵卷积核, 等变CNN, 对称群, Lorentz群, Clebsch-Gordan系数

## 一句话总结

提出一种绕过 Clebsch-Gordan 系数计算、直接从群表示矩阵元素构造可操纵核显式基的方法，通过"稳定子约束 + Schur 引理 + Steering"三步策略统一覆盖 SO(2)、O(2)、SO(3)、O(3) 和非紧致 Lorentz 群，大幅简化等变 CNN 的核设计流程。

## 研究背景与动机

**领域现状**: 等变 CNN 通过将对称先验（如旋转不变性）硬编码到网络结构中，在分子模拟、粒子物理、3D 视觉等任务上显著提升了性能和数据效率。其核心组件是可操纵卷积核，需满足约束 $K(g \cdot x) = \rho_{\text{out}}(g) K(x) \rho_{\text{in}}(g)^{-1}$。

**现有痛点**: 现有求解该约束的主流方法（Lang et al. 2021）依赖于计算 Clebsch-Gordan (CG) 系数，需要在"耦合基"与"非耦合基"之间反复变换。对于某些群（特别是非紧致群如 Lorentz 群），CG 系数的计算非常困难甚至不可行。另一些方法（Finzi et al. 2021, Zhdanov et al. 2023）采用 MLP 隐式参数化，但丧失了解析解的可解释性。

**核心矛盾**: 理论上只需找到满足线性约束的核空间基，但 CG 系数的引入将一个概念上简单的问题变得计算复杂。

**本文目标**: 能否绕过 CG 系数，直接从群表示矩阵元素出发，为任意对称群和任意张量类型的特征图提供即用的可操纵核基？

**切入角度**: 利用轨道上某点的稳定子群将全局约束退化为局部不变性条件，再通过 Schur 引理直接求解同态空间。

**核心 idea**: 在稳定子群固定点处求解简化约束，然后通过 steering 操作推广到整个轨道，谐波基函数作为表示矩阵元素自然出现。

## 方法详解

### 整体框架

整个方法由三步构成：(1) 在群 $G$ 的轨道上选取参考点 $x_0$，确定其稳定子群 $H = \text{Stab}_{x_0}$；(2) 将可操纵约束退化为 $x_0$ 处的不变性条件 $K(x_0) = \rho_j(h) K(x_0) \rho_l(h)^{-1}, \forall h \in H$，利用 Schur 引理在 $H$ 的不可约分解上直接求解同态基；(3) 通过 steering 操作 $K(g \cdot x_0) = \rho_j(g) K(x_0) \rho_l(g)^{-1}$ 将结果推广到轨道上的任意点。这一思路在文献中虽有零散提及，但本文首次系统化地在多个群上展开并给出完整闭合公式。

### 关键设计

1. **稳定子约束简化与 Schur 引理求解**:

    - 功能：将全局可操纵约束退化为参考点处的局部不变性条件
    - 核心思路：选定 $x_0$ 后，将 $\rho_j$ 和 $\rho_l$ 限制到 $H$ 上得到可约表示 $\rho_j^H$ 和 $\rho_l^H$，将其分解为不可约表示的直和。由 Schur 引理，同态矩阵 $K(x_0)$ 的块结构几乎完全确定——不等价不可约表示之间的块为零，等价不可约表示之间的块正比于恒等映射（实数情况下可包含 $\mathbb{I}$ 和 $J$ 两个生成元）
    - 设计动机：对于 SO(2)，$H$ 仅为恒等矩阵，约束自动满足，任何矩阵都是解；对于 SO(3)，$H \simeq$ SO(2)，约束通过对角分解块求解，自由参数维度为 $2\min(j,l)+1$

2. **Steering 操作生成完整核基**:

    - 功能：将 $x_0$ 处的同态基推广到轨道上的任意点
    - 核心思路：对每个基元素 $T_m$，通过 $K_m(g \cdot x_0) = \rho_j(g) T_m \rho_l(g)^{-1}$ 得到完整的核函数。对 SO(3) 的复数表示，核基的矩阵元素形如 $D^j_{m_j m}(g) D^l_{m m_l}(g)^{-1}$，其中 $D$ 为 Wigner-D 矩阵
    - 设计动机：谐波基函数不需要预先选择，而是作为表示矩阵元素自然出现。与传统方法相比，免去了耦合基/非耦合基变换的全部步骤

3. **非紧致 Lorentz 群的处理**:

    - 功能：将方法扩展到物理学中重要的 Lorentz 群 $\text{SO}^+(1,3)$
    - 核心思路：分两种物理情形。**大质量粒子**：轨道为类时双曲面，$H \simeq$ SO(3)，通过 SU(2) 的 CG 系数（而非 Lorentz 群的 CG 系数）做子群分解，整数自旋的互缠器表现为射影算子（如 $\Delta^\mu{}_\nu = \delta^\mu{}_\nu - u^\mu u_\nu$）。**无质量粒子**：轨道为光锥，$H \simeq$ ISO(2)，仅考虑 SO(2) 子群，互缠器为横向射影算子。半整数自旋通过荷共轭算子 $\mathcal{C}$ 和 $\gamma$ 矩阵构造四元数结构
    - 设计动机：Lorentz 群的 CG 系数极其复杂（Bogatskiy et al. 2020），本文通过在时空张量表示中直接构造射影算子完全绕过了这一困难，且结果具有清晰的物理解释

### 损失函数 / 训练策略

本文为纯理论贡献，不涉及训练过程。作者指出在实现时可对输入表示标签 $l$ 和输出表示标签 $j$ 分别独立截断，而传统耦合基方法只能对耦合后的单一标签 $J$ 截断。这种灵活性可能带来更强的网络表达能力，但需通过实验验证。

## 实验关键数据

### 主实验

本文无数值实验。通过解析推导验证方法正确性，与已知结果进行了系统对比：

| 对称群 | 验证内容 | 验证结果 |
|--------|---------|---------|
| SO(2) 复数 | 核基 $e^{i(j-l)\phi}$ | 与 Weiler et al. (2019) Table 8 完全一致 |
| SO(2) 实数 | 4维核基（三角函数矩阵） | 严格复现 Table 8 所有案例 |
| O(2) 实数 | 反射约束下的简化基 | 匹配 Table 9 对应案例（含 $\rho_{\tilde{0}}$ 表示） |
| SO(3) 实数 | 基维度 $2\min(j,l)+1$ | 与已知理论一致 |
| SO(3) 复数 vs 实数 | 实参数维度比 $4\min(j,l)+2$ vs $2\min(j,l)+1$ | 通过荷共轭对称性统一解释 |
| O(3) | 奇偶性分类的互缠器 | 每对不可约表示有 $\min(l,j)+1$ 个复互缠器 |
| Lorentz 群（大质量） | 自旋 0/1/2 射影算子 | 给出物理直觉清晰的 $u^\mu u_\nu$, $\Delta^\mu{}_\nu$ 等形式 |
| Lorentz 群（无质量） | 自旋 1/2 横向射影 | 含正/负能量投影 $P_\pm(u)$ 和四元数结构 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 复数 vs 实数 (SO(3)) | 实参数: $4\min(j,l){+}2$ vs $2\min(j,l){+}1$ | 荷共轭将参数减半，两组基生成同一解空间 |
| 整数自旋 vs 半整数自旋 (Lorentz) | 互缠器类型: $\mathbb{R}$ vs $\mathbb{H}$ | 半整数自旋引入四元数结构 $\{𝕀, I, J, K\}$ |
| 独立截断 vs 耦合截断 | 灵活性 | $j,l$ 可独立选择，比单一 $J$ 截断更灵活 |

### 关键发现

- 所有紧致群的结果中，谐波基函数无需预先选择，自动从表示矩阵元素中产生
- Lorentz 群的整数自旋互缠器退化为射影算子形式，具有直接的物理对应（能量-动量分解）
- 复数与实数表示的参数维度差异可通过荷共轭对称性统一解释

## 亮点与洞察

- **极致简洁性**: SO(2) 的全部推导仅需几行——稳定子约束自动满足，steering 一步得到 $e^{i(j-l)\phi}$，与传统需要选择基函数再分解的方法形成鲜明对比
- **理论统一性**: 同一个三步框架（选点→Schur 引理→steering）无缝覆盖从最简单的 2D 旋转到非紧致 Lorentz 群，方法论上的统一性非常优雅
- **物理直觉清晰**: Lorentz 群结果天然以射影算子形式出现（$u^\mu u_\nu$, $\Delta^\mu{}_\nu$），与量子场论中的标准工具完美对接
- **独立截断灵活性**: 传统方法在耦合基上截断 $J$，本文可在输入/输出表示标签 $j,l$ 上独立截断，为网络设计提供更大自由度

## 局限与展望

- 纯理论论文，未在任何视觉或物理任务上进行数值实验验证实际效果提升
- 未与现有等变框架（e3nn、escnn）进行计算效率或内存开销的定量对比
- 仅考虑了完全可约表示，对非紧致群的更一般不可约表示情况未做讨论
- 混叠（aliasing）处理和离散化实现细节留待未来工作
- 独立截断 $j,l$ 带来的表达能力优势需要实验设计来量化验证

## 相关工作与启发

- **vs Lang et al. (2021)**: 后者需计算 CG 系数 + 耦合基变换 + 逆变换三步，本文直接从表示矩阵元素一步到位，概念更清晰计算更简洁
- **vs Weiler et al. (2019, E2CNN)**: 两者得到相同的解向量空间但基元素不同，本文通过 steering 操作更直接地获得结果
- **vs Bogatskiy et al. (2020, LorentzNet)**: 后者为 Lorentz 群计算完整 CG 系数（困难且不直观），本文利用 SO(3) 子群分解 + 射影算子绕过
- **vs Finzi et al. (2021), Zhdanov et al. (2023/2024)**: 后者用 MLP 隐式参数化可操纵核（适用于任意矩阵群），本文给出显式解析解，可解释性更强
- 为在粒子物理模拟中构建 Lorentz 等变网络提供了更实用的核构造工具

## 评分

- 新颖性: ⭐⭐⭐⭐ 核心思路在文献中有零散提及，但首次系统化展开到 Lorentz 群并给出完整闭合公式
- 实验充分度: ⭐⭐ 纯理论工作无数值实验，但与已知解析结果的交叉验证充分
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，同时为非表示论专家读者提供了循序渐进的讲解层次
- 价值: ⭐⭐⭐ 为等变网络研究者提供了更直接的理论工具，但缺少实验验证限制了当前实际影响力

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization](physmodpo_physically-plausible_humanoid_motion_with_preference_optimization.md)
- [\[CVPR 2026\] LocalDPO: Direct Localized Detail Preference Optimization for Video Diffusion Models](mind_the_generative_details_direct_localized_detail_preference_optimization_for_.md)
- [\[CVPR 2026\] Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models](principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la.md)
- [\[CVPR 2026\] Bias at the End of the Score: Demographic Biases in Reward Models for T2I](bias_reward_models_t2i.md)
- [\[CVPR 2026\] GlyphPrinter: Region-Grouped Direct Preference Optimization for Glyph-Accurate Visual Text Rendering](glyphprinter_region-grouped_direct_preference_optimization_for_glyph-accurate_vi.md)

<!-- RELATED:END -->
