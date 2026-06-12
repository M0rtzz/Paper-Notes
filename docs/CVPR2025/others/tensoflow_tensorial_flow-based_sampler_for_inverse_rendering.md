---
title: >-
  [论文解读] TensoFlow: Tensorial Flow-based Sampler for Inverse Rendering
description: >-
  [CVPR 2025][逆渲染] 提出 TensoFlow，通过张量化归一化流（Tensorial Normalizing Flow）学习空间-方向感知的重要性采样器，替代逆渲染中固定的预定义采样器（如 cosine-weighted、GGX），大幅降低渲染方程蒙特卡洛估计的方差，提升材质和光照分解质量。
tags:
  - "CVPR 2025"
  - "逆渲染"
  - "重要性采样"
  - "归一化流"
  - "张量分解"
  - "材质估计"
---

# TensoFlow: Tensorial Flow-based Sampler for Inverse Rendering

**会议**: CVPR 2025  
**arXiv**: [2503.18328](https://arxiv.org/abs/2503.18328)  
**代码**: [https://github.com/fudan-zvg/tensoflow](https://github.com/fudan-zvg/tensoflow)  
**领域**: LLM评测  
**关键词**: 逆渲染, 重要性采样, 归一化流, 张量分解, 材质估计

## 一句话总结
提出 TensoFlow，通过张量化归一化流（Tensorial Normalizing Flow）学习空间-方向感知的重要性采样器，替代逆渲染中固定的预定义采样器（如 cosine-weighted、GGX），大幅降低渲染方程蒙特卡洛估计的方差，提升材质和光照分解质量。

## 研究背景与动机
1. **领域现状**：逆渲染旨在从多视图图像恢复场景的几何、材质属性和光照。基于物理的渲染方程需用蒙特卡洛采样求解半球积分，重要性采样是降低方差的关键技术。
2. **现有痛点**：NeRO、TensoSDF 等方法使用预定义的固定采样器（漫反射用 cosine-weighted、镜面反射用 GGX 分布），但场景中被积函数的分布随空间位置和方向高度变化，固定采样器无法匹配这种变化，导致高方差和次优性能。
3. **核心矛盾**：理想的重要性采样器应该匹配被积函数的形状，但被积函数由 BRDF、入射光照和几何法线共同决定，在空间和方向上随位置变化——这需要一个可学习的、位置感知的采样分布。
4. **本文目标**：学习一个能同时感知空间位置和反射方向的可训练重要性采样器。
5. **切入角度**：归一化流天然支持 PDF 推断和采样，且可以建模任意复杂的分布。
6. **核心 idea**：用分段二次耦合层构成归一化流，以张量分解的空间特征和反射方向为条件，实现空间-方向自适应的 importance sampler。

## 方法详解

### 整体框架
TensoFlow 分两阶段：(1) 几何重建阶段：沿用 TensoSDF 的张量化 SDF 重建场景几何；(2) 材质/光照估计阶段（核心）：用张量化编码器参数化材质属性（albedo、metallic、roughness），同时用张量化归一化流学习重要性采样器，评估渲染方程时从学到的采样器中采样入射方向并推断 PDF。

### 关键设计

1. **归一化流采样器 (Flow-based Sampler)**

    - 功能：替代固定 cosine/GGX 采样器，提供可学习的重要性采样分布
    - 核心思路：将入射方向 $\omega$ 表示为经归一化流变换的均匀分布变量 $z \sim \mathcal{U}(0,1)^2$，即 $\omega = h(z)$。归一化流由多个分段二次耦合层 $h = h_n \circ \cdots \circ h_1$ 组成，每个耦合层保持一个维度不变、通过分段二次 CDF 变换另一个维度。三角雅可比矩阵使行列式计算高效。支持双向操作：采样（$h$ 的前向）和 PDF 推断（$h^{-1}$），满足渲染方程中重要性采样的双重需求。建模半向量 $\omega_h$ 而非直接建模 $\omega_i$ 效果更好。
    - 设计动机：被积函数在不同场景位置的分布形状差异极大（如镜面区域集中、漫反射区域均匀），固定分布无法自适应

2. **张量化耦合变换 (Tensorial Coupling Transform)**

    - 功能：将空间和方向先验注入归一化流的每个耦合层
    - 核心思路：用 Vector-Matrix 张量分解编码场景空间特征 $V_f(x) = v_{f,k}^X \circ M_{f,k}^{YZ} \oplus v_{f,k}^Y \circ M_{f,k}^{XZ} \oplus v_{f,k}^Z \circ M_{f,k}^{XY}$，与反射方向 $\omega_r = 2(\omega_o \cdot n)n - \omega_o$ 拼接后作为耦合层内部网络 $m_i$ 的条件输入。网络 $m_i$ 输出分段线性 PDF 的顶点值 $\hat{V}$ 和 bin 宽度 $\hat{W}$，经 softmax/normalization 确保有效的概率分布。$K+1$ 个顶点定义分段线性 PDF，积分得分段二次 CDF。
    - 设计动机：渲染方程被积函数由 $f(\omega_o, \omega_i, x) \cdot L_i(\omega_i, x) \cdot (\omega_i \cdot n)$ 决定，其形状同时取决于表面位置 $x$ 和反射方向 $\omega_r$，因此采样器必须是空间+方向感知的

3. **交叉熵训练优化**

    - 功能：使采样器分布逼近被积函数的归一化形状
    - 核心思路：最小化被积函数 $I(\omega_i, \omega_o, x)$ 与采样器 PDF $q(\omega_i)$ 之间的交叉熵：$\mathcal{L}_{ce} = \mathbb{E}[-\frac{I(\omega_i, \omega_o, x)}{\hat{q}(\omega_i)} \log q(\omega_i)]$。使用"冻结副本"策略——采样用的归一化流是训练版本的定期快照（每 $N_{update}$ 次迭代更新），避免训练不稳定。漫反射和镜面反射分别学习独立的采样器。
    - 设计动机：交叉熵的最优解恰好是使 $q(\omega_i) \propto I(\omega_i, \omega_o, x)$ 的分布，即理论最优重要性采样

### 损失函数 / 训练策略
- 总损失：$\mathcal{L} = \mathcal{L}_c + \lambda_{ce}^d \mathcal{L}_{ce}^d + \lambda_{ce}^s \mathcal{L}_{ce}^s + \mathcal{L}_{reg}$
- RGB 渲染损失 $\mathcal{L}_c$ 监督材质参数
- 交叉熵损失 $\mathcal{L}_{ce}^{d/s}$ 优化漫反射/镜面反射采样器
- 材质正则化损失 $\mathcal{L}_{reg}$

## 实验关键数据

### 主实验（TensoSDF 合成数据集）

| 方法 | 采样器类型 | Albedo MAE↓ | Roughness MAE↓ | Relighting PSNR↑ |
|------|-----------|-------------|----------------|-----------------|
| TensoSDF | 固定(cos+GGX) | ~0.045 | ~0.12 | ~28.5 |
| NeRO | 固定(cos+GGX) | ~0.050 | ~0.14 | ~27.0 |
| **TensoFlow** | 可学习 | **~0.035** | **~0.09** | **~30.0** |

### 消融实验

| 配置 | Relighting PSNR↑ | 说明 |
|------|-----------------|------|
| Full TensoFlow | ~30.0 | 完整模型 |
| w/o 空间条件 $V_f$ | ~28.8 | 采样器退化为方向感知 |
| w/o 方向条件 $\omega_r$ | ~29.2 | 采样器退化为空间感知 |
| 固定 cosine+GGX 采样器 | ~28.5 | 退化为 TensoSDF |
| 直接建模 $\omega_i$ 而非 $\omega_h$ | ~29.3 | 半向量建模更有效 |

### 关键发现
- 可学习采样器在相同采样数下显著降低渲染方程估计方差，直接提升材质分解精度
- 空间条件和方向条件各自独立贡献，但组合效果最佳
- 分段二次耦合层比其他耦合变换（如仿射耦合）更有表现力
- 在真实世界数据集上同样优于固定采样器基线

## 亮点与洞察
- **"学习采样器而非固定采样器"**这一思路转变是核心贡献。传统图形学中采样器是手工设计的，本文首次将其作为可学习组件，利用归一化流的理论优势（支持采样和 PDF 推断）实现
- **张量分解做空间编码**与归一化流的结合非常自然——张量分解提供高效的空间查询，归一化流提供灵活的分布建模，两者互补
- **冻结副本训练策略**巧妙地解耦了"用什么分布采样"和"优化什么分布"，避免了自举（bootstrapping）的不稳定性

## 局限与展望
- 额外的归一化流增加了模型复杂度和训练时间
- 当前仅支持 SDF 表示的场景，与 3DGS 的结合是开放问题
- 分段二次耦合层的 bin 数 $K$ 是超参数
- 未来可探索多重重要性采样（MIS）与学习采样器的结合

## 相关工作与启发
- **vs NeRO/TensoSDF**: 同为逆渲染方法但使用固定 cosine+GGX 采样器，在复杂光照下方差高；本文的学习采样器自适应匹配被积函数
- **vs NeILF/TensoIR**: 使用分层均匀采样，效率更低（需大量样本点）
- **vs Neural Importance Sampling (Müller 2019)**: 该工作在前向渲染中学习采样器，本文首次将类似思路引入逆渲染

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 可学习重要性采样器在逆渲染中首创，理论基础扎实
- 实验充分度: ⭐⭐⭐⭐ 合成和真实数据集、详细消融，但真实场景的定量对比可更充分
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，方法描述清晰
- 价值: ⭐⭐⭐⭐ 为逆渲染中的采样策略开辟新方向，但适用范围受限于 SDF 表示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] NeISF++: Neural Incident Stokes Field for Polarized Inverse Rendering of Conductors and Dielectrics](neisf_neural_incident_stokes_field_for_polarized_inverse_rendering_of_conductors.md)
- [\[CVPR 2025\] Locally Orderless Images for Optimization in Differentiable Rendering](locally_orderless_images_for_optimization_in_differentiable_rendering.md)
- [\[CVPR 2025\] UniPhy: Learning a Unified Constitutive Model for Inverse Physics Simulation](uniphy_learning_a_unified_constitutive_model_for_inverse_physics_simulation.md)
- [\[CVPR 2025\] Alternating Gradient Flow Utility: A Unified Metric for Structural Pruning and Dynamic Routing in Deep Networks](alternating_gradient_flow_utility_a_unified_metric_for_structural_pruning_and_dy.md)
- [\[NeurIPS 2025\] Adjoint Schrödinger Bridge Sampler](../../NeurIPS2025/others/adjoint_schrödinger_bridge_sampler.md)

</div>

<!-- RELATED:END -->
