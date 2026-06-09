---
title: >-
  [论文解读] GeoDynamics: A Geometric State-Space Neural Network for Understanding Brain Dynamics on Riemannian Manifolds
description: >-
  [NeurIPS 2025][医学图像][state-space model] 提出GeoDynamics，将经典状态空间模型(SSM)从欧几里得空间推广到对称正定(SPD)流形，通过加权Frechet均值聚合和正交群平移实现流形上的状态演化…
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "state-space model"
  - "SPD manifold"
  - "Riemannian geometry"
  - "brain dynamics"
  - "functional connectivity"
---

# GeoDynamics: A Geometric State-Space Neural Network for Understanding Brain Dynamics on Riemannian Manifolds

**会议**: NeurIPS 2025  
**arXiv**: [2601.13570](https://arxiv.org/abs/2601.13570)  
**代码**: 无  
**领域**: 脑动力学 / 状态空间模型 / 黎曼几何  
**关键词**: state-space model, SPD manifold, Riemannian geometry, brain dynamics, functional connectivity

## 一句话总结
提出GeoDynamics，将经典状态空间模型(SSM)从欧几里得空间推广到对称正定(SPD)流形，通过加权Frechet均值聚合和正交群平移实现流形上的状态演化，在脑连接组（AD/PD/ASD早期诊断）和人体动作识别上均取得SOTA。

## 研究背景与动机

**领域现状**：理解脑动力学的主流方法分两类：(a)分析BOLD信号时间波动（RNN/LSTM/SSM/Mamba）；(b)分析FC矩阵拓扑变化（GNN/SPDNet）。SSM因Mamba在CV/NLP中的成功而受关注。

**现有痛点**：(a) SSM假设状态在欧几里得空间演化，但FC矩阵天然是SPD矩阵，生活在弯曲黎曼流形上——欧几里得运算违反SPD约束；(b) 滑动窗口对窗口大小敏感；(c) SPDNet等缺乏时间建模；(d) 现有方法只看空间或只看时间。

**核心矛盾**：FC矩阵的时间演化是SPD流形上的曲线，需在保持几何一致性前提下同时捕捉时空动力学。

**核心 idea**：用黎曼几何的Frechet均值和正交群作用替换SSM中的欧几里得线性运算，使系统状态在SPD流形上几何一致地演化。

## 方法详解

### 整体框架
输入：时序FC矩阵序列（滑动窗口从BOLD信号构建）。经GeoDynamics的状态更新、观测、注意力、对数映射后输出分类结果。所有中间表示均在SPD流形上。

### 关键设计

1. **SPD流形上的状态更新方程**:

    - 用加权Frechet均值(wFM)替换线性组合：在流形上沿测地线做内在平均，自动保持SPD
    - 用正交群平移 $\mathcal{T}(U,V) = g(V)Ug(V)^\top$ 替换加法更新，Stein度量下等距
    - 使用Stein距离避免重复特征分解

2. **离散化与全局卷积**:

    - 矩阵指数离散化保稳定性
    - 卷积核参数化为 $\hat{\mathcal{K}} = \mathcal{K}^\top\mathcal{K} + \epsilon I$ 保证SPD

3. **SPD保持注意力(SPA)**:

    - 在流形卷积响应上定义注意力权重，exp映射保证输出SPD
    - 学到的注意力权重可解释为脑区连接异常程度

4. **任务读出**:

    - 对数映射到切空间 + softmax分类器 + 交叉熵损失

## 实验关键数据

### 主实验：脑连接组（准确率%）

| 数据集 | 任务 | GeoDynamics | 次优方法 | 提升 |
|--------|------|-------------|---------|------|
| HCP-WM | 8类工作记忆 | **98.29** | Mamba 97.22 | +1.07 |
| OASIS | AD vs CN | **71.43** | SPDNet ~68 | +~3 |
| ADNI | 4类认知状态 | **56.00** | GSN 52.80 | +3.2 |
| PPMI | 4类PD分期 | **72.01** | — | — |

### 模型复杂度对比（HCP-WM, N=360）

| 模型 | 参数量(M) | Accuracy |
|------|----------|----------|
| Mamba (2048d, 5层) | 132 | 97.22 |
| Mamba (1024d, 2层) | 14.07 | 95.92 |
| **GeoDynamics (2层)** | **14.60** | **98.29** |

### 动作识别验证

| 数据集 | GeoDynamics | 次优方法 |
|--------|-------------|---------|
| Florence | **94.1** | GR-GCN 93.3 |
| HDM05 | **72.3** | F-DMT-Net 71.3 |
| UTKinect | **98.0** | F-DMT-Net 98.0 |

### 关键发现
- 任务态fMRI上序列模型大幅优于空间模型（高达30%），GeoDynamics在更低参数量下超越Mamba
- 神经退行性疾病静息态fMRI上，GeoDynamics显著优于空间和序列模型——流形建模同时捕获时空信息
- SPA注意力图与临床知识一致：AD异常在DMN+体感皮层，PD在感觉运动区+额叶+小脑，ASD在颞叶+视觉皮层
- 对滑动窗口大小相对鲁棒（35-45最优）
- HAR数据集同样SOTA，验证跨领域通用性

## 亮点与洞察
- **SSM到SPD流形的数学推广**：用wFM替换线性组合、正交群平移替换加法更新，每步有严格几何保证。可推广到其他流形值时间序列
- **Stein距离的选择**：避免重复特征分解，计算效率高，实用的工程选择
- **SPA注意力可解释性**：注意力权重对应脑区连接异常程度，将黑盒模型与临床解剖学知识对齐

## 局限与展望
- 仍依赖滑动窗口构建FC矩阵——可探索无窗口的连续时间FC构建
- SPD流形上wFM优化是迭代的，高维FC矩阵(N=360)计算成本较高
- 脑疾病分类绝对准确率仍不高（ADNI 56%、PPMI 72%）
- HAR数据集规模很小（199-686样本），应在NTU-RGB+D等大规模benchmark测试
- 实际计算的相关矩阵可能半正定或有数值问题
- 与基于Transformer注意力的脑网络分析方法的比较缺失
- 代码未开源，可复现性存在疑问

## 相关工作与启发
- **vs Mamba**: 欧几里得空间SSM，在相同参数量(~14M)下GeoDynamics准确率高2.4%(98.3 vs 95.9)。优势来自流形感知的状态演化能保持FC矩阵的内在几何结构
- **vs SPDNet**: 纯空间流形模型缺乏时间建模，在任务态fMRI上表现差（序列模型高出30%）。GeoDynamics继承流形表示但加入时空联合建模
- **vs STAGIN/BNT/ContrastPool**: GNN脑网络方法将FC矩阵视为图而非流形元素，消息传递在欧几里得空间操作无法保证SPD约束
- **vs 传统dFC方法**: 滑动窗口对窗口大小敏感，GeoDynamics的SSM模块部分补偿了窗口选择偏差
- 该框架可自然推广到其他流形值时间序列分析，如Grassmann流形上的子空间追踪或Stiefel流形上的正交帧演化
- SPA注意力机制的可解释性可作为临床神经影像分析的辅助工具，帮助定位疾病相关脑区

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ SSM到SPD流形的推广是优雅的数学贡献，理论分析严谨
- 实验充分度: ⭐⭐⭐⭐ 5个脑数据集+3个HAR数据集覆盖广，但部分绝对性能不高
- 写作质量: ⭐⭐⭐⭐ 数学公式严谨清晰，但符号较多对非专家有门槛
- 价值: ⭐⭐⭐⭐ 为脑动力学和流形时间序列分析提供新范式
<!-- NeurIPS 2025 | video_understanding -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Generalizable, Real-Time Neural Decoding with Hybrid State-Space Models](generalizable_real-time_neural_decoding_with_hybrid_state-space_models.md)
- [\[NeurIPS 2025\] DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs](dyg-mamba_continuous_state_space_modeling_on_dynamic_graphs.md)
- [\[NeurIPS 2025\] Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry](riemannian_flow_matching_for_brain_connectivity_matrices_via_pullback_geometry.md)
- [\[CVPR 2026\] Modeling Spatiotemporal Neural Frames for High Resolution Brain Dynamics](../../CVPR2026/medical_imaging/modeling_spatiotemporal_neural_frames_for_high_resolution_brain_dynamic.md)
- [\[NeurIPS 2025\] Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)

</div>

<!-- RELATED:END -->
