---
title: >-
  [论文解读] Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning
description: >-
  [ICML 2025][图像生成][扩散规划] 提出LoMAP——训练无关的扩散规划修正方法，在每个反向扩散步将引导后样本投影到由离线数据近邻构建的局部低秩子空间上，防止不可行轨迹生成，理论证明引导误差随维度以 $O(\sqrt{d})$ 增长。 领域现状：扩散模型用于轨迹规划（如Diffuser）通过建模整条轨迹分布避免了…
tags:
  - "ICML 2025"
  - "图像生成"
  - "扩散规划"
  - "流形偏离"
  - "低秩投影"
  - "离线RL"
  - "轨迹优化"
---

# Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning

**会议**: ICML 2025  
**arXiv**: [2506.00867](https://arxiv.org/abs/2506.00867)  
**代码**: [GitHub](https://github.com/leekwoon/lomap)  
**领域**: 扩散规划 / 离线强化学习  
**关键词**: 扩散规划, 流形偏离, 低秩投影, 离线RL, 轨迹优化

## 一句话总结
提出LoMAP——训练无关的扩散规划修正方法，在每个反向扩散步将引导后样本投影到由离线数据近邻构建的局部低秩子空间上，防止不可行轨迹生成，理论证明引导误差随维度以 $O(\sqrt{d})$ 增长。

## 研究背景与动机

**领域现状**：扩散模型用于轨迹规划（如Diffuser）通过建模整条轨迹分布避免了逐步自回归的误差累积，结合奖励引导采样可生成高回报行为。**现有痛点**：奖励引导采样存在根本性问题——MSE训练的引导函数 $\mathcal{J}_\phi^{\text{MSE}}(\tau^i) = \mathbb{E}_{q(\tau^0|\tau^i)}[\mathcal{J}(\tau^0)]$ 系统性低估了真实引导 $\mathcal{J}_t(\tau^i) = \log\mathbb{E}_{q(\tau^0|\tau^i)}[e^{\mathcal{J}(\tau^0)}]$（Jensen不等式），在高维长horizon任务中引导误差以 $O(\sqrt{d})$ 增长，导致采样轨迹偏离数据流形产生不可行路径。**核心矛盾**：奖励引导越强，轨迹越可能偏离流形；不引导则无法获得高回报。**本文目标**：在保持奖励引导的同时将生成轨迹拉回数据流形。**切入角度**：利用离线数据集中的轨迹作为流形的局部线性近似基础。**核心idea**：去噪估计→检索近邻→前向扩散→PCA得局部子空间→投影。

## 方法详解

### 整体框架
在标准Diffuser的每个反向扩散步之后插入一个LoMAP投影模块：先按正常流程做奖励引导去噪得到 $\tau^{i-1}$，再将其投影到由离线数据近邻构建的局部低秩子空间上。整个过程无需额外训练。

### 关键设计

1. **引导误差的理论下界（Proposition 3.2）**:
    - 功能：证明MSE引导与真实引导之间的差距不可避免地随维度增长
    - 核心思路：利用Jensen不等式将差距分解为 $\delta(\tau^0) = e^{\mathcal{J}(\tau^0)}/\mathbb{E}[e^{\mathcal{J}(\tau^0)}] - \mathcal{J}(\tau^0)$ 与前向噪声 $\epsilon$ 的关联。高维中 $\|\epsilon\|_2 \approx \sqrt{d}$，且 $\delta$ 与 $\epsilon$ 对齐时引导误差下界为 $c\sqrt{d}/\sqrt{1-\alpha_i}$
    - 设计动机：理论证明了修正的必要性——不是模型能力不足，而是MSE目标本身的固有限制

2. **局部流形近似与投影（LoMAP核心）**:
    - 功能：在每个去噪步构建数据流形的局部线性近似并投影
    - 核心思路：(1) 用Tweedie公式得到去噪估计 $\hat{\tau}^{0|i-1}$；(2) 从离线数据检索 $k$ 个最近邻轨迹 $\{\tau_{(n_j)}^0\}$；(3) 将这些近邻前向扩散到时间步 $i{-}1$：$\tau_{(n_j)}^{i-1} = \sqrt{\alpha_{i-1}}\tau_{(n_j)}^0 + \sqrt{1-\alpha_{i-1}}\epsilon_{(n_j)}$；(4) 对 $\{\tau_{(n_j)}^{i-1}\}$ 做PCA得到正交基 $U\in\mathbb{R}^{d\times r}$；(5) 投影 $\tau^{i-1} \leftarrow UU^\top\tau^{i-1}$
    - 设计动机：前向扩散的近邻天然位于时间步 $i{-}1$ 的流形附近（低维流形假设），PCA提取主方向去除正交于流形的偏离分量

3. **与层次化扩散规划器的兼容（HD + LoMAP）**:
    - 功能：将LoMAP作为即插即用模块集成到更复杂的规划架构中
    - 核心思路：LoMAP只在引导更新后添加一步投影操作，不改变任何前置模块——可直接嵌入Hierarchical Diffuser等层次化规划器
    - 设计动机：AntMaze等复杂任务需要层次化分解（先规划子目标再规划底层动作），LoMAP在两个层级都可独立应用

### 损失函数 / 训练策略
- LoMAP本身无需训练（training-free），仅在推理时操作
- 底层Diffuser训练不变：噪声预测器 $\epsilon_\theta$ 的MSE损失 $\mathcal{L}(\theta) = \mathbb{E}_{i,\epsilon,\tau^0}[\|\epsilon - \epsilon_\theta(\tau^i)\|^2]$
- 引导网络 $\mathcal{J}_\phi$ 的MSE损失不变
- PCA保留方差比例 $\lambda=0.99$，近邻数 $k$ 默认5-10

## 实验关键数据

### 主实验：Maze2D单任务规划

| 环境 | IQL | RGG | TAT | Diffuser | **Diffuser$^\mathcal{P}$** |
|------|-----|-----|-----|----------|---------------------------|
| U-Maze | 47.4 | 108.8 | 114.5 | 113.9 | **126.0±0.26** |
| Medium | 34.9 | 131.8 | 130.7 | 121.5 | **131.0±0.46** |
| Large | 58.6 | 135.4 | 133.4 | 123.0 | **151.9±2.66** |
| 平均 | 47.0 | 125.3 | 126.2 | 119.5 | **136.3** |

### 消融实验：不可行轨迹比例（Artifact Ratio）

| 采样数量 | Diffuser | RGG | **Diffuser$^\mathcal{P}$** |
|---------|----------|-----|---------------------------|
| 100条（Medium） | ~15% | ~8% | **<1%** |
| 100条（Large） | ~30% | ~18% | **<3%** |
| 500条（Large） | ~50% | ~35% | **<5%** |

### 关键发现
- Maze2D-Large提升最显著（123.0→151.9,+23.5%）——越复杂环境流形偏离越严重，LoMAP收益越大
- 不可行轨迹比例从Diffuser的~30%降至<3%（Large环境），验证了流形投影的有效性
- RGG虽减少了artifact但同时降低了轨迹多样性（Fig.3聚集到少数路径），LoMAP保持高可靠性和多样性
- Multi-task（随机目标）设置中IQL性能急剧下降（58.6→24.8），扩散规划器+LoMAP保持稳定
- 近邻数 $k=5{-}10$ 和PCA方差保留 $\lambda=0.99$ 在多数场景下表现稳健
- 与HD层次化规划器结合可进一步提升AntMaze性能

## 亮点与洞察
- 引导误差下界 $O(\sqrt{d})$ 的证明是关键理论贡献——不是说模型不够好，而是MSE目标本身在高维中的固有偏差（Jensen不等式）使得引导必然偏离，维度越高偏离越大。
- "去噪→检索→前向扩散→PCA→投影"的流程简洁优雅，计算开销很小（PCA在低维 $k$ 个样本上），且完全training-free使其成为真正的即插即用模块。
- 在安全关键场景（如机器人规划）中保证轨迹不穿墙/不越界具有重大实际价值——从>30%失败率降到<3%是质的飞跃。

## 局限与展望
- 每步检索 $k$ 个近邻和PCA增加推理延迟（但对标Diffuser本身的多步去噪开销相对较小）
- PCA的线性子空间假设在高度非线性流形区域可能不足——可考虑kernel PCA或自编码器
- 离线数据覆盖外的区域无法构建有效子空间——LoMAP无法帮助泛化到未见环境
- 近邻数 $k$ 和PCA维度 $r$ 需要调优
- 未探索与在线RL或model-based方法的结合

## 相关工作与启发
- **vs RGG (Lee et al., 2023b)**: RGG用OOD检测度量做样本精炼，但依赖精细调整的引导步数且牺牲多样性；LoMAP直接在几何上修正
- **vs TAT (Feng et al., 2024)**: TAT也用轨迹精炼策略，但LoMAP操作更直接（投影 vs 重采样）
- **vs MPGD (Chung et al., 2022)**: 图像域中类似的流形投影思路，LoMAP将其适配到轨迹规划并基于前向扩散近邻构建子空间

## 评分
- 新颖性: ⭐⭐⭐⭐ 引导误差的理论下界+局部低秩投影思路新颖
- 实验充分度: ⭐⭐⭐⭐ Maze2D/Multi2D/AntMaze多环境+artifact分析+可视化
- 写作质量: ⭐⭐⭐⭐⭐ 理论动机→方法设计→实验验证逻辑紧密
- 价值: ⭐⭐⭐⭐⭐ 训练无关+即插即用，对扩散规划的可靠性有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] What We Don't C: Manifold Disentanglement for Structured Discovery](../../NeurIPS2025/image_generation/what_we_dont_c_manifold_disentanglement_for_structured_discovery.md)
- [\[CVPR 2025\] Derivative-Free Diffusion Manifold-Constrained Gradient for Unified XAI](../../CVPR2025/image_generation/derivative-free_diffusion_manifold-constrained_gradient_for_unified_xai.md)
- [\[NeurIPS 2025\] Generative Model Inversion Through the Lens of the Manifold Hypothesis](../../NeurIPS2025/image_generation/generative_model_inversion_through_the_lens_of_the_manifold_hypothesis.md)
- [\[NeurIPS 2025\] StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold](../../NeurIPS2025/image_generation/stella_subspace_learning_in_low-rank_adaptation_using_stiefel_manifold.md)
- [\[ECCV 2024\] Scalable Group Choreography via Variational Phase Manifold Learning](../../ECCV2024/image_generation/scalable_group_choreography_via_variational_phase_manifold_learning.md)

</div>

<!-- RELATED:END -->
