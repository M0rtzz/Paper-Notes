---
title: >-
  [论文解读] Graph Neural Networks for Interferometer Simulations
description: >-
  [NeurIPS 2025 (AI for Science Workshop)][图学习][图神经网络] 首次将图神经网络应用于光学干涉仪仿真，通过 GATv2 + KAN 架构预测 LIGO 干涉仪中的电磁场功率和空间强度分布，实现比标准仿真软件（FINESSE）快 815 倍的推理速度…
tags:
  - "NeurIPS 2025 (AI for Science Workshop)"
  - "图学习"
  - "图神经网络"
  - "干涉仪仿真"
  - "LIGO"
  - "引力波"
  - "光学模拟"
---

# Graph Neural Networks for Interferometer Simulations

**会议**: NeurIPS 2025 (AI for Science Workshop)  
**arXiv**: [2512.16051](https://arxiv.org/abs/2512.16051)  
**代码**: [LIGO GitLab](https://git.ligo.org/uc_riverside/gnn-ifosim)  
**领域**: 图学习 / 科学计算  
**关键词**: GNN, 干涉仪仿真, LIGO, 引力波, 光学模拟

## 一句话总结

首次将图神经网络应用于光学干涉仪仿真，通过 GATv2 + KAN 架构预测 LIGO 干涉仪中的电磁场功率和空间强度分布，实现比标准仿真软件（FINESSE）快 815 倍的推理速度，同时保持较好的物理精度。

## 研究背景与动机

引力波（Gravitational Waves）是时空结构的拉伸/收缩，由爱因斯坦在 1916 年预言。LIGO（激光干涉仪引力波天文台）通过双循环 Michelson 干涉仪来探测引力波。为探测引力波，LIGO 和未来的 Cosmic Explorer 需要极高灵敏度，这意味着干涉仪设计必须对实际制造误差具有鲁棒性。

**现有痛点**: 搜索最优鲁棒设计参数是一个极具挑战性的计算问题，需要运行数千次高保真光学仿真，并在高维非凸损失景观中进行优化。主要计算瓶颈在于：
1. 为捕获更精细的空间特征，需包含高阶模式，散射矩阵维度以 $\mathcal{O}(n^2)$ 增长
2. 计算矩阵元素本身极其耗时（尤其是考虑热透镜效应和有限口径散射时）
3. 干涉仪需要"锁定"——微调腔镜位置以达到谐振，每次调整都需重新运行仿真

**核心矛盾**: 训练数据收集本身就极耗时，若模型泛化能力有限则实用价值大打折扣。但即使是粗略的仿真近似也有价值——能帮助优化程序快速剪枝参数空间中不良设计区域。

**切入角度**: 干涉仪中电磁场与一系列光学组件交互的过程，天然适合图表示。每个光学组件可分解为入射/出射场节点，边表示场之间的空间连接。GNN 的消息传递可有效捕获这种局部交互和远程依赖。

**核心idea**: 用 GNN 学习干涉仪的稳态电磁场（功率和强度分布），作为 FINESSE 仿真的替代模型（surrogate model），在保持物理精度的同时大幅加速推理。

## 方法详解

### 整体框架

**输入**: 干涉仪的图表示。每个光学组件（镜面）被拆分为 4 个节点（每侧 2 个，分别对应入射场和出射场）。每个节点有 3 个特征：波前曲率半径、光学组件反射率、取向角度。每条边有 2 个特征：长度和折射率。

**输出**: 两个任务——(1) 每个节点处的入射/出射场功率；(2) 每个节点处的空间强度分布（2D 强度图）。

**数据集**: 三种干涉仪拓扑：
- Fabry-Perot (FP) 腔：30,000 个图，每图 10 个节点
- 简单耦合腔 (Simple CC)：5,000 个图，每图 18 个节点
- Arm-SRC 耦合腔：30,000 个图，每图 74 个节点

### 关键设计

1. **功率预测模型（Power Prediction）**:

    - 功能：预测干涉仪中每个光学点的入射和出射功率
    - 核心思路：20 层 GATv2 + 6 层前馈网络 + LeakyReLU 激活 + 残差连接。由于干涉仪内功率跨越数个数量级（腔内 kW 级 vs 出射 mW 级），模型预测 $\log P$ 而非原始功率
    - 设计动机：GATv2 的注意力机制允许网络学习不同光学组件之间的优先级关系，20 层深度确保信息在复杂干涉仪拓扑中充分传播

2. **带物理约束的损失函数**:

    - 功能：在标准重建损失之外加入能量守恒正则化
    - 核心公式：
    $\mathcal{L} = \frac{1}{n} \sum_{n}^{N} \| \mathbf{y}_n - \hat{\mathbf{y}}_n \|_1 + \lambda \| \hat{\mathbf{y}}_n - \mathbf{A}^T \hat{\mathbf{y}}_n \|_1$
      第一项为标准 MAE 损失，第二项惩罚违反能量守恒的预测（节点入射功率之和应等于该节点功率）
    - 设计动机：将物理先验（能量守恒）嵌入训练目标，确保预测的物理可行性

3. **强度分布预测模型（Intensity Prediction）**:

    - 功能：预测每个光学点的 2D 空间强度分布
    - 核心思路：15 层 GAT → Deep Kolmogorov-Arnold Network (KAN)。利用场的径向对称性，先用 KAN 学习径向强度分布 $I(r)$，再旋转生成完整 2D 强度图
    - 设计动机：(1) 利用物理对称性将 $\mathcal{O}(n^2)$ 自由度降为 $\mathcal{O}(n)$；(2) KAN 在学习物理特殊函数（如球谐函数）方面优于 MLP，参数更少

4. **图表示设计**:

    - 功能：将干涉仪光学系统转化为图结构
    - 核心思路：每个镜面拆分为 4 个节点（两侧 × 入射/出射），边连接入射场到反射场和透射场，以及出射场到下一个光学组件的入射场
    - 设计动机：这种分解方式精确反映了电磁场在干涉仪中的物理传播路径

### 损失函数 / 训练策略

- **功率预测**: L1 损失 + 能量守恒正则化项
- **强度预测**: L1 损失（$\frac{W}{m^2}$）
- **数据采集**: 从"理想"干涉仪配置出发，随机游走扰动参数（曲率半径、反射率、间距），每步运行 FINESSE 仿真获取真值
- **训练集构成**: Mixed 模型用 20,000 FP + 4,000 Arm-SRC CC 样本训练

## 实验关键数据

### 主实验（功率预测 L1 损失）

| 训练数据 | 测试: Arm-SRC CC | 测试: Fabry-Perot | 测试: Simple CC |
|----------|:-:|:-:|:-:|
| GAT+MLP (FP) | ∞ | 0.52 | 2.94 |
| GAT+MLP (Mixed) | **0.25** | **0.54** | **3.01** |
| GAT+MLP (Arm-SRC) | 0.24 | 1.36 | 2.98 |
| GAT+KAN (Mixed) | 0.38 | 0.76 | 1.09 |
| MLP Only (Mixed) | 0.41 | 1.32 | 1380.19 |
| KAN Only (Mixed) | 0.19 | 33.98 | 38.91 |
| GraphTransformer+MLP (Mixed) | 0.34 | 0.65 | 1.71 |

### 消融实验

**GAT 层数影响**:

| GAT 层数 | FP | Simple CC | Arm-SRC CC |
|----------|:-:|:-:|:-:|
| 1 | 1.06 | 2.86 | 0.43 |
| 3 | 0.86 | 2.20 | 0.42 |
| 8 | 0.54 | 0.83 | 0.39 |
| 15 | 0.53 | 0.70 | 0.39 |
| 20 | 0.53 | 0.71 | 0.38 |

**强度预测模型对比**:

| 模型 | L1 损失 ($W/m^2$) |
|------|:-:|
| GAT + KAN | **27.2** |
| GAT + MLP（同参数量） | 58.4 |

### 计算效率对比

| 方法 | 单次仿真 (s) | FP 优化 (s) |
|------|:-:|:-:|
| FINESSE | 2.857 | 170.8 |
| SIS | 14.932 | - |
| GNN (Power) | **0.018** | 53.7 |
| GNN (Intensity) | **0.011** | - |

单次仿真加速比：FINESSE 的 **159 倍**，SIS 的 **815 倍**！

### 关键发现

1. **GNN 大幅优于无图结构模型**: MLP Only 和 KAN Only 在未见拓扑上泛化极差（CC 数据集出现 1380 甚至 ∞ 的损失），而 GNN 模型保持低且稳定的损失
2. **GAT 层数边际递减**: 从 1 层到 8 层损失显著下降，15-20 层后改善微弱，说明信息已充分传播
3. **KAN 优于 MLP 用于强度预测**: 利用 KAN 学习径向分布，L1 损失降低 53%（27.2 vs 58.4），因为 KAN 更擅长学习物理特殊函数
4. **混合训练增强泛化**: 仅注入少量 Arm-SRC CC 训练样本（4,000 个）即可使 Mixed 模型在该拓扑上达到与专用模型相当的性能
5. **跨拓扑泛化有限但有用**: FP 训练模型在 CC 上失败（∞），但在架构设计搜索中即使粗略估计也有价值
6. **能量守恒损失有效**: 物理约束正则化项确保预测满足基本物理定律

## 亮点与洞察

1. **首创性应用**: 首次将深度学习应用于光学干涉仪仿真中的电磁场传播预测，开辟了 "ML for instrumentation design" 的新方向
2. **KAN + 对称性利用**: 利用场的径向对称性将 2D 预测降维为 1D，再结合 KAN 的函数逼近优势，既降低了计算量又提高了精度——这是物理先验与网络架构设计深度结合的优秀范例
3. **物理约束损失**: 能量守恒正则化项是一种简单但有效的方式将物理先验注入训练
4. **实用性优先**: 论文明确指出模型不需要极高精度——即使是粗略估计也能帮助优化程序快速剪枝，这种务实的定位很有价值
5. **完整数据集贡献**: 提供了三种干涉仪拓扑的高保真仿真数据集作为 benchmark

## 局限与展望

1. **跨拓扑泛化不足**: 在训练集未包含的干涉仪拓扑上泛化能力有限，这是最关键的局限，因为干涉仪设计优化的核心就是探索新拓扑
2. **仅建模部分物理**: 未包含点吸收体、热透镜效应、散光光束形状等高阶效应，距全尺度干涉仪设计尚有差距
3. **不执行锁定程序**: 模型在已锁定干涉仪数据上训练，但锁定过程本身也需要大量仿真
4. **FP 优化加速有限**: 理论单次加速 159x，但实际优化加速仅 3.2x（170.8s vs 53.7s），瓶颈在于 FINESSE 到 GNN 输入格式转换的开销
5. **缺乏深入消融**: 论文自述为概念验证，许多架构选择（层数、GATv2 vs 其他）缺乏系统性消融
6. **功率跨尺度问题**: 虽然用 $\log P$ 缓解了 kW-mW 的尺度差异，但对极小功率的预测精度仍有限

## 相关工作与启发

- **Pfaff et al. (2021)**: 基于网格的 GNN 物理仿真 → 本文将类似思路应用于光学仿真，但不关注时间演化
- **Alkin et al. (2024)**: Universal Physics Transformers → 通用物理 foundation model 的方向，但本文的场景不涉及时间演化和完整场预测
- **Liu et al. (2024)**: KAN 在学习球谐函数等物理特殊函数上的优势 → 启发了用 KAN 代替 MLP 预测径向强度分布
- **Paganini et al. (2018)**: GAN 加速粒子物理仿真 → 未来可用生成模型建模干涉仪参数扰动的条件分布

**启发**: 本文的核心贡献不在于 GNN 架构创新，而在于将 GNN 应用于一个全新的科学问题，并证明了其可行性。"即使粗略估计也有用" 的定位策略值得其他科学计算加速工作借鉴。KAN 与物理对称性结合的设计模式具有通用价值。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将 GNN 应用于干涉仪仿真，问题选择新颖且有实际价值
- 实验充分度: ⭐⭐⭐ 三种拓扑 + 多架构对比 + 消融 + 速度对比，但缺乏对高阶物理效应的测试
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，物理背景解释到位，附录完整
- 价值: ⭐⭐⭐⭐ 开辟了 ML for instrumentation design 的新方向，数据集贡献有长期价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Over-squashing in Spatiotemporal Graph Neural Networks](over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[NeurIPS 2025\] Sound Logical Explanations for Mean Aggregation Graph Neural Networks](sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)
- [\[NeurIPS 2025\] Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization](logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids](graph_neural_networks_for_efficient_ac_power_flow_prediction_in_power_grids.md)

</div>

<!-- RELATED:END -->
