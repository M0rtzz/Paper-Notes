---
title: >-
  [论文解读] SENC: Handling Self-collision in Neural Cloth Simulation
description: >-
  [ECCV2024][图学习][cloth simulation] 提出 SENC，通过基于 Global Intersection Analysis (GIA) 的自碰撞损失和自碰撞感知图神经网络，首次在自监督神经布料模拟中有效解决布料自碰撞问题。 自监督神经布料模拟因无需昂贵的真实数据而具有吸引力，但现有方法普遍面临一个…
tags:
  - "ECCV2024"
  - "图学习"
  - "cloth simulation"
  - "self-collision"
  - "图神经网络"
  - "自监督学习"
  - "Global Intersection Analysis"
---

# SENC: Handling Self-collision in Neural Cloth Simulation

**会议**: ECCV2024  
**arXiv**: [2407.12479](https://arxiv.org/abs/2407.12479)  
**代码**: [zycliao/senc](https://zycliao.github.io/senc)  
**领域**: 图学习  
**关键词**: cloth simulation, self-collision, graph neural network, self-supervised learning, Global Intersection Analysis  
**机构**: 香港大学

## 一句话总结

提出 SENC，通过基于 Global Intersection Analysis (GIA) 的自碰撞损失和自碰撞感知图神经网络，首次在自监督神经布料模拟中有效解决布料自碰撞问题。

## 背景与动机

自监督神经布料模拟因无需昂贵的真实数据而具有吸引力，但现有方法普遍面临一个关键问题：**布料自碰撞**。自碰撞在多种场景下频繁出现，例如手臂紧贴身体时腋下区域的衣物自穿透、裙子前后摆的重叠等。

现有碰撞检测与响应技术无法直接应用于自监督神经模拟器，主要原因包括：

1. **Incremental Potential Contact (IPC)**：采用 barrier 方法防止碰撞，假设模拟从无碰撞状态开始。碰撞能量在碰撞对接近时趋于无穷大，导致梯度爆炸，不适用于训练时随机实例化衣物状态的神经模拟器
2. **基于边-边碰撞的惩罚函数方法**：同样需要从无碰撞状态出发
3. **基于 Signed Distance Function (SDF) 的方法**：SDF 梯度仅局部推动粒子向最近表面移动，容易陷入局部最优——当一部分深度穿透另一部分时，中间区域可能被困在 mesh 内部

## 核心问题

如何在自监督神经布料模拟框架中有效处理布料自碰撞，既不要求无碰撞初始化，又能正确引导穿透区域的顶点解除碰撞？

## 方法详解

### 1. 自碰撞感知图神经网络 (Self-collision-aware GNN)

基于 MeshGraphNets 和 HOOD 的 GNN 架构进行扩展。先前方法仅根据 mesh 拓扑连接构建图，消息沿拓扑传播，无法处理拓扑距离远但空间距离近的自碰撞。

**关键改进**：根据顶点的空间距离构建额外的自碰撞边。对每个布料顶点 $\mathbf{v}_i$，搜索空间距离 $\|\mathbf{v}_i - \mathbf{v}_j\| < r$（$r = 2$ cm）的其他顶点 $\mathbf{v}_j$，并排除原始 mesh 边（相邻顶点不会自碰撞），构建自碰撞边 $\mathbf{e}^{\text{self-col}}_{ij}$。

节点特征更新公式扩展为：

$$\mathbf{x}'_i \leftarrow f_{\text{node}}\left(\mathbf{x}, \sum_j \mathbf{e}'^{\text{ body}}_{ij}, \sum_j \mathbf{e}'^{\text{ cloth}}_{ij}, \sum_j \mathbf{e}'^{\text{ self-col}}_{ij}\right)$$

此外，通过将外力附加到输入节点特征来建模可变外力。训练时每次迭代生成随机方向和大小的力，叠加到重力上。

### 2. 基于 GIA 的自碰撞损失

自碰撞损失基于布料自穿透形成的体积，通过以下四个步骤计算：

**步骤一：衣物闭合 (Garment Closure)**。填充衣物的开口（如袖口），使穿透区域形成闭合 mesh。方法是计算每个开口边界的平均位置，添加扇形三角形连接。

**步骤二：重新网格化 (Remeshing)**。对存在自交叉的配置进行重新网格化，使所有交叉点恰好位于重新网格化后的边上。交叉点用重心坐标表示，以支持反向传播。

**步骤三：全局交叉分析 (Global Intersection Analysis)**。找到所有交叉路径（intersection paths），区分两种自碰撞类型：
- 单一区域折叠产生的一条交叉路径（如严重弯曲的肘部）
- 两个不同区域相交产生的两条交叉路径（如环面自交叉）

通过 loop vertex（两个相交三角形共享的顶点）区分这两种情况。使用并行 flood fill 算法提取穿透面集合：同时从交叉路径两侧遍历，先完成遍历的一侧被视为穿透区域。

**步骤四：计算穿透体积**。对由 GIA 提取的穿透面，通过对每个穿透面与原点构成的四面体的有符号体积求和来计算穿透体积。由于重新网格化步骤中的新顶点均用原始顶点的重心坐标表示，体积损失可通过神经网络反向传播。

### 3. 整体能量模型

总损失函数包含七项：

$$\mathcal{L} = \mathcal{L}_{\text{self-col}} + \mathcal{L}_{\text{col}} + \mathcal{L}_{\text{stretching}} + \mathcal{L}_{\text{bending}} + \mathcal{L}_{\text{ext-force}} + \mathcal{L}_{\text{inertia}} + \mathcal{L}_{\text{friction}}$$

其中 $\mathcal{L}_{\text{self-col}}$ 惩罚自穿透体积，$\mathcal{L}_{\text{col}}$ 基于人体 SDF 防止衣物-人体碰撞，$\mathcal{L}_{\text{stretching}}$ 和 $\mathcal{L}_{\text{bending}}$ 分别基于 St.Venant-Kirchhoff 材料模型和弯曲能量。

### 4. 训练策略

- 使用 AMASS 数据集的 52 个动作捕捉序列训练
- 两阶段训练：先无自碰撞损失预训练 120K 轮，再加入全部损失训练 70K 轮（早期网络输出噪声大，GIA 计算耗时且易产生数值误差）
- 单张 RTX 4090 总训练约 48 小时

## 实验关键数据

在 AMASS 数据集的 4 个测试序列（2175 帧）上评估，使用平均自碰撞损失和超过阈值的帧比例作为指标：

| 方法 | T-shirt $\mathcal{L}_{\text{self-col}}$ (×10⁻³) | T-shirt %(0.01) | Skirt $\mathcal{L}_{\text{self-col}}$ (×10⁻³) | Skirt %(0.01) |
|------|:---:|:---:|:---:|:---:|
| SNUG | 52.00 | 38.76 | N/A | N/A |
| NCS | 75.05 | 70.58 | 476.03 | 100 |
| HOOD | 26.10 | 22.62 | 7.33 | 9.84 |
| **SENC** | **1.80** | **3.72** | **1.59** | **2.71** |

SENC 在所有指标上大幅超越现有方法，T-shirt 上自碰撞损失降低约 14 倍（vs HOOD），skirt 上降低约 4.6 倍。

### 消融实验

| 变体 | T-shirt $\mathcal{L}_{\text{self-col}}$ | Skirt $\mathcal{L}_{\text{self-col}}$ |
|------|:---:|:---:|
| 面积损失替代体积损失 | 28.81 | 8.80 |
| 无自碰撞边 | 1.14 | 7.92 |
| 保留 mesh 边 | 2.37 | 2.02 |
| **SENC (完整)** | **1.80** | **1.59** |

- 面积损失几乎无效：减少穿透面积的梯度无法有效引导顶点解除碰撞
- 无自碰撞边在 T-shirt 有效但 skirt 失败：skirt 碰撞区域拓扑距离更大
- 保留 mesh 边略差于排除版本，且增加计算量

## 亮点

1. **首次解决神经布料模拟中的自碰撞**：通过 GIA 提取穿透体积构造损失，梯度自然形成解除碰撞的力
2. **自碰撞感知 GNN**：通过空间距离构建额外边，弥补拓扑图对远距离碰撞的盲区
3. **可微分设计**：所有新顶点用重心坐标表示，穿透体积损失可端到端反向传播
4. **外力建模**：支持用户施加风力等外力，增强模拟动态性
5. **通用性**：可无缝集成到现有神经布料模拟框架中

## 局限与展望

1. **计算效率**：穿透体积计算中的重新网格化和 GIA 步骤耗时较长
2. **衣物闭合限制**：仅能处理容易闭合的 mesh；复杂拓扑衣物（如长裙底部开口）的闭合可能抑制合理变形或遗漏碰撞
3. **闭合启发式的两难**：以长裙为例——闭合底部会阻止下摆正常上移，不闭合则无法检测前后摆碰撞
4. **未扩展到多层布料和 3D 可变形角色**

## 与相关工作的对比

| 方法 | 自监督 | 处理自碰撞 | 动态模拟 | 任意拓扑 |
|------|:---:|:---:|:---:|:---:|
| SNUG | ✓ | ✗ | 有限 | ✗ |
| NCS | ✓ | ✗ | ✓ | ✗ |
| HOOD | ✓ | ✗ | ✓ | ✓ |
| ClothCombo | ✓ | 简单排斥力 | 准静态 | ✓ |
| ContourCraft | ✓ | 轮廓长度（可能反向） | ✓ | ✓ |
| **SENC** | **✓** | **✓ (GIA体积)** | **✓** | **✓** |

与 ClothCombo 的简单排斥损失不同，SENC 基于穿透体积的梯度能正确引导碰撞解除方向。与 ContourCraft 基于自交叉轮廓长度的损失相比，SENC 不会在真正解除方向与减少轮廓长度方向相反时加剧碰撞。

## 启发与关联

- GIA 提取穿透体积的思路可推广到其他需要处理 mesh 自交叉的任务，如 3D 生成中的 mesh 质量优化
- 空间距离构建额外图边的策略对任何需要跨拓扑远距离交互的 GNN 任务都有参考价值
- 未来可探索 winding number 或 electronic flux 来计算无需闭合边界的近似穿透体积

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将 GIA 引入神经布料模拟处理自碰撞，自碰撞感知 GNN 设计合理
- 实验充分度: ⭐⭐⭐⭐ — 多种衣物类型、定量定性对比全面，消融实验充分
- 写作质量: ⭐⭐⭐⭐ — 问题动机清晰，方法描述详尽，图示直观
- 价值: ⭐⭐⭐⭐ — 解决了神经布料模拟中长期未解决的关键问题，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](../../NeurIPS2025/graph_learning/self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](../../ACL2026/graph_learning/graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[ECCV 2024\] Confidence Self-Calibration for Multi-Label Class-Incremental Learning](confidence_self-calibration_for_multi-label_class-incremental_learning.md)
- [\[ICML 2025\] HGOT: Self-supervised Heterogeneous Graph Neural Network with Optimal Transport](../../ICML2025/graph_learning/hgot_self-supervised_heterogeneous_graph_neural_network_with_optimal_transport.md)
- [\[ICML 2025\] EvoMesh: Adaptive Physical Simulation with Hierarchical Graph Evolutions](../../ICML2025/graph_learning/evomesh_adaptive_physical_simulation_with_hierarchical_graph_evolutions.md)

</div>

<!-- RELATED:END -->
