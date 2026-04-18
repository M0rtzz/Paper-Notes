---
title: >-
  [论文解读] S2AM3D: Scale-controllable Part Segmentation of 3D Point Clouds
description: >-
  [CVPR2026][3D视觉][点云部件分割] 提出融合2D预训练先验与3D对比监督的点云部件分割框架S2AM3D，通过点一致性编码器获得全局一致的点特征，并设计尺度感知提示解码器实现连续可控的分割粒度调节，在多个基准上大幅超越现有方法。
tags:
  - CVPR2026
  - 3D视觉
  - 点云部件分割
  - 多粒度控制
  - 对比学习
  - 2D-3D联合监督
  - SAM
---

# S2AM3D: Scale-controllable Part Segmentation of 3D Point Clouds

**会议**: CVPR2026  
**arXiv**: [2512.00995](https://arxiv.org/abs/2512.00995)  
**代码**: [项目主页](https://sumuru789.github.io/S2AM3D-website/)  
**领域**: 3d_vision  
**关键词**: 点云部件分割, 多粒度控制, 对比学习, 2D-3D联合监督, SAM

## 一句话总结

提出融合2D预训练先验与3D对比监督的点云部件分割框架S2AM3D，通过点一致性编码器获得全局一致的点特征，并设计尺度感知提示解码器实现连续可控的分割粒度调节，在多个基准上大幅超越现有方法。

## 背景与动机

点云部件级分割是连接细粒度几何细节与高层语义理解的关键任务，在3D内容创建、机器人操控和逆向工程等领域有重要应用。现有方法面临两大挑战：

1. **原生3D方法泛化性差**：高质量3D部件标注成本高昂，现有数据集规模和类别多样性有限（如ShapeNet-Part、PartNet），导致在开放域形状上的泛化性能严重受限
2. **2D先验方法跨视图不一致**：利用SAM等2D模型对3D渲染图进行分割再融合的方法，在遮挡、细长结构和复杂拓扑下会产生跨视图不一致性，累积误差损害全局3D一致性
3. **粒度控制不灵活**：PartField等基于特征聚类的方法依赖后处理聚类，控制不连续且不直观；Point-SAM等基于点提示的方法缺乏显式粒度控制机制

## 方法详解

### 整体框架

S2AM3D采用**解耦训练**策略，分为两个阶段：

- **阶段一**：训练点一致性部件编码器（Point-Consistent Part Encoder），融合2D分割先验与3D对比监督，产生全局一致的逐点特征
- **阶段二**：冻结编码器，训练尺度感知提示解码器（Scale-Aware Prompt Decoder），以点提示索引 $p$ 和可选尺度提示 $s \in [0,1]$ 为条件，实现灵活的部件分割

输入点云 $\mathbf{P} \in \mathbb{R}^{N \times 3}$，编码器输出逐点特征 $\mathbf{F} \in \mathbb{R}^{N \times D}$，解码器根据提示生成概率掩码 $\hat{\mathbf{m}} \in [0,1]^N$。

### 点一致性部件编码器

编码器的核心思想是**在2D蒸馏的基础上叠加3D对比监督**，解决仅靠多视图2D蒸馏带来的跨视图不一致问题。

**基础架构**：采用PVCNN体素编码器提取点特征，转换为三平面（tri-plane）表示 $\mathbf{T} \in \mathbb{R}^{3 \times D \times H \times W}$（$xy$、$yz$、$zx$ 三个正交平面），再通过Transformer块聚合特征。三平面特征从随机视角渲染为2D潜变量，用SAM蒸馏进行监督。

**三平面特征提取**：给定3D坐标 $(x,y,z)$，将点反投影到三个特征平面并求和：

$$\mathbf{F} = \Big[\mathbf{T}_{xy}(x_n, y_n) + \mathbf{T}_{yz}(y_n, z_n) + \mathbf{T}_{zx}(z_n, x_n)\Big]_{n=1}^{N}$$

**3D对比监督**：关键创新在于引入原生3D标签数据的对比学习。约束对比在实例内部（intra-instance）进行，每个mini-batch仅包含一个物体，避免跨实例语义不匹配。对于锚点 $i$ 及其标签 $y_i$，正样本集为：

$$\hat{P}(i) = \{j \in \hat{P} \setminus \{i\} \mid y_j = y_i\}$$

使用带温度 $\tau$ 的余弦相似度 $s_{ij} = \mathbf{f}_i^\top \mathbf{f}_j / \tau$，对比损失为：

$$\mathcal{L}_{\text{contr}} = \frac{1}{|\hat{P}|} \sum_{i \in \hat{P}} -\log \frac{\sum_{j \in \hat{P}(i)} e^{s_{ij}}}{\sum_{j \in \hat{P} \setminus \{i\}} e^{s_{ij}}}$$

该目标使同一部件的点特征紧凑聚合，不同部件的点特征远离，产生全局一致的嵌入和清晰的边界。

### 尺度感知提示解码器

解码器接收编码器输出的点特征 $\mathbf{F}$ 和3D坐标 $\mathbf{P}$，加入3D正弦位置编码得到基础表示：

$$\mathbf{X}^{(0)} = \mathbf{F} + \mathrm{PE}(\mathbf{P})$$

#### 尺度调制器（Scale Modulator）

尺度 $s$ 定义为部件的相对大小（该部件点数占总点数的比例）。对连续尺度 $s \in [0,1]$，构建**可学习正弦嵌入**：

$$\mathbf{e}(s) = \big[\sin(\omega_k s + \phi_k), \ \cos(\omega_k s + \phi_k)\big]_{k=1}^{M}$$

其中 $\{\omega_k, \phi_k\}$ 为可学习参数，$M$ 为频率对数。然后通过**FiLM（Feature-wise Linear Modulation）**在通道维度调制全局特征：

$$[\boldsymbol{\gamma}, \boldsymbol{\beta}] = \text{Linear}(\mathrm{LN}(\mathbf{e}(s)))$$

$$\mathrm{FiLM}(\mathbf{X}; s) = \mathbf{X} \odot (1 + \alpha \boldsymbol{\gamma}) + \alpha \boldsymbol{\beta}$$

其中 $\alpha$ 为可学习标量门控。FiLM与Transformer块交替堆叠 $L_m$ 层：

$$\mathbf{X}^{(\ell+1)} = T_\ell\big(\mathrm{FiLM}(\mathbf{X}^{(\ell)}; s)\big), \quad \ell = 0, \dots, L_m - 1$$

最终得到尺度条件增强表示 $\tilde{\mathbf{F}} = \mathbf{X}^{(L_m)}$。

**尺度Dropout**：训练时以概率0.1随机将 $\mathbf{e}(s)$ 置零，此时FiLM退化为恒等映射，保证推理时无需尺度输入也能工作。

#### 双向交叉注意力（Bi-directional Cross-Attention）

单向交叉注意力难以在一次前向传播中同时完成上下文聚合和细粒度细化。双向交叉注意力让提示点特征 $\tilde{\mathbf{F}}_p \in \mathbb{R}^{1 \times D}$ 与全局特征 $\tilde{\mathbf{F}} \in \mathbb{R}^{N \times D}$ 双向交互：

$$\mathbf{q}^{(\ell+1)} = \mathbf{q}^{(\ell)} + \mathrm{CAttn}(\mathbf{q}^{(\ell)}; \mathbf{Y}^{(\ell)})$$

$$\mathbf{Y}^{(\ell+1)} = \mathrm{FFN}\Big(\mathbf{Y}^{(\ell)} + \mathrm{CAttn}(\mathbf{Y}^{(\ell)}; \mathbf{q}^{(\ell+1)})\Big)$$

堆叠 $L_d$ 层后，经MLP和Sigmoid输出逐点概率掩码：

$$\hat{\mathbf{m}} = \sigma(\mathrm{MLP}(\mathbf{H})) \in [0,1]^N$$

### 损失函数

分割损失采用**动态加权BCE + Dice**的混合目标：

$$\mathcal{L}_{\text{seg}} = \lambda_{\text{bce}} \mathrm{BCE}_{\text{dyn}}(\hat{\mathbf{m}}, \mathbf{m}) + \lambda_{\text{dice}} \left(1 - \frac{2\hat{\mathbf{m}}^\top \mathbf{m}}{\|\hat{\mathbf{m}}\|_1 + \|\mathbf{m}\|_1}\right)$$

动态BCE根据每个样本的正样本比例 $\pi$ 自适应计算权重 $\beta = (1-\pi)/(\pi + \varepsilon)$，缓解类别不平衡。Dice项直接优化集合级别的重叠度，对小部件和长尾分布更鲁棒。

### 数据集构建

构建了包含**10万+点云实例、约120万部件标注**的大规模数据集，数据来源于Objaverse，覆盖400+类别。自动化流水线包括三个步骤：

1. **部件标注**：基于表面积比例采样并分配部件标签
2. **质量过滤**：训练二分类PointNet验证器，自动筛选标注不合理的样本
3. **连通性细化**：对同一标签下空间不连通的区域，用DBSCAN聚类拆分为独立标签

## 实验

### 主实验结果

**交互式分割（点提示 → 单部件掩码）**：

| 方法 | PartObjaverse-Tiny (IoU%) | PartNet-E (IoU%) | 平均 |
|------|:---:|:---:|:---:|
| Point-SAM | 31.46 | 50.23 | 40.85 |
| P3-SAM | 35.05 | 39.98 | 37.52 |
| **S2AM3D** | **46.47** | **62.52** | **54.50** |
| **S2AM3D (+scale)** | **61.19** | **77.51** | **69.35** |

**全分割（预测所有点的部件标签）**：

| 方法 | PartObjaverse-Tiny (IoU%) | PartNet-E (IoU%) | 平均 |
|------|:---:|:---:|:---:|
| Find3D | 20.76 | 21.69 | 21.23 |
| SAMPart3D | 48.79 | 56.17 | 52.48 |
| SAMesh | - | 26.66 | - |
| PartField | 51.54 | 59.10 | 55.32 |
| P3-SAM | 58.10 | 65.39 | 61.75 |
| **S2AM3D** | **63.29** | **77.98** | **70.64** |

### 消融实验

| 设置 | PartObjaverse-Tiny | PartNet-E | 平均 |
|------|:---:|:---:|:---:|
| **+scale 完整模型** | **61.19** | **77.51** | **69.35** |
| +scale 去掉3D监督 | 53.94 | 64.11 | 59.03 |
| +scale 去掉自建数据 | 53.12 | 66.12 | 59.62 |
| **No scale 完整模型** | **46.47** | **62.52** | **54.50** |
| No scale 去掉3D监督 | 41.14 | 55.39 | 48.27 |
| No scale 去掉自建数据 | 42.12 | 58.56 | 50.34 |
| No scale 去掉尺度嵌入 | 42.31 | 58.28 | 50.30 |

### 关键发现

- **3D对比监督是最大性能贡献者**：去掉后 +scale 设置下平均IoU下降10.32%，特征可视化显示边界模糊和内部不一致
- **尺度提示带来显著提升**：加入尺度条件后，交互式分割平均IoU从54.50%提升至69.35%（+14.85%）
- **自建数据集的关键作用**：替换为PartNet训练数据后性能明显下降，证明大规模高质量数据的分布互补价值
- **尺度嵌入增强解码鲁棒性**：即使推理时不提供尺度，有尺度嵌入训练的模型仍优于无尺度嵌入版本（54.50 vs 50.30）
- **仅需XYZ坐标**：相比Point-SAM需要颜色、P3-SAM需要法线，S2AM3D仅用坐标即可取得最优结果

## 亮点

- 2D-3D联合训练范式设计巧妙：用2D先验提供泛化能力，用3D对比监督保证全局一致性，互补性强
- 尺度感知解码器通过FiLM + 双向交叉注意力实现连续粒度控制，支持从细到粗的平滑过渡
- 自动化数据流水线（标注→过滤→细化）具有可扩展性，构建了目前最大规模的3D部件分割数据集之一
- 同一框架统一了交互式分割和全分割两种任务

## 局限与展望

- 仅支持点提示和尺度信号交互，未来可引入文本指令实现更直观的语义交互
- 依赖PartField预训练参数初始化编码器，编码器本身的创新有限
- 数据集标注依赖自动化流水线，过滤策略（PointNet验证器）的泛化性未充分验证
- 未讨论推理速度和内存消耗，10000个采样点的三平面 $448 \times 512 \times 512$ 维度较大
- 评估数据集规模偏小（PartObjaverse-Tiny仅200样本），更大规模评估可增强说服力

## 评分

- 新颖性: ⭐⭐⭐⭐ — 2D-3D联合监督 + 连续尺度控制的组合设计有创意，但编码器基础架构沿用PartField
- 实验充分度: ⭐⭐⭐⭐ — 两种任务设置+消融+可视化较完整，但评估数据集规模偏小
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，公式推导规范，图表质量高
- 价值: ⭐⭐⭐⭐ — 为3D部件分割提供了实用的粒度控制方案，数据集贡献有附加价值
