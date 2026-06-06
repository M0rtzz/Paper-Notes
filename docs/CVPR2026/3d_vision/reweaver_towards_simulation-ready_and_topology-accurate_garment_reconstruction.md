---
title: >-
  [论文解读] ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction
description: >-
  [CVPR2026][3D视觉][服装重建] 提出 ReWeaver 框架，从最少4张多视图RGB图像中联合重建3D服装几何与2D缝纫图案（sewing pattern），通过双路径Transformer预测3D曲面片/曲线及其拓扑连接，再经组内注意力将3D结构展平为2D面板边缘…
tags:
  - "CVPR2026"
  - "3D视觉"
  - "服装重建"
  - "缝纫模式"
  - "拓扑重建"
  - "多视图重建"
  - "物理仿真"
---

# ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction

**会议**: CVPR2026  
**arXiv**: [2601.16672](https://arxiv.org/abs/2601.16672)  
**作者**: Ming Li, Hui Shan, Kai Zheng, Chentao Shen, Siyu Liu, Yanwei Fu, Zhen Chen, Xiangru Huang
**机构**: 浙江大学, 上海创新研究院, 西湖大学, 复旦大学, Adobe, 西安电子科技大学
**代码**: 待确认  
**领域**: 3D视觉  
**关键词**: 服装重建, 缝纫模式, 拓扑重建, 多视图重建, 物理仿真

## 一句话总结

提出 ReWeaver 框架，从最少4张多视图RGB图像中联合重建3D服装几何与2D缝纫图案（sewing pattern），通过双路径Transformer预测3D曲面片/曲线及其拓扑连接，再经组内注意力将3D结构展平为2D面板边缘，首次实现拓扑准确且可直接用于物理仿真的服装资产恢复。

## 背景与动机

高质量3D服装重建在虚拟试穿、数字人、游戏和机器人操作等应用中至关重要。然而现有方法存在两大痛点：

1. **非结构化表示的局限**：现有方法（点云、SDF、3D高斯泼溅等）虽能近似服装几何，但缺乏显式的缝纫结构（seam/panel），难以直接用于物理仿真、服装编辑或重定向。这些表示与工业标准的服装设计流程（以2D缝纫图案为核心）天然不兼容。

2. **已有缝纫图案方法的不足**：
    - 依赖预定义拓扑的方法（如 DiffAvatar）仅适用于简单服装，无法处理未见过的版型
    - 基于视觉-语言模型的方法（如 ChatGarment、AIpparel）通过token化JSON描述生成2D图案，虽然拓扑泛化性更强，但几何精度不足
    - 大多数方法只关注2D图案，忽略了精确的3D几何理解

**核心目标**：同时重建准确的服装**拓扑**（哪些面板/缝线相连）和**几何**（各元素的精确3D形状），使输出既能用于3D感知，又能用于高保真物理仿真。

## 方法详解

### 整体框架

ReWeaver 采用编码器-解码器架构，流程分为四个阶段：

1. **多视图视觉编码**（Section 3.2）：基于 VGGT 的多视图编码器提取统一特征
2. **3D曲线与曲面片预测**（Section 3.3）：双路径 Transformer 预测3D几何与拓扑
3. **2D图案预测**（Section 3.4）：组内注意力将3D结构展平为2D面板边缘
4. **拓扑/几何精炼**：后处理保证面板闭合与拓扑一致性

**术语定义**（2D/3D双空间）：

| 空间 | 表面区域 | 边界线 |
|:---:|:---:|:---:|
| 3D | Patch（曲面片） | Curve（曲线/缝线） |
| 2D | Panel（面板） | Edge（边缘） |

### 多视图视觉编码器

采用 VGGT 的设计思路，处理流程如下：

- 每张输入图像被划分为不重叠的 $16\times16$ patch，经 DINOv2 backbone 嵌入为 token
- 交替堆叠**帧内自注意力**（intra-frame attention，细化单视图特征）和**帧间自注意力**（inter-frame attention，跨视图聚合信息）
- 将最后的帧内和帧间输出拼接，展平所有帧的 token 得到序列 $T_i \in \mathbb{R}^{N_i \times D}$，其中 $D=768$

这种交替注意力设计能渐进式地整合**局部纹理线索**和**全局几何信息**，适应稀疏且未知分布的多视图输入。

### 3D 曲线与曲面片预测（双路径 Transformer）

这是 ReWeaver 的核心模块，负责预测3D几何元素及其拓扑连接关系。

**输入**：视觉 token $T_i$，可学习的面片查询 $Q_p \in \mathbb{R}^{N_p \times D}$（$N_p=200$）和曲线查询 $Q_c \in \mathbb{R}^{N_c \times D}$（$N_c=70$）。查询数量约为训练数据中最大数量的两倍。

**双路径 Transformer 架构**（灵感来自 ComplexGen）：

每一层中，patch 路径和 curve 路径各自执行：

1. **组内自注意力**：同类元素之间交换信息（patch-patch 或 curve-curve）
2. **跨组交叉注意力**：从图像 token 和另一类元素中检索上下文信息
3. **LayerNorm + FFN + 残差连接**

经过多层后得到精炼的 token：$T_p \in \mathbb{R}^{N_p \times D}$ 和 $T_c \in \mathbb{R}^{N_c \times D}$。

**三个解码头**：

#### (1) 概率预测头

通过3层MLP + Sigmoid预测每个查询是否对应有效元素：

$$\sigma_p^i = \text{sigmoid}(f_p^{\text{prob}}(T_p^i)), \quad \sigma_c^i = \text{sigmoid}(f_c^{\text{prob}}(T_c^i))$$

低概率元素通过阈值 $\epsilon_p$、$\epsilon_c$ 过滤，并经**拓扑精炼**得到二值有效掩码 $\boldsymbol{\sigma}_p^{\star}$、$\boldsymbol{\sigma}_c^{\star}$。

#### (2) 几何预测头（HyperNetwork）

这是一个关键创新：**不直接回归点坐标**，而是用超网络（HyperNetwork）生成参数化映射MLP的权重。

对于曲线，超网络 $f_c^{\text{geo}}$ 根据 token $T_c^i$ 生成一个3层MLP，将 $[0,1]$ 映射到 $\mathbb{R}^3$：

$$g_c^i(u) = f_c^{\text{geo}}(T_c^i)(u) \in \mathbb{R}^3, \quad \forall u \in [0,1]$$

对于曲面片，超网络 $f_p^{\text{geo}}$ 生成的MLP将 $[0,1]^2$ 映射到 $\mathbb{R}^3$：

$$g_p^i(u,v) = f_p^{\text{geo}}(T_p^i)(u,v) \in \mathbb{R}^3, \quad \forall u,v \in [0,1]$$

**超网络的优势**：

- 训练时可在任意密度均匀采样，不影响几何的光滑性和连续性
- 推理时支持**自适应采样密度**——小面片稀采、大面片密采，产生近均匀的3D点分布
- 每个 token 参数化一个独立的连续映射，编码丰富的形状信息

#### (3) 连通性预测头

预测 patch $i$ 和 curve $j$ 之间的连接概率：

$$\sigma_{pc}(i,j) = \text{sigmoid}(f_p^{\text{adj}}(T_p^i) \cdot f_c^{\text{adj}}(T_c^j))$$

通过线性投影后做点积再 Sigmoid，得到邻接矩阵 $\sigma_{pc}$，经阈值 $\epsilon_{\text{adj}}$ 过滤和拓扑精炼得到最终的二值连接矩阵 $\sigma_{pc}^{\star} \in \{0,1\}^{N_p \times N_c}$。

### 2D 图案预测（组内注意力展平）

给定有效的 patch/curve token 和精炼后的拓扑，这个模块将3D结构"展平"为2D缝纫图案。

**核心思路**：根据 $\sigma_{pc}^{\star}$，将每个有效 patch token 与其相连的 curve token 分为一组，在**组内**执行注意力操作。

具体流程：

1. 组内的 curve token 先自注意力交换信息
2. 再与关联的 patch token 做交叉注意力
3. 经 LayerNorm + FFN + 残差连接后，得到 edge token $T_e$

对每条连接的曲线 $j \in \partial_i$，用另一个超网络生成MLP，将1D参数映射到归一化2D坐标：

$$g_e^{ij}(u) = f_e^{\text{edge}}(T_e^j)(u) \in [0,1]^2, \quad \forall u \in [0,1]$$

**尺度恢复**：由于2D面板在 $[0,1]^2$ 归一化空间中预测，额外引入一个MLP $f_p^{\text{scale}}$ 从 patch token 预测缩放因子 $s_i$，乘以归一化坐标恢复真实物理尺寸。

**几何精炼**：由于超网络不保证相邻边端点完美对齐，后处理步骤强制边缘闭合，使面板形成封闭回路，可直接用于三角化和仿真。

### 损失函数

通过**匈牙利匹配**建立预测元素与真值的对应关系，总损失包含三项：

**几何损失**（Chamfer Distance）：

$$L_{\text{geo}} = \sum_{g \in \mathcal{G}} w_{\text{geo}}^{(g)} \cdot \text{CD}(V(g), V(m(g)))$$

对所有参数化映射（patch、curve、edge）输出的点集与真值计算倒角距离。

**分类与连通性损失**（BCE）：

$$L_{\text{cls}} = \sum_{\sigma \in \{\boldsymbol{\sigma}_p, \boldsymbol{\sigma}_c, \sigma_{pc}\}} w_{\text{cls}}^{(\sigma)} \cdot \text{BCE}(\sigma, m(\sigma))$$

**尺度损失**（$\ell_2$）：

$$L_{\text{scale}} = \sum_{i=1}^{N_p} w_{\text{scale}} \|s_i - s_{m(i)}^{\text{gt}}\|_2^2$$

## 实验

### 数据集：GCD-TS

在 GarmentCodeData (GCD) 基础上扩展，主要改进：

- 替换了 GCD 中包含强缝线线索的默认纹理，改用约50种 BEDLAM 人体纹理和大量可平铺（tileable）服装纹理
- 每个服装-人体配对从4个视角（前/后/左/右）渲染，带小尺度相机姿态扰动
- 总计约 **100,000** 个带纹理的多视图样本，覆盖广泛的复杂几何与拓扑

### 主实验结果

| 指标 | AIpparel-MV | ReWeaver | 说明 |
|:---|:---:|:---:|:---|
| $\text{Acc}_p$ ↑ 面板数准确率 | 0.4561 | **0.8923** | ReWeaver 高出 **+43.6%** |
| $\text{Acc}_e$ ↑ 边数准确率 | **0.6774** | 0.6570 | 两者相当 |
| $\text{Acc}_o$ ↑ 整体拓扑准确率 | 0.3090 | **0.5863** | ReWeaver 高出 **+27.7%** |
| $\text{CD}_e$ ↓ 2D边倒角距离 | 0.0648 | **0.0395** | 几何更精确 |
| IoU ↑ 面板交并比 | 0.7084 | **0.8080** | 高出 **+10.0%** |

ReWeaver 在6项指标中5项显著优于多视图增强的 AIpparel（AIpparel-MV），尤其在**面板数准确率**上从45.6%跃升到89.2%，表明模型能更可靠地识别服装的拓扑结构。

### 消融实验：拓扑与几何精炼的效果

| 配置 | $\text{CD}_p^{\text{base}}$ ↓ | $\text{CD}_p^{\text{adapt}}$ ↓ | $\text{CD}_c$ ↓ | $\text{Acc}_p$ ↑ | $\text{Acc}_e$ ↑ | $\text{Acc}_o$ ↑ | $\text{CD}_e$ ↓ | IoU ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 有精炼 | 0.0225 | **0.0187** | 0.0264 | 0.8923 | **0.6570** | **0.5863** | **0.0395** | **0.8080** |
| 无精炼 | 0.0225 | 0.0188 | **0.0255** | **0.9101** | 0.5361 | 0.4880 | 0.0416 | 0.7775 |

**关键发现**：

- **拓扑精炼**移除冗余/重复边，使边数准确率（$\text{Acc}_e$）从53.6%提升到65.7%（+12.1%），整体准确率从48.8%提升到58.6%（+9.8%）
- **几何精炼**闭合2D空间中边缘间的细小间隙，产生完全封闭的面板边界，使 IoU 从77.8%提升到80.8%
- 精炼对3D几何指标影响很小（$\text{CD}_p$、$\text{CD}_c$ 几乎不变），说明它主要改善了拓扑一致性和2D面板质量
- 注意无精炼时 $\text{Acc}_p$ 反而略高（0.91 vs 0.89），因为精炼会移除一些被误认为冗余的有效元素，但整体拓扑质量仍有大幅提升

### 自适应采样

- 训练时使用固定 $20\times20$ 的面片采样密度
- 推理时先预采样 $20\times20$ 网格，再根据**空间方差**自适应保留点。小面片剔除过密的点，大面片保持密集采样
- 自适应采样后的 $\text{CD}_p^{\text{adapt}}$ 从0.0225降至0.0187，证明自适应策略有效

## 亮点

- ⭐ **首次联合重建**：同时输出3D服装几何与2D缝纫图案，并维护显式的2D-3D对应关系，使输出直接可用于物理仿真
- ⭐ **超网络参数化**：用 HyperNetwork 生成连续参数化映射，推理时支持任意密度采样和自适应采样，兼具灵活性与几何光滑性
- ⭐ **双路径 Transformer**：patch/curve 双路径的自注意力+交叉注意力设计，有效融合多视图图像证据与结构几何约束
- ⭐ **GCD-TS 数据集**：10万级规模，修复了原GCD中纹理泄漏缝线信息的问题，提升泛化能力
- ⭐ 面板数准确率 89.2%，远超基线的 45.6%，拓扑泛化能力强

## 局限性

- 高质量复杂拓扑+真实感纹理的3D服装数据仍然稀缺，实验输入存在明显的**仿真-真实域差距**（sim-to-real gap）
- 边数准确率（$\text{Acc}_e = 0.657$）相对面板数准确率偏低，说明**细粒度拓扑**预测仍有较大改进空间
- 缺少真实世界图像的定量评估，仅在合成数据集上验证
- 几何精炼依赖后处理启发式规则，不保证百分百闭合成功
- 视图数固定为4个标准视角（前后左右），对更自由的拍摄条件未做充分验证

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将3D几何+2D缝纫图案+拓扑连接联合建模，超网络参数化设计优雅
- 实验充分度: ⭐⭐⭐ — 合成数据上验证全面但缺少真实图像评估，基线仅对比了一个方法
- 写作质量: ⭐⭐⭐⭐ — 术语定义清晰，2D/3D双空间描述到位，框架图易懂
- 价值: ⭐⭐⭐⭐ — 直接输出可仿真资产，对数字人/虚拟试穿/机器人操作领域有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins](motionanymesh_physics-grounded_articulation_for_simulation-ready_digital_twins.md)
- [\[CVPR 2026\] PhysHead: Simulation-Ready Gaussian Head Avatars](physhead_simulation-ready_gaussian_head_avatars.md)
- [\[CVPR 2026\] SwiftTailor: Efficient 3D Garment Generation with Geometry Image Representation](swifttailor_efficient_3d_garment_generation_with_geometry_image_representation.md)
- [\[AAAI 2026\] Pb4U-GNet: Resolution-Adaptive Garment Simulation via Propagation-before-Update Graph Network](../../AAAI2026/3d_vision/pb4u-gnet_resolution-adaptive_garment_simulation_via_propagation-before-update_g.md)
- [\[ICML 2026\] STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics–Physics Dual System](../../ICML2026/3d_vision/stable_simulation-ready_tabletop_layout_generation_via_a_semantics-physics_dual_.md)

</div>

<!-- RELATED:END -->
