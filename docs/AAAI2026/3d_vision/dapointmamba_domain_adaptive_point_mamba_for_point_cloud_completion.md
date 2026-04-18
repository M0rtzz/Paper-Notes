---
title: >-
  [论文解读] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion
description: >-
  [AAAI 2026][3D视觉][点云补全] 首次将 Mamba（SSM）引入无监督域自适应点云补全（UDA PCC），提出 DAPointMamba 框架，通过跨域 Patch 级扫描、空间 SSM 对齐和通道 SSM 对齐三个模块，在保持线性复杂度和全局感受野的同时实现了跨域高质量点云补全。
tags:
  - AAAI 2026
  - 3D视觉
  - 点云补全
  - 域自适应
  - State Space Model
  - Mamba
  - 跨域对齐
---

# DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion

**会议**: AAAI 2026  
**arXiv**: [2511.20278](https://arxiv.org/abs/2511.20278)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 点云补全, 域自适应, State Space Model, Mamba, 跨域对齐

## 一句话总结

首次将 Mamba（SSM）引入无监督域自适应点云补全（UDA PCC），提出 DAPointMamba 框架，通过跨域 Patch 级扫描、空间 SSM 对齐和通道 SSM 对齐三个模块，在保持线性复杂度和全局感受野的同时实现了跨域高质量点云补全。

## 研究背景与动机

点云补全（PCC）是 3D 视觉的基础任务，广泛应用于自动驾驶、机器人和虚拟现实。然而现有监督方法在跨域部署时性能严重下降，因为不同传感器和场景带来的分布偏移。

现有 UDA PCC 方法主要面临两个瓶颈：

**CNN 架构受限于局部感受野**：基于卷积的骨干网络无法建模全局几何结构，限制了域不变特征学习

**Transformer 架构面临二次复杂度**：DAPoinTr 虽然引入了全局建模，但注意力机制的二次复杂度导致计算效率低下，尤其对长序列 patch 不友好

Mamba/SSM 模型天然具备全局感受野和线性复杂度的优势，但直接将 SSM 应用于 UDA PCC 会遇到以下挑战：

- **空间拓扑破坏**：将稀疏非结构化 3D 点云直接序列化为 1D 序列会破坏空间拓扑和局部几何特征
- **缺乏域不变特征设计**：现有 Point Mamba 架构缺少针对域迁移的专门设计

## 方法详解

### 整体框架

DAPointMamba 包含三个核心组件，形成从局部到全局的跨域对齐体系：

1. **Cross-Domain Patch-Level Scanning (CDPS)**：跨域 Patch 级扫描，保证序列化过程中的空间对应
2. **Cross-Domain Spatial SSM Alignment (CDSA)**：跨域空间 SSM 对齐，解决细粒度空间不一致
3. **Cross-Domain Channel SSM Alignment (CDCA)**：跨域通道 SSM 对齐，解决全局语义不一致

### 关键设计

#### CDPS：跨域 Patch 级扫描

CDPS 的核心思想是通过共享坐标归一化和 Z-order 序列化，确保源域和目标域的 patch 在空间上对齐。

具体步骤：
1. 计算源域和目标域的共享最小坐标：$C_{min} = min(min(X_s, dim=1), min(X_t, dim=1))$
2. 归一化并离散化到共享网格空间：$G_s = [X_s - C_{min} * scale]$，$G_t = [X_t - C_{min} * scale]$
3. 使用一致的 Z-order 曲线编码将 3D 坐标映射为 1D 序列
4. 按 Z-order 值排序后，将序列划分为 $G$ 个 patch，每个包含 $K$ 个点

由于统一归一化和 Z-order 序列化，第 $g$ 个 patch 在两个域中对应同一空间区域，实现精确的 patch 级对齐。

#### CDSA：跨域空间 SSM 对齐

CDSA 通过基于相似度的特征调制来强化局部空间对齐：

1. 对 patch 级特征应用深度可分离 1D 卷积：$\mathcal{D}_s = DWConv(P_s^G)$，$\mathcal{D}_t = DWConv(P_t^G)$
2. 计算余弦相似度作为空间相似性权重：$\mathcal{W}_{spatial} = cos(D_s, D_t)$
3. 使用相似性权重调制 patch 特征：$\tilde{X}_s = P_s^G \odot W_{spatial}$
4. 通过 MSE 损失促进跨域局部特征一致性：$\mathcal{L}_{sp} = \frac{1}{BDG}\sum(\tilde{X}_s - \tilde{X}_t)^2$

设计直觉：高相似度区域保持不变，低相似度区域的特征被抑制，从而引导模型关注域间一致的空间结构。

#### CDCA：跨域通道 SSM 对齐

CDCA 针对全局语义不一致问题，通过通道混合和自适应调制实现全局对齐：

1. **全局特征计算**：对各 patch 求平均得到 $g_s, g_t \in \mathbb{R}^{B \times D}$
2. **对齐强度估计**：$\alpha = Sigmoid(MLP([g_s, g_t])) \in \mathbb{R}^{B \times 1}$
3. **通道交叉混合**：将特征通道分为 $S$ 段，交替拼接源域和目标域片段

    - $X_{s,mix} = [X_s^{(1)}, X_t^{(2)}, X_s^{(3)}, X_t^{(4)}, \cdots]$
4. **自适应相似度调制**：计算混合表示的余弦相似度，结合 $\alpha$ 生成自适应权重
5. **通道对齐损失**：$\mathcal{L}_{ch} = \frac{1}{BDG}\sum(\tilde{F}_s - \tilde{F}_t)^2$

设计亮点：通过信息交叉混合打破域边界，让源域和目标域的全局语义特征能够互相感知。

### 损失函数 / 训练策略

总损失函数：

$$\mathcal{L}_{total} = Loss_{(CD)} + \lambda L_{sp} + \beta L_{ch}$$

- $Loss_{(CD)}$：Chamfer Distance 重建损失
- $\lambda = 0.1$，$\beta = 0.1$
- 初始学习率 $1 \times 10^{-3}$，权重衰减 $5 \times 10^{-2}$，batch size 32
- 骨干：PointMamba 的 refinement 模块

## 实验关键数据

### 主实验

**3D-FUTURE 数据集（CD↓，×10⁴）：**

| 方法 | Avg | Cabinet | Chair | Lamp | Sofa | Table |
|------|-----|---------|-------|------|------|-------|
| DAPoinTr | 22.35 | 18.46 | 17.60 | 27.91 | 23.08 | 24.71 |
| **DAPointMamba** | **20.40** | 19.35 | **16.21** | **22.81** | **22.38** | **21.25** |

**ModelNet 数据集（CD↓，×10⁴）：**

| 方法 | Avg | Plane | Car | Chair | Lamp | Sofa | Table |
|------|-----|-------|-----|-------|------|------|-------|
| DAPoinTr | 13.79 | 2.38 | 8.04 | 13.83 | 33.26 | 12.72 | 12.51 |
| **DAPointMamba** | **13.11** | **2.30** | **7.58** | **13.15** | **32.04** | **12.48** | **11.08** |

**Real-World Scans（UCD↓/UHD↓，×10⁴/×10²）：**

| 方法 | ScanNet-Chair | KITTI-Car |
|------|--------------|-----------|
| DAPoinTr | 1.1/2.7 | 0.45/1.8 |
| **DAPointMamba** | **0.95**/2.8 | **0.40**/2.1 |

### 消融实验

**各模块逐步添加效果（3D-FUTURE Avg CD↓）：**

| Baseline | +CDPS | +CDSA | +CDCA |
|----------|-------|-------|-------|
| 23.38 | 21.73 | 21.17 | **20.40** |

**计算效率对比：**

| 模型 | Params(M) | FLOPs(G) | Time(ms) |
|------|-----------|----------|----------|
| DAPoinTr | 36.904 | 24.912 | 23.774 |
| **DAPointMamba** | **9.571** | **5.192** | **3.820** |

### 关键发现

- 相比 DAPoinTr，参数量减少 74%，FLOPs 减少 79%，推理延迟减少 84%
- CDPS 贡献最大（降低 1.65），CDCA 在高方差类别（lamp、table）上改进最显著
- 在真实世界扫描数据上，UCD 全面优于前方法，但 UHD（最大误差）略逊——这是因为方法优化的是整体形状而非极端点

## 亮点与洞察

1. **首次探索 SSM 在 UDA PCC 中的适配性**，填补了 Mamba 在域自适应点云任务上的研究空白
2. **三层对齐体系设计精巧**：CDPS（patch 空间对应）→ CDSA（细粒度空间对齐）→ CDCA（全局语义对齐），从局部到全局逐层递进
3. **Z-order 曲线共享归一化**是一个简洁但有效的跨域空间对齐手段
4. **线性复杂度 + 高精度的平衡**：在性能超越 Transformer 方案的同时（CD 降低 1.95），复杂度却大幅降低

## 局限与展望

1. UHD 指标（最大点误差）表现一般，可能需要引入边界点的特殊处理
2. 仅验证了 synthetic→real 和 synthetic→synthetic 的迁移，缺少 real→real 场景
3. 通道混合策略（偶数/奇数交替）较为固定，可以探索自适应混合比例
4. 评估指标中 Cabinet 类别表现不如 DAPoinTr，表明某些几何形状的适配仍有改进空间

## 相关工作与启发

- **DAPoinTr**（SOTA 基线）：Transformer 架构在 UDA PCC 上的先驱，但二次复杂度是瓶颈
- **PointMamba**：验证了 SSM 在点云分析中的有效性，DAPointMamba 在此基础上引入域自适应能力
- **Z-order 曲线**：经典空间索引方法，在此被巧妙用于跨域空间对齐的统一序列化
- 启发：SSM/Mamba 在其他 3D 跨域任务（如检测、分割）中的适配性值得进一步探索

## 评分

- 新颖性: ⭐⭐⭐⭐ （首次将 Mamba 引入 UDA PCC，三个模块设计有新意）
- 实验充分度: ⭐⭐⭐⭐ （多个 benchmark + 真实数据 + 效率对比 + 可视化）
- 写作质量: ⭐⭐⭐⭐ （逻辑清晰，图表丰富）
- 价值: ⭐⭐⭐⭐ （为 Mamba 在域自适应 3D 任务中的应用开辟了新方向）
