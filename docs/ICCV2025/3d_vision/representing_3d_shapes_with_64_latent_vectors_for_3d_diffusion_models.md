---
title: >-
  [论文解读] Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models
description: >-
  [ICCV 2025][3D视觉][3D VAE] 提出COD-VAE，通过两阶段自编码器方案（渐进式编码器 + Triplane解码器 + 不确定性引导Token剪枝），将3D形状编码为仅64个1D潜在向量，在保持重建质量的同时实现16×压缩比和20.8×生成加速。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D VAE
  - 紧凑潜在空间
  - 3D扩散模型
  - Triplane解码
  - 剪枝
---

# Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models

**会议**: ICCV 2025  
**arXiv**: [2503.08737](https://arxiv.org/abs/2503.08737)  
**代码**: [GitHub](https://github.com/join16/COD-VAE)  
**领域**: 3D视觉  
**关键词**: 3D VAE, 紧凑潜在空间, 3D扩散模型, Triplane解码, Token Pruning

## 一句话总结

提出COD-VAE，通过两阶段自编码器方案（渐进式编码器 + Triplane解码器 + 不确定性引导Token剪枝），将3D形状编码为仅64个1D潜在向量，在保持重建质量的同时实现16×压缩比和20.8×生成加速。

## 研究背景与动机

在潜在扩散模型（LDM）框架中，通过VAE构建压缩潜在空间是高效3D扩散模型的关键。然而3D领域面临特殊挑战：3D物体本质上是不规则、稀疏且连续的，直接生成更加困难。

**现有方法的问题**：

**显式3D表示（点云/体素/八叉树）**：需要更大的潜在空间和专门设计的网络，训练和扩展困难

**VecSet框架**：3DShape2VecSet建立了将3D形状编码为1D向量集合的基础框架，但其核心的cross-attention层作为可学习的下采样/插值，压缩比有限——VecSet需要512甚至1024个潜在向量才能实现高质量重建

**解码瓶颈**：VecSet直接将潜在向量映射到neural fields，需要处理超过200万个查询点（$128^3$网格），计算开销巨大

**核心问题**：大量潜在向量导致后续扩散模型计算成本极高（self-attention的二次复杂度），严重制约3D生成效率。

**核心思路**：用中间表示空间（intermediate representation spaces）作为桥梁替代直接映射，实现更高压缩比的编码和更高效的解码。

## 方法详解

### 整体框架

COD-VAE采用两阶段自编码器方案：
- **编码器**：点云 → 中间点patches $\mathcal{H}$（适度压缩）→ 紧凑潜在向量 $\mathcal{Z}$（高压缩）→ 回传全局信息到高分辨率点
- **解码器**：潜在向量 → triplane特征重建 → 浅层MLP解码neural fields

关键创新是引入中间表示空间（点patches和triplanes）作为3D点和潜在向量之间的桥梁。

### 关键设计

1. **渐进式编码器（Progressive Encoder）**：

   三组不同分辨率的特征集合：高分辨率点特征 $\mathcal{G}^{(0)} \in \mathbb{R}^{N \times C}$（$N$个）、中间点patches $\mathcal{H}^{(0)} \in \mathbb{R}^{L \times C}$（$L$个）、紧凑向量 $\mathcal{F} \in \mathbb{R}^{M \times C}$（$M$个），其中 $M \ll L \ll N$。
   
   每个编码器block执行渐进转换：
   
   $$\mathcal{H}^{(l)} = \text{SelfAttn}^3(\text{CrossAttn}(\mathcal{H}^{(l-1)}, \mathcal{G}^{(l-1)}))$$
   $$\mathcal{F}^{(l)} = \text{SelfAttn}(\text{CrossAttn}(\mathcal{F}^{(l-1)}, \mathcal{H}^{(l)}))$$
   $$\mathcal{G}^{(l)} = \text{CrossAttn}(\mathcal{G}^{(l-1)}, \mathcal{F}^{(l)})$$
   
   核心思想：高分辨率 → 中等分辨率 → 紧凑向量的渐进压缩，再将全局信息回传到高分辨率点以进一步精炼。这种渐进方式比VecSet的直接cross-attention映射实现了更高的压缩比。

2. **Triplane解码器**：

   与VecSet直接将潜在向量映射到查询点occupancy值不同，COD-VAE先重建密集triplane特征，再用bilinear插值 + 浅层MLP解码neural fields。
   
   - 使用可学习token序列 $\mathbf{e} \in \mathbb{R}^{(R/f \times R/f) \times C}$ 表示triplane tokens位置
   - 初始化：通过cross-attention从解压向量 $\mathcal{F}'$ 查询初始triplane tokens
   - Token处理：ViT-style transformer blocks处理（含不确定性剪枝）
   - 最终：linear层将tokens投影为triplane特征
   
   对查询点 $\mathbf{q}$，从三个平面检索特征并求和，通过浅层MLP得到occupancy值。这消除了VecSet中处理200万+查询点的cross-attention瓶颈。

3. **不确定性引导Token剪枝（Uncertainty-Guided Token Pruning）**：

   虽然triplane解码高效，但处理密集triplane的transformer计算仍然昂贵（与triplane分辨率二次增长）。
   
   设计：在解码器开头使用辅助不确定性head预测重建误差：
   
   $$u(\mathbf{q}) = \psi_{xy}(U_{xy}, \mathbf{q}) \cdot \psi_{yz}(U_{yz}, \mathbf{q}) \cdot \psi_{xz}(U_{xz}, \mathbf{q})$$
   
   只保留不确定性最高的25% tokens进行后续处理，剪枝75%的简单区域计算。不确定性head通过MSE损失训练预测重建误差 $\mathcal{L}_{rec}(\mathbf{q})$。

### 损失函数 / 训练策略

**两阶段训练**：

- **第一阶段**：训练自编码器（无KL block和latent decoder），使用binary cross entropy loss + 不确定性损失
- **第二阶段**：冻结第一阶段组件，训练KL block和latent decoder，使用MSE损失对齐latent decoder输出 $\mathcal{F}'$ 和编码器输出 $\mathcal{F}$，加KL散度正则 + 重建误差

两阶段策略让latent decoder专注于通道解压，提高VAE精度。

**实现细节**：4个encoder blocks + 12个decoder layers，$C = 512$, $D = 32$, patch size $f = 8$。训练采样4096个查询点（均匀）+ 4096个近表面点。

## 实验关键数据

### 主实验（ShapeNet重建）

| 方法 | M | IoU↑ | CD↓ | F1↑ |
|------|---|------|-----|-----|
| VecSet (AE) | 32 | 87.8 | 0.021 | 91.3 |
| VecSet (AE) | 64 | 91.2 | 0.017 | 94.7 |
| VecSet (AE) | 512 | 96.3 | 0.013 | 98.0 |
| **COD-VAE (AE)** | **32** | **96.1** | **0.012** | **98.0** |
| **COD-VAE (AE)** | **64** | **96.5** | **0.012** | **98.2** |
| VecSet (VAE) | 512 | 96.2 | 0.013 | 98.0 |
| **COD-VAE (VAE)** | **64** | **96.3** | **0.012** | **98.0** |

COD-VAE仅用64个向量即超越VecSet-512的重建质量。用32个向量就达到VecSet-512的水平（IoU 96.1 vs 96.2），实现**16×压缩**。

### 生成实验（ShapeNet类别条件生成）

| 方法 | M | Rendering-FID↓ | Surface-FPD↓ | 采样吞吐量↑ | 全流程吞吐量↑ |
|------|---|----------------|-------------|------------|------------|
| VecSet | 32 | 65.56 | 0.800 | 39.77 | 2.81 |
| VecSet | 64 | 54.47 | 0.629 | 21.92 | 2.63 |
| VecSet | 512 | 44.18 | 0.521 | 2.59 | 1.16 |
| **COD-VAE** | **32** | **37.34** | **0.473** | **41.19** | **24.17** |
| **COD-VAE** | **64** | **37.05** | **0.460** | **22.49** | **16.09** |

COD-VAE M=64实现Rendering-FID 37.05（VecSet-512的44.18），全流程吞吐量**13.9×/20.8×提速**。

### 消融实验

**编码器设计消融（ShapeNet AE, M=32）**：

| 编码器 | IoU↑ | CD↓ | F1↑ |
|--------|------|-----|-----|
| CrossAttn (VecSet) | 91.2 | 0.016 | 94.7 |
| CrossAttn ×12 | 93.4 | 0.014 | 96.5 |
| CrossAttn ×24 | 93.5 | 0.014 | 96.6 |
| **渐进式（本文）** | **96.1** | **0.012** | **98.0** |

**解码器设计消融**：

| 解码器 | IoU↑ | CD↓ | F1↑ | 重建速度(sample/s)↑ |
|--------|------|-----|-----|---------------------|
| CrossAttn (VecSet) | 95.8 | 0.013 | 97.8 | 3.43 |
| CrossAttn ×3 | 96.3 | 0.013 | 98.0 | 1.47 |
| **Triplane（本文）** | **96.1** | **0.012** | **98.0** | **42.68** |

Triplane解码器重建速度达42.68 sample/s，是VecSet CrossAttn的12.4×，IoU几乎相同。

### Objaverse复杂物体结果

| 方法 | M | IoU↑ | CD↓ | F1↑ | 吞吐量↑ | 显存(GB) |
|------|---|------|-----|-----|---------|----------|
| VecSet | 64 | 71.2 | 0.054 | 85.2 | 1.99 | 5.46 |
| VecSet | 1024 | 79.8 | 0.051 | 87.0 | 0.23 | 9.96 |
| **COD-VAE** | **64** | **79.9** | **0.046** | **88.0** | **4.97** | **3.08** |

在更复杂的Objaverse数据集上，COD-VAE(M=64)甚至超越VecSet(M=1024)，吞吐量提升21.6×，显存减少69%。

### 关键发现

1. **中间表示空间是关键**：渐进式编码器比VecSet的直接映射IoU提升4.9%（91.2→96.1），增加cross-attention层数无法弥补
2. **大量潜在向量并非高质量重建的前提**：64个向量足以超越512个向量的质量
3. **Triplane解码兼具质量和效率**: 重建速度提升12×，同时质量相当
4. **Token剪枝75%不影响重建质量**：不确定性引导的剪枝精准识别简单区域

## 亮点与洞察

- **16×压缩比突破**：证明了3D形状的信息冗余度远高于之前的认知，64个向量足以表示复杂3D形状
- **两阶段方案设计精妙**：编码端用点patches做桥梁，解码端用triplane做桥梁，各最适合各自的功能
- **不确定性剪枝思路通用**：自适应计算资源分配的思路可推广到其他需要处理大量token的3D任务
- **训练策略聪明**：两阶段训练让latent decoder专注通道解压，比端到端训练更稳定

## 局限与展望

- Triplane表示对非凸物体可能存在表达局限（xy/yz/xz三平面的信息遗漏）
- 不确定性head预测基于初始triplane tokens，精度受限于此阶段的特征质量
- 25%保留比例是固定的，自适应调整可能进一步提升效率
- 纹理/颜色重建未涉及，当前仅处理geometry（occupancy field）

## 相关工作与启发

- VecSet是本文的直接基线，本文通过引入中间表示空间系统性地解决了其压缩比不足的问题
- TiTok在2D领域的压缩latent token思路启发了本文3D方向的探索
- Triplane表示（EG3D等）在3D生成中广泛使用，本文创新性地将其作为解码的中间表示
- 不确定性估计的token剪枝可借鉴到视觉Transformer的效率优化

## 评分

- **新颖性**: ⭐⭐⭐⭐ 两阶段自编码方案思路清晰，中间表示空间的设计有深度
- **实验充分度**: ⭐⭐⭐⭐⭐ ShapeNet+Objaverse双数据集、重建+生成双任务、全面的消融和效率分析
- **写作质量**: ⭐⭐⭐⭐⭐ 论文结构优秀，motivation到method的逻辑链完整，图表清晰
- **价值**: ⭐⭐⭐⭐⭐ 20.8×生成加速对3D扩散模型有重大实际意义，compact latent space可作为下游多种生成方法的基础
