---
title: >-
  [论文解读] CuMPerLay: Learning Cubical Multiparameter Persistence Vectorizations
description: >-
  [ICCV 2025][医学图像][多参数持久同调] 提出 CuMPerLay，一个可微的立方多参数持久同调 (Cubical Multiparameter Persistence, CMP) 向量化层，将 CMP 分解为多条可学习的单参数持久同调线，通过联合学习双滤过 (bifiltration) 函数实现端到端训练，嵌入 Swin Transformer 后在医学图像分类和语义分割任务上（尤其小数据场景）取得显著提升。
tags:
  - ICCV 2025
  - 医学图像
  - 多参数持久同调
  - 立方复形
  - 可微向量化
  - 拓扑数据分析
  - Transformer
---

# CuMPerLay: Learning Cubical Multiparameter Persistence Vectorizations

**会议**: ICCV 2025  
**arXiv**: [2510.12795](https://arxiv.org/abs/2510.12795)  
**代码**: [circle-group/cumperlay](https://github.com/circle-group/cumperlay) (有)  
**领域**: 医学图像 / 拓扑深度学习  
**关键词**: 多参数持久同调, 立方复形, 可微向量化, 拓扑数据分析, Swin Transformer  
**机构**: Imperial College London + UT Dallas

## 一句话总结

提出 CuMPerLay，一个可微的立方多参数持久同调 (Cubical Multiparameter Persistence, CMP) 向量化层，将 CMP 分解为多条可学习的单参数持久同调线，通过联合学习双滤过 (bifiltration) 函数实现端到端训练，嵌入 Swin Transformer 后在医学图像分类和语义分割任务上（尤其小数据场景）取得显著提升。

## 背景与动机

**持久同调 (Persistent Homology, PH)** 是拓扑数据分析 (TDA) 的核心工具，通过跟踪数据在不同尺度下的拓扑特征（连通分量、环、空洞等）的"出生"和"死亡"来捕获全局结构信息。PH 在深度学习中的整合已取得进展（如 PersLay），但现有方法几乎局限于**单参数持久同调 (Single-Parameter Persistence, SPP)**，即只沿一个方向对数据进行滤过。

**多参数持久同调 (Multiparameter Persistence, MP)** 沿多个方向同时滤过，能捕获更丰富的拓扑信息。然而 MP 面临两个核心困难：

1. **结构复杂性**：MP 模块没有类似单参数下 barcode 的完备离散不变量，无法简洁地用持久图表示
2. **向量化困难**：由于不存在完备的 MP 描述子，将 MP 信息转为可供机器学习使用的向量非常困难

**立方复形 (Cubical Complex)** 是处理图像拓扑的天然选择——像素直接对应立方体的顶点，邻接关系对应边和面。相比 Vietoris-Rips 或 Alpha 复形，立方复形完美匹配图像的网格结构，计算效率更高。

然而，**立方多参数持久同调 (CMP)** 至今未能有效融入深度学习，核心瓶颈在于缺乏一种既可微又稳定的 CMP→向量映射。本文正是解决这一问题。

## 方法详解

### 核心思想：CMP 分解为可学习的单参数持久同调

CuMPerLay 的关键洞察是：**将不可直接向量化的多参数持久同调，分解为多条可学习的单参数持久同调线 (line persistence)**。具体来说：

给定图像的特征图 $x \in \mathbb{R}^{B \times C \times H \times W}$，方法包含三个核心步骤：

#### 1. 可学习双滤过 (Learnable Bifiltration)

通过 **Filtration Decoder** 从特征图生成双滤过函数：

$$G = f_\theta(x), \quad G \in \mathbb{R}^{B \times C_0 \times H \times W}$$

其中 Filtration Decoder 由多层反卷积/上采样 + GroupNorm + ReLU + Conv 组成，将高级特征映射到 $C_0$ 个滤过通道。每个通道对应一个独立的滤过函数。

为了得到离散的阈值来构建 bifiltration，使用 **Stair Combined Thresholding**：对连续滤过值应用 sigmoid 归一化到 $[0,1]$，再通过可学习的阈值采样 $n_T$ 个离散水平（默认 $n_T=16$），生成紧凑滤过表示：

$$\hat{G}_{\text{norm}} \in \mathbb{R}^{B \times C_0 \times H' \times W'}$$

#### 2. 立方持久同调计算 (Cubical Persistence)

对每个滤过通道独立计算 2D 立方持久同调。具体实现基于 C++/CUDA 加速的 `cubical_persistence_v_2d_full` 函数：

- 输入：紧凑滤过表示 $\hat{G}_{\text{norm}}$
- 输出：持久配对 $P \in \mathbb{R}^{B \times C_0 \times d \times F \times 2}$，其中 $d$ 为同调维数（0维=连通分量，1维=环），$F$ 为最大特征数
- 同时输出配对长度 $L$ 用于生成有效掩码

每对 $(b_i, d_i)$ 表示一个拓扑特征的出生值和死亡值。通过对 $C_0$ 个通道分别计算 SPP，CuMPerLay 将原本的双参数问题转化为 $C_0$ 个独立的单参数问题，而 $C_0$ 个滤过函数本身是联合学习的。

#### 3. 可微向量化：Silhouette v2

使用改进的 **PersLay 风格向量化**，包含两个可学习组件：

**权重函数 (Weight)**：对每个持久配对计算重要性权重：

$$w_i = c \cdot |d_i - b_i|^p$$

其中 $c$ (常数) 和 $p$ (幂次) 均为可学习参数，按通道和维度独立学习。支持跨特征归一化。

**Phi 函数 (Tent Phi)**：将持久配对映射到 $S$ 维采样空间（默认 $S=128$），使用可学习的采样点 $\{s_k\}_{k=1}^S$：

$$\phi_k(b_i, d_i) = \text{ReLU}\left(\frac{d_i - b_i}{2} - \left|s_k - \frac{b_i + d_i}{2}\right|\right)$$

加权后沿特征维度求和，得到最终向量化表示：

$$v_k = \sum_i w_i \cdot \phi_k(b_i, d_i)$$

输出维度为 $\mathbb{R}^{B \times C_0 \times S \times d}$。

### 整体架构：TopoSwin-MP

CuMPerLay 层嵌入 **Swin Transformer V2** 的每个 stage 之后，形成 TopoSwin-MP 架构：

```
Input → Patch Embed → [Stage1 → Topo1 → Gate1] → [Stage2 → Topo2 → Gate2]
      → [Stage3 → Topo3 → Gate3] → [Stage4 → Topo4 → Gate4] → Norm → Pool → Head
```

每个 Stage 后的 TopoBlock 包含：Filtration Decoder → Cubical PH → Silhouette Vectorization → MLP

**Gated Topology Linear** 将拓扑特征融合回主干特征：

$$x' = x + x \cdot \sigma(\text{Linear}(t)) + \text{bias}(t)$$

其中 $t$ 为拓扑向量，$\sigma$ 为 sigmoid 门控。类似 SE 注意力但以拓扑特征为条件。

此外还有 **Input Guidance**：直接对原始输入图像也计算一路 CuMPerLay 拓扑特征，拼接到每个 stage 的拓扑向量中，提供低级拓扑先验。

**辅助拓扑分类头**：将所有 stage 的持久表示拼接后通过独立 MLP 做分类，与主分类头共同优化：

$$\mathcal{L} = \mathcal{L}_{CE} + 0.25 \cdot \mathcal{L}_{topo\_CE} + 0.01 \cdot \mathcal{L}_{multifilt\_reg}$$

其中 $\mathcal{L}_{multifilt\_reg}$ 是多滤过正则化损失，鼓励不同滤过函数学到不同的模式。

### 理论保证：稳定性定理

CuMPerLay 在广义 Wasserstein 度量下具有稳定性保证：对于两个立方复形 $K_1, K_2$，其向量化表示的差异被输入滤过函数的差异所上界。这保证了滤过函数的微小扰动不会导致向量化结果的剧烈变化，为端到端梯度训练提供理论基础。

## 实验关键数据

### 数据集

| 数据集 | 任务 | 规模 | 特点 |
|-------|------|------|------|
| ISIC 2018 | 皮肤病变 7 分类 | ~10K 图像 | 类别严重不平衡 |
| CBIS-DDSM | 乳腺 X 光 2 分类 | ~2K 裁剪图像 | 小数据集 |
| Glaucoma | 眼底图像 2 分类 | ~1.5K 图像 | 小数据集 |
| Pascal VOC + SBD | 语义分割 21 类 | ~11K 图像 | 通用视觉 |

### 分类性能对比

**ISIC 2018 (AUC-ROC)**：
- ResNet-50 baseline: ~80%
- Swin-V2-B baseline: ~85%
- TopoSwin-MP (本文): **显著提升**，达到最优

**关键结论**：CuMPerLay 在所有分类数据集上均超过 baseline 和现有 TDA 方法：
- 超过 PersLay（单参数持久同调层）
- 超过 ATOL（自适应拓扑层）
- 超过基于 Betti 曲线的方法

### 小数据场景

CuMPerLay 在**有限数据场景下优势尤为突出**：
- 使用 25%、50% 训练数据时，拓扑增强模型相对 baseline 的提升幅度更大
- 原因：拓扑特征提供了与数据量无关的全局结构先验

### 分割性能

在 Pascal VOC + SBD 语义分割任务上，将 CuMPerLay 接入分割网络后，mIoU 同样获得提升，验证方法的通用性。

### 消融实验

- **多参数 vs 单参数**：多参数持久同调（多通道滤过）显著优于单个灰度滤过
- **可学习滤过 vs 固定滤过**：学习的滤过函数优于手工设计的密度/体素滤过
- **Input Guidance**：添加原始输入的拓扑引导进一步提升性能
- **辅助拓扑头**：联合训练拓扑分类头有助于学习更好的滤过函数

## 亮点与洞察

1. **优雅的分解策略**：将数学上不可直接处理的多参数持久同调，分解为可学习的多条单参数线，既保留了多参数的表达力，又利用了成熟的单参数工具链。这是一个"化繁为简"的典范
2. **端到端可微**：整个流程（滤过生成→立方 PH→向量化）都是可微的，梯度可以从分类损失一直回传到滤过函数的参数，使拓扑特征真正任务驱动
3. **立方复形的天然适配**：选择立方复形而非 simplicial 复形，完美利用了图像数据的网格结构，避免了点云采样和 Rips 复形构建的开销
4. **C++/CUDA 加速**：持久同调计算通过自定义 CUDA kernel 实现，使得训练在实际中可行。代码已开源 ([circle-group/cmp](https://github.com/circle-group/cmp))
5. **Gated 融合机制**：拓扑特征通过 sigmoid 门控乘性融合回主干，而非简单拼接，让网络自适应决定拓扑信息的贡献度

## 局限与展望

1. **仅支持 2D 立方复形**：当前 CUDA 实现仅支持 2D 图像，论文提到 dim=3 虽在框架中预留但未实现。扩展到 3D 医学影像（CT/MRI）是自然方向
2. **计算开销**：每个 stage 都需要计算完整的立方 PH，虽有 CUDA 加速但仍增加约 30-50% 训练时间
3. **分解的信息损失**：将双参数持久同调分解为独立的单参数线会损失参数间的交互信息，某些仅在双参数空间中可见的拓扑特征可能被遗漏
4. **仅实验了分类和分割**：检测、生成等任务未探索
5. **Filtration Decoder 较重**：每个 stage 的 Filtration Decoder 包含多层反卷积，参数量不小。轻量化设计值得探索
6. **阈值数 $n_T$ 和通道数 $C_0$ 的选择**：目前依赖手动调参（默认 16 和 8），自适应选择机制可能更好

## 相关工作与启发

- **PersLay** (NeurIPS 2020)：首个通用可微持久同调向量化层，但仅支持单参数 PH。CuMPerLay 可视为 PersLay 在多参数+立方复形方向的扩展
- **PLLay** (ICML 2021)：将 PersLay 推广到持久 landscape，同样限于单参数
- **Multiparameter Persistence Image** (NeurIPS 2020, Carrière & Blumberg)：提出多参数持久图像，但不可微
- **Smart Vectorizations** (Coskunuzer et al., 2021)：本文合著者的前期工作，探索单/多参数的智能向量化
- **Differentiability of MP** (ICML 2024, Scoccola et al.)：证明多参数 PH 的可微性，为本文提供理论基础
- **拓扑+Transformer**：将功能性拓扑特征与 Swin Transformer 结合是新趋势，有望推广到更多视觉 Transformer 架构

**对研究的启发**：CuMPerLay 展示了 TDA 与深度学习结合的新范式——不是将 TDA 当作固定的特征提取预处理，而是让拓扑计算本身成为可学习的网络组件。这种思路可以推广到图神经网络、点云处理等领域。

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [ReCoN-Ipsundrum: An Inspectable Recurrent Persistence Loop Agent with Affect-Coupled Cognition](../../AAAI2026/medical_imaging/recon-ipsundrum_an_inspectable_recurrent_persistence_loop_agent_with_affect-coup.md)
- [PVChat: Personalized Video Chat with One-Shot Learning](pvchat_personalized_video_chat_with_one-shot_learning.md)
- [An OpenMind for 3D Medical Vision Self-supervised Learning](an_openmind_for_3d_medical_vision_selfsupervised_learning.md)
- [SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality](simmlm_a_simple_framework_for_multi-modal_learning_with_missing_modality.md)
- [Vector Contrastive Learning for Pixel-wise Pretraining in Medical Vision](vector_contrastive_learning_for_pixel-wise_pretraining_in_medical_vision.md)

<!-- RELATED:END -->
