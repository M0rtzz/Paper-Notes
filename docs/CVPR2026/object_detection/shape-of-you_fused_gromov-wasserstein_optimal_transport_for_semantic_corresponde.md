---
title: >-
  [论文解读] Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild
description: >-
  [CVPR2026][目标检测][语义对应] 将语义对应问题重新建模为 Fused Gromov-Wasserstein (FGW) 最优传输问题，利用 3D 基础模型提供的几何结构约束来生成全局一致的伪标签，解决了传统最近邻匹配因局部性和 2D 外观歧义导致的几何不一致问题。
tags:
  - CVPR2026
  - 目标检测
  - 语义对应
  - 最优传输
  - Gromov-Wasserstein
  - 3D几何约束
  - 伪标签
  - 基础模型
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild

**会议**: CVPR2026  
**arXiv**: [2603.11618](https://arxiv.org/abs/2603.11618)  
**代码**: [Shape-of-You](https://github.com/im-jiin/Shape-of-You)  
**领域**: object_detection  
**关键词**: 语义对应, 最优传输, Gromov-Wasserstein, 3D几何约束, 伪标签, 基础模型

## 一句话总结

将语义对应问题重新建模为 Fused Gromov-Wasserstein (FGW) 最优传输问题，利用 3D 基础模型提供的几何结构约束来生成全局一致的伪标签，解决了传统最近邻匹配因局部性和 2D 外观歧义导致的几何不一致问题。

## 研究背景与动机

**语义对应的核心挑战**：在 in-the-wild 场景中，极端视角变化、光照差异和类内形状差异使得建立像素级语义对应极为困难，而全监督方法依赖昂贵的逐像素标注，可扩展性差。

**伪标签方法的固有缺陷**：当前无显式几何标注的方法（如 ASIC）依赖 DINO/SD 等基础模型的特征进行最近邻 (NN) 匹配生成伪标签，但 NN 匹配是局部操作，忽略了全局结构关系。

**2D 特征的几何歧义**：纯粹基于 2D 外观训练的模型无法反映物体真实的 3D 几何结构，导致语义上合理但几何上错误的对应（如对称部件、重复纹理区域的误匹配）。

**局部匹配无法保持结构一致性**：NN 匹配逐点独立决策，不保证匹配集合在全局上的几何连贯性，引入的训练噪声会降低模型性能。

**GW 计算瓶颈**：Gromov-Wasserstein 是二次规划的非凸问题，直接求解在计算上不可行，需要高效的近似方案。

**部分可见性与遮挡**：真实场景中物体常被遮挡或截断，平衡最优传输的硬边际约束（所有质量必须传输）不适用于这种部分匹配情况。

## 方法详解

### 整体框架

Shape-of-You (SoY) 采用两阶段流程：(1) **伪标签生成**：通过 FGW 最优传输融合语义特征相似性和 3D 几何结构一致性，生成高质量伪标签；(2) **适配器训练**：使用软目标损失训练轻量级适配器网络，推理时仅需前向传播无需迭代优化。

**输入表示**：给定源/目标图像，用 SAM 分割出实例区域后划分为 patch 网格。每个 patch 获取两种表示——DINOv2/SD 提取的语义特征 $\mathbf{f}_i \in \mathbb{R}^d$ 和 VGGT（3D 基础模型）提升的 3D 坐标 $\mathbf{v}_i \in \mathbb{R}^3$。

### 关键设计

**Stage 1 — 语义 UOT 初始化 (t=0)**：

- 计算语义代价矩阵 $C^{\text{sem}}_{ij} = 1 - \cos(\mathbf{f}_i^A, \mathbf{f}_j^B)$
- 采用非平衡最优传输 (UOT) 而非经典 OT，通过 KL 散度惩罚放松边际约束，处理遮挡/非重叠区域
- 用 Sinkhorn 算法求解得到初始传输计划 $\pi^{(0)}$

**Stage 2 — 基于锚点的 FGW 迭代精炼 (t>0)**：

- **锚点选择**：从 $\pi^{(t-1)}$ 中选取 K=64 个高置信度锚点对，要求满足循环一致性（前向匹配→反向匹配的 3D 循环误差低于阈值 $\delta$）
- **GW 线性化**：用稀疏锚点传输计划 $\hat{\pi}$ 替代二次项中的一个 $\pi$，将 $O(N^2M^2)$ 的二次问题转化为 $O(NMK)$ 的线性问题：$C^{\text{geo}}_{ij} = \frac{1}{K}\sum_{(a_A,a_B)\in\mathcal{A}} |D^A_{i,a_A} - D^B_{j,a_B}|$
- **融合代价**：归一化后按 $C^{\text{total}} = (1-\alpha)\tilde{C}^{\text{sem}} + \alpha\tilde{C}^{\text{geo}}$（$\alpha=0.3$）融合，再求解 UOT 得到 $\pi^{(t)}$
- 迭代 T 轮，锚点逐步改善，最终 $\pi^{(T)}$ 作为伪标签

**软目标损失（Soft Target Loss）**：

- 从 $\pi^{(T)}$ 提取 top-k 候选构建多热二值目标 $\pi^{\text{hard}}$
- 用网络当前特征计算 OT 得到语义软目标 $\pi^{\text{curr}}$（停止梯度）
- 混合：$\pi^{\text{soft}} = (1-\beta)\pi^{\text{hard}} + \beta\pi^{\text{curr}}$（$\beta=0.5$），防止对语义相似但未被选中的候选过度惩罚

### 损失函数

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{soft}} + \mathcal{L}_{\text{dense}}$$

- $\mathcal{L}_{\text{soft}}$：对称交叉熵损失，在预测相似度分布与 $\pi^{\text{soft}}$ 之间计算，含可学习温度参数 $\tau$
- $\mathcal{L}_{\text{dense}}$：标准密集对应损失，通过 soft-argmax 将梯度传播到所有特征图位置，加高斯噪声 $\epsilon$ 正则化

## 实验

### 主实验结果

| 方法 | SPair-71k PCK@0.1 | SPair-71k PCK@0.05 | AP-10k I.S. | AP-10k C.S. |
|------|-------------------|---------------------|-------------|-------------|
| ASIC | 36.9 | - | - | - |
| DINOv2 | 55.7 | - | - | - |
| DistillDIFT | 59.8 | 42.7 | 65.5 | 62.8 |
| DINOv2 + SD | 63.5 | 48.3 | 65.5 | 63.3 |
| **SoY (Ours)** | **67.9** | **50.8** | **68.0** | **65.8** |

在 SPair-71k 上以 67.9% PCK@0.1 超越零样本基线 DINOv2+SD 4.4 个百分点，18 个类别中 17 个达到最优或次优。AP-10k 零样本评估中三种设置均取得最优。

### 消融实验

| 伪标签策略 | Geometry-aware PCK_label@0.1 |
|-----------|----------------------------|
| Nearest Neighbor | 53.8% |
| Semantic OT | 54.5% |
| Fused OT (balanced) | 55.7% |
| **Fused UOT (ours)** | **56.1%** |

| 组件消融 | PCK@0.1 |
|---------|---------|
| Backbone (DINOv2+SD) 零样本 | 63.5 |
| Adapter + NN 标签 | 64.6 |
| Adapter + FGW 标签 | 66.8 |
| + relaxed c.c. + soft target loss | **67.9** |

### 关键发现

- **3D 几何距离是关键**：在 intra-structure 对比实验中，2D 距离 (65.7%) 和语义距离 (66.2%) 均不如 3D 几何距离 (67.6%)，2D 距离反而低于不用 GW 的基线 (66.5%)
- **几何约束对歧义场景尤为重要**：GW 项在 Geometry-aware 子集上提升 +2.3%p，而整体提升相对较小，证明 3D 约束主要解决几何歧义
- **各组件协同增效**：FGW 伪标签 (+2.0%p) > 循环一致性 (+0.2%p)，说明全局结构匹配远比局部后处理有效；软目标损失在 FGW 标签基础上进一步提升

## 亮点

- **问题建模新颖**：首次将语义对应建模为 FGW 最优传输问题，天然融合了跨域语义相似性（Wasserstein）和域内几何一致性（Gromov-Wasserstein）
- **锚点线性化高效实用**：通过 64 个锚点将 NP-hard 的二次 GW 问题近似为可用 Sinkhorn 求解的线性 OT，工程上可行
- **3D 基础模型的巧妙引入**：用 VGGT 将 2D 图像提升到 3D，为 GW 项提供有意义的几何度量，而非直接在 2D 坐标上计算距离
- **推理高效**：训练完成后推理仅需轻量适配器前向传播 + 最近邻匹配，无需迭代 OT 求解

## 局限性

- **依赖 3D 基础模型质量**：VGGT 在透明表面或平坦重建失败时会导致错误锚点
- **对称部件仍有困难**：如中等视角下对称汽车部件，可能导致锚点匹配错误
- **软目标可能过度平滑**：$\beta=0.5$ 的混合比例在某些情况下可能模糊几何信号
- **SAM 分割依赖**：伪标签生成依赖 SAM 的实例分割质量，分割不准会影响下游匹配

## 相关工作

- **无几何标注的语义对应**：ASIC（NN 伪标签）、SHIC（图像到形状）、DistillDIFT（特征蒸馏）——均依赖局部匹配或 2D 蒸馏，缺乏全局结构感知
- **最优传输用于结构匹配**：经典 Wasserstein OT 用于对应学习，GECO 引入非平衡 OT 进行几何一致特征学习——但未结合 Gromov-Wasserstein 的结构比较能力
- **3D 几何基础模型**：DUSt3R、MASt3R 需要多阶段，VGGT 单次前向预测多种 3D 属性——本文利用 VGGT 为 GW 提供几何先验
- **GW 近似方法**：图匹配、视频动作分割等领域的 GW 近似——本文的锚点线性化与 anchor-based GW surrogate 思路一致

## 评分

- 新颖性: ⭐⭐⭐⭐ (FGW 建模 + 3D 基础模型几何先验的组合是首创)
- 实验充分度: ⭐⭐⭐⭐ (两个数据集 + 详尽消融 + PCK_label 指标验证伪标签质量)
- 写作质量: ⭐⭐⭐⭐ (OT 背景清楚, 方法推导完整, 图表信息量大)
- 价值: ⭐⭐⭐⭐ (为无标注语义对应提供了新范式, 3D+OT 的思路可迁移到其他视觉匹配任务)

<!-- RELATED:START -->

## 相关论文

- [How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts](../../AAAI2026/object_detection/how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)
- [Boosting Domain Incremental Learning: Selecting the Optimal Parameters Is All You Need](../../CVPR2025/object_detection/boosting_domain_incremental_learning_selecting_the_optimal_parameters_is_all_you.md)
- [CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning](clcr_cross-level_semantic_collaborative_representation_for_multimodal_learning.md)
- [SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking](slice_semantic_latent_injection_via_compartmentali.md)
- [Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval](beyond_semantic_search_towards_referential_anchoring_in_composed_image_retrieval.md)

<!-- RELATED:END -->
