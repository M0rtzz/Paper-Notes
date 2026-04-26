---
title: >-
  [论文解读] Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians
description: >-
  [ICCV 2025][3D视觉][3D高斯溅射] 提出 Can3Tok，首个可将场景级3DGS编码到低维潜空间的变分自编码器，通过规范化查询（canonical query）的交叉注意力实现高效tokenization，配合3DGS归一化和语义感知过滤解决尺度不一致问题，在DL3DV-10K上成功泛化到新场景。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D高斯溅射
  - 变分自编码器
  - 场景级3D生成
  - 规范化token
  - 潜空间建模
---

# Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians

**会议**: ICCV 2025  
**arXiv**: [2508.01464](https://arxiv.org/abs/2508.01464)  
**代码**: [https://github.com/Zerg-Overmind/Can3Tok](https://github.com/Zerg-Overmind/Can3Tok)  
**领域**: 3D Vision / 3D Generation  
**关键词**: 3D高斯溅射, 变分自编码器, 场景级3D生成, 规范化token, 潜空间建模

## 一句话总结

提出 Can3Tok，首个可将场景级3DGS编码到低维潜空间的变分自编码器，通过规范化查询（canonical query）的交叉注意力实现高效tokenization，配合3DGS归一化和语义感知过滤解决尺度不一致问题，在DL3DV-10K上成功泛化到新场景。

## 研究背景与动机

3D生成取得了长足进展，但主要集中在物体级——基于NeRF/3DGS的方法可高质量生成单个物体。场景级3D生成面临根本性挑战：

**3DGS数据结构不兼容现有VAE**：3DGS本质上是高度非结构化的——包含异构特征（几何、外观、光照）且像点云一样不规则；场景级3DGS包含大量Gaussian基元（>10K），压缩到低维embedding很困难

**尺度不一致**：由 COLMAP SfM 初始化导致，不同场景的全局尺度和每个Gaussian的缩放值都不一致，无法直接用于大规模训练

**噪声伪影**：场景级3DGS重建中，由于观测不充分，常存在大量浮点（floaters）噪声

实验表明现有3D VAE（PointNet VAE、L3DG等）完全无法收敛于场景级3DGS数据——哪怕几百个场景也训练失败，更无泛化能力。

## 方法详解

### 整体框架

Can3Tok 是基于 Transformer 的 VAE，包含编码器和解码器。编码器通过交叉注意力将大量3DGS基元（40K个）压缩到低维潜空间，解码器从潜空间重建原始3DGS参数。整体流程：输入 $\mathcal{G} \in \mathbb{R}^{N \times (2L_B + C)}$ → 交叉注意力 → 自注意力×8 → 潜空间($64 \times 64 \times 4$) → 自注意力×16 → MLP → 输出3DGS。

### 关键设计

1. **规范化查询的交叉注意力 Tokenization**：

    - 输入3DGS有N=40K个基元，直接自注意力计算量巨大
    - 使用可学习查询 $query \in \mathbb{R}^{M \times (P+Q)}$（M=256）做交叉注意力，将N个输入压缩为M个token
    - 关键创新：查询用规则体素网格坐标初始化（canonical space），引入结构化几何先验
    - 输入特征包含 Fourier 位置编码 $\gamma(\mathbf{x})$ 和最近体素坐标编码 $\gamma(\mathbf{v})$
    - 体素坐标提供"锚点"位置信息，降低编码器学习难度

2. **3DGS 归一化 (Normalization)**：解决尺度不一致的核心方案。

    - 平移：将场景3D中心移到原点 $translate = -\frac{1}{n}\sum_{i=1}^n \mathbf{x}_i$
    - 缩放：将所有3DGS约束在半径 $r$ 的球内 $scale = \frac{r}{\max|\mathbf{x}+translate|_2 \times 1.1}$
    - 同步缩放每个Gaussian的 scaling 参数 $\hat{\mathbf{s}} = \mathbf{s} \times scale$
    - 对应调整相机位置 $\hat{T}_i = (T_i + translate) \times scale$
    - 保持其他属性（旋转、透明度、颜色、SH）不变
    - 另一优势：可通过单目深度估计恢复真实尺度

3. **语义感知过滤 (Semantic-aware Filtering)**：

    - 使用 LangSam（文本引导的SAM变体）在场景中间帧检测"最显著区域"
    - 在分割掩码内选择一个Gaussian作为种子，通过K-NN逐步扩展到预设数量N=40K
    - 去除浮点噪声和非显著区域，保留最干净、最有语义意义的3DGS子集
    - 实验表明无过滤时高频细节严重丢失

### 损失函数 / 训练策略

$$\mathcal{L} = \text{Dist}(GS_{output}, GS_{input}) + \lambda \mathcal{L}_{KL}(\mathbf{z}, \mathcal{N}(\mathbf{0}, \mathbf{I}))$$

- $\text{Dist}$：所有3DGS特征通道的 L2 距离
- $\lambda = 1 \times 10^{-6}$：KL散度权重极小，优先保证重建质量
- 数据增强：对输入3DGS施加随机 SO(3) 旋转
- 训练细节：8× A100 GPU，5天训练；推理编解码仅需 ~0.06s

架构：编码器1层线性+1层交叉注意力+8层自注意力+2层投影；解码器1层线性+16层自注意力+3层MLP。注意力使用Flash-Attention，12头×64维。潜空间 $\mathbf{z} \in \mathbb{R}^{64 \times 64 \times 4}$，与Stable Diffusion潜空间相同大小。

## 实验关键数据

### 主实验

在 DL3DV-10K 测试集上的定量对比：

| 方法 | L2 误差↓ | 失败率↓ |
|------|---------|--------|
| L3DG（3DGS编码器，卷积） | 1200.4 | 100% |
| PointNet VAE | 1823.0 | 100% |
| PointTransformer | 230.7 | 70% |
| **Can3Tok (ours)** | **30.1** | **2.5%** |

失败率定义为重建L2误差超过1000的比例。所有对比方法几乎完全失败，仅Can3Tok成功泛化。PointNet和L3DG甚至在训练集中超过500个场景就无法收敛。

### 消融实验

| 设置 | L2 误差↓ | 失败率↓ |
|------|---------|--------|
| w/o Learnable Query | $10^{25}$ | 100% |
| w/o Normalization | 1889.7 | 100% |
| w/o Voxel Appending | 50.5 | 4.3% |
| w/o Data Filtering | 73.3 | 6.1% |
| w/o Data Augmentation | 53.3 | 4.6% |
| **Full (ours)** | **30.1** | **2.5%** |

### 关键发现

- **归一化是必要条件**：没有归一化即使Can3Tok也完全无法泛化（失败率100%），说明尺度不一致是场景级3D表示学习的根本障碍
- **可学习查询不可或缺**：移除后误差爆炸至 $10^{25}$，交叉注意力tokenization是模型成功的基石
- **语义过滤显著提升质量**：L2误差从73.3降至30.1，过滤噪声3DGS防止高频细节被潜空间"淹没"
- **潜空间保持空间信息**：t-SNE可视化显示同一场景不同SO(3)旋转的latent呈闭环，相似场景在潜空间中聚集
- **潜空间具有语义编码能力**：同一场景不同子采样（覆盖相同内容）的latent互相接近，不同场景的远离
- 推理速度快（~0.06s编解码），可无缝对接扩散模型做前馈生成

## 亮点与洞察

- **首次实现场景级3DGS VAE**：所有之前的3D VAE方法（PointNet、L3DG等）在场景级数据上完全失败，Can3Tok是唯一成功的方案
- **简洁有效的归一化策略**：借鉴2D图像将RGB归一化到[-1,1]的做法，对3DGS做中心平移+球面缩放，解决了3D场景表示学习的开放问题
- **潜空间与SD兼容**：$64 \times 64 \times 4$ 的潜空间形状与Stable Diffusion完全一致，可直接使用现有扩散架构（UNet/DiT）做条件生成
- **从数据到模型的完整方案**：不仅设计了模型架构，还提出了3DGS预处理管线（归一化+过滤+增强），对社区有参考价值

## 局限与展望

- 仅限于3DGS表示，无法直接应用于NeRF、mesh等其他3D表示
- 2.5%的失败率主要来自训练数据中质量较差的3DGS重建（运动模糊、远近视角不平衡）
- 语义过滤截取最显著区域，可能丢失完整场景信息（仅保留前景）
- 生成质量受限于VAE重建精度，细节恢复仍有提升空间
- text-to-3DGS生成依赖BLIP标注的简短描述，更丰富的文本条件有待探索
- 场景级数据集DL3DV-10K规模有限，更大数据集可能进一步提升泛化

## 相关工作与启发

- PerceiverIO 的交叉注意力压缩思路被巧妙地用于3DGS的tokenization
- 3DGS的归一化问题类似于NeRF/SfM中的坐标系统一问题，但更复杂（包含scaling参数）
- 潜空间与SD兼容的设计降低了下游生成任务的门槛，可直接复用文本/图像编码器
- Bolt3D等并发工作也发现卷积VAE在3DGS上失败的问题，验证了Can3Tok发现的普遍性

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次解决场景级3DGS潜空间建模问题，核心创新（canonical query + 归一化）简洁有效
- **实验充分度**: ⭐⭐⭐⭐ 定量+定性+t-SNE潜空间分析+消融全面覆盖，但生成应用展示较初步
- **写作质量**: ⭐⭐⭐⭐ 问题分析透彻，为什么现有方法失败解释清楚
- **价值**: ⭐⭐⭐⭐⭐ 开辟了场景级3DGS生成的新方向，潜空间+扩散模型的范式具有广阔前景

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [\[ICCV 2025\] AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering](aaa_gaussians_anti_aliased_artifact_free_3d_gaussian_rendering.md)
- [\[ICCV 2025\] LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling](localdygs_multi-view_global_dynamic_scene_modeling_via_adaptive_local_implicit_f.md)
- [\[ICCV 2025\] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)
- [\[ICCV 2025\] Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction](momentum-gs_momentum_gaussian_self-distillation_for_high-quality_large_scene_rec.md)

<!-- RELATED:END -->
