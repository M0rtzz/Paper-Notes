---
title: >-
  [论文解读] Auto-Regressively Generating Multi-View Consistent Images
description: >-
  [ICCV 2025][3D视觉][多视图生成] 提出 MV-AR，首次将自回归模型引入多视图图像生成，利用所有先前视图作为条件逐步生成后续视图，配合统一的多模态条件注入模块和 Shuffle View 数据增强，在文本/图像/形状条件下均达到与扩散模型可比的一致性。
tags:
  - ICCV 2025
  - 3D视觉
  - 多视图生成
  - 自回归模型
  - 3D内容创建
  - 多模态条件
  - 视图一致性
---

# Auto-Regressively Generating Multi-View Consistent Images

**会议**: ICCV 2025  
**arXiv**: [2506.18527](https://arxiv.org/abs/2506.18527)  
**代码**: 有（论文中提及 released）  
**领域**: 3D视觉  
**关键词**: 多视图生成, 自回归模型, 3D内容创建, 多模态条件, 视图一致性

## 一句话总结
提出 MV-AR，首次将自回归模型引入多视图图像生成，利用所有先前视图作为条件逐步生成后续视图，配合统一的多模态条件注入模块和 Shuffle View 数据增强，在文本/图像/形状条件下均达到与扩散模型可比的一致性。

## 研究背景与动机
从文本、参考图像或几何形状生成多视图一致图像是3D内容创建的基础。当前主流方法基于扩散模型（如 MVDream、Zero123++、SyncDreamer），通过交叉注意力或拼接多视图来实现跨视图信息交换。

**现有痛点**：

**远视图一致性差**：扩散方法使用单一参考视图（如正面图）来生成所有视图。当生成背面图时，参考视图的重叠内容几乎为零，引导信息失效，导致纹理不一致

**条件切换不灵活**：扩散方法在改变条件类型（文本→图像→形状）时通常需要大幅修改架构

**本质矛盾**：扩散方法建模联合概率分布 $p(v_1, v_2, ..., v_n)$，这种同时生成所有视图的方式不符合人类观察3D物体的渐进过程

**核心 idea**：自回归模型天然适合渐进式生成——生成第 $n$ 个视图时可以利用前 $n-1$ 个视图的全部信息。相邻视图之间总有较大重叠，即使背面视图也能从侧面视图获得有效参考。

## 方法详解

### 整体框架
MV-AR 基于预训练的文本到图像 AR 模型（LLamaGen），将2D VQVAE 编码的多视图 token 序列拼接后逐 token 自回归生成。条件模块包括文本（prefill token + SSA）、相机位姿（Shift Position Embedding）、参考图像（Image Warp Controller）和几何形状（预训练形状编码器 token）。通过渐进学习策略统一训练。

### 关键设计

1. **Split Self-Attention (SSA)**：在 Transformer 中将文本 token 和图像 token 拼接后做自注意力，但将自注意力输出中文本位置的残差置零，公式为 $SSA(X_{in}) = X_{in} + Concat(0 \cdot O_{text}, O_{image})$。

    - **设计动机**：常规自注意力中，后续图像 token 会反向干扰文本 token 的表示，降低文本引导效果。SSA 让文本 token 只影响图像生成但不被图像反向影响，提升了文本-图像对齐（CLIP Score 提升）。

2. **Shift Position Embedding (SPE)**：将 Plücker 射线编码 $r_{i,j} = (o \times d, d)$ 作为位移位置编码，直接加到 token 嵌入上。

    - **设计动机**：不同视图的 token 需要知道自己对应的3D位置和视角方向。SPE 提供的物理角度信息让模型能区分不同视图中的同一位置，提升空间理解。

3. **Image Warp Controller (IWC)**：利用当前视图的相机位姿和参考图像特征预测重叠区域的纹理特征，公式为 $X_{IWC} = FFN(CA(SA(X_{ref}), r))$，以残差方式注入模型。

    - **设计动机**：高层图像特征（如 CLIP）丢失低层细节，IWC 使用编码器级特征保留颜色和纹理细节，逐 token 注入保证精准控制。实验证明 IWC 远优于 In-context 和 Cross-Attention 方式。

4. **Shuffle View 数据增强**：训练时随机打乱 $N$ 个视图的顺序，将 1 个3D物体扩展为 $\frac{N(N-1)}{2}$ 种训练序列。

    - **设计动机**：AR 模型需要海量数据防止过拟合，但高质量多视图数据稀缺。ShufV 利用多视图的顺序灵活性大幅扩充数据量，同时让模型学习正反两个方向的视图转换。

### 损失函数 / 训练策略
- 标准 AR 损失：$\mathcal{L}_{ar} = -\frac{1}{T}\sum_{t=1}^T \log p(q_t | q_{<t})$
- **渐进学习**：先训练 t2mv（文本→多视图）模型，再在 t2mv 基础上训练 X2mv（任意条件→多视图）
- 文本条件随机 drop 并替换为指令模板（如"Generate multi-view images of the following <<<img>>>"）
- Drop 概率从 0 线性增加到 0.5（前10K迭代），之后保持 0.5
- 优化器 AdamW，16×A800 GPU，batch 1024，lr $4 \times 10^{-4}$，训练 30K 迭代

## 实验关键数据

### 主实验

**文本→多视图 (GSO 30 objects)**

| 方法 | FID↓ | IS↑ | CLIP-Score↑ |
|------|------|-----|-------------|
| MVDream (扩散) | 141.05 | 7.49 | 28.71 |
| LLamaGen (基线AR) | 146.11 | 5.78 | 28.36 |
| **MV-AR** | **144.29** | **8.00** | **29.49** |

**图像→多视图 (GSO 30 objects)**

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Zero123 | 18.93 | 0.779 | 0.166 |
| SyncDreamer | 19.89 | 0.801 | 0.129 |
| Wonder3D | 22.82 | 0.892 | 0.062 |
| **MV-AR** | **22.99** | 0.907 | 0.084 |

MV-AR 在 PSNR 上取得最优（22.99），CLIP-Score 上也最高（29.49），证明 AR 方法在多视图一致性和文本对齐上与扩散方法可比甚至更好。

### 消融实验

| 组件 | FID↓ / IS↑ (t2mv) | PSNR↑ / SSIM↑ / LPIPS↓ (i2mv) |
|------|-------------------|-------------------------------|
| w/o SPE | 147.29 / 7.26 | 21.30 / 0.843 / 0.118 |
| w/o ShufV | 173.51 / 4.77 | 18.27 / 0.778 / 0.194 |
| **Full MV-AR** | **144.29 / 8.00** | **22.99 / 0.907 / 0.084** |

| 图像条件方式 | PSNR↑ | SSIM↑ | LPIPS↓ |
|-------------|-------|-------|--------|
| In-context | 11.92 | 0.538 | 0.477 |
| Cross Attention | 15.13 | 0.709 | 0.310 |
| **IWC** | **22.99** | **0.907** | **0.084** |

Shuffle View 是贡献最大的组件（去掉后 FID 恶化 +29，PSNR 下降 4.72），IWC 远优于其他图像条件注入方式。

### 关键发现
- AR 生成在前后视图一致性上天然优于扩散模型，因为它可以利用中间过渡视图
- SSA 相比常规自注意力显著提升了 CLIP Score（文本-图像对齐）
- IWC 利用低层特征指导纹理生成，效果远超使用 CLIP/DINO 等高层语义特征
- MV-AR 是首个能同时处理文本、图像和形状条件的统一多视图生成模型

## 亮点与洞察
- **首次将 AR 范式引入多视图生成**，为该领域提供了扩散模型之外的新路线
- **统一多模态条件（X2mv）**：文本+图像+形状可同时使用，灵活性强
- Shuffle View 数据增强简单有效，适用于所有基于序列的多视图方法
- SSA 的设计解决了 in-context 条件中文本被图像干扰的通用问题，可迁移到其他多模态 AR 模型

## 局限与展望
- AR 模型的**单向性**和**离散编码**是固有限制，生成质量受 VQVAE 重建质量约束
- **累积误差**：前序视图质量差会影响后续视图，虽然提到了但未完全解决
- 当前使用2D VQVAE 编码，未来使用因果3D VAE可能进一步提升跨视图一致性
- 生成分辨率受限（256×256），需要更高分辨率的 tokenizer
- 渲染速度比扩散模型慢（需要逐 token 生成所有视图的全部 token）

## 相关工作与启发
- **MVDream / Zero123++ / SyncDreamer**：扩散路线的多视图生成方法，本文证明 AR 路线也可行
- **LLamaGen / VQGAN / VAR**：AR 视觉生成的基础工作，本文在此基础上扩展到多视图
- **PixelCNN → VQVAE → VAR**：AR 图像生成的演进路线
- **启发**：AR 的渐进生成天然适合需要序列化结构的任务（如视频、多视图、全景图），未来可能统一生成与理解

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次成功将 AR 引入多视图生成，开辟新路线
- 实验充分度: ⭐⭐⭐⭐ 三种条件任务全面评估，消融充分；但仅在 GSO 30 objects 上评估，规模偏小
- 写作质量: ⭐⭐⭐⭐ 问题-方案对应清晰，但符号较多
- 价值: ⭐⭐⭐⭐ 证明了 AR 路线的可行性，启发后续工作

<!-- RELATED:START -->

## 相关论文

- [PanSt3R: Multi-view Consistent Panoptic Segmentation](panst3r_multi-view_consistent_panoptic_segmentation.md)
- [MV-Adapter: Multi-view Consistent Image Generation Made Easy](mv-adapter_multi-view_consistent_image_generation_made_easy.md)
- [3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement](../../CVPR2025/3d_vision/3denhancer_consistent_multi-view_diffusion_for_3d_enhancement.md)
- [SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models](spinmeround_consistent_multi-view_identity_generation_using_diffusion_models.md)
- [DeepMesh: Auto-Regressive Artist-Mesh Creation with Reinforcement Learning](deepmesh_auto-regressive_artist-mesh_creation_with_reinforcement_learning.md)

<!-- RELATED:END -->
