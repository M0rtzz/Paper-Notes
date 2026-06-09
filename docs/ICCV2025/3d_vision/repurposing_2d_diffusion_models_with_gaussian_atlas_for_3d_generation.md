---
title: >-
  [论文解读] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation
description: >-
  [3D视觉] 提出 Gaussian Atlas 表示法，将无序3D高斯通过最优传输映射到球面再展平为规整2D网格，从而直接微调预训练2D Latent Diffusion模型实现高质量文本到3D生成。
tags:
  - "3D视觉"
---

# Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2503.15877](https://arxiv.org/abs/2503.15877)
- **代码**: [项目页面](https://cs.stanford.edu/~xtiange/projects/gaussianatlas)
- **领域**: 3D生成 · 扩散模型
- **关键词**: 3D高斯, 2D扩散模型迁移, Gaussian Atlas, 大规模数据集, 文本到3D

## 一句话总结
提出 Gaussian Atlas 表示法，将无序3D高斯通过最优传输映射到球面再展平为规整2D网格，从而直接微调预训练2D Latent Diffusion模型实现高质量文本到3D生成。

## 研究背景与动机

3D扩散模型的发展受限于高质量3D数据的稀缺，性能远不如2D对应模型。现有方法尝试利用2D扩散模型先验来辅助3D生成，但存在以下问题：

**数据瓶颈**：高质量3D模型的创建和标注成本极高，数据规模远不及2D图像（十亿级）

**间接利用**：Score Distillation Sampling (SDS) 等方法仅以冻结权重间接使用2D模型，效率低且质量受限

**表示鸿沟**：3D高斯的无序点集结构无法直接输入2D网络，需要结构化的2D表示才能实现迁移学习

**核心思路**：设计一种将3D高斯映射为规整2D网格的方法（Gaussian Atlas），使预训练2D扩散模型能够直接微调用于3D生成。

## 方法详解

### 整体框架

流水线分两个阶段：
1. **3DGS预拟合阶段**：为大量3D物体预拟合高质量3D高斯，构建GaussianVerse数据集
2. **扩散模型训练阶段**：将3D高斯转换为Gaussian Atlas 2D表示，微调预训练Latent Diffusion的UNet

### GaussianVerse 数据集

基于Scaffold-GS构建了包含**205,737个高质量3DGS拟合**的大规模数据集。关键改进：

- **可见性排序剪枝策略**：不同于固定数量约束，设定上界 $\tau=36,864$（即 $192 \times 192$），根据每个高斯在随机相机视角下的不透明度进行排序，剪枝可见性最低的高斯
- **感知损失增强**：在拟合损失中加入LPIPS感知损失，以更少的高斯数实现更高渲染保真度

拟合损失函数：

$$\lambda'_{rgb}\mathcal{L}_{rgb} + \lambda'_{ssim}\mathcal{L}_{ssim} + \lambda'_{lpips}\mathcal{L}_{lpips} + \lambda'_{reg}\mathcal{R}$$

每个物体约10分钟拟合时间（A100），总计超过3.8 A100 GPU年。

### Gaussian Atlas：3D到2D的映射

将无序3D高斯转换为规整2D网格的三步过程：

**第一步：球面偏移（Sphere Offsetting）**

假设一个单位球 $\mathcal{S}$，表面均匀分布 $N$ 个点 $\{s_i \in \mathbb{R}^3\}$。通过**最优传输（Optimal Transport）**将3D高斯的位置映射到球面上。与GaussianCube不同，本方法映射到球面而非体积网格内部。

**第二步：等距矩形投影（Equirectangular Projection）**

将球面上的3D高斯通过等距矩形投影 $\mathcal{M}$ 展平为2D平面坐标 $\{p_i \in \mathbb{R}^2\}$。由于 $\mathcal{M}$ 是确定性函数，投影是所有物体一致的。

**第三步：平面偏移（Plane Offsetting）**

再次使用最优传输将展平后的2D坐标映射到 $\sqrt{N} \times \sqrt{N}$ 的规整方形网格顶点 $\{q_i \in \mathbb{R}^2\}$。由于投影的确定性，此步OT只需计算一次，索引可复用。

最终得到的Gaussian Atlas形状为 $\sqrt{N} \times \sqrt{N} \times C$, 其中 $C = ||\mathbf{x}-\mathbf{s}|| + ||\mathbf{c}|| + ||\mathbf{o}|| + ||\mathbf{s}|| + ||\mathbf{r}||$，包含所有高斯属性。

### 微调Latent Diffusion

- **跳过VAE编解码**：高斯属性分布与自然图像差异太大，不适合直接使用VAE自编码
- **标准化对齐**：使用GaussianVerse全集的逐像素均值和标准差对Atlas进行标准化，使其分布对齐VAE编码的图像潜变量
- **通道适配**：每个3通道属性与1通道不透明度拼接为4通道，UNet输入层重复4次适配 $128 \times 128 \times 16$ 的输入

训练损失：

$$\lambda_{diff}\mathcal{L}_{diff} + \lambda_{rgb}\mathcal{L}_{rgb} + \lambda_{mask}\mathcal{L}_{mask} + \lambda_{lpips}\mathcal{L}_{lpips}$$

其中 $\mathcal{L}_{diff}$ 为v参数化扩散损失，$\mathcal{L}_{rgb}$, $\mathcal{L}_{mask}$ 为渲染L1损失，$\mathcal{L}_{lpips}$ 为感知损失。

### 推理

从随机2D噪声出发，使用DPMsolver++进行反向扩散，引导尺度3.5。生成并渲染一个3DGS样本仅需**不到5秒**。

## 实验

### 主实验：文本到3D生成对比

| 方法 | CLIP score ↑ | VQA score ↑ | 高斯数量 ↓ |
|------|-------------|-------------|-----------|
| DreamGaussian | 20.52 | 0.37 | 40K |
| LGM | 20.28 | 0.35 | 66K |
| TriplaneGaussian | 21.10 | 0.46 | 16K |
| GaussianCube | 22.31 | 0.52 | 33K |
| **GaussianAtlas (Ours)** | **23.20** | **0.61** | **16K** |

在CLIP分数上超过GaussianCube 0.9，VQA分数高17%，且仅使用其一半的高斯数量和训练步数。

### 消融实验：预训练2D模型 vs 从零训练

| 方法 | 训练步数 | CLIP score ↑ | VQA score ↑ |
|------|---------|-------------|-------------|
| 从零训练 | 500K | 19.33 | 0.23 |
| 迁移2D LD | 500K | 21.61 | 0.49 |
| 从零训练 | 1M | 20.85 | 0.40 |
| 迁移2D LD | 1M | **23.20** | **0.61** |

预训练2D模型在相同步数下显著优于从零训练，验证了2D知识的可迁移性。

### 用户研究

从2,500+有效回复中统计：
- vs GaussianCube：**65%** 用户偏好本方法
- vs TriplaneGaussian：**88%** 用户偏好本方法

### 关键发现

1. **确定性映射至关重要**：基于优化的展平方法（学习每个物体独立的UV映射）会导致不一致的视觉模式，微调后只能生成噪声
2. **权重变化微小**：微调后UNet权重与预训练权重的差异极小（最大变化层也比随机初始化小8倍），说明预训练权重是优良的3D生成起点
3. **无需VAE**：直接在标准化后的Atlas空间训练UNet，避免了VAE对非自然图像的不适配

## 亮点与洞察

1. **统一2D/3D生成**：首次证明预训练text-to-image扩散模型可直接微调用于3D高斯生成，无需复杂的中间步骤
2. **巧妙的2D表示设计**：球面OT→等距投影→平面OT的三步流程既保持3D拓扑连续性，又确保跨物体映射一致性
3. **大规模数据集贡献**：GaussianVerse（205K高质量拟合）为社区提供了重要基础设施
4. **推理极快**：生成+渲染不到5秒，远优于SDS优化方法（分钟级）

## 局限性

- 生成分辨率受限于Atlas网格大小（$128 \times 128 = 16K$ 高斯），复杂物体细节不足
- 数据集构建成本极高（3.8+ A100 GPU年）
- 等距矩形投影在极点区域存在固有的面积畸变
- 仅支持单物体生成，未扩展到场景级别
- 缺乏与TRELLIS等最新方法的直接对比

## 相关工作与启发

- **2D扩散模型迁移**：Marigold（深度预测）, GeoWizard（几何预测）启发了对3D任务的迁移
- **3D高斯生成**：GaussianCube（3D网格OT）, GVGen（体素偏移）, DiffGS（三平面映射）
- **2D表示3D**：Triplane系列（NFD, CRM, InstantMesh）, Omages（UV映射）, DiffSplat（多视图潜变量）
- **数据集**：ShapeSplat, Objaverse系列

**启发**：确定性、一致性的3D→2D映射是成功迁移的关键——学习式映射虽灵活但破坏了跨样本的模式一致性。

## 评分
- 新颖性：★★★★★ — 首创Gaussian Atlas表示，开辟2D模型直接微调做3D生成的全新路径
- 技术深度：★★★★☆ — OT映射+投影的设计简洁优雅，但扩散模型训练本身较标准
- 实用性：★★★★☆ — 推理极快但数据集构建成本巨大，需要显著计算资源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM)](unleashing_vecset_diffusion_model_for_fast_shape_generation.md)
- [\[ICCV 2025\] CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation](cuts3d_cutting_semantics_in_3d_for_2d_unsupervised_instance_segmentation.md)
- [\[ICCV 2025\] RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors](ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)
- [\[ICCV 2025\] NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement](neuraleaf_neural_parametric_leaf_models_with_shape_and_deformation_disentangleme.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
