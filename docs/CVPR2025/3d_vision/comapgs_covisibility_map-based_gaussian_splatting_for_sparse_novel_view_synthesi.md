---
title: >-
  [论文解读] CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis
description: >-
  [CVPR 2025][3D视觉][稀疏视角合成] 提出CoMapGS，利用像素级共视性图（covisibility map）来指导稀疏视角3DGS中初始点云增强和自适应加权监督，首次显式关注并恢复高不确定性的单视角区域。 稀疏视角新视角合成面临三个核心挑战： 1. 区域不平衡监督：高共视区域因多视角存在被过度优化…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "稀疏视角合成"
  - "3D高斯溅射"
  - "共视性图"
  - "点云增强"
  - "不确定性感知"
---

# CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis

**会议**: CVPR 2025  
**arXiv**: [2503.20998](https://arxiv.org/abs/2503.20998)  
**代码**: [youngkyoonjang.github.io/projects/comapgs](https://youngkyoonjang.github.io/projects/comapgs)  
**领域**: 3D视觉  
**关键词**: 稀疏视角合成, 3D高斯溅射, 共视性图, 点云增强, 不确定性感知

## 一句话总结

提出CoMapGS，利用像素级共视性图（covisibility map）来指导稀疏视角3DGS中初始点云增强和自适应加权监督，首次显式关注并恢复高不确定性的单视角区域。

## 研究背景与动机

稀疏视角新视角合成面临三个核心挑战：

1. **区域不平衡监督**：高共视区域因多视角存在被过度优化，单视角区域（仅在一个训练视角可见）缺乏多视角约束而被忽视
2. **稀疏点云初始化**：COLMAP在少量训练图像下的特征匹配点云非常稀疏，缺乏几何细节
3. **高不确定性区域**：单视角区域无多视角几何约束，现有方法要么忽略要么惩罚这些区域，而非利用其中的信息

现有方法（如FSGS、CoR-GS）主要关注高共视区域的优化，对单视角区域缺乏有效的恢复策略。本文核心创新是：**利用共视性图量化每个像素的不确定性水平，并据此进行差异化的点云增强和监督加权**。

## 方法详解

### 整体框架

CoMapGS基于CoR-GS等3DGS稀疏视角方法，增加三个关键步骤：(1) 使用MASt3R密集对应预测生成像素级共视性图；(2) 通过密集对应三角化和单目深度估计对齐，增强低/高不确定性区域的初始点云；(3) 训练proximity MLP分类器并结合共视性图进行自适应加权的proximity loss监督。

### 关键设计

1. **共视性图生成与初始点云增强**:
    - 功能：量化每个像素的多视角共视次数，并据此分区域增强稀疏点云
    - 核心思路：对每对训练图像使用MASt3R预测密集对应，累计每个像素的匹配次数$M_i(x,y)$得到共视性图。低不确定性区域（$M_i \geq 1$）通过三角化密集对应点$P_T$补充COLMAP稀疏点云$P_C$，保留距离$P_C$超过阈值$\epsilon$的新点。高不确定性区域（$M_i = 0$的单视角区域）通过单目深度估计反投影生成$P_d^{high}$，学习一个各向异性缩放变换$f_{scale}$将其对齐到三角化点云的坐标系
    - 设计动机：COLMAP的关键点匹配在少量图像下极度稀疏，而MASt3R的密集对应可大幅增加点云密度；单目深度虽然尺度任意但可通过已知区域学习对齐

2. **共视性图加权的Proximity Loss**:
    - 功能：根据区域不确定性水平自适应调整监督强度
    - 核心思路：训练3层MLP分类器$f_p$区分增强点云$P_{final}$（正样本）和随机远离点（负样本），输出proximity score $s \in [0,1]$。对视锥内高斯，权重$w_{in} = 1/(M_i(\pi(g,\mathbf{H}_i)) + 1)$与共视次数成反比——单视角区域权重最大（=1），高共视区域权重小。对视锥外高斯，当场景平均共视分数$S > 0.7$时启用基于$S$的线性衰减权重$w_{out}$
    - 设计动机：高共视区域已被标准重建损失充分监督，proximity loss应更多关注欠约束的单视角区域；高共视场景中视锥外的高斯也应被适度约束

3. **增强点云的分区域策略**:
    - 功能：将增强点云按共视性分为不同置信度区域分别处理
    - 核心思路：最终点云$P_{final} = P_u^{low} \cup P_s^{high}$，其中$P_u^{low}$来自三角化（高置信度），$P_s^{high}$来自对齐后的单目深度反投影（低置信度）。分类器和加权监督自然区分不同来源点的可靠性
    - 设计动机：三角化点有多视角验证更可靠，深度反投影点虽精度较低但对填补空白区域至关重要

### 损失函数 / 训练策略

总损失添加proximity loss项：

$$\mathcal{L} = (1-\lambda)\mathcal{L}_1(I, I^*) + \lambda\mathcal{L}_{D\text{-}SSIM}(I, I^*) + \mathcal{L}_p$$

其中proximity loss $\mathcal{L}_p = \frac{1}{|G|}\sum_{g \in G}(\chi(g)w_{in} + (1-\chi(g))w_{out}) \cdot (1-s)$，$\chi(g)$指示高斯是否在视锥内。该方法可无缝集成到FSGS和CoR-GS等现有方法中。

## 实验关键数据

### 主实验

| 数据集/视角 | 指标 | CoR-GS | CoR-GS + CoMapGS | 提升 |
|------------|------|--------|-------------------|------|
| LLFF 3-view | PSNR/SSIM/LPIPS | 20.47/0.717/0.199 | 21.11/0.747/0.182 | +0.64/+0.030/-0.017 |
| LLFF 6-view | PSNR/SSIM/LPIPS | 24.78/0.844/0.116 | 25.20/0.854/0.108 | +0.42/+0.010/-0.008 |
| LLFF 9-view | PSNR/SSIM/LPIPS | 26.48/0.881/0.086 | 26.73/0.886/0.082 | +0.25/+0.005/-0.004 |
| Mip-NeRF 360 12-view | PSNR/SSIM/LPIPS | 19.16/0.574/0.414 | 19.68/0.591/0.394 | +0.52/+0.017/-0.020 |
| Mip-NeRF 360 24-view | PSNR/SSIM/LPIPS | 23.32/0.729/0.271 | 23.46/0.734/0.264 | +0.14/+0.005/-0.007 |

### 消融实验（LLFF 6-view）

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|------|-------|-------|--------|------|
| CoR-GS baseline | 24.777 | 0.844 | 0.116 | 基线 |
| + Proximity loss only | 24.787 | 0.845 | 0.116 | 无点云增强时提升有限 |
| + 低不确定性点云增强△ | 24.90 | 0.849 | 0.112 | 密集共视区域点云 |
| + △ + Proximity loss | 25.153 | 0.854 | 0.109 | 协同效应明显 |
| + 完整点云增强 | 25.076 | 0.852 | 0.109 | 加入单视角区域点 |
| **完整CoMapGS** | **25.204** | **0.854** | **0.108** | 所有组件 |

### 关键发现

- **初始点云增强和加权监督有协同效应**：单独使用proximity loss仅提升0.01 PSNR，但配合增强点云后提升0.25+
- 视角越少提升越大（3-view提升0.64 PSNR vs 9-view提升0.25），说明该方法对极稀疏场景尤其有效
- 即使仅增强低不确定性区域的点云（△）也有显著提升（PSNR +0.12，LPIPS -0.004），说明点云密度是关键瓶颈
- Mip-NeRF 360的LPIPS提升特别明显（-0.020），因为室外场景有更多高不确定性区域

## 亮点与洞察

- **首次显式关注单视角区域**：以往方法忽略或惩罚高不确定性区域，本文反其道行之，通过proximity loss更强力地约束这些区域
- 共视性图的概念简洁有力，将复杂的多视角几何关系压缩为每个像素一个整数计数
- **即插即用**设计：CoMapGS可直接集成到FSGS和CoR-GS中，不改变原有策略

## 局限与展望

- 依赖MASt3R进行密集对应预测，增加了预处理计算成本
- Proximity MLP分类器是离线训练的，不参与3DGS的在线优化，可能无法充分利用训练过程中几何的变化
- 单目深度对齐使用简单的线性回归（各向异性缩放），可能不足以处理复杂的非线性深度尺度变化
- 在3-view的PSNR上略低于ReconFusion（扩散模型方法），但SSIM/LPIPS更优

## 相关工作与启发

- 与DyCheck提出的共视性概念不同，本文将covisibility从评估工具扩展为训练信号
- 点云增强策略（密集对应三角化+深度对齐）可独立使用，对所有3DGS方法有增益
- Proximity classifier的思路类似于SDF场的占用预测，但更轻量化，值得在其他场景重建任务中尝试

## 评分

- 新颖性: ⭐⭐⭐⭐ 共视性图指导的自适应监督思路新颖，首次关注单视角区域恢复
- 实验充分度: ⭐⭐⭐⭐ LLFF和Mip-NeRF 360多设置评估，消融完整，与多个baseline对比
- 写作质量: ⭐⭐⭐⭐ 符号定义严谨，方法描述系统化，图例清晰
- 价值: ⭐⭐⭐⭐ 实用的即插即用模块，对稀疏视角合成有持续性的贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)
- [\[CVPR 2025\] SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis](splatflow_multi-view_rectified_flow_model_for_3d_gaussian_splatting_synthesis.md)
- [\[CVPR 2025\] Novel View Synthesis with Pixel-Space Diffusion Models](novel_view_synthesis_with_pixel-space_diffusion_models.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](../../ICCV2025/3d_vision/self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [\[CVPR 2025\] DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting](dropgaussian_structural_regularization_for_sparse-view_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
