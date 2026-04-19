---
title: >-
  [论文解读] UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction
description: >-
  [3D视觉] 提出 UrbanGS，一个面向城市级场景的可扩展 3DGS 重建框架，通过深度一致的 D-Normal 正则化、空间自适应高斯剪枝和统一分区策略，同时提升几何精度、渲染质量和内存效率。
tags:
  - 3D视觉
---

# UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction

- **会议**: ICLR 2026
- **arXiv**: [2602.02089](https://arxiv.org/abs/2602.02089)
- **代码**: 未公开
- **领域**: 3D 视觉 / 大规模场景重建
- **关键词**: 3D Gaussian Splatting, Large-Scale Reconstruction, Depth-Normal Regularization, Gaussian Pruning, Urban Scene

## 一句话总结

提出 UrbanGS，一个面向城市级场景的可扩展 3DGS 重建框架，通过深度一致的 D-Normal 正则化、空间自适应高斯剪枝和统一分区策略，同时提升几何精度、渲染质量和内存效率。

## 研究背景与动机

3DGS 在有限场景中表现优异，但扩展到大规模城市环境面临三大挑战：

**几何一致性差**：仅监督渲染法线只能更新旋转参数，无法更新位置参数，导致表面重建不精确

**内存效率低**：均匀区域（天空、远处建筑立面）生成大量冗余高斯基元

**计算可扩展性差**：分区方案引入边界不连续性，无关视角的处理浪费计算资源

## 方法详解

### 整体框架

UrbanGS 包含三个核心模块：

1. **深度一致 D-Normal 正则化**（几何精度）
2. **空间自适应高斯剪枝 SAGP**（内存效率）
3. **统一分区和视角分配**（可扩展性）

### 1. 深度一致 D-Normal 正则化

**问题**：直接用伪法线 $N$ 监督渲染法线 $\hat{N}$ 只能通过梯度更新旋转参数 $R$，无法有效更新位置参数 $u$。

**解决方案**：从渲染深度图推导 D-Normal $\bar{N}_d$：

$$\bar{N}_d(n,p) = \frac{\nabla_v d(n,p) \times \nabla_h d(n,p)}{|\nabla_v d \times \nabla_h d|}$$

其中 $d$ 是深度图反投影得到的 3D 坐标。D-Normal 正则化：

$$\mathcal{L}_{dn} = \|\bar{N}_d - N\|_1 + (1 - \bar{N}_d \cdot N)$$

通过 D-Normal 建立几何约束与深度的内在联系，使位置和旋转参数同时更新。

### 深度一致性正则化

为确保多视角深度一致性，引入逆深度损失和自适应置信度加权：

**逆深度损失**：

$$\mathcal{L}_{id}(u,v) = |\hat{D}^{-1}(u,v) - D_{ext}^{-1}(u,v)|$$

**几何感知置信度**：

$$w_d = \exp\left(\frac{\cos\phi - 1}{0.01}\right) \cdot \exp\left(-\frac{\epsilon_d}{0.1}\right)$$

其中 $\cos\phi$ 衡量深度梯度一致性，$\epsilon_d$ 衡量归一化逆深度偏差。

**总损失**：

$$\mathcal{L}_{total} = \mathcal{L}_{RGB} + \lambda_1 \mathcal{L}_n + \lambda_2 \mathcal{L}_{dn} + \lambda_3 (w_d \cdot \mathcal{L}_{id})$$

### 2. 空间自适应高斯剪枝 (SAGP)

**场景分区**：将场景划分为体素单元，特征长度与全局高斯密度相关：

$$\ell = \lambda \left(\frac{\mathcal{V}_{scene}}{\mathcal{N}}\right)^{1/3}$$

**局部体积归一化**（亚线性变换抑制过大基元）：

$$w_{v,i} = \left(\min\left(\frac{v_i}{\vartheta_{local}^{(t)}}, 1\right)\right)^{\kappa}$$

$\kappa=0.5$（平方根）放大精细结构的重要性。

**重要性评分**（三因素乘积）：

$$S_i = \phi_i \cdot \tau_i \cdot w_{v,i}$$

- $\phi_i$：归一化射线相交频率
- $\tau_i$：Sigmoid 映射的不透明度
- $w_{v,i}$：亚线性体积权重

高斯仅在同时具有高可见性、频繁观测和适当几何尺度时才被保留。

### 3. 分区策略

基于 CityGS 改进：
- 先通过 SAGP 剪枝全局粗糙 3DGS，减少冗余高斯吸引无关视角
- 子块边界保留共享高斯基元，避免几何不连续
- 基于几何和 SSIM 的相机视角分配

## 实验

### 数据集

- **Mill19**：Building, Rubble（航拍场景）
- **UrbanScene3D**：Residence, Sci-Art（城市场景）

### 主要结果（渲染质量）

| 方法 | Building PSNR | Rubble PSNR | Residence PSNR | Sci-Art PSNR |
|------|--------------|-------------|----------------|-------------|
| 3DGS | 22.53 | 25.51 | 22.36 | 24.13 |
| CityGS-v2 | - | - | - | - |
| VCR-GauS | - | - | - | - |
| **UrbanGS** | **最优** | **最优** | **最优** | **最优** |

UrbanGS 在所有数据集上的 SSIM、PSNR、LPIPS 均达到 SOTA 或接近 SOTA。

### 几何精度

通过渲染深度图的定性对比：
- UrbanGS 的物体表面更平滑
- CityGS-v2 和 VCR-GauS 在远处建筑和复杂区域出现失真

### 内存效率

SAGP 剪枝实现显著模型压缩（具体压缩比见消融），同时保持渲染质量。VCR-GauS 在 A5000 GPU 上因 OOM 失败，UrbanGS 可正常运行。

### 消融实验

| 消融 | 效果 |
|------|------|
| w/o D-Normal 正则化 | 位置参数无法有效更新，表面粗糙 |
| w/o 深度一致性 | 多视角深度不对齐 |
| w/o 置信度加权 | 不靠谱深度预测干扰优化 |
| w/o SAGP | 高斯数量爆炸，内存不足 |
| 全局 vs 自适应剪枝 | 自适应保留更多细节 |

## 亮点

1. **D-Normal 正则化**巧妙解决了法线监督无法更新位置参数的问题
2. **深度+法线双重监督**的理论动机充分，有数学证明
3. **SAGP** 是首个专为城市级 3DGS 设计的剪枝框架
4. **系统性方案**：几何精度 + 内存效率 + 可扩展性三者兼顾
5. 在 A5000 等消费级 GPU 上实现大规模场景重建

## 局限性

1. 依赖外部深度估计器（DepthAnything-v2）和法线估计器的质量
2. SAGP 的超参数（$\lambda, t, \kappa$）需调整
3. 分区策略主要继承 CityGS，创新有限
4. 仅在航拍/城市场景评估，室内大场景未验证
5. 逆深度损失对近处物体可能过度平滑

## 相关工作

- **大规模 3DGS**：VastGaussian (Lin et al., 2024) 分块但有边界不一致；CityGaussian (Liu et al., 2024a) 需耗时后处理；CityGS-v2 (Liu et al., 2024b) 用 2DGS 但降低渲染质量
- **几何优化**：2DGS (Huang et al., 2024a), VCR-GauS (Chen et al., 2024b) 引入深度/法线正则化但未充分更新位置
- **高斯剪枝**：Fan et al. (2023) 基于全局指标的简单剪枝在大场景中过度简化

## 评分

- **创新性**: ⭐⭐⭐⭐ — D-Normal 正则化和 SAGP 均为有针对性的贡献
- **实用性**: ⭐⭐⭐⭐⭐ — 直接解决城市级重建的实际痛点
- **清晰度**: ⭐⭐⭐⭐ — 方法描述系统，理论分析充分
- **意义**: ⭐⭐⭐⭐ — 为大规模 3DGS 提供了完整解决方案

<!-- RELATED:START -->

## 相关论文

- [S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction](../../ICCV2025/3d_vision/s3r-gs_streamlining_the_pipeline_for_large-scale_street_scene_reconstruction.md)
- [UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)
- [TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](../../CVPR2026/3d_vision/tehor_text-guided_3d_human_and_object_reconstruction_with_textures.md)
- [Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk](topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)
- [Universal Beta Splatting](universal_beta_splatting.md)

<!-- RELATED:END -->
