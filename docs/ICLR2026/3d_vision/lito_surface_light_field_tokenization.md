---
title: >-
  [论文解读] LiTo: Surface Light Field Tokenization
description: >-
  [ICLR 2026][3D视觉][表面光场] 提出LiTo——通过将表面光场(surface light field)编码为紧凑latent向量集合来同时建模3D几何和视角依赖外观：输入RGB-D多视角图像的光场随机子采样 -> Perceiver IO编码器(支持100万token输入的3D局部attention) + flow-matching几何解码器 + 高阶球谐Gaussian解码器 -> 实现重建和单图到3D生成都超越TRELLIS，首次在latent 3D表示中建模高光/菲涅尔反射等视角依赖效果。
tags:
  - ICLR 2026
  - 3D视觉
  - 表面光场
  - 3D latent表示
  - 视角依赖外观
  - Gaussian Splatting
  - flow matching
---

# LiTo: Surface Light Field Tokenization

**会议**: ICLR 2026  
**arXiv**: [2603.11047](https://arxiv.org/abs/2603.11047)  
**代码**: 无(Apple内部)  
**领域**: 3D视觉/生成  
**关键词**: 表面光场, 3D latent表示, 视角依赖外观, Gaussian Splatting, flow matching

## 一句话总结
提出LiTo——通过将表面光场(surface light field)编码为紧凑latent向量集合来同时建模3D几何和视角依赖外观：输入RGB-D多视角图像的光场随机子采样 -> Perceiver IO编码器(支持100万token输入的3D局部attention) + flow-matching几何解码器 + 高阶球谐Gaussian解码器 -> 实现重建和单图到3D生成都超越TRELLIS，首次在latent 3D表示中建模高光/菲涅尔反射等视角依赖效果。

## 研究背景与动机

**领域现状**：3D latent表示领域分为两类——几何only表示(3DShape2VecSet/TripoSG/ShapeTokens)只建模形状不含外观；几何+外观表示(TRELLIS/3DTopia-XL)加入外观但只支持视角无关的漫反射颜色(view-independent diffuse color)。

**现有痛点**：(1) 几何only方法无法渲染逼真的3D内容——缺少颜色/材质/光照效果；(2) TRELLIS虽然包含外观但用DINOv2特征的mean pooling，丢弃了视角方向信息——无法建模高光、菲涅尔反射等view-dependent效果；(3) 3DTopia-XL虽建模PBR材质但需要从mesh优化primitive表示的预处理步骤。

**核心矛盾**：真实物体外观强烈依赖观察角度(金属反射、菲涅尔效应等)，但现有3D latent表示丢弃了方向信息。要建模视角依赖效果，需要编码表面光场(surface light field)——不仅是表面位置和颜色，还要包含观察方向。

**切入角度**：RGB-D多视角图像就是表面光场的离散采样——每个像素提供一个(表面点位置, 观察方向, 颜色)元组。通过随机子采样这些光场样本作为输入，用编码器插值、用3阶球谐Gaussian解码器输出。

**核心 idea**：将表面光场的随机子采样编码为紧凑latent tokens，用双路解码器(flow-matching几何 + 球谐Gaussian外观)实现几何和视角依赖外观的统一3D表示。

## 方法详解

### 整体框架
输入：150张多视角RGB-D渲染图 -> 提取~1.6亿表面光场采样 -> 随机选100万作为编码器输入。输出：k=8192个d=32维latent tokens。

三个组件：(1) Perceiver IO编码器(支持100万token) (2) Flow-matching几何解码器 (3) 视角依赖Gaussian解码器

### 关键设计

1. **表面光场采样与编码**：

    - 功能：把RGB-D多视角图像转化为表面光场采样 $\{(\mathbf{x}_i, \hat{\mathbf{d}}_i, \mathbf{c}_i)\}$，编码为latent
    - 核心思路：从RGB-D反投影得到表面点 $\mathbf{x}$，从针孔相机模型得到观察方向 $\hat{\mathbf{d}}$，像素颜色得 $\mathbf{c}$。随机采样 $N=2^{20}$ 个作为编码器输入。编码器使用Perceiver IO，输出8192个32维latent tokens
    - 设计动机：完整表面光场信息量巨大(1.6亿采样)，但有大量冗余。随机子采样让编码器学会插值，generalize到完整光场。每个采样包含方向信息 $\hat{\mathbf{d}}$ 是捕获视角依赖效果的关键

2. **3D局部Attention实现百万级输入**：

    - 功能：让Perceiver IO高效处理100万token输入
    - 核心思路：设计3D patchification——将输入采样按K-NN分配到8192个query对应的空间patch中，每个query只attend其patch内的采样(类似ViT的16x16 patch但推广到3D表面)。自attention用voxel-based windowed attention(每层shift半个voxel)
    - 设计动机：标准Perceiver IO的cross attention对100万token计算量巨大。3D patchification用L2距离(非geodesic)近似表面局部性，是速度和精度的好权衡

3. **双路解码器(几何 + 视角依赖外观)**：

    - 功能：从latent同时恢复3D几何和视角依赖外观
    - 几何解码器：flow-matching建模3D表面分布 $p(\mathbf{x}|\mathcal{S}) \approx \delta(\mathbf{x} \in \partial\Omega)$。Loss: $\mathcal{L}_{geo} = \mathbb{E}_{t,\mathbf{x}} \|V(\mathbf{x}_t; t) - (\mathbf{x} - \epsilon)\|^2$。可在推理时采样点云
    - **Gaussian解码器**：输出3阶球谐(SH degree 3)的3D Gaussians用于视角依赖渲染。输入sparse occupancy grid作为query，cross attend到latent tokens，MLP输出每个occupied voxel 64个Gaussians。Loss: $\mathcal{L}_{radiance} = \|I_{est} - I_{gt}\|^2 + 0.2 \cdot \text{LPIPS}$
    - 设计动机：几何解码器不依赖mesh/occupancy/SDF的预处理——直接从点云学习。3阶球谐比TRELLIS的view-independent color多捕获高频视角依赖效果

4. **单阶段生成(vs TRELLIS的两阶段)**：

    - 功能：latent直接编码完整物体信息用于单阶段生成
    - 核心思路：训练DiT flow-matching模型(623M参数)，DINOv2编码输入图像，生成latent条件化于图像。训练时旋转世界坐标系使输入视角相机为identity -> 输出自动与输入视角对齐
    - 设计动机：TRELLIS需要先生成粗糙occupancy再生成SLAT(两阶段)。LiTo的latent已包含完整信息，单阶段更简洁。坐标系旋转策略确保生成物体与输入对齐(TRELLIS在canonical坐标生成需后处理)

### 训练策略
- 编码器+解码器：256 batch，64 GPU，90K iterations，9天
- 生成模型(DiT)：256 batch，128 H100 GPU，600K iterations，20天
- 数据：Objaverse-XL 500K物体子集(TRELLIS同源)，每个物体3种光照x150视角

## 实验关键数据

### 主实验：重建质量 (Toys4k)
| 方法 | PSNR↑ (simple) | SSIM↑ | LPIPS↓ | PSNR↑ (hard) | SSIM↑ | LPIPS↓ |
|------|---------------|-------|--------|-------------|-------|--------|
| TRELLIS | 31.12 | 0.974 | 0.034 | 27.57 | 0.941 | 0.090 |
| **LiTo** | **34.16** | **0.985** | **0.023** | **32.36** | **0.967** | **0.055** |

### 消融：生成质量 (Toys4k)
| 方法 | CLIP↑ | Conditioning View FID↓ | KID↓ | Novel View FID↓ |
|------|-------|----------------------|------|----------------|
| TRELLIS | 0.899 | 12.84 | 0.088 | 7.600 |
| **LiTo** | **0.905** | **6.219** | **0.009** | **6.216** |

### 关键发现
- **重建PSNR提升3dB**: 在hard设置(近距离相机)上从27.57提升到32.36，说明视角依赖建模在近距离观察时尤为重要
- **几何质量不降反升**: 额外建模外观不损害几何精度——在不使用GT粗糙几何的方法中，LiTo几何最优(Chamfer distance最低)
- **生成输入忠实度大幅提升**: Conditioning view FID从12.84降到6.219(坐标系旋转策略起作用)
- **球谐不同阶捕获不同特征**: degree 0=漫反射，degree 1=大致方向性，degree 2-3=高光/菲涅尔
- **Latent空间紧凑**: 8192x32 = 262K参数，比TRELLIS的20Kx11=220K和TripoSG的2048x64=131K大，但无需GT粗糙几何

## 亮点与洞察
- **表面光场作为3D表示的统一框架**：表面光场理论上可以重建任意相机位姿的图像——是最完整的3D外观表示。将其latent化是自然但之前未被充分探索的方向
- **随机子采样+编码器插值**的训练策略非常优雅：不需要完整的表面光场(不可能获得)，只需随机子集让编码器学会generalize
- **3D patchification**巧妙解决了百万级token输入的效率问题，且方法概念上与ViT的patch一致，易于理解
- **单阶段生成 + 坐标系旋转**比TRELLIS的两阶段更简洁，且自然解决了输出对齐问题

## 局限与展望
- 训练数据需要RGB-D多视角渲染(150视角)，获取成本较高
- 3D patchification用L2距离近似表面局部性，当多个表面靠近时会跨表面attend
- 未建模透明/半透明物体(深度图假设第一个交点)
- 计算开销大：编码器+解码器训练需64 GPU 9天，DiT需128 H100 20天

## 相关工作与启发
- **vs TRELLIS**: TRELLIS用DINOv2 mean pooling丢弃方向信息只有diffuse color。LiTo编码方向信息实现view-dependent。且TRELLIS需两阶段生成+canonical坐标，LiTo单阶段+输入对齐
- **vs TripoSG**: 几何only方法无外观。LiTo额外建模视角依赖外观但在无GT粗糙几何条件下几何质量也更好
- **vs 3DTopia-XL**: PrimX需要mesh到primitive的优化预处理。LiTo直接从RGB-D渲染构建输入更scalable

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在3D latent表示中实现视角依赖外观建模，表面光场tokenization概念新
- 实验充分度: ⭐⭐⭐⭐ 重建和生成都有详细对比，多数据集验证
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，与TRELLIS等对比明确
- 价值: ⭐⭐⭐⭐⭐ 对3D生成表示方法有重要推进，解决了视角依赖的关键盲点
