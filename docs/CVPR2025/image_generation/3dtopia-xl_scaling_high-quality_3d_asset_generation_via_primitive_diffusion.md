---
title: >-
  [论文解读] 3DTopia-XL: Scaling High-Quality 3D Asset Generation via Primitive Diffusion
description: >-
  [CVPR 2025][图像生成][3D生成] 提出基于新型原语表示PrimX和Diffusion Transformer的原生3D生成模型3DTopia-XL，能从文本或图像输入生成带有高分辨率几何、纹理和PBR材质的高质量3D资产，在质量和效率上显著超越现有方法。
tags:
  - CVPR 2025
  - 图像生成
  - 3D生成
  - PBR材质
  - 原语表示
  - Transformer
  - 纹理网格
---

# 3DTopia-XL: Scaling High-Quality 3D Asset Generation via Primitive Diffusion

**会议**: CVPR 2025  
**arXiv**: [2409.12957](https://arxiv.org/abs/2409.12957)  
**代码**: https://github.com/3DTopia/3DTopia-XL (有)  
**领域**: 3D生成 / 图像生成  
**关键词**: 3D生成, PBR材质, 原语表示, 扩散Transformer, 纹理网格

## 一句话总结
提出基于新型原语表示PrimX和Diffusion Transformer的原生3D生成模型3DTopia-XL，能从文本或图像输入生成带有高分辨率几何、纹理和PBR材质的高质量3D资产，在质量和效率上显著超越现有方法。

## 研究背景与动机

高质量3D资产在影视、游戏、虚拟现实等领域需求巨大，但手动创建代价极高。现有自动3D生成方法主要分三类：

**SDS方法**（如DreamFusion）：通过逐场景优化将2D扩散先验提升至3D，但优化耗时、几何质量差、多面不一致。

**稀疏视图重建**（如LRM、InstantMesh）：用大模型从少量视图回归3D，但多数基于triplane-NeRF表示，参数效率低导致分辨率受限，且为确定性方法缺乏多样性。

**原生3D扩散模型**（如Shap-E、3DTopia）：建模3D分布直接生成3D物体，但几乎都无法生成包含几何+纹理+材质的PBR资产。

**核心矛盾**：现有3D表示要么参数效率低（triplane）、要么无法编码PBR材质、要么张量化速度慢，难以在大规模数据上训练高质量3D扩散模型。

**本文切入角度**：设计一种新的基于原语的3D表示PrimX，同时编码形状/颜色/材质为紧凑的 $N \times D$ 张量，然后在此基础上用DiT进行扩散生成。核心idea：**用表面锚定的小体素原语集合高效表示textured mesh，再用Transformer建模原语间的全局关联。**

## 方法详解

### 整体框架
输入为文本prompt或单张图像，经过两阶段流程输出带PBR材质的GLB网格：
1. **PrimX表示**：将3D textured mesh编码为 $N \times D$ 紧凑张量
2. **Primitive Patch Compression**：用3D VAE将每个原语的payload压缩到潜空间
3. **Latent Primitive Diffusion**：用DiT在潜空间原语集合上做扩散生成
4. **PBR资产提取**：从生成的PrimX反解出高质量纹理网格（GLB文件）

### 关键设计

1. **PrimX表示**:

    - 功能：将textured mesh的形状、颜色、材质统一编码为一组表面锚定的体素原语集合
    - 核心思路：在网格表面采样 $N$ 个锚点，每个原语 $\mathcal{V}_k = \{\mathbf{t}_k, s_k, \mathbf{X}_k\}$ 包含位置、尺度和 $a^3 \times 6$ 的payload（1维SDF + 3维RGB + 2维材质）。空间任一点的属性通过加权插值得到：
    $F_{\mathcal{V}}(\mathbf{x}) = \sum_{k=1}^N w_k(\mathbf{x}) \cdot \mathcal{I}(\mathbf{X}_k, (\mathbf{x}-\mathbf{t}_k)/s_k)$
   权重 $w_k$ 基于L∞距离归一化，保证局部支撑。整个mesh可表示为 $N \times D$ 张量（$D = 3+1+a^3 \times 6$）。
    - 设计动机：相比triplane（参数效率低、分辨率受限）和MLP（慢），PrimX在相同参数预算下拟合精度最高且速度快10倍。表面锚定确保参数集中在有意义的区域。

2. **Primitive Patch Compression（3D VAE）**:

    - 功能：将每个原语的高维payload压缩为低维潜表示
    - 核心思路：用3D卷积构建VAE，对每个原语独立压缩。编码器将 $\mathbf{X}_k \in \mathbb{R}^{a^3 \times 6}$ 压缩为 $\hat{\mathbf{X}}_k \in \mathbb{R}^{(a/2)^3 \times 1}$，压缩率48倍。训练目标为重建损失+KL正则：
    $\mathcal{L}_{\text{ae}} = \mathbb{E}[\|\mathbf{X}_k - D(E(\mathbf{X}_k))\|_2 + \lambda_{\text{kl}} \mathcal{L}_{\text{kl}}]$
    - 设计动机：全局压缩（如对整个triplane做VAE）会使潜空间过于复杂；局部patch独立压缩简单高效，全局语义交给后续扩散模型建模。

3. **Latent Primitive Diffusion（DiT）**:

    - 功能：在潜空间原语集合上建模分布，实现条件3D生成
    - 核心思路：将每个原语当做一个token，用28层DiT建模原语间的全局关联。包含cross-attention接入条件信号（文本/图像编码）、self-attention建模原语间关系、AdaLN注入时间步。采用v-prediction + CFG + cosine调度训练。PrimX的排列不变性天然适配Transformer，无需位置编码。
    - 设计动机：得益于PrimX的紧凑性，可以直接在高分辨率上训练而无需超分辨率后处理，框架简洁统一。

4. **PBR资产提取**:

    - 功能：从PrimX无损转回textured mesh（GLB格式）
    - 核心思路：用Marching Cubes在SDF零等值面提取几何；在1024×1024 UV空间采样颜色和材质值；对UV贴图做膨胀+最近邻插值抗锯齿。
    - 设计动机：大多数3D生成方法导出mesh时用顶点着色，质量急剧下降。PrimX的高质量SDF表面支持高分辨率UV采样，避免质量损失。

### 损失函数 / 训练策略
- **PrimX拟合**：两阶段微调——先优化SDF（$\lambda_{\text{SDF}}=10, \lambda=0$，1k迭代），再优化颜色+材质（$\lambda_{\text{SDF}}=0, \lambda=1$，1k迭代），总耗时约1.5分钟/样本
- **VAE**：重建损失 + KL正则
- **DiT**：v-prediction目标 + CFG（10%概率drop条件），cosine噪声调度，1000步

## 实验关键数据

### 主实验：表示质量对比（相同1.05M参数预算）

| 表示方法 | 拟合时间 | CD ×10⁻⁴ ↓ | PSNR-SDF ↑ | PSNR-RGB ↑ | PSNR-Mat ↑ |
|---------|---------|-----------|-----------|-----------|-----------|
| MLP | 14 min | 4.502 | 40.73 | 21.19 | 13.99 |
| MLP w/ PE | 14 min | 4.638 | 40.82 | 21.78 | 12.75 |
| Triplane | 16 min | 9.678 | 39.88 | 18.28 | 16.46 |
| Dense Voxels | 10 min | 7.012 | 41.70 | 20.01 | 15.98 |
| **PrimX** | **1.5 min** | **1.310** | **41.74** | **21.86** | **16.50** |

### 消融实验：原语数量和分辨率

| N（原语数） | a³（分辨率） | 参数量 | PSNR-SDF ↑ | PSNR-RGB ↑ |
|-----------|-----------|-------|-----------|-----------|
| 64 | 32³ | 2.10M | 61.05 | 22.18 |
| 256 | 16³ | 1.05M | 59.05 | 23.50 |
| 2048 | 8³ | 1.05M | **62.52** | **24.23** |

### 关键发现
- **更多小原语优于更少大原语**：相同参数预算下，N=2048/a=8的配置在几何和纹理上都显著优于N=64/a=32
- PrimX拟合速度是triplane的**10倍以上**（1.5min vs 16min），且几何质量显著更好
- 生成的PBR资产可直接用于Blender等图形引擎，展现真实的高光和光泽效果
- Text-to-3D的CLIP Score优于Shap-E和3DTopia

## 亮点与洞察
- PrimX表示设计精妙：表面锚定+局部体素，兼顾参数效率和表达能力
- 局部压缩+全局扩散的分治策略，使高分辨率3D生成无需超分辨率模块
- 在同类原生3D扩散模型中首个支持完整PBR材质（几何+纹理+金属度/粗糙度）
- 3D inpainting和插值等应用展示了原生3D扩散模型相对重建方法的独特优势

## 局限与展望
- 生成质量仍受训练数据规模和多样性限制
- PrimX原语数量固定，不能自适应复杂度不同的物体
- 当前只支持单物体生成，场景级生成有待探索
- UV unwrapping在极端拓扑下可能不稳定

## 相关工作与启发
- M-SDF（Yariv et al.）提出了mosaic SDF概念，但只编码形状；PrimX扩展到形状+颜色+材质
- LRM系列的triplane表示参数效率瓶颈是本文核心动力
- DiT在2D图像生成的成功启发了将其扩展到3D原语集合上
- 核心启发：**3D生成的瓶颈在表示而非模型**——好的表示能让经典扩散框架直接work

## 评分
- 新颖性: ⭐⭐⭐⭐ PrimX表示设计有新意，但整体pipeline是VAE+DiT的标准范式
- 实验充分度: ⭐⭐⭐⭐⭐ 表示对比、生成对比、大量消融实验都很充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ 首个支持PBR材质的高质量原生3D生成模型，实用价值高

<!-- RELATED:START -->

## 相关论文

- [OmniStyle: Filtering High Quality Style Transfer Data at Scale](omnistyle_filtering_high_quality_style_transfer_data_at_scale.md)
- [EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation](eden_enhanced_diffusion_for_high-quality_large-motion_video_frame_interpolation.md)
- [EmoDubber: Towards High Quality and Emotion Controllable Movie Dubbing](emodubber_towards_high_quality_and_emotion_controllable_movie_dubbing.md)
- [ArtiFade: Learning to Generate High-quality Subject from Blemished Images](artifade_learning_to_generate_high-quality_subject_from_blemished_images.md)
- [StableAnimator: High-Quality Identity-Preserving Human Image Animation](stableanimator_high-quality_identity-preserving_human_image_animation.md)

<!-- RELATED:END -->
