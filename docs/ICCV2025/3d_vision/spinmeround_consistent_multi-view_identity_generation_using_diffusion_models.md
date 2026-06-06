---
title: >-
  [论文解读] SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models
description: >-
  [ICCV2025][3D视觉][多视角生成] 提出 SpinMeRound，一种基于身份嵌入的多视角扩散模型，能从单张或少量人脸图像生成 360° 全头部一致性肖像及对应法线图，在人脸新视角合成任务上超越现有多视角扩散方法。
tags:
  - "ICCV2025"
  - "3D视觉"
  - "多视角生成"
  - "扩散模型"
  - "人脸新视角合成"
  - "身份保持"
  - "法线估计"
  - "3D Gaussian Splatting"
---

# SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models

**会议**: ICCV2025  
**arXiv**: [2504.10716](https://arxiv.org/abs/2504.10716)  
**代码**: 暂未开源  
**领域**: 3D视觉  
**关键词**: 多视角生成, 扩散模型, 人脸新视角合成, 身份保持, 法线估计, 3D Gaussian Splatting

## 一句话总结

提出 SpinMeRound，一种基于身份嵌入的多视角扩散模型，能从单张或少量人脸图像生成 360° 全头部一致性肖像及对应法线图，在人脸新视角合成任务上超越现有多视角扩散方法。

## 研究背景与动机

从单张人脸图像生成任意视角的高质量头部肖像是计算机视觉中的长期难题。主要挑战来自以下几方面：

**3D 人脸数据稀缺**：大规模多视角全头部数据集极度匮乏，制约了模型训练

**传统方法局限**：3DMM 方法仅能建模面部区域，无法处理发型等复杂结构；NeRF 方法（如 PanoHead）的后脑合成质量差，野外图像反演困难

**现有扩散模型的不足**：
   - 通用多视角扩散模型（如 Cat3D）未针对人脸优化，存在恐怖谷效应
   - Zero123 系列生成质量低且多视角一致性差
   - DiffPortrait3D 仅能生成近正面视角
   - Era3D/Morphable Diffusion 只支持固定视角
   - 视频扩散模型（SV3D）计算量大且受限于特定相机轨迹

作者认为，需要一种专门针对人脸设计的多视角扩散方法，能够在保持身份一致性的同时生成覆盖全头部的高保真视角。

## 方法详解

### 整体架构

SpinMeRound 基于潜空间多视角 UNet，核心设计包含三个关键组件：身份条件化机制、多视角扩散模型和新视角采样策略。

### 1. 身份条件化机制

- 使用预训练 ArcFace 网络提取身份嵌入向量 $\mathbf{w} \in \mathbb{R}^{512}$
- 采用 Arc2Face 的注入方式：构造文本提示 "a photo of \<\<id\>\> person"，将身份嵌入替换对应 token
- 经 CLIP 文本编码器处理后得到条件向量 $\mathbf{c} \in \mathbb{R}^{N \times 768}$

### 2. 多视角扩散模型

**输入表示**：模型同时处理 $P = (M + K) = 8$ 对人脸图像及其法线图，其中 $M \in \{1, 3\}$ 为条件视角数，$K$ 为目标视角数。

**编码流程**：
- 使用 SD1.5 预训练 AutoEncoder 将图像和法线编码到潜空间：$\mathbf{z} \in \mathbb{R}^{4 \times 64 \times 64}$
- 图像潜向量与法线潜向量按通道拼接
- 每个视角的潜向量与 ray coordinate map $\mathbf{r} \in \mathbb{R}^{149 \times 64 \times 64}$（编码光线原点和方向）拼接
- 附加二值掩码 $\mathbf{m} \in \{0, 1\}^{1 \times 64 \times 64}$ 区分条件视角与目标视角

**网络结构**：
- 以 Arc2Face 为基础初始化
- 在原有 2D 自注意力层之间插入 3D 注意力层（参考 Cat3D），实现多视角信息共享
- 扩展输入/输出卷积层通道以适配法线和相机信息

### 3. 训练策略

- 基于 EDM 框架训练，分两阶段：
    - **阶段一**（600k 迭代）：单条件视角训练
    - **阶段二**（额外 1M 迭代）：随机选择 0、1 或 3 个条件视角，各概率 1/3
- 50% 概率将白色背景替换为随机颜色以增强数据多样性
- CFG 训练：以 $P_{uncond} = 0.15$ 概率随机替换身份向量为空字符串、条件图像为零图
- 对数信噪比偏移 $\log(N)$，$N = 7$ 为目标视图数

### 4. 三步采样策略（单图输入）

给定单张野外图像，通过三步策略生成覆盖全头部的一致视角：

**步骤 A — 对齐与法线生成**：
- 使用 PanoHead 对齐方式裁剪对齐输入图像
- 将法线生成视为 channel-wise inpainting 任务，利用条件引导采样获取对应法线图

**步骤 B — 锚点视角生成**：
- 生成 7 个锚点图像，覆盖 $\pm 45°, \pm 90°, \pm 135°, 180°$ 共 360° 范围

**步骤 C — 中间视角生成**：
- 以输入图像和最近两个锚点图像为条件三元组，生成任意中间视角
- 可生成 48、88 或更多视角，取决于角度步长

采样使用 EDM 采样器，50 步，guidance scale = 3。

### 5. 训练数据集

- 使用 PanoHead 生成约 7k 合成身份（从 ~10k 中手动去除后脑伪影的样本）
- 每个身份渲染 125 个视角的图像和法线图
- 通过 marching cubes 从三平面特征图提取形状，用 PyTorch3D 渲染法线

## 实验关键数据

### 定量比较（NeRSemble 数据集，222 个身份，16 个角度）

| 方法 | L2↓ | LPIPS↓ | SSIM↑ | ID Sim (ArcFace)↑ | ID Sim (VGGFace)↑ |
|------|------|--------|-------|---------------------|---------------------|
| EG3D (NeRF) | 0.025 | 0.4 | 0.55 | 0.31 | 0.89 |
| PanoHead (NeRF) | 0.012 | 0.32 | 0.65 | 0.27 | 0.88 |
| Zero123 | 0.195 | 0.515 | 0.55 | 0.169 | 0.44 |
| Zero123-XL | 0.198 | 0.51 | 0.563 | 0.118 | 0.442 |
| SV3D | 0.087 | 0.41 | 0.660 | 0.36 | 0.881 |
| DiffPortrait3D | 0.1 | 0.5 | 0.35 | 0.55 | 0.887 |
| **SpinMeRound** | **0.033** | **0.3** | **0.73** | **0.61** | **0.911** |

SpinMeRound 在 LPIPS、SSIM 和双身份相似度指标上均达到 SOTA，L2 距离与 NeRF 方法相当。

### 消融实验

| 变体 | L2↓ | LPIPS↓ | SSIM↑ |
|------|------|--------|-------|
| 无输入图像（仅 ID 嵌入） | 0.1246 | 0.4299 | 0.568 |
| 无身份嵌入 | 0.028 | 0.26 | 0.70 |
| 无法线生成 | 0.056 | 0.32 | 0.65 |
| **完整 SpinMeRound** | **0.018** | **0.22** | **0.75** |

三个组件（输入图像、身份嵌入、法线生成）均对最终性能有显著贡献。

### 其他能力
- **无条件采样**：利用 CFG 训练方案，设置空身份嵌入即可生成全新身份的多视角图像
- **3D 重建**：48 视角输入 3DGS 可重建出一致的 3D 头部
- **身份插值**：在身份嵌入空间进行线性插值，可实现平滑的身份渐变

## 亮点与洞察

1. **身份嵌入 + 多视角扩散的巧妙结合**：将 ArcFace 身份特征注入扩散过程，既利用了人脸识别模型的泛化能力，又保证了多视角间的身份一致性
2. **法线图联合生成**：同时生成 RGB 图像和法线图，不仅提供了 3D 形状先验，消融实验证明这一设计显著提升了一致性和细节质量
3. **三步锚点采样策略**：优雅地解决了模型单次仅能生成有限视角的限制，通过先生成锚点再填充中间视角，实现任意密度的 360° 覆盖
4. **纯合成数据训练却泛化到野外图像**：仅使用 PanoHead 生成的约 7k 合成身份训练，却能在真实的 NeRSemble 数据集和野外图像上取得最佳表现
5. **支持无条件生成和身份插值**：CFG 训练方案带来了额外的生成灵活性

## 局限与展望

1. **训练数据依赖 PanoHead**：合成数据的质量上限受限于 PanoHead，特别是后脑区域的质量；若能获取真实多视角数据或更先进的合成工具，性能有望进一步提升
2. **仅支持静态头部**：不支持表情变化或动态建模，无法生成可动画化的 avatar（同期工作 Pippo/DiffPortrait360 已部分探索此方向）
3. **分辨率受限于 SD1.5**：基于 SD1.5 的 512×512 分辨率，升级到更强大的基础模型（如 SDXL/SD3）可能带来质量飞跃
4. **采样效率**：生成 48 个视角需要多轮 50 步采样，速度方面有优化空间
5. **有限的训练身份数**：仅约 7k 身份可能限制了泛化能力，扩展合成数据规模值得探索
6. **未与最新闭源方法对比**：Cat3D 为闭源，无法公平对比

## 相关工作与启发

- **Cat3D [Gao et al., 2024]**：通用多视角扩散框架，本文采用其 3D 注意力层和相机编码方式，但将其专门化到人脸领域并增加身份条件和法线生成
- **Arc2Face [Deng et al., 2019]**：提供了身份嵌入注入扩散模型的优雅机制，本文以此为基础模型初始化
- **PanoHead [An et al., 2023]**：360° 全头部 GAN，本文用于训练数据生成，也是对齐算法的来源
- **Zero123 / SV3D**：代表性的多视角/视频扩散基线，本文在人脸重建指标上全面超越
- **3DGS [Kerbl et al., 2023]**：用于验证多视角一致性，表明生成的视角可直接用于高质量 3D 重建

**对后续研究的启发**：将领域先验（身份嵌入）融入通用多视角扩散框架的思路可推广到其他物体类别（如车辆、建筑），辅助模态的联合生成（法线图）也是提升一致性的有效手段。

## 评分
- 新颖性: ⭐⭐⭐⭐ — 身份嵌入 + 法线联合生成 + 锚点采样的组合有效且新颖
- 实验充分度: ⭐⭐⭐⭐ — 定量定性比较全面，消融充分，但缺少与闭源 Cat3D 的对比
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，方法阐述完整
- 价值: ⭐⭐⭐⭐ — 在人脸全头部新视角合成方向推进了 SOTA，对 3D avatar 构建有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MV-Adapter: Multi-view Consistent Image Generation Made Easy](mv-adapter_multi-view_consistent_image_generation_made_easy.md)
- [\[ICCV 2025\] MVGBench: a Comprehensive Benchmark for Multi-view Generation Models](mvgbench_a_comprehensive_benchmark_for_multi-view_generation_models.md)
- [\[ICCV 2025\] MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion](materialmvp_illumination-invariant_material_generation_via_multi-view_pbr_diffus.md)
- [\[ICCV 2025\] FlexGen: Flexible Multi-View Generation from Text and Image Inputs](flexgen_flexible_multi-view_generation_from_text_and_image_inputs.md)
- [\[ECCV 2024\] MVDD: Multi-View Depth Diffusion Models](../../ECCV2024/3d_vision/mvdd_multi-view_depth_diffusion_models.md)

</div>

<!-- RELATED:END -->
