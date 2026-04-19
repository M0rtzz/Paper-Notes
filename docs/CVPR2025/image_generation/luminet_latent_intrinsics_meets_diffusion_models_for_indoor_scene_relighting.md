---
title: >-
  [论文解读] LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting
description: >-
  [CVPR 2025][relighting][diffusion] 提出 LumiNet，结合潜在内在表征和条件扩散模型实现室内场景级光照迁移，通过 Variational-StyLitGAN 数据策略和潜在 ControlNet 架构，仅用图像输入即可迁移复杂光照效果。
tags:
  - CVPR 2025
  - relighting
  - diffusion model
  - ControlNet
  - latent intrinsics
  - indoor scene
  - StyleGAN
---

# LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting

**会议**: CVPR 2025  
**arXiv**: [2412.00177](https://arxiv.org/abs/2412.00177)  
**代码**: [https://luminet-relight.github.io](https://luminet-relight.github.io)  
**领域**: image_generation  
**关键词**: relighting, latent intrinsics, diffusion model, ControlNet, StyleGAN, indoor scene, light transfer, cross-attention

## 一句话总结

提出 LumiNet，将源图像的潜在内在特征（128 维 albedo-like 表征）和目标图像的潜在外在光照码（16 维）注入改造后的 ControlNet，实现仅用图像输入的室内场景级光照迁移，包含镜面高光、阴影和间接照明等复杂效果。

## 研究背景与动机

**领域现状**: 图像级重光照在影视制作、建筑可视化和混合现实中有重要应用。扩散模型+ControlNet 已展示出强大的条件生成能力。

**现有痛点**: (1) **传统逆渲染**: 需要精确 3D 重建和材质分解，计算昂贵且误差累积；(2) **StyLitGAN**: 在 StyleGAN 潜在空间中能操控光照，但无法泛化到真实图像；(3) **Latent Intrinsics (Zhang et al.)**: 能分解潜在内在/外在表征，但不能泛化到复杂任意场景；(4) **IC-Light**: 擅长人像重光照但无法处理全场景；(5) **RGB↔X**: 需要 G-buffer 输入（法线、深度等），依赖显式几何信息。

**核心矛盾**: 室内场景光照迁移需要理解光源位置、材质交互和间接照明 — 这些信息高度场景特定，传统显式建模难以覆盖所有情况。

**本文切入角度**: 在潜在空间中操作 — 利用预训练的潜在内在分解模型提取几何/albedo 不变特征和光照码，将光照迁移转化为条件图像生成问题。

## 方法详解

### 整体框架

三部分：
1. **数据准备**: Variational-StyLitGAN 生成合成训练对 + 真实数据集互补
2. **Latent ControlNet**: 处理源图的潜在内在特征，保持几何和材质
3. **光照适配器**: 通过交叉注意力注入目标光照码

### 关键设计

**1. Variational-StyLitGAN 数据策略**
- **功能**: 解决训练数据稀缺问题 — 真实场景的不同光照配对极难获取。
- **核心机制**: 用 ConvNeXt 变分编码器将真实 LSUN 卧室图像映射到 StyLitGAN 潜在空间，在冻结的预训练生成器上产生每场景 7 种光照变化：
  $$\mathcal{L} = \underbrace{\text{MSE}(\mathbf{x}, \hat{\mathbf{x}}) + \text{LPIPS}(\mathbf{x}, \hat{\mathbf{x}})}_{\mathcal{L}_{rec}} + \underbrace{D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| \mathcal{N}(0,I))}_{\mathcal{L}_{KL}}$$
- **CLIP 过滤**: 用"photo-realistic"、"good lighting"、"illumination"等关键词筛选 ~1K 高质量样本。
- **设计动机**: 原始 StyLitGAN 随机采样会 mode collapse（每 10-20 次迭代输出几乎相同），变分映射引入真实图像多样性打破此问题。FID 从 47.99 降至 35.81。

**2. 潜在内在 ControlNet（Latent Intrinsic Control）**
- **功能**: 改造 ControlNet 使其在潜在空间而非图像空间操作，处理来自**两张不同图像**的表征。
- **核心机制**:
    - 从源图提取潜在内在特征 $\mathcal{A}_o \in \mathbb{R}^{H \times W \times 128}$（albedo-like，光照不变）
    - 从目标图提取光照外在码 $\mathcal{I}_{L_t} \in \mathbb{R}^{16}$
    - 将 $\mathcal{I}_{L_t}$ 广播至空间维度后与 $\mathcal{A}_o$ 拼接为 $\mathbb{R}^{H \times W \times 144}$，经卷积投射为 ControlNet 输入
- **vs 传统 ControlNet**: 传统 ControlNet 处理单场景的条件图；LumiNet 处理两张不同场景的潜在表征 — 保留源图几何/albedo、迁移目标图光照。

**3. 光照适配器网络（Lighting Adaptor）**
- **功能**: 将低维目标光照码注入预训练扩散模型的交叉注意力层。
- **核心机制**: MLP (3072→4096→4096→4096→3072) 将光照码映射为 $\mathcal{I}_{E_t} \in \mathbb{R}^{3 \times 1024}$，替代文本 embedding 送入交叉注意力。训练时不使用任何文本 prompt。
- **设计动机**: 交叉注意力提供了对扩散模型全局行为的控制通路，光照码通过此通路影响二阶效果（如桌面反射、间接照明）。

### 损失函数

$$\mathcal{L}_\text{Lumi} = \|\epsilon - \theta(\epsilon(S^{L_t})_t, t, \{\mathcal{A}_o, \mathcal{I}_{L_t'}\}, \mathcal{I}_{E_t}, \epsilon(S^{L_o}))\|_2^2$$

标准潜在扩散去噪损失。仅训练 ControlNet 和交叉注意力层，其余参数冻结。

### 推理增强

- **Bypass Decoder**: 替换默认 VAE 解码器，微调后更好地保留几何细节
- **最近邻种子选择**: 多种子生成后根据目标光照码选最优
- **Flow-Based Clean Up**: 用 rectified-flow inversion ($\eta=0.99$) 去除伪影

## 实验关键数据

### 主实验 — MIIW 定量评估

| 方法 | 标签需求 | RMSE↓ | SSIM↑ |
|---|---|---|---|
| SA-AE | Light | 0.232 | 0.559 |
| Latent-Intrinsic | - | 0.222 | 0.571 |
| RGB↔X (same scene) | G-Buffer | 0.340 | 0.350 |
| **LumiNet** | **-** | **0.240** | **0.527** |

LumiNet 作为通用模型（训练在多样数据上），接近专用 MIIW 模型的性能。
  
### 用户研究 — Real-world 评估

| 方法 | 法线一致 AE↓ | 图像质量↓ | 光照质量↓ | Prompt 对齐↓ |
|---|---|---|---|---|
| RGB↔X | 3.14 | 2.21 | 2.88 | 2.70 |
| IC-Light-v2 | 3.42 | 3.06 | 2.57 | 2.74 |
| Latent-Intrinsic | 3.61 | 2.24 | 2.52 | 2.40 |
| **LumiNet** | **2.74** | **1.71** | **1.30** | **1.40** |

31 位参与者的用户研究中全面第一。法线一致性中位角误差 <3°，几何保持最优。

### 消融实验

- **去掉 Variational-StyLitGAN 数据**: 无法学习场景级光照效果（如灯开关）
- **去掉潜在内在条件**: 退化为普通 ControlNet，仅改变平均颜色而非光照
- **去掉适配器 + 交叉注意力微调**: 失去二阶效果（桌面光泽反射等）
- **去掉 Flow Inversion**: 重光照合理但有伪影
- **Var-StyLitGAN 消融**: FID 47.98→37.07→35.81（加变分+CLIP 过滤）

### 关键发现

1. **潜在 >> 像素空间条件控制**: 潜在内在特征比显式 albedo/法线更鲁棒。
2. **虽然仅在同场景对上训练，却能跨场景迁移**: 潜在空间的抽象性使模型学到了场景无关的光照操控能力。
3. **合成数据对场景级效果关键**: 没有 Var-StyLitGAN 数据就无法学习灯开关等光源操控。
4. **二阶光照效果需要交叉注意力**: ControlNet 处理空间条件不够，光照码需通过全局注意力通路影响间接照明。

## 亮点与洞察

1. **首个场景级图像重光照扩散方法**: 超越了人像/物体级别的限制，直接处理完整室内场景。
2. **潜在内在表征的巧妙利用**: 不做显式 intrinsic decomposition（albedo/roughness/normal），而是用预训练模型的隐含分解，避免了误差累积。
3. **两图输入的 ControlNet 改造**: 标准 ControlNet 处理同场景条件，LumiNet 首次处理跨场景的"源几何+目标光照"组合。
4. **从同场景训练到跨场景泛化**: 泛化能力来源于潜在空间的抽象性 — 在预训练的 intrinsic 空间中，光照和内容已被显式解耦。

## 局限与展望

1. 无法识别过小或背对相机的光源（Figure 7 失败案例）。
2. 无法迁移极端色温变化（如 KTV 室的彩色灯光）。
3. 不控制光照强度 — 输出亮度可能与目标不一致。
4. 推理需多种子+最近邻选择+Flow Inversion 后处理，pipeline 较重。
5. 依赖 Latent Intrinsic 预训练模型的质量上限。
6. 未在视频场景中验证时序一致性。

## 相关工作与启发

- **StyLitGAN (Bhattad et al.)**: 在 StyleGAN 潜在空间操控光照 → 仅限合成图像；LumiNet 用其作为数据引擎而非推理工具。
- **Latent Intrinsics (Zhang et al.)**: 证明潜在空间中存在内在/外在分解 → LumiNet 将此分解注入扩散模型实现端到端重光照。
- **IC-Light**: 人像重光照 SOTA → 无法处理全场景的多光源复杂交互。
- **启发**: "潜在内在表征 + 扩散模型"的组合范式可推广到其他场景编辑任务（如材质迁移、风格迁移），关键是找到合适的潜在分解作为条件信号。

## 评分

⭐⭐⭐⭐ — 场景级重光照的新突破，数据策略+架构设计+推理增强形成完整方案，用户研究全面胜出；但推理管线较重、极端场景处理不足。

<!-- RELATED:START -->

## 相关论文

- [Comprehensive Relighting: Generalizable and Consistent Monocular Human Relighting and Harmonization](comprehensive_relighting_generalizable_and_consistent_monocular_human_relighting.md)
- [EchoScene: Indoor Scene Generation via Information Echo over Scene Graph Diffusion](../../ECCV2024/image_generation/echoscene_indoor_scene_generation_via_information_echo_over_scene_graph_diffusio.md)
- [Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes](channel-wise_noise_scheduled_diffusion_for_inverse_rendering_in_indoor_scenes.md)
- [Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)
- [MixerMDM: Learnable Composition of Human Motion Diffusion Models](mixermdm_learnable_composition_of_human_motion_diffusion_models.md)

<!-- RELATED:END -->
