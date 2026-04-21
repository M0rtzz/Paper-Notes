---
title: >-
  [论文解读] ScribbleLight: Single Image Indoor Relighting with Scribbles
description: >-
  [CVPR 2025][图像生成][室内重光照] ScribbleLight 提出一个基于涂鸦引导的单张室内图像重光照生成模型，通过 Albedo-conditioned Stable Image Diffusion 保持原图纹理颜色，并设计编码器-解码器 ControlNet 架构实现几何保持的精细光照控制，用户只需简单涂鸦即可实现开关灯、投射阴影等多种光照效果。
tags:
  - CVPR 2025
  - 图像生成
  - 室内重光照
  - 涂鸦控制
  - 扩散模型
  - ControlNet
  - 内在图像分解
---

# ScribbleLight: Single Image Indoor Relighting with Scribbles

**会议**: CVPR 2025  
**arXiv**: [2411.17696](https://arxiv.org/abs/2411.17696)  
**代码**: 无（项目页: https://chedgekorea.github.io/ScribbleLight/ ）  
**领域**: 扩散模型 / 图像生成  
**关键词**: 室内重光照, 涂鸦控制, Stable Diffusion, ControlNet, 内在图像分解

## 一句话总结

ScribbleLight 提出一个基于涂鸦引导的单张室内图像重光照生成模型，通过 Albedo-conditioned Stable Image Diffusion 保持原图纹理颜色，并设计编码器-解码器 ControlNet 架构实现几何保持的精细光照控制，用户只需简单涂鸦即可实现开关灯、投射阴影等多种光照效果。

## 研究背景与动机

**领域现状**：图像重光照在房地产、虚拟布景和室内设计等领域有重要应用。户外重光照因主要光源（太阳）单一且可预测而相对简单。室内场景涉及多个光源（天花板灯、台灯、窗户透射光等），产生复杂的叠加软阴影，是最具挑战性的重光照场景。

**现有痛点**：现有 3D 重光照方法需要密集场景采集；隐式方法（如潜空间编辑）只能实现粗粒度全局光照变化，无法控制局部细节；显式光照表示（如球面高斯、辐照度场）对用户而言控制界面间接且复杂。用户想要的是直接标注"哪里变亮、哪里变暗"，但没有方法支持涂鸦驱动的室内重光照。

**核心矛盾**：涂鸦是极稀疏的控制信号，仅提供高层级引导。如何从如此稀疏的输入生成物理合理的光照效果，同时保持原始图像的颜色和纹理（即 albedo），是核心技术矛盾。

**本文目标**：设计一个生成模型，使用户通过简单二值涂鸦（1=变亮, 0=变暗）就能实现多种室内光照效果，包括开关灯、添加高光和投射阴影等。

**切入角度**：利用大规模预训练扩散模型（Stable Diffusion v2）嵌入的通用图像先验解决涂鸦引导的歧义性，通过 albedo 条件化保留原图内在属性。

**核心 idea**：两阶段训练——先微调 Albedo-conditioned SD 学会在保持 albedo 的条件下生成不同光照图像，再训练 ControlNet 接受涂鸦+法线图来引导光照效果。

## 方法详解

### 整体框架

ScribbleLight 采用两阶段训练。第一阶段：将 albedo 图像编码为潜在表示并与加噪图像拼接作为 U-Net 输入，训练 Albedo-conditioned SD。第二阶段：训练 ScribbleLight ControlNet，包含编码器-解码器结构——编码器将涂鸦图和法线图编码为光照特征图，解码器重建法线和目标 shading 以正则化编码表示，编码器输出注入第一阶段的 SD 模型引导生成。

### 关键设计

1. **Albedo-conditioned Stable Image Diffusion**:

    - 功能：在生成重光照图像时保持原图的颜色和纹理
    - 核心思路：用 VAE 编码器分别编码图像 $I$ 和 albedo $A$ 为潜在向量 $z^I$ 和 $z^A$。对图像潜在向量按时间步 $t$ 加噪，对 albedo 潜在向量加固定量级噪声（T=200）。两者沿特征维拼接送入 SD 的 U-Net（输入通道翻倍，新增权重零初始化）。训练损失为 $\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_{\theta^S}(z_t^I, z_T^A, t, p)\|_2^2]$
    - 设计动机：直接送精确 albedo 会导致模型过度依赖使得光照变化不足，且 albedo 预测器误差直接传播为伪影。加固定噪声引入不确定性，既保留基本颜色结构又迫使模型更多依赖图像先验

2. **ScribbleLight ControlNet 编码器-解码器**:

    - 功能：从涂鸦和法线图提取包含 3D 几何和光照信息的控制特征
    - 核心思路：编码器 $\mathcal{E}^C$ 将涂鸦 M 和法线 N 拼接编码为光照特征图 $f$。解码器 $\mathcal{D}^C$ 从特征图重建法线和单色 shading：$\mathcal{L}_D = \|\mathcal{D}^C(\mathcal{E}^C(M,N)) - (S_{mono}, N)\|_2^2$。ControlNet 以特征图 $f$、加噪潜在向量 $z_t^I$ 和文本提示 $p$ 为输入。ControlNet 初始化为原始 SD v2 权重（非 albedo 条件化版本），联合训练
    - 设计动机：纯编码器缺乏约束可能使潜在特征丢失几何信息。解码器重建法线和 shading 确保特征包含重光照所需的完整信息。去掉法线会导致创建随机物体，去掉解码器则产生幻觉

3. **涂鸦自动生成策略**:

    - 功能：从真实图像数据集自动创建训练涂鸦标注
    - 核心思路：基于 shading 强度分布阈值化——$I(x)>\mu+\sigma$ 标为亮（1），$I(x)<\mu-\sigma$ 标为暗（0），其余标为中性（0.5）。为模拟真实涂鸦的粗糙感，做随机大小（3-19 kernel）的形态学膨胀和腐蚀
    - 设计动机：没有配对涂鸦-重光照数据集。阈值化边缘与图像内容高度对齐不符合真实用户涂鸦，形态学操作打破这种对齐

### 损失函数 / 训练策略

- 第一阶段在 LSUN Bedrooms 100K 子集上训练，albedo 由 IID 方法预测，文本提示由 BLIP-2 生成
- 第二阶段冻结 Albedo SD，单独训练 ControlNet + 编码器-解码器，法线由 DSINE 预测，shading 由 IID 方法提取
- Albedo 噪声固定在 T=200 步，经验发现为最优值

## 实验关键数据

### 主实验

| 方法 | RMSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|--------|---------|
| LightIt* | 0.341(0.302) | 9.61(10.65) | 0.232(0.332) | 0.564(0.518) |
| RGB↔X | 0.269(0.251) | 12.47(12.99) | 0.416(0.437) | 0.439(0.425) |
| **ScribbleLight** | **0.206(0.190)** | **14.29(15.01)** | **0.436(0.504)** | **0.394(0.370)** |

在 BigTime 时间推移数据集 206 对测试图像上评估，报告 mean(best) 值，5 个随机种子。

### 消融实验

| Albedo 条件方式 | 加噪 | RMSE ↓ | PSNR ↑ | LPIPS ↓ |
|----------------|------|--------|--------|---------|
| ControlNet 输入 | - | 0.2305 | 13.19 | 0.4839 |
| SD 条件化 | 否 | 0.2082 | 14.07 | 0.4193 |
| SD 条件化 | 是 | **0.2059** | **14.29** | **0.3942** |

| 法线图 | 解码器 | RMSE ↓ | PSNR ↑ | LPIPS ↓ |
|--------|--------|--------|--------|---------|
| × | ✓ | 0.2224 | 13.61 | 0.4251 |
| ✓ | × | 0.2098 | 14.06 | 0.4093 |
| ✓ | ✓ | **0.2059** | **14.29** | **0.3942** |

### 关键发现

- Albedo 条件化 SD 比将 albedo 注入 ControlNet 大幅好（LPIPS 0.3942 vs 0.4839）
- 对 albedo 潜在空间加噪对鲁棒性和光照多样性都有显著帮助
- 法线图和控制解码器各自独立贡献，缺少任一会导致伪影或几何不一致
- 即使涂鸦物理不一致，模型仍能生成视觉合理结果（会"想象"画外光源）
- 不同随机种子生成的结果都一致遵循涂鸦引导，同时提供多样化光照变体
- 支持渐进式涂鸦（coarse-to-fine），用户可迭代精炼

## 亮点与洞察

1. 涂鸦作为重光照控制信号是非常自然直觉的交互方式，大幅降低用户门槛
2. 对 albedo 条件加固定噪声的做法简单有效——容忍预测误差又促进光照多样性
3. 编码器-解码器正则化确保潜在特征编码了有用几何和光照信息
4. 模型自动生成合理的二次光照效果（如灯周围的柔和光晕），即使涂鸦中未指定

## 局限与展望

- 无法修正强烈物理不一致的涂鸦，可能生成不合理的光照效果
- 不支持彩色光照调整，生成结果偏向常见颜色（黄色、蓝色）
- 训练数据仅 LSUN Bedrooms，其他室内场景泛化需更多数据
- 未来可支持彩色涂鸦控制光源颜色

## 相关工作与启发

- 与 LightIt 相比，albedo 条件化显著改善纹理/颜色保持
- 与 RGB↔X 的 intrinsic 分解-重组相比，涂鸦不需要像素级精确 shading
- 涂鸦控制思路可推广到其他需要用户精细局部控制的图像编辑任务

## 评分

- **新颖性**: 7/10 — 涂鸦+重光照组合新颖，但技术组件（albedo条件SD、ControlNet）有先例
- **实验充分度**: 7/10 — 有定量对比和消融，但测试集有限，无用户研究
- **写作质量**: 8/10 — 结构清晰，图示丰富，问题定义明确
- **价值**: 7/10 — 为室内光照编辑提供了实用直觉的工具

<!-- RELATED:START -->

## 相关论文

- [LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting](luminet_latent_intrinsics_meets_diffusion_models_for_indoor_scene_relighting.md)
- [Comprehensive Relighting: Generalizable and Consistent Monocular Human Relighting and Harmonization](comprehensive_relighting_generalizable_and_consistent_monocular_human_relighting.md)
- [RoomPainter: View-Integrated Diffusion for Consistent Indoor Scene Texturing](roompainter_view-integrated_diffusion_for_consistent_indoor_scene_texturing.md)
- [ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models](ice_intrinsic_concept_extraction_from_a_single_image_via_diffusion_models.md)
- [Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes](channel-wise_noise_scheduled_diffusion_for_inverse_rendering_in_indoor_scenes.md)

<!-- RELATED:END -->
