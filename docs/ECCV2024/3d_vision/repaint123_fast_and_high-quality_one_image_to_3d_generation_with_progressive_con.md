---
title: >-
  [论文解读] Repaint123: Fast and High-Quality One Image to 3D Generation with Progressive Controllable Repainting
description: >-
  [ECCV 2024][3D视觉][单图生成3D] Repaint123 提出了一种渐进式可控重绘策略，用 2D 扩散模型生成多视角一致的高质量图像，再通过简单的 MSE 损失快速优化 3D 表征，仅需 2 分钟即可从单张图像生成纹理精细、多视角一致的 3D 内容，大幅超越基于 SDS 的方法。 领域现状：单图到 3D 生成…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "单图生成3D"
  - "渐进式重绘"
  - "多视角一致性"
  - "Score Distillation Sampling"
  - "3D Gaussian Splatting"
---

# Repaint123: Fast and High-Quality One Image to 3D Generation with Progressive Controllable Repainting

**会议**: ECCV 2024  
**arXiv**: [2312.13271](https://arxiv.org/abs/2312.13271)  
**代码**: [https://pku-yuangroup.github.io/repaint123/](https://pku-yuangroup.github.io/repaint123/)  
**领域**: 3D视觉 / 扩散模型  
**关键词**: 单图生成3D, 渐进式重绘, 多视角一致性, Score Distillation Sampling, 3D Gaussian Splatting

## 一句话总结

Repaint123 提出了一种渐进式可控重绘策略，用 2D 扩散模型生成多视角一致的高质量图像，再通过简单的 MSE 损失快速优化 3D 表征，仅需 2 分钟即可从单张图像生成纹理精细、多视角一致的 3D 内容，大幅超越基于 SDS 的方法。

## 研究背景与动机

**领域现状**：单图到 3D 生成是计算机视觉与图形学的重要交叉任务，广泛应用于机器人、VR/AR 等领域。当前主流方法利用 2D 扩散模型的先验知识，通过 Score Distillation Sampling（SDS）将可学习的 3D 表征（如 NeRF）渲染成多视角图像后蒸馏优化。

**现有痛点**：SDS 方法存在三个严重缺陷：(1) 多视角不一致（Multi-face 问题），不同视角生成的纹理可能矛盾；(2) 纹理质量退化，表现为过饱和和过平滑；(3) 生成速度慢，通常需要 30 分钟到数小时。这些问题源于 SDS 损失与 3D 表征优化之间存在的内在冲突。

**核心矛盾**：SDS 在每一步采样时对不同视角独立生成梯度，无法保证相邻视角间的纹理一致性。同时 SDS 本身是一个高方差的优化目标，容易导致纹理的过饱和/过平滑。速度和质量之间存在难以调和的 trade-off。

**本文目标** (1) 如何生成多视角一致的高质量新视角图像；(2) 如何避免 SDS 导致的纹理退化；(3) 如何将生成速度提升到分钟级别。

**切入角度**：作者观察到 2D 扩散模型本身具有强大的图像生成能力，问题在于如何约束不同视角的生成结果保持一致。如果能先生成高质量、多视角一致的 2D 图像，就可以用简单的 MSE 损失快速优化 3D 表征，完全绕过 SDS。

**核心 idea**：用渐进式重绘策略从参考视角出发逐步生成相邻视角的一致纹理，结合深度引导和参考注意力注入保证一致性，再用 MSE 损失直接优化 3D 网格纹理。

## 方法详解

### 整体框架

Repaint123 采用两阶段框架。粗阶段：使用 3D Gaussian Splatting + SDS 在约 1 分钟内获得粗糙的 3D 模型。精细阶段：将粗模型转为 Mesh 表征，从参考视角出发双向旋转相机，每次渐进增加 40°，对每个新视角用 2D 扩散模型重绘不可见区域（遮挡区域），同时保持可见区域（重叠区域）的像素对齐。生成的多视角一致高质量图像用简单 MSE 损失优化网格纹理，整个精细阶段约 1 分钟。

### 关键设计

1. **渐进式重绘策略 (Progressive Repainting)**:

    - 功能：生成多视角一致的新视角图像
    - 核心思路：对粗糙渲染的新视角图像进行 DDIM Inversion 得到确定性的中间噪声潜变量，保留粗糙 3D 模型中一致的颜色信息。在去噪过程中，将重叠区域的潜变量替换为反转后的潜变量以保持像素对齐：$x_{t-1} = x_{t-1}^{inv} \odot (1-M) + x_{t-1}^{rev} \odot M$，其中 $M$ 是遮挡掩码。同时使用 ControlNet 施加深度图约束保证几何一致性。双向旋转策略确保了前后视角的 junction 处也能保持一致
    - 设计动机：与直接用 SDS 独立优化各视角不同，渐进式重绘利用相邻视角的重叠区域作为"锚点"，逐步传播纹理信息，天然保证了短程视角一致性

2. **参考纹理注入 (Mutual Self-Attention)**:

    - 功能：缓解累积的纹理偏差，保证长程视角一致性（尤其是背面视角）
    - 核心思路：在每个去噪步骤中，将新视角的 Key/Value 特征替换为参考视角的注意力特征：$\text{Attention}(Q_t, K_r, V_r) = \text{Softmax}(Q_t K_r^T / \sqrt{d}) V_r$。这使得新视角图像可以直接查询参考图像的高质量纹理细节，避免随着重绘角度增大纹理质量逐渐退化
    - 设计动机：渐进式重绘虽然保证了邻近视角一致性，但随着角度累积，纹理偏差会逐渐放大。通过注入参考视角的注意力特征，提供全局一致性约束

3. **可见性感知自适应重绘强度 (Visibility-aware Adaptive Repainting)**:

    - 功能：自适应调节重叠区域的重绘强度，平衡忠实度与图像质量
    - 核心思路：根据法线图计算可见性图 $V$，反映每个像素在历史视角中的最佳观察角度（$\cos\theta^*$）。基于正交投影定理，片元的投影分辨率正比于 $\cos\theta$，因此将重绘强度设为 $1 - \cos\theta^*$。使用时间步感知二值化将软可见性图转为硬掩码：$M_t^{i,j} = 1$ if $V^{i,j} > 1 - t/T$，else $0$。这使得低可见性区域（之前只在侧面观察过）获得更强的重绘，高可见性区域则保持原有纹理
    - 设计动机：之前方法对所有区域使用固定重绘强度，无法处理因斜视角导致的低分辨率纹理问题。自适应策略实现了忠实度与真实感的最优权衡

### 损失函数 / 训练策略

粗阶段使用 SDS 损失优化 3D Gaussian Splatting；精细阶段使用简单的像素级 MSE 损失 $\mathcal{L}_{MSE} = \|I_{fine} - I\|_2^2$ 直接优化网格纹理。此外使用 IP-Adapter 将参考图像编码为 16 个 image prompt token，用于分类器自由引导（Classifier-Free Guidance），提升生成质量。

## 实验关键数据

### 主实验

| 方法 | 类型 | CLIP↑ | Contextual↓ | PSNR↑ | LPIPS↓ | 时间 |
|------|------|-------|-------------|-------|--------|------|
| RealFusion | NeRF | 0.71 | 2.20 | 19.24 | 0.194 | 20min |
| Make-It-3D | NeRF | 0.81 | 1.82 | 16.56 | 0.177 | 1h |
| Zero123-XL | NeRF | 0.83 | 1.59 | 19.56 | 0.108 | 30min |
| Magic123 | NeRF | 0.82 | 1.64 | 19.68 | 0.107 | 1h(+2h) |
| DreamGaussian | GS | 0.77 | 1.61 | 18.94 | 0.111 | 2min |
| **Repaint123** | **GS** | **0.85** | **1.55** | 19.00 | **0.101** | **2min** |

### 消融实验

| 配置 | CLIP↑ | Contextual↓ | PSNR↑ | LPIPS↓ |
|------|-------|-------------|-------|--------|
| Coarse only | 0.71 | 1.78 | 21.17 | 0.133 |
| + Repaint | 0.71 | 1.62 | 22.41 | 0.049 |
| + Mutual Attention | 0.78 | 1.56 | 22.42 | 0.048 |
| + Image Prompt | 0.84 | 1.52 | 22.40 | 0.048 |
| + Adaptive (Full) | **0.88** | **1.50** | 22.38 | **0.048** |

### 关键发现

- 渐进式重绘本身是贡献最大的组件，Contextual Distance 从 1.78 降到 1.62，LPIPS 从 0.133 大幅降到 0.049
- Mutual Attention 和 Image Prompt 主要提升 CLIP 相似度（多视角语义一致性），分别带来 +0.07 和 +0.06 的提升
- 角度间隔 40° 是最优选择，60° 虽然指标略高但容易出现 multi-face 问题
- NeRF 版 Repaint123 也能显著超越 Magic123，验证了方法的通用性

## 亮点与洞察

- **彻底绕过 SDS 的思路**：本文的核心洞察是将 3D 生成问题转化为"先生成一致的多视角 2D 图像，再用 MSE 重建"的两步走策略。这比直接用 SDS 优化 3D 表征更可控、更稳定、更快速
- **时间步感知二值化的巧妙设计**：将连续的可见性图与去噪时间步关联，在早期去噪（大噪声）时只重绘高需求区域，后期（小噪声）扩大重绘范围，实现渐进式精细化。这个 trick 可以迁移到任何涉及 inpainting + denoising 的场景
- **速度优势**：2 分钟的生成速度比 NeRF 方法快 10-30 倍，这主要得益于 MSE 损失替代 SDS 以及 3DGS 的高效表征

## 局限与展望

- 当前 3D Gaussian Splatting 的技术成熟度有限，提取的 mesh 可能存在空洞等几何伪影
- 参考视角重建的 PSNR 低于 NeRF 方法（19.00 vs 24.69），说明 GS 到 Mesh 的转换损失了精度
- 40° 间隔的重绘策略对于高度非对称物体可能需要更密的采样
- 未探索多参考图像输入的场景，可能限制了背面纹理的真实感
- 可以考虑结合最新的多视角扩散模型（如 Zero123++）生成初始多视角图像，再用重绘策略精细化

## 相关工作与启发

- **vs DreamGaussian**: 同样使用 3DGS + 2 分钟生成，但 DreamGaussian 在 refine 阶段仍用 SDS，导致纹理过平滑。Repaint123 用重绘 + MSE 替代 SDS，CLIP 从 0.77 提升到 0.85
- **vs Magic123**: Magic123 使用 Zero123 的 2D SDS + DMTet，耗时 1h+ 但 PSNR 更高（19.68 vs 19.00），说明 NeRF 级别的几何质量仍有优势
- **vs HiFi-123**: 虽然都用了 inversion + attention injection，但 Repaint123 选用 ControlNet + 深度引导而非 depth-based diffusion model，更灵活通用

## 评分

- 新颖性: ⭐⭐⭐⭐ 渐进式重绘替代SDS的思路清晰有效，可见性感知重绘强度设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 多数据集对比、详细消融、NeRF版本验证、角度分析等较为充分
- 写作质量: ⭐⭐⭐⭐ 图示清晰，方法描述条理分明，pipeline易于理解
- 价值: ⭐⭐⭐⭐ 提供了一种实用的快速高质量单图3D生成方案，对后续工作有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] Compress3D: a Compressed Latent Space for 3D Generation from a Single Image](compress3d_a_compressed_latent_space_for_3d_generation_from_a_single_image.md)
- [\[ECCV 2024\] DreamView: Injecting View-specific Text Guidance into Text-to-3D Generation](dreamview_injecting_view-specific_text_guidance_into_text-to-3d_generation.md)
- [\[ECCV 2024\] Track Everything Everywhere Fast and Robustly](track_everything_everywhere_fast_and_robustly.md)
- [\[ECCV 2024\] CityGaussian: Real-Time High-Quality Large-Scale Scene Rendering with Gaussians](citygaussian_real-time_high-quality_large-scale_scene_rendering_with_gaussians.md)

</div>

<!-- RELATED:END -->
