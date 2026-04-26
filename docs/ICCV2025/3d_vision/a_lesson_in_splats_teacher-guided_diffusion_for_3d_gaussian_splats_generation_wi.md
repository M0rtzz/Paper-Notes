---
title: >-
  [论文解读] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision
description: >-
  [ICCV 2025][3D视觉][3D高斯喷溅] 本文提出了一种仅使用2D图像监督来训练3D扩散模型的新框架——通过将确定性3D重建模型作为"噪声教师"生成3D噪声样本，并结合多步去噪策略和循环一致性正则化，实现了超越教师模型的3D高斯喷溅生成质量（PSNR提升0.5-0.85）。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D高斯喷溅
  - 扩散模型
  - 2D监督
  - 教师引导
  - 新视角合成
---

# A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision

**会议**: ICCV 2025  
**arXiv**: [2412.00623](https://arxiv.org/abs/2412.00623)  
**代码**: [https://lesson-in-splats.github.io/](https://lesson-in-splats.github.io/)  
**领域**: 3D视觉 / 3D生成  
**关键词**: 3D高斯喷溅, 扩散模型, 2D监督, 教师引导, 新视角合成

## 一句话总结
本文提出了一种仅使用2D图像监督来训练3D扩散模型的新框架——通过将确定性3D重建模型作为"噪声教师"生成3D噪声样本，并结合多步去噪策略和循环一致性正则化，实现了超越教师模型的3D高斯喷溅生成质量（PSNR提升0.5-0.85）。

## 研究背景与动机
- **领域现状**：从2D图像恢复3D结构是一个天然的不适定问题，生成式模型（如扩散模型）可以建模多种可能的3D结构。但现有的3D扩散模型几乎都需要完整的3D ground truth作为监督信号
- **确定性方法的局限**：Splatter Image、Flash3D等前馈式3D重建方法可以仅用稀疏2D视图训练，但作为确定性模型无法捕获多种合理重建的多样性，在不确定区域产生模糊预测
- **核心矛盾**：标准扩散训练要求去噪过程和监督信号处于同一模态——即噪声加在3D上，监督也需要3D ground truth；但大规模3D数据极其稀缺
- **本文解决方案**：解耦去噪模态（3D）和监督模态（2D）。利用确定性重建模型的不完美3D预测作为"噪声教师"生成噪声输入，再通过可微渲染将3D输出渲染为2D图像进行监督
- **核心洞察**：受SDEdit启发，在足够大的噪声级别 $t > t^*$ 下，教师模型的噪声3D样本与真实3D ground truth的噪声分布几乎一致

## 方法详解

### 整体框架
训练分为两个阶段：Stage 1（Bootstrapping）使用教师模型的3D预测作为直接监督，高效初始化扩散模型；Stage 2（Multi-step Denoising Fine-tuning）通过多步去噪并渲染到2D进行图像级监督，使模型超越教师。两个阶段都使用循环一致性正则化。

### 关键设计

1. **噪声教师与模态解耦**:

    - 功能：用预训练的确定性3D重建模型（如Splatter Image或Flash3D）作为"噪声教师" $T_\phi$，其输出 $s_0^{\text{teacher}}$ 虽不完美但可作为3D噪声样本的来源
    - 核心思路：对教师输出加噪 $s_t = \sqrt{\alpha_t}s_0^{\text{teacher}} + \sqrt{1-\alpha_t}\epsilon$，关键是选择临界时间步 $t^*$，使得 $t \geq t^*$ 时教师噪声分布与真实GT噪声分布足够接近
    - 设计动机：传统扩散训练必须在同一模态中完成噪声添加和监督。这种解耦策略打破了这一限制，使得3D扩散模型可以仅用2D图像训练

2. **多步去噪训练策略**:

    - 功能：在训练时不再使用单步去噪（只在 $t > t^*$ 采样），而是从 $t > t^*$ 开始进行多步迭代去噪直到 $t=0$，得到最终3D预测
    - 核心公式：$\hat{s}_0 = D_\theta(\hat{s}_1, 1, x_{\text{src}}) \circ \cdots \circ D_\theta(s_t, t, x_{\text{src}})$
    - 渲染监督损失：$\mathcal{L}_{\text{mlt-stp}} = \mathbb{E}[\lambda_t \|x_{\text{tgt}}^v - \mathcal{R}(\hat{s}_0, v)\|_2^2]$
    - 设计动机：如果只在高噪声级别训练（$t > t^*$），模型无法学习低噪声级别的细节恢复。多步去噪"展开"了整个去噪链，使梯度可以反向传播到所有时间步，让模型在低噪声级别也能产生高质量结果

3. **Bootstrap预热阶段**:

    - 功能：在多步去噪前先用教师的3D输出进行单步去噪训练，快速初始化模型
    - 核心公式：$\mathcal{L}_{\text{bootstrap}} = \mathbb{E}[\ell_{\text{3DGS}} + \ell_{\text{image}}]$，其中 $\ell_{\text{3DGS}} = \|s_0^{\text{teacher}} - D_\theta(s_t,t,x_{\text{src}})\|^2$，$\ell_{\text{image}} = \|x_{\text{tgt}}^v - \mathcal{R}(D_\theta(s_t,t,x_{\text{src}}),v)\|_2^2$
    - 设计动机：直接用多步去噪从头训练代价极高（需维护多步梯度），Bootstrap阶段用单步去噪+教师3D监督高效初始化模型到教师水平

4. **循环一致性正则化**:

    - 功能：将预测的3DGS渲染到目标视角 $\hat{x}_{\text{tgt}}$，再用这张图像驱动第二次3D预测 $\tilde{s}_0$，然后渲染回源视角对比
    - 核心公式：$\mathcal{L}_{\text{cyc}} = \|x_{\text{src}} - \mathcal{R}(\tilde{s}_0, v_{\text{src}})\|_2^2$
    - 设计动机：受CycleGAN启发，循环一致性约束预测不仅要在外观上匹配目标图像，还要足够可靠以驱动反向重建

### 损失函数 / 训练策略
- Stage 1：Bootstrap损失 = 3D L2损失 + 2D渲染损失 + 循环一致性损失，batch size=100/GPU，全时间步采样
- Stage 2：多步去噪渲染损失 + 循环一致性损失，DDIM采样器10步，batch size=10/GPU
- 使用4块NVIDIA A6000 GPU训练

## 实验关键数据

### 主实验

| 数据集 | 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|------|--------|--------|---------|
| ShapeNet Cars (1-view) | Splatter Image (Large) | 24.00 | 0.92 | 0.078 |
| ShapeNet Cars (1-view) | **SplatDiffusion (Medium)** | **24.84** | **0.93** | **0.077** |
| ShapeNet Chairs (1-view) | Splatter Image (Large) | 24.43 | 0.93 | 0.067 |
| ShapeNet Chairs (1-view) | **SplatDiffusion (Medium)** | **25.21** | **0.93** | **0.066** |
| RealEstate10k (5帧) | Flash3D | 28.46 | 0.899 | 0.100 |
| RealEstate10k (5帧) | **SplatDiffusion** | **29.12** | **0.932** | **0.087** |
| RealEstate10k (10帧) | Flash3D | 25.94 | 0.857 | 0.133 |
| RealEstate10k (10帧) | **SplatDiffusion** | **26.54** | **0.887** | **0.122** |

### 消融实验

| 配置 | Novel View PSNR ↑ | Source View PSNR ↑ | 说明 |
|------|-------------------|-------------------|------|
| Splatter Image (Large) 教师 | 24.20 | 31.12 | 教师模型基线 |
| Stage I（仅渲染损失） | 18.82 | 20.98 | 缺少3D监督效果差 |
| Stage I（扩散+渲染损失） | 22.61 | 28.20 | Bootstrap有效 |
| Stage II（仅渲染损失） | 24.49 | 31.98 | 已超越教师 |
| Stage I+II 无循环一致性 | 24.69 | 33.06 | 缺少正则化 |
| **完整模型** | **24.91** | **33.71** | 所有组件协同 |

### 关键发现
- 使用更小的Medium模型（295MB）超越了更大的Large教师模型（646MB），GPU显存也更低（1.15GB vs 1.71GB）
- 多步去噪训练是核心创新——Stage II的渲染损失训练使模型从22.61提升到24.49 PSNR
- 循环一致性在两个阶段都带来稳定提升
- 框架灵活性强，可适配Splatter Image（物体级）和Flash3D（场景级）两种不同教师

## 亮点与洞察
- **范式突破**：首次系统性地解决了"3D空间去噪、仅2D监督"的技术难题，突破了扩散模型训练的同模态限制
- **以小胜大**：Medium模型超越Large教师模型，说明扩散模型的生成能力可以弥补模型容量的不足
- **通用框架**：教师模型可替换，在物体级和场景级数据集上都验证了有效性
- SDEdit中的噪声级别临界点 $t^*$ 的洞察非常巧妙——足够大的噪声可以抹平不完美教师和真实GT之间的分布差异

## 局限与展望
- 计算成本：多步去噪训练仍较昂贵，batch size被迫降至10
- 教师模型质量的下限：如果教师模型太差，噪声分布可能即使在高噪声级别也难以对齐
- 目前仅验证了3DGS表征，是否可推广到NeRF或Mesh等其他3D表征有待探索
- 生成多样性评估不足——虽然框架天然支持多样化生成，但实验主要关注重建质量

## 相关工作与启发
- **vs Holodiffusion**：Holodiffusion也尝试2D监督3D扩散，但用额外去噪pass处理分布差异；本文更优雅地通过噪声级别选择和多步去噪解决
- **vs Score Distillation (SDS)**：SDS从2D扩散模型"提升"到3D，但存在视角一致性问题（Janus问题）；本文直接在3D空间去噪避免了此问题
- **vs ViewsetDiffusion**：ViewsetDiff依赖多视角与3D的双射关系，在视角数量有限时效果受限

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 模态解耦+多步去噪训练的想法极具创新性
- 实验充分度: ⭐⭐⭐⭐ 物体和场景级数据集覆盖，消融完整
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，方法阐述循序渐进
- 价值: ⭐⭐⭐⭐ 为3D生成模型的可扩展训练开辟了新方向

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models](learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)
- [\[ICCV 2025\] Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_a.md)
- [\[ICCV 2025\] CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization](cl-splats_continual_learning_of_gaussian_splatting_with_local_optimization.md)
- [\[ICCV 2025\] MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion](materialmvp_illumination-invariant_material_generation_via_multi-view_pbr_diffus.md)
- [\[ICCV 2025\] Amodal3R: Amodal 3D Reconstruction from Occluded 2D Images](amodal3r_amodal_3d_reconstruction_from_occluded_2d_images.md)

<!-- RELATED:END -->
