---
title: >-
  [论文解读] SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates
description: >-
  [3D视觉] 提出SuperMat，一个单步推理的PBR材质分解框架，通过结构化专家分支和调度器修正实现端到端训练，引入re-render loss确保物理一致性，将推理速度从秒级提升至毫秒级。 从图像中分解PBR材质（albedo、metallic、roughness）是3D资产创建的核心挑战，现有方法存在三大瓶颈： 模型…
tags:
  - "3D视觉"
---

# SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2411.17515](https://arxiv.org/abs/2411.17515)
- **代码**: [项目页面](https://hyj542682306.github.io/SuperMat/)
- **领域**: 3D视觉
- **关键词**: PBR材质分解, 单步推理, 端到端训练, Re-render Loss, 3D材质估计

## 一句话总结

提出SuperMat，一个单步推理的PBR材质分解框架，通过结构化专家分支和调度器修正实现端到端训练，引入re-render loss确保物理一致性，将推理速度从秒级提升至毫秒级。

## 研究背景与动机

从图像中分解PBR材质（albedo、metallic、roughness）是3D资产创建的核心挑战，现有方法存在三大瓶颈：

**模型冗余**：每种材质属性需要独立的扩散模型，训练和推理开销翻倍

**推理缓慢**：DDIM需要30-50步去噪，无法满足交互式应用需求

**分解效果不足**：基于噪声预测的训练策略无法直接监督最终材质输出，阻碍了perceptual loss和re-render loss等高级技术的应用

## 方法详解

### 整体框架

SuperMat基于Stable Diffusion微调，包含三个核心设计：

### 1. 结构化专家分支

在UNet的最后一个UpBlock处复制为两个专家分支：
- **Albedo分支**：专门预测漫反射贴图
- **RM分支**：专门预测roughness-metallic联合贴图

共享模块提取通用特征，专家分支捕获材质特定特征。新增参数仅占UNet总参数的2.23%（19.3M），实现了单模型多材质输出。

### 2. 单步推理与端到端训练

**调度器修正**：发现DDIM默认的leading timestep设置存在缺陷——单步预测时，模型接收到的timestep（$t=1$，暗示输入几乎无噪声）与实际输入（纯噪声）不匹配。修正为trailing设置（$t=T$），实现真正的单步推理。

**端到端训练**：单步推理使得反向传播可行，可以直接在最终预测材质上计算损失：

$$\mathcal{L}_m = \mathcal{L}_p(\hat{k}_d, k_d) + \mathcal{L}_p(\hat{k}_{rm}, k_{rm})$$

其中 $\mathcal{L}_p$ 是基于VGG-16的感知损失。

### 3. Re-render Loss

利用预测材质在新光照条件下渲染，与GT渲染结果对比：

$$\mathcal{L}_{re} = \mathcal{L}_p(\mathcal{R}_{\hat{k}_n, \hat{k}_p, \hat{c}}(\hat{k}_d, \hat{k}_m, \hat{k}_r), \mathcal{R}_{\hat{k}_n, \hat{k}_p, \hat{c}}(k_d, k_m, k_r))$$

这确保了不同材质属性之间的物理一致性——即使单独看每个材质贴图接近GT，渲染结果也可能不正确。

### 多视角扩展（SuperMatMV）

基于MVDream架构，加入3D self-attention和相机外参条件，实现6视角同时分解，保证跨视角一致性。

### UV精修网络（3D扩展）

将SuperMatMV的多视角分解结果反投影到UV空间，通过UV精修网络补全未覆盖区域、提升质量。整个3D流程仅需约3秒。

## 实验

### 主实验：图像空间材质分解

| 方法 | Albedo PSNR↑ | Metallic PSNR↑ | Roughness PSNR↑ | Relighting PSNR↑ | 时间(s)↓ |
|------|------|------|------|------|------|
| IIR | 21.94 | 17.95 | 19.73 | 20.98 | 0.04 |
| RGB→X | 22.30 | 15.36 | 20.40 | 21.51 | 3.32 |
| StableMaterial | 23.44 | 20.29 | 21.01 | 22.56 | 0.53 |
| SuperMat w/o e2e | 24.26 | 20.79 | 20.81 | 23.90 | 3.09 |
| SuperMat w/o re-render | 26.70 | 24.54 | 23.52 | 26.41 | 0.07 |
| **SuperMat** | **27.68** | **25.48** | **24.25** | **27.66** | **0.07** |
| SuperMatMV | 27.56 | 26.11 | 24.84 | 27.64 | 0.09 |

### 消融实验

| 配置 | Albedo PSNR | Relighting PSNR | 时间 |
|------|------|------|------|
| 无调度器修正（多步推理） | 24.26 | 23.90 | 3.09s |
| 有调度器修正，无re-render | 26.70 | 26.41 | 0.07s |
| 完整SuperMat | **27.68** | **27.66** | 0.07s |

### 关键发现
1. **调度器修正是核心贡献**：使推理速度提升约40倍（3.09s → 0.07s），同时PSNR提升2-4dB
2. **Re-render loss贡献显著**：Relighting PSNR提升1.25dB，验证了跨材质属性交互监督的重要性
3. **单模型多材质 vs 多模型**：结构化专家分支在仅增加2.23%参数的情况下实现了双模型等效功能

## 亮点与洞察

1. **发现并修正DDIM调度器缺陷**：这个简单修正释放了单步扩散模型的巨大潜力，具有广泛影响
2. **Re-render loss实现跨属性物理约束**：首次在扩散模型材质分解中引入渲染一致性约束
3. **毫秒级推理**：将材质分解从学术研究推向实际应用

## 局限性

- 依赖训练数据的光照多样性和材质覆盖度
- UV精修网络对未见过的复杂几何形状的泛化性有待验证
- 单步推理可能在极端光照条件下损失细节

## 相关工作

- **扩散材质分解**: RGB→X, IntrinsicAnything, StableMaterial
- **传统方法**: Derender3D, IIR
- **单步扩散**: DMD, InstaFlow

## 评分

- 新颖性: ⭐⭐⭐⭐ (调度器修正+re-render loss组合创新)
- 技术深度: ⭐⭐⭐⭐ (系统性解决三大瓶颈)
- 实验质量: ⭐⭐⭐⭐⭐ (全面消融+SOTA大幅提升)
- 实用价值: ⭐⭐⭐⭐⭐ (毫秒级推理，极高实用性)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)
- [\[ICCV 2025\] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[ICCV 2025\] One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images](one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)
- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](../../CVPR2025/3d_vision/multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->
