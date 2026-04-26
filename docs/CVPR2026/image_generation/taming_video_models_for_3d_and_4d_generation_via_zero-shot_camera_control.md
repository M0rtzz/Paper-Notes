---
title: >-
  [论文解读] Taming Video Models for 3D and 4D Generation via Zero-Shot Camera Control
description: >-
  [CVPR 2026][图像生成][视频扩散模型] WorldForge 提出一个完全无训练的推理时引导框架，通过三个协同组件——步内递归精化（IRR）、光流门控潜变量融合（FLF）和双路径自校正引导（DSG）——将预训练视频扩散模型改造为精确相机轨迹可控的 3D/4D 生成工具，在轨迹精度和感知质量上同时超越训练式和推理式基线。
tags:
  - CVPR 2026
  - 图像生成
  - 视频扩散模型
  - 3D生成
  - 4D生成
  - 相机控制
  - 无训练推理
---

# Taming Video Models for 3D and 4D Generation via Zero-Shot Camera Control

**会议**: CVPR 2026  
**arXiv**: [2509.15130](https://arxiv.org/abs/2509.15130)  
**代码**: https://worldforge-agi.github.io (项目页)  
**领域**: 扩散模型 / 3D视觉  
**关键词**: 视频扩散模型, 3D生成, 4D生成, 相机控制, 无训练推理

## 一句话总结
WorldForge 提出一个完全无训练的推理时引导框架，通过三个协同组件——步内递归精化（IRR）、光流门控潜变量融合（FLF）和双路径自校正引导（DSG）——将预训练视频扩散模型改造为精确相机轨迹可控的 3D/4D 生成工具，在轨迹精度和感知质量上同时超越训练式和推理式基线。

## 研究背景与动机

1. **领域现状**：视频扩散模型（VDM）在海量视频数据上训练，编码了丰富的时空先验，能生成逼真的视觉内容。研究者开始利用 VDM 做 3D/4D 任务（新视角合成、场景生成、动态重渲染等）。

2. **现有痛点**：
    - **可控性差**：VDM 难以精确遵循 6-DoF 相机轨迹，导致空间不一致
    - **场景-相机运动耦合**：改变视角时会引起不期望的物体变形和场景不稳定
    - **微调方案代价高**：在运动条件数据上微调（如 LoRA、ControlNet）计算昂贵、泛化差、可能破坏预训练先验
    - **Warp-and-repaint 方案不鲁棒**：将帧按新相机路径投影后用生成模型修补——但扭曲的输入是分布外（OOD）的，预训练模型处理不好，产生 artifact

3. **核心矛盾**：精细的相机可控性与生成质量/泛化性之间存在根本矛盾——注入控制信号会破坏模型先验，而保留先验则无法精确控制。

4. **本文目标** 在推理时注入精确的轨迹控制，同时完整保留 VDM 的世界先验，且不需要任何训练或微调。

5. **切入角度**：采用 warp-and-repaint 流程获取轨迹引导帧，但通过三个精心设计的推理时干预机制解决其固有的 OOD 问题。核心观察：VAE 潜变量的不同通道编码了不同信息（运动 vs 外观），可以选择性注入控制信号。

6. **核心 idea**：通过选择性地在运动相关通道注入轨迹信号、在每个去噪步进行微校正、并利用引导与非引导路径的差异进行自校正，实现无训练的精确相机控制。

## 方法详解

### 整体框架
给定单张图像或视频帧，先用视觉基础模型（深度估计）重建场景点云，沿用户指定的相机轨迹做 warp 渲染得到引导视频。输入图像同时转为文本 prompt 和潜变量表示，送入 Image-to-Video 扩散模型。在去噪过程中，通过 IRR、FLF、DSG 三个模块注入轨迹控制信号。输出为轨迹精确对齐的高质量视频。

### 关键设计

1. **步内递归精化（Intra-Step Recursive Refinement, IRR）**:

    - 功能：在每个去噪步嵌入微级别的"predict-correct"循环，持续将轨迹信息注入生成过程
    - 核心思路：在每个去噪步 $t$，模型先通过标准去噪得到中间估计 $\hat{\mathbf{x}}_0^{(t)}$。然后用融合算子 $\mathbf{F}$ 将轨迹潜变量 $\mathbf{x}_{traj}$（warp 帧编码到潜空间）注入可观测区域：$\mathbf{F}(\hat{\mathbf{x}}_0^{(t)}, \mathbf{x}_{traj}) = \mathbf{M} \cdot \mathbf{x}_{traj} + (1-\mathbf{M}) \cdot \hat{\mathbf{x}}_0^{(t)}$，其中 $\mathbf{M}$ 是由 warp 产生的有效像素掩码。融合后重新加噪送入下一步去噪。关键区别：在干净空间 $\hat{\mathbf{x}}_0^{(t)}$ 上做融合（而非像之前 inpainting 工作在噪声空间 $\mathbf{x}_{t-1}$ 上），为后续 FLF 模块奠定基础
    - 设计动机：逐步校正确保轨迹信号在整个采样过程中持续作用，避免一次性注入导致的信号衰减

2. **光流门控潜变量融合（Flow-Gated Latent Fusion, FLF）**:

    - 功能：识别潜空间中与运动高度相关的通道，仅在这些通道上注入轨迹信息，保护外观通道不受干扰
    - 核心思路：对 $\hat{\mathbf{x}}_0^{(t)}$ 的每个通道 $c$ 计算帧间光流 $\mathcal{F}_{pred}^{(t,c)}$，同样对 $\mathbf{x}_{traj}$ 计算参考光流 $\mathcal{F}_{gt}^{(t,c)}$。对比两者（使用 M-EPE、M-AE、Fl-all 三种光流指标）得到运动相似度分数 $S^{(t,c)}$。用动态阈值 $\delta^{(t)} = \mu_S^{(t)} - \lambda^{(t)}\sigma_S^{(t)}$ 筛选运动相关通道。高分通道注入轨迹信息，低分通道保留原始预测。$\lambda^{(t)}$ 从松到紧调度——早期保留更多通道建立结构，后期逐步收紧保护细节
    - 设计动机：实验发现不同 VAE 通道的功能差异显著——如通道 13 几乎总被过滤（低运动相关），通道 8从不被过滤（高运动相关）。无差别覆写所有通道会严重破坏视觉质量

3. **双路径自校正引导（Dual-Path Self-Corrective Guidance, DSG）**:

    - 功能：抑制 warp 帧带入的深度误差、遮挡和对齐不良导致的 artifact，平衡控制精度与生成质量
    - 核心思路：受 CFG 启发但解决不同问题。每步维护两条路径：(1) 非引导路径 $\mathbf{v}_t^{ori}$（从原始 $\mathbf{x}_t$ 去噪），质量高但无轨迹控制；(2) 引导路径 $\mathbf{v}_t^{traj}$（从校正后 $\mathbf{x}_t'$ 去噪），遵循轨迹但可能有噪声。关键发现：两条路径的余弦相似度仅 0.3-0.6（角度 50°-70°），远大于 CFG 中条件/无条件路径的差异（接近 1）。因此不能直接用 CFG 公式。解决方案：取引导方向在非引导方向上的正交分量做校正 $\mathbf{v}_t^{corr} = \mathbf{v}_t^{traj} + \rho \cdot \beta_t(\mathbf{v}_t^{traj} - \alpha_t \cdot \mathbf{v}_t^{ori})$，其中 $\beta_t = \sqrt{1-\alpha_t^2}$ 自适应缩放——路径分歧大时强力校正，一致时保留自然预测
    - 设计动机：直接用 CFG 公式会因两条路径角度差过大而产生严重 artifact（消融实验证实）。正交分量投影巧妙避免了大角度发散的不利影响

### 推理细节
使用 Wan2.1 I2V-14B 模型，50 步 UniPC 采样，IRR 在前 20 步（约 35-45%）应用。FLF 早期（前 5 步）禁用以保留结构完整性，后期逐步收紧筛选。单 GPU（≥69GB VRAM）运行，可生成最高 1280×720 视频。

## 实验关键数据

### 主实验（3D 静态场景生成）

| 方法 | 训练免 | FID ↓ | CLIP_sim ↑ | ATE ↓ | RPE-T ↓ | RPE-R ↓ |
|------|-------|------|-----------|------|---------|---------|
| ViewCrafter | ✗ | 117.50 | 0.930 | 0.236 | 0.315 | 0.728 |
| TrajectoryCrafter | ✗ | 111.49 | 0.910 | 0.090 | 0.152 | 0.267 |
| NVS-Solver | ✓ | 118.64 | 0.937 | 0.224 | 0.268 | 1.056 |
| See3D | ✗ | 123.26 | 0.941 | 0.091 | 0.089 | 0.250 |
| **WorldForge** | **✓** | **96.08** | **0.948** | **0.077** | **0.086** | **0.221** |

### 4D 动态场景控制

| 方法 | 训练免 | FVD ↓ | CLIP-V_sim ↑ | ATE ↓ | RPE-T ↓ |
|------|-------|------|-------------|------|---------|
| ViewExtrapolator | ✓ | 108.48 | 0.913 | 1.040 | 1.208 |
| TrajectoryCrafter | ✗ | 97.31 | 0.923 | 0.431 | 1.078 |
| **WorldForge** | **✓** | **93.17** | **0.938** | 0.527 | **0.826** |

### 消融实验

| 配置 | FID ↓ (3D) | CLIP_sim ↑ (3D) | FVD ↓ (4D) | CLIP-V_sim ↑ (4D) |
|------|-----------|----------------|-----------|-------------------|
| w/o DSG | 109.43 | 0.943 | 95.69 | 0.937 |
| w/o FLF | 112.69 | 0.945 | 99.79 | 0.932 |
| w/o DSG & FLF | 113.12 | 0.943 | 103.17 | 0.931 |
| DSG (用 CFG 公式) | 120.91 | 0.936 | 109.10 | 0.919 |
| **完整模型** | **96.08** | **0.948** | **93.17** | **0.938** |

### 关键发现
- 作为无训练方法，WorldForge 在生成质量和轨迹精度上同时超越所有训练式基线（FID 96.08 vs 第二名 111.49）
- 三个组件缺一不可——去掉 IRR 完全丧失轨迹控制；去掉 FLF 产生不自然输出（FID 从 96.08 升至 112.69）；去掉 DSG 引入 warp artifact（FID 升至 109.43）
- 直接使用 CFG 公式做 DSG 效果最差（FID 120.91），验证了大角度发散下 CFG 失效的分析
- 方法可跨模型迁移——在 SVD 和 LongCat-Video 上均有效
- 通道角色稳定：通道 13 几乎总是低运动相关被过滤，通道 8 总是高运动相关被保留，4D 场景比 3D 场景表现出更多样化的通道分数

## 亮点与洞察
- **选择性通道注入的精妙设计**：核心洞察——VAE 潜空间的不同通道编码不同信息。FLF 用光流作为运动通道的直接度量（而非 PCA 等统计方法），在不需要梯度优化的情况下实现了运动-外观解耦
- **正交投影解决大角度 CFG 问题**：标准 CFG 假设条件和无条件方向接近（角度~0°），本文场景中两条路径角度 50-70°，直接相减会灾难性失败。取正交分量的思路借鉴自 APG 但在完全不同的任务中验证了有效性
- **无训练即插即用的通用性**：一个框架支持 3D 场景生成、4D 重渲染、视频编辑、视频稳定、虚拟试穿等 12+ 种应用，model-agnostic（Wan2.1、SVD、LongCat-Video 均适用）

## 局限与展望
- 迭代引导过程导致无法实时运行
- 依赖深度估计质量——虽然 VDM 先验能缓解部分深度误差，但极端深度错误仍会影响结果
- 单通道扫描分析 VDM 上下文较长时可能增加计算开销
- 未来方向：蒸馏引导过程到更少步数、结合更强生成模型、提升分辨率

## 相关工作与启发
- **vs ViewCrafter/See3D（训练式）**: 在 warp+微调数据上训练，计算成本高且可能破坏 VDM 先验；WorldForge 完全保留先验且泛化更好
- **vs NVS-Solver（无训练式）**: 同为推理时引导但缺少运动-外观解耦和自校正，轨迹精度差（ATE 0.224 vs 0.077）
- **vs ReCamMaster（训练式 4D）**: 使用 T2V 模型+专门训练的相机控制模块，但无法接受相同的 warp 输入，可控性和泛化性受限

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 三个推理时引导组件的设计原创性强，FLF 和 DSG 的设计理由充分且新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 70+ 场景、50+ 视频、3D/4D 双任务、跨模型验证、完整消融
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，但三个组件的符号较多，阅读有一定门槛
- 价值: ⭐⭐⭐⭐⭐ 无训练即插即用的通用性使其有极高的实用价值，12+ 种应用展示令人印象深刻

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] DiP: Taming Diffusion Models in Pixel Space](dip_taming_diffusion_models_in_pixel_space.md)
- [\[ICCV 2025\] AnyPortal: Zero-Shot Consistent Video Background Replacement](../../ICCV2025/image_generation/anyportal_zero-shot_consistent_video_background_replacement.md)
- [\[CVPR 2026\] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [\[CVPR 2026\] Taming Sampling Perturbations with Variance Expansion Loss for Latent Diffusion Models](taming_sampling_perturbations_with_variance_expansion_loss_for_latent_diffusion_.md)
- [\[CVPR 2026\] Exploring Conditions for Diffusion Models in Robotic Control](exploring_conditions_for_diffusion_models_in_robotic_control.md)

<!-- RELATED:END -->
