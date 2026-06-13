---
title: >-
  [论文解读] Matrix3D: Large Photogrammetry Model All-in-One
description: >-
  [CVPR 2025][3D视觉][摄影测量] Matrix3D 提出一个基于多模态扩散 Transformer 的统一摄影测量模型，通过掩码学习策略在单一模型中同时完成位姿估计、深度预测和新视角合成三大任务，在 CO3D 上位姿估计旋转精度达 96.5%，显著超越所有专用方法。 领域现状：传统摄影测量流程由多个独立阶段组成…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "摄影测量"
  - "多模态扩散"
  - "位姿估计"
  - "深度预测"
  - "新视角合成"
---

# Matrix3D: Large Photogrammetry Model All-in-One

**会议**: CVPR 2025  
**arXiv**: [2502.07685](https://arxiv.org/abs/2502.07685)  
**代码**: [https://nju-3dv.github.io/projects/matrix3d](https://nju-3dv.github.io/projects/matrix3d) (项目页)  
**领域**: 3D视觉  
**关键词**: 摄影测量, 多模态扩散, 位姿估计, 深度预测, 新视角合成

## 一句话总结
Matrix3D 提出一个基于多模态扩散 Transformer 的统一摄影测量模型，通过掩码学习策略在单一模型中同时完成位姿估计、深度预测和新视角合成三大任务，在 CO3D 上位姿估计旋转精度达 96.5%，显著超越所有专用方法。

## 研究背景与动机
**领域现状**：传统摄影测量流程由多个独立阶段组成——特征检测、SfM、MVS 等，每个阶段使用不同算法且无法联合优化。同时，稀疏视角输入下重建质量严重下降，通常需要数百张图像才能获得可靠结果。

**现有痛点**：第一，传统流程需要密集图像采集，实际应用中难以满足；第二，多阶段管线中各步骤独立运行、误差逐级累积，导致最终结果次优。已有方法如 PF-LRM 和 DUSt3R 尝试用单一前馈模型完成位姿估计和重建，但仍无法同时支持位姿估计、深度和新视角合成。

**核心矛盾**：多任务统一建模面临的关键挑战是训练数据不完整——现有数据集往往只提供部分模态标注（如只有图像+位姿对或图像+深度对），而非三种模态的完整组合。

**本文目标** (1) 构建一个能同时支持位姿估计、深度预测和新视角合成的统一模型；(2) 解决部分标注数据下的多模态训练问题；(3) 在稀疏视角输入下实现高质量重建。

**切入角度**：作者观察到可以将所有模态统一为 2D 表示——相机用 Plücker 射线图、深度用视差图、图像用 VAE 潜码——从而利用图像生成模型的能力。借鉴 MAE 的掩码策略，通过随机遮掩不同视角/模态的输入进行训练，使得即使只有双模态数据也能训练完整的三模态模型。

**核心 idea**：用多模态扩散 Transformer + 掩码学习将摄影测量的多个子任务统一为"模态转换"问题，并利用不完整数据实现全模态训练。

## 方法详解

### 整体框架
Matrix3D 的输入是一组稀疏视角图像（最多 8 视角），输出可以是目标视角的相机位姿（射线图）、深度图、或新视角 RGB 图像的任意组合。整体架构采用基于 Hunyuan-DiT 预训练权重的编码器-解码器扩散 Transformer：编码器处理已知的条件信息（各视角各模态的已知数据），解码器对带噪声的目标模态进行去噪。最终结合 3DGS 优化生成完整的三维重建。

### 关键设计

1. **多模态统一表示与编码**:

    - 功能：将图像、相机位姿和深度统一为可被扩散模型处理的 2D 表示
    - 核心思路：RGB 图像通过 SDXL 的 VAE 编码到低维潜空间；相机位姿用 Plücker 射线图表示，每个像素编码一条从相机出发的射线，天然形成图像状 2D 数据；深度转为视差图（深度倒数）以压缩数据范围。所有模态都经过固定偏移和缩放使分布接近标准高斯
    - 设计动机：统一的 2D 表示让三种截然不同的模态可以共享同一个生成框架，免去了为每种模态设计专用网络的需求

2. **视角级掩码学习策略**:

    - 功能：在多视角多模态场景中实现灵活的输入/输出配置，并允许使用不完整标注数据训练
    - 核心思路：不同于 MAE 在单张图像内遮掩 patch，Matrix3D 在视角级别和模态级别进行遮掩。训练时按 NVS：位姿估计：深度预测：全随机 = 3:3:3:1 的比例分配任务，被遮掩的视角/模态用噪声填充作为待去噪目标。推理时通过选择不同的"条件-目标"组合执行不同任务。此外以 10% 概率丢弃条件以支持 classifier-free guidance
    - 设计动机：现有数据集往往只有双模态（如图像-位姿或图像-深度），视角级掩码使得模型可以从这些不完整数据中学习三模态生成能力，大幅增加可用训练数据量

3. **多视角编码器-解码器 DiT 架构**:

    - 功能：实现跨视角、跨模态的特征融合与生成
    - 核心思路：编码器由多个自注意力块组成，处理所有条件数据的潜码并投射到共享潜空间；解码器也由自注意力块组成，但额外增加了交叉注意力层以整合编码器的条件特征。所有视角和模态的潜码按序拼接后通过 Transformer 层，捕捉跨视角对应关系。三种位置编码分别编码 patch token 位置（RoPE）、视角 ID 和模态 ID（正弦位置编码，不同基频）
    - 设计动机：纯自注意力机制允许每个 token 关注到所有视角所有模态的信息，天然适合需要多视角一致性的摄影测量任务

### 损失函数 / 训练策略
采用 v-prediction 损失进行训练：$\mathcal{L} = \|D(E(\mathbf{x}_c), \mathbf{x}_{g,t}, t) - \mathbf{v}\|^2$，其中 $\mathbf{v} = \alpha_t \epsilon - \sigma_t \mathbf{x}_0$。训练分三阶段：先在 256 分辨率下训练 4 视角 180K 步，再扩展到 8 视角 20K 步，最后提升到 512 分辨率 20K 步。混合使用 Objaverse、MVImgNet、CO3D-v2、RealEstate10k、Hypersim、ARKitScenes 六个数据集，涵盖物体级和场景级。

## 实验关键数据

### 主实验

| 数据集/任务 | 指标 | Matrix3D | 之前SOTA | 提升 |
|------------|------|----------|----------|------|
| CO3D 位姿(8视角) | RRA@15° | 96.1% | 92.4% (RayDiffusion) | +3.7% |
| CO3D 位姿(8视角) | CCA@0.1 | 87.8% | 81.9% (RayDiffusion) | +5.9% |
| GSO NVS | PSNR | 20.45 | 19.22 (SyncDreamer) | +1.23 |
| DTU 单目深度 | AbsRel | 0.036 | 0.064 (Metric3D v2) | -43.8% |
| DTU 单目深度 | δ₁ | 0.985 | 0.969 (Metric3D v2) | +1.6% |

### 消融实验

| 配置 | RRA@15°(2视角) | CCA@0.1(2视角) | 说明 |
|------|---------------|----------------|------|
| RGB Only | 95.6% | 100% | 仅用RGB条件 |
| RGB + Depth | 95.8% | 100% | 加入深度条件 |

### 关键发现
- 位姿估计在所有视角数（2-8）上均大幅超越 RayDiffusion 和 DUSt3R，尤其在相机中心精度上优势更为明显
- 单目深度预测性能显著优于 Metric3D v2 和 Depth Anything v2，尽管模型设计初衷是多视角方法
- 当额外提供深度条件时，位姿估计和 NVS 质量均有进一步提升，验证了多模态联合推理的有效性
- NVS 支持任意视角合成，不受固定视角配置限制，灵活性远超 SyncDreamer、Wonder3D 等方法

## 亮点与洞察
- **统一多任务建模的优雅方案**：将位姿、深度、NVS 全部转化为"模态转换"问题，通过改变掩码模式实现任务切换，避免了为每个任务训练单独模型的繁琐。巧妙之处在于任务间可以互相增益
- **掩码策略解决数据不完整问题**：借鉴 MAE 思想但推广到视角×模态级别，使得双模态标注的数据也能参与三模态训练。这个思路可以迁移到任何需要多模态数据但标注不完整的场景
- **多轮交互式 3D 创建**：用户可以通过分步提供条件（先估计位姿，再生成新视角，再补充深度）实现精细控制的 3D 内容创建，这在实际应用中非常有价值

## 局限与展望
- 最大支持 8 视角，对于大规模场景（如城市级重建）仍显不足，受限于 Transformer 的平方复杂度
- 依赖预训练 VAE 进行图像编码，VAE 的重建误差会传递到最终 NVS 质量
- 论文未报告运行时间和显存消耗，推理效率可能是实际部署的瓶颈
- 3DGS 优化阶段需要额外处理多视角不一致性问题，说明扩散生成的多视角图像仍存在一定不一致

## 相关工作与启发
- **vs DUSt3R**: DUSt3R 通过预测 3D 点图再用 PnP 恢复位姿，是判别式方法；Matrix3D 是生成式方法，通过扩散直接生成射线图。Matrix3D 精度更高但计算量可能更大
- **vs RayDiffusion**: 二者都用射线表示和扩散模型做位姿估计，但 RayDiffusion 仅支持位姿估计，而 Matrix3D 统一支持三种任务，且通过多模态训练获得更好的特征
- **vs InstantMesh/Wonder3D**: 这些方法只做 NVS 且通常限制在固定视角配置，Matrix3D 支持任意视角且可同时输出深度和位姿

## 评分
- 新颖性: ⭐⭐⭐⭐ 将掩码学习推广到多模态多视角层面，统一三种任务的思路新颖
- 实验充分度: ⭐⭐⭐⭐ 在三个任务上均与多种SOTA对比，但缺少效率分析
- 写作质量: ⭐⭐⭐⭐ 论文逻辑清晰，从问题定义到方法设计层层递进
- 价值: ⭐⭐⭐⭐⭐ All-in-one 摄影测量模型的实用价值非常高，代表了3D视觉向统一模型发展的趋势

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] One Diffusion to Generate Them All](one_diffusion_to_generate_them_all.md)
- [\[ICCV 2025\] RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation](../../ICCV2025/3d_vision/robotron-mani_all-in-one_multimodal_large_model_for_robotic_manipulation.md)
- [\[CVPR 2025\] LIM: Large Interpolator Model for Dynamic Reconstruction](lim_large_interpolator_model_for_dynamic_reconstruction.md)
- [\[CVPR 2025\] Gaussian Splatting for Efficient Satellite Image Photogrammetry (EOGS)](gaussian_splatting_for_efficient_satellite_image_photogrammetry.md)
- [\[CVPR 2025\] PartRM: Modeling Part-Level Dynamics with Large Cross-State Reconstruction Model](partrm_modeling_part-level_dynamics_with_large_cross-state_reconstruction_model.md)

</div>

<!-- RELATED:END -->
