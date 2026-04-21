---
title: >-
  [论文解读] BokehDiff: Neural Lens Blur with One-Step Diffusion
description: >-
  [ICCV 2025][3D视觉][散景渲染] BokehDiff提出基于预训练扩散模型的单步推理散景渲染方法，通过物理启发的自注意力模块（PISA）融入能量守恒、弥散圆约束和自遮挡效果，配合合成前景数据训练，在深度不连续区域显著优于传统方法。
tags:
  - ICCV 2025
  - 3D视觉
  - 散景渲染
  - 扩散模型
  - 单步推理
  - 物理启发注意力
  - 数据合成
---

# BokehDiff: Neural Lens Blur with One-Step Diffusion

**会议**: ICCV 2025  
**arXiv**: [2507.18060](https://arxiv.org/abs/2507.18060)  
**代码**: https://github.com/FreeButUselessSoul/bokehdiff  
**领域**: 3d_vision  
**关键词**: 散景渲染, 扩散模型, 单步推理, 物理启发注意力, 数据合成

## 一句话总结
BokehDiff提出基于预训练扩散模型的单步推理散景渲染方法，通过物理启发的自注意力模块（PISA）融入能量守恒、弥散圆约束和自遮挡效果，配合合成前景数据训练，在深度不连续区域显著优于传统方法。

## 研究背景与动机
散景效果（Bokeh）是大光圈镜头产生的焦外模糊，广泛用于人像摄影突出主体。由于大光圈镜头成本高昂，计算散景渲染成为热门研究方向。

**现有痛点**：
1. 传统神经渲染方法（DeepLens、BokehMe等）高度依赖深度估计精度，在深度不连续区域（如头发、毛发边缘）产生明显伪影
2. 多层方法（MPIB、Dr.Bokeh）需要将场景分层并逐层inpaint，复杂场景易失败
3. 扩散模型虽具有强大的生成先验，但迭代去噪过程会改变图像内容且速度慢；其自注意力机制忽略了散景渲染的3D物理特性

**切入角度**：将全焦图像视为散景图像加"需去除的噪声"的组合，不添加任何噪声，仅一步前向传播实现变换，同时设计物理约束的注意力模块。

## 方法详解

### 整体框架
基于预训练SDXL模型，将全焦图像直接作为输入（不添加噪声），固定timestep $T=499$，微调U-Net的LoRA和编码器 $\mathcal{E}$，解码器 $\mathcal{D}$ 冻结。在下采样层引入PISA模块施加物理约束。最终只需一步前向传播生成散景图像。

### 关键设计

1. **单步扩散推理方案**:

    - 功能：将散景渲染建模为从全焦图像到散景图像的一步变换
    - 核心思路：利用扩散模型的降噪公式 $\hat{z}_0 = \frac{z_t - \beta_t \cdot \epsilon_\theta(z_t; c_{\text{txt}})}{\alpha_t}$，将全焦图像的latent当作 $z_t$，学习转换为散景图像的latent。固定 $T=499$ 作为平衡点，仅微调LoRA（rank=8）
    - 设计动机：扩散模型在特定timestep上表现最佳，one-step方案避免了迭代去噪引入的累积误差和内容改变。不添加噪声确保了原始结构的保留

2. **PISA（Physics-Inspired Self-Attention）模块**:

    - 功能：替换标准自注意力，使其遵循散景渲染的物理规律
    - 核心思路包含三个物理约束：
        - **能量守恒归一化**：将softmax从key维度改为query维度归一化 $A_{qk}^{(Q)} = \frac{\exp(A_{qk})}{\sum_s \exp(A_{sk})}$，保证每个光源的总贡献为1
        - **弥散圆空间约束**：根据焦平面视差差计算弥散圆半径 $r_c(k) = |d_f - d_k| \cdot A$，限制像素影响范围，通过可微soft边缘函数应用mask $C_{qk}$
        - **自遮挡掩码**：通过共线关系计算采样点位置 $\tilde{P}$，判断光源到目标像素路径是否被遮挡，生成可见性掩码 $M_{\text{vis}}$
    - 设计动机：标准自注意力的全局感受野和"忽略不重要像素"的特性与散景渲染的物理模型矛盾——散景需要对邻域像素的贡献进行能量守恒的加权聚合

3. **数据合成流水线**:

    - 功能：合成高质量配对训练数据
    - 核心思路：使用预训练T2I模型生成带透明度的逼真前景（而非从照片分割），配合小光圈拍摄的真实背景，随机放置不同深度和角度的图层，通过光线追踪渲染散景效果
    - 设计动机：解决真实配对数据的对齐问题和传统合成数据（分割前景/3D引擎）的质量瓶颈，兼顾数据规模和质量

### 损失函数 / 训练策略
在像素空间计算损失（而非latent空间），包含四项：
- MSE损失 $\mathcal{L}_{\text{MSE}}$：基础重建
- 感知损失 $\mathcal{L}_{\text{VGG}}$：LPIPS距离
- 多尺度边缘损失 $\mathcal{L}_{\text{edge}}$：使用扩展Sobel算子关注焦内/焦外的边缘变化
- 对抗损失 $\mathcal{L}_{\text{adv}}$：以预训练ConvNext为backbone的判别器

总损失：$\mathcal{L} = \lambda_{\text{MSE}} \mathcal{L}_{\text{MSE}} + \lambda_{\text{VGG}} \mathcal{L}_{\text{VGG}} + \lambda_{\text{edge}} \mathcal{L}_{\text{edge}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}$，其中 $\lambda_{\text{MSE}}=1, \lambda_{\text{VGG}}=5, \lambda_{\text{adv}}=0.5, \lambda_{\text{edge}}=1$。

训练仅需单张NVIDIA L40s GPU约12小时。

## 实验关键数据

### 主实验

| 数据集 | 指标 | BokehDiff | BokehMe | MPIB | Dr.Bokeh | DeepLens |
|--------|------|-----------|---------|------|----------|----------|
| EBB Val294 (真实) | PSNR↑ | **24.652** | 24.014 | 23.334 | 23.479 | 22.703 |
| EBB Val294 | SSIM↑ | **0.8357** | 0.8134 | 0.7920 | 0.8221 | 0.7623 |
| EBB Val294 | DISTS↓ | **0.1155** | 0.1460 | 0.1581 | 0.1225 | 0.1483 |
| EBB Val294 | LPIPS↓ | **0.3737** | 0.3921 | 0.4031 | 0.3771 | 0.4191 |
| BLB Level 5 (合成) | LPIPS↓ | **0.0888** | 0.1404 | 0.2561 | 0.4539 | 0.2976 |
| 用户研究 | 准确性↑ | **4.42** | 3.81 | 1.83 | 3.41 | 1.55 |
| 用户研究 | 真实性↑ | **4.37** | 3.93 | 1.89 | 3.38 | 1.68 |
| 用户研究 | 偏好↑ | **4.56** | 4.03 | 2.04 | 3.64 | 1.96 |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|------|-------|-------|--------|------|
| w/o $\mathcal{L}_{\text{adv}}$ | 24.623 | 0.8322 | 0.3768 | 对抗损失有小幅贡献 |
| w/o $\mathcal{L}_{\text{VGG}}$ | 24.285 | 0.8196 | 0.4218 | 感知损失非常重要 |
| w/o $\mathcal{L}_{\text{edge}}$ | 24.628 | 0.8346 | 0.3785 | 边缘损失有贡献 |
| w/o CoC | 22.217 | 0.6881 | 0.4280 | 弥散圆约束极为关键 |
| w/o SoftmaxQ | 24.468 | 0.8325 | 0.3800 | 能量守恒归一化有效 |
| w/o occlusion | 24.399 | 0.8291 | 0.3808 | 自遮挡有贡献 |
| $T=249$ | 24.646 | 0.8335 | 0.3781 | timestep选择稳健 |
| $T=749$ | 24.481 | 0.8319 | 0.3838 | 极端timestep性能下降 |
| **完整模型** | **24.652** | **0.8357** | **0.3737** | 全部组件协同最优 |

### 关键发现
- 弥散圆约束（CoC）是PISA模块中最关键的组件，去除后PSNR下降2.4 dB，SSIM下降0.15
- 50名摄影爱好者的用户研究中，BokehDiff在准确性、真实性和偏好三个维度均大幅领先
- BokehDiff对深度估计误差具有鲁棒性：在视差图侵蚀/膨胀实验中性能下降最小
- 可支持任意焦面调节，生成完整的焦距堆栈（focal stack）

## 亮点与洞察
- 单步扩散推理的创新思路：不添加噪声、不迭代去噪，将图像质量差异重新定义为"噪声"
- PISA模块优雅地将光学成像物理（能量守恒、弥散圆、遮挡）融入自注意力机制
- 数据合成方案利用T2I模型生成带透明度的前景，解决了长期困扰的数据质量与规模矛盾
- 一步推理实现了实时渲染的可能性

## 局限与展望
- VAE解码器仍会对不显著的结构引入不可避免的变化
- 目前基于SDXL，更换信息压缩更少的backbone（如更好的VAE）可能进一步提升
- 深度估计本身的误差仍是上游瓶颈，但BokehDiff对此展现了较好的鲁棒性
- 未讨论对非圆形光圈（如星芒效果）的支持

## 相关工作与启发
- 继承了one-step diffusion（SinSR等）和"无噪声去噪"的思想脉络
- 物理约束注意力机制可推广到其他需要物理先验的图像处理任务
- 启发：预训练扩散模型的先验可通过轻量级微调适配到特定物理变换任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 将散景渲染重构为one-step diffusion问题，PISA模块物理约束设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 真实+合成数据集评估全面，用户研究规模较大
- 写作质量: ⭐⭐⭐⭐ 物理动机清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ 为计算摄影领域提供了扩散模型的新范式，实用性强

<!-- RELATED:START -->

## 相关论文

- [Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models](../../CVPR2025/3d_vision/difix3d_improving_3d_reconstructions_with_single-step_diffusion_models.md)
- [Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation](../../CVPR2026/3d_vision/ada3drift_adaptive_training-time_drifting_for_one-step_3d_visuomotor_robotic_man.md)
- [From One to More: Contextual Part Latents for 3D Generation](from_one_to_more_contextual_part_latents_for_3d_generation.md)
- [One Diffusion to Generate Them All](../../CVPR2025/3d_vision/one_diffusion_to_generate_them_all.md)
- [PolarAnything: Diffusion-based Polarimetric Image Synthesis](polaranything_diffusion-based_polarimetric_image_synthesis.md)

<!-- RELATED:END -->
