---
title: >-
  [论文解读] Coordinate-based Speed of Sound Recovery for Aberration-Corrected Photoacoustic Computed Tomography
description: >-
  [ICCV 2025][医学图像][光声计算断层成像] 本文提出一种高效的自监督联合重建方法，通过将声速（SOS）参数化为像素网格或神经场，并通过可微成像前向模型反向传播梯度来恢复SOS和高质量光声图像，在精度上超越现有SOTA的同时实现35倍加速（40秒 vs 23分钟）。 光声计算断层成像（PACT）是一种类似超声的非侵…
tags:
  - "ICCV 2025"
  - "医学图像"
  - "光声计算断层成像"
  - "声速恢复"
  - "神经场"
  - "自监督学习"
  - "波前像差校正"
---

# Coordinate-based Speed of Sound Recovery for Aberration-Corrected Photoacoustic Computed Tomography

**会议**: ICCV 2025  
**arXiv**: [2409.10876](https://arxiv.org/abs/2409.10876)  
**代码**: [https://lukeli0425.github.io/Coord-SoS-PACT/](https://lukeli0425.github.io/Coord-SoS-PACT/)  
**领域**: 医学影像 / 计算成像  
**关键词**: 光声计算断层成像, 声速恢复, 神经场, 自监督学习, 波前像差校正

## 一句话总结

本文提出一种高效的自监督联合重建方法，通过将声速（SOS）参数化为像素网格或神经场，并通过可微成像前向模型反向传播梯度来恢复SOS和高质量光声图像，在精度上超越现有SOTA的同时实现35倍加速（40秒 vs 23分钟）。

## 研究背景与动机

光声计算断层成像（PACT）是一种类似超声的非侵入性成像技术，在医学应用中前景广阔。其原理是：生物组织吸收激光能量后发射超声波，外部换能器阵列采集信号后重建初始压力（IP）分布图像。

然而，传统的图像重建算法（如延迟求和DAS、滤波反投影）假设组织内声速（SOS）恒定，忽略了实际组织的声学异质性：

- **波前畸变**：组织内SOS的空间变化导致超声波的波前扭曲，到达换能器的时间偏离假设，产生振铃、拖尾、双影等伪影
- **直接测量SOS代价高**：超声CT（USCT）可以直接测量SOS，但需要额外设备，实施困难且昂贵
- **现有联合重建方法计算昂贵**：APACT（当前SOTA）采用穷举搜索波前系数，需23分钟/样本，且其简化波前模型无法处理图像整体缩放问题
- **有监督学习不可行**：PACT是较新的成像模态，缺乏大规模真实数据集（配对测量+真实像差+IP和SOS真值），域偏移问题严重

核心矛盾：**准确的SOS信息对高质量成像至关重要，但获取SOS既昂贵又困难**。本文的解决思路：利用基于坐标的表示（像素网格/神经场）参数化SOS，通过可微物理模型自监督优化，无需外部训练数据。

## 方法详解

### 整体框架

1. 从光声信号用不同延迟参数 $d$ 重建一组DAS图像栈（类似焦点栈）
2. 将SOS参数化为像素网格（PG）或神经场（NF），通过物理模型计算每个图像patch的点扩散函数（PSF）
3. 用多通道反卷积恢复清晰图像patch
4. 自监督损失：让预测的像差与观测像差一致，通过反向传播更新SOS参数

### 关键设计

1. **可微成像前向模型**:

    - 功能：基于物理模型计算波前误差和PSF，支持端到端反向传播
    - 核心思路：假设直线声学射线模型，从样本点 $\mathbf{r}'$ 到换能器 $\mathbf{r}$ 的飞行时间为路径积分：$t(\mathbf{r}',\mathbf{r},\mathbf{v})=\int_{\mathbf{r}'}^{\mathbf{r}}\frac{1}{v(\mathbf{l})}dl$。波前误差为：$w(\theta;\mathbf{r}',\mathbf{v})=\int_{\mathbf{r}'}^{\mathbf{r}(\theta)}(1-\frac{v_0}{v(\mathbf{l})})dl$。PSF通过傅里叶域传递函数计算：$H_i(\mathbf{k};d,\mathbf{v})=\frac{1}{2}(e^{-j|\mathbf{k}|(d-w(\angle\mathbf{k};\mathbf{r}'_i,\mathbf{v}))}+e^{j|\mathbf{k}|(d-w(\angle\mathbf{k}+\pi;\mathbf{r}'_i,\mathbf{v}))})$
    - 设计动机：全可微的物理前向模型使得SOS可以通过梯度优化端到端学习，而非穷举搜索

2. **多通道反卷积**:

    - 功能：利用多个延迟值提供的冗余信息实现鲁棒的图像恢复
    - 核心思路：对每个图像patch，不同延迟值产生不同的PSF，但对应同一清晰图像。频域表示为 $\mathbf{Y}_i = \mathbf{H}_i(\mathbf{v})X_i$，通过伪逆重建：$\hat{X}_i = \frac{\mathbf{H}_i(\mathbf{v})^\top \mathbf{Y}_i}{\mathbf{H}_i(\mathbf{v})^\top \mathbf{H}_i(\mathbf{v})}$
    - 设计动机：单通道反卷积严重受噪声和PSF形状影响，多通道（默认16）提供超定方程，大幅降低振铃伪影并恢复patch边缘丢失的特征。且额外通道是数字生成的，不引入新的测量误差

3. **坐标基SOS表示（PG与NF）**:

    - 功能：以紧凑的参数形式表示2D声速分布
    - 核心思路：
        - **像素网格（PG）**：与IP图像同分辨率的插值网格，配合TV正则化约束平滑性
        - **神经场（NF）**：单层256特征的全连接网络，SIREN激活函数（正弦激活），将像素坐标映射到SOS值。参数量仅1027，远小于像素数（~200K），自带隐式正则化
    - 设计动机：NF的低自由度（1027参数 vs ~200K像素）减少了逆问题的自由度，作为隐式正则化使问题更良定

### 损失函数 / 训练策略

自监督像差匹配损失：
$$L(\mathbf{v}_\phi) = \sum_{i=1}^{N} |\mathbf{k}| \|\mathbf{Y}_i - \mathbf{H}_i(\mathbf{v}_\phi)\hat{X}_i(\mathbf{Y}_i, \mathbf{H}_i(\mathbf{v}_\phi))\|_2^2 + \lambda\text{TV}(\mathbf{v}_\phi)$$

- NF：$\lambda=0$（依赖网络隐式正则化），10 epoch，学习率 $5\times10^{-3}$
- PG：$\lambda=10^{-4}$（TV正则化），30 epoch，学习率 $10^{-1}$
- 优化器：Adam，硬件：NVIDIA RTX A6000 GPU
- 默认延迟通道数 M=16

## 实验关键数据

### 主实验

**数值模拟（5个数值体模平均）：**

| 方法 | IP PSNR ↑ | IP SSIM ↑ | SOS PSNR ↑ | SOS SSIM ↑ | 时间 |
|------|-----------|-----------|------------|------------|------|
| Conventional DAS | 21.49 | 0.372 | - | - | 0.13s |
| Dual-SOS DAS | 24.42 | 0.446 | - | - | 0.12s |
| APACT (SOTA) | 21.49 | 0.434 | 17.74 | 0.908 | **23 min** |
| Ours (PG) | 25.05 | 0.514 | 21.26 | 0.903 | 113.3s |
| **Ours (NF)** | **25.08** | **0.519** | **22.29** | **0.931** | **40.3s** |
| Ours (SOS oracle) | 25.61 | 0.537 | - | - | 2.6s |

NF方法在所有指标上均优于APACT，且速度快**35倍**（40s vs 23min）。

### 消融实验

**延迟通道数影响（NF方法）：**

| 延迟通道数M | 4 | 8 | 16 | 24 | 32 |
|------------|---|---|----|----|-----|
| 计算时间(s) | ~10 | ~20 | 40 | ~60 | ~80 |
| IP质量 | 较低 | 中等 | 良好 | 略优 | 微优 |

16通道后性能收益递减，选择16作为默认值实现最佳性能-时间平衡。

**NF网络大小消融**：网络过小（隐式正则化太强）或过大（正则化太弱、高频能力过强）均失败，单层256特征为最优配置。

**PG的TV正则化权重**：$\lambda=10^{-4}$ 为最优，更大的 $\lambda$ 鼓励更平滑的SOS但可能过度平滑。

### 关键发现
- **波前误差分析**：APACT仅建模0阶和2阶傅里叶分量，遗漏了1阶余弦分量（导致图像缩放）和高阶细节。本方法可准确建模低阶和高阶分量
- **APACT的图像缩放问题**：由于忽略1阶波前分量，PSF偏心导致图像patch向组织中心偏移，APACT无法解决此问题，而本方法通过完整波前建模自然修复
- **PG的条纹伪影**：PG在SOS重建上存在条纹状伪影，源于波前计算的路径积分特性（直线声学射线模型），但IP图像质量仍然不错
- **真实数据泛化**：方法在叶片体模（3种SOS材料）和体内小鼠肝脏数据上均表现优异，无需任何针对真实数据的修改

## 亮点与洞察
- **自监督 + 物理模型**：完全不需要外部训练数据，通过物理一致性约束实现端到端SOS恢复，对数据稀缺的医学成像领域意义重大
- **信号处理与神经网络的优雅结合**：NF不是作为黑盒网络，而是作为SOS的高效紧凑表示，在物理框架中使用，增强了可解释性和可靠性
- **焦点栈类比精妙**：将不同延迟参数的DAS图像栈类比为焦点栈，每个延迟对应不同的"聚焦深度"，多通道联合处理提供冗余信息
- **35倍加速**：对临床应用意义重大，使得SOS恢复从不实用变为可行

## 局限与展望
- 直线声学射线模型是简化假设，在强声学异质性下实际声波会弯曲
- PG表示的条纹伪影问题需要更好的正则化策略来解决
- 仅验证了2D环形阵列PACT系统，向3D系统和其他阵列配置的扩展尚未探索
- NF的SIREN架构和网络大小需要手动调参，自适应机制可能更好
- 缺乏大规模定量评估（仅5个数值体模），且真实数据缺少SOS真值对比

## 相关工作与启发
- **APACT**：自适应光声CT方法，将图像分patch穷举搜索波前系数，是本文的主要对比方法
- **Dual-SOS DAS**：假设体内均匀SOS进行一阶校正，快速但精度有限
- **SIREN（Neural Fields）**：正弦激活的隐式神经表示，本文用作SOS的紧凑参数化
- **UniDepth / DiffMorpher**：类似的坐标基表示在其他计算成像任务中的应用
- **Coordinate-based Neural Representations for AO**：在自适应光学中使用坐标基表示估计波前

## 评分
- 新颖性: ⭐⭐⭐⭐ 将神经场引入PACT的SOS恢复是新颖的跨领域应用，可微前向模型设计优雅
- 实验充分度: ⭐⭐⭐⭐ 数值模拟+体模+体内实验全覆盖，消融充分，但数值体模数量有限
- 写作质量: ⭐⭐⭐⭐⭐ 物理推导严谨清晰，图示精美直观，与现有方法的对比分析非常深入
- 价值: ⭐⭐⭐⭐ 35倍加速使得SOS恢复在临床上变得可行，自监督框架对数据稀缺领域有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](../../ICLR2026/medical_imaging/dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)
- [\[ICCV 2025\] SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images](seganypet_universal_promptable_segmentation_from_positron_emission_tomography_im.md)
- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-supervised Learning](an_openmind_for_3d_medical_vision_selfsupervised_learning.md)
- [\[CVPR 2025\] Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](../../CVPR2025/medical_imaging/developing_foundation_models_for_universal_segmentation_from_3d_whole-body_posit.md)
- [\[ICCV 2025\] GDKVM: Echocardiography Video Segmentation via Spatiotemporal Key-Value Memory with Gated Delta Rule](gdkvm_echocardiography_video_segmentation_via_spatiotemporal_key-value_memory_wi.md)

</div>

<!-- RELATED:END -->
