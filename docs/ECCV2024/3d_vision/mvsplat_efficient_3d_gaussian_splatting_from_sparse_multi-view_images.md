---
title: >-
  [论文解读] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images
description: >-
  [ECCV 2024][3D视觉][3D Gaussian Splatting] 提出MVSplat，通过plane-sweep构建代价体（cost volume）来精确定位Gaussian中心，以极少参数量（pixelSplat的1/10）和最快推理速度（22fps）实现了稀疏视角前馈式3D Gaussian预测的SOTA。
tags:
  - ECCV 2024
  - 3D视觉
  - 3D Gaussian Splatting
  - 前馈式重建
  - 代价体
  - 稀疏视角
  - 新视角合成
---

# MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images

**会议**: ECCV 2024  
**arXiv**: [2403.14627](https://arxiv.org/abs/2403.14627)  
**代码**: [GitHub](https://github.com/donydchen/mvsplat)  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, 前馈式重建, 代价体, 稀疏视角, 新视角合成

## 一句话总结

提出MVSplat，通过plane-sweep构建代价体（cost volume）来精确定位Gaussian中心，以极少参数量（pixelSplat的1/10）和最快推理速度（22fps）实现了稀疏视角前馈式3D Gaussian预测的SOTA。

## 研究背景与动机

1. **领域现状**: 从稀疏视角（低至2张图）前馈式重建3D场景是近期热门方向。NeRF-based方法（pixelNeRF、MuRF）需要昂贵的体渲染采样，速度慢。3DGS凭借光栅化渲染天然避免了体采样开销，近期pixelSplat等开始探索前馈式3DGS。

2. **现有痛点**: pixelSplat虽然引入了epipolar Transformer学习跨视角特征，但仍然是从图像特征直接回归概率深度分布——这种从特征到深度的映射本质上是模糊且不可靠的，导致几何质量差、存在大量浮动Gaussians。为了获得合理几何需要额外50K步深度正则化微调。此外pixelSplat有125M参数，较重。

3. **核心矛盾**: 精确定位3D Gaussian中心是高质量渲染的关键，但from-feature-to-depth的数据驱动回归方式难以提供可靠的几何感知。需要更具几何感知的深度估计方式。

4. **本文要解决什么**: 设计一个轻量高效的前馈模型，从稀疏多视角图像直接预测高质量3D Gaussians。

5. **切入角度**: 引入经典多视角立体视觉（MVS）中的**代价体**（cost volume），通过特征匹配而非特征回归来估计深度，将问题从"从特征猜深度"变为"从匹配找深度"。

6. **核心idea一句话**: 用plane-sweep代价体编码的跨视角特征匹配信息来定位Gaussian中心，比直接回归概率深度更可靠、更轻量。

## 方法详解

### 整体框架

MVSplat的管线：(1) 用CNN+Transformer提取多视角交互感知特征$\{\boldsymbol{F}^i\}$；(2) 通过plane-sweep在逆深度空间均匀采样$D$个深度候选，构建每个视角的代价体$\boldsymbol{C}^i \in \mathbb{R}^{\frac{H}{4} \times \frac{W}{4} \times D}$；(3) U-Net细化代价体并预测深度图；(4) 深度图反投影得到Gaussian中心，同时并行预测opacity、covariance和球谐颜色；(5) 用3DGS光栅化渲染新视角，仅用RGB photometric loss端到端训练。

### 关键设计

1. **代价体构建（Cost Volume Construction）**: 对于视角$i$，将其他视角$j$的特征按深度候选$d_m$进行单应性warp：

$$\boldsymbol{F}_{d_m}^{j \to i} = \mathcal{W}(\boldsymbol{F}^j, \boldsymbol{P}^i, \boldsymbol{P}^j, d_m)$$

然后计算点积相似度得到correlation：

$$\boldsymbol{C}_{d_m}^i = \frac{\boldsymbol{F}^i \cdot \boldsymbol{F}_{d_m}^{j \to i}}{\sqrt{C}}$$

堆叠$D$个correlation得到代价体$\boldsymbol{C}^i = [\boldsymbol{C}_{d_1}^i, \ldots, \boldsymbol{C}_{d_D}^i]$。多于两个视角时对correlation做像素级平均，使模型可接受任意数量输入。设计动机：代价体捕捉的是特征间**相对相似度**，不依赖特征绝对尺度，天然具有跨数据集泛化性。

2. **代价体细化（Cost Volume Refinement）**: 一个轻量2D U-Net以Transformer特征和代价体的拼接为输入，输出残差$\Delta\boldsymbol{C}^i$：

$$\tilde{\boldsymbol{C}}^i = \boldsymbol{C}^i + \Delta\boldsymbol{C}^i$$

在U-Net最低分辨率层注入3层跨视角注意力来交换不同视角代价体的信息，跨视角注意力不依赖视角数量。最终通过CNN upsampler将代价体上采样到全分辨率$\hat{\boldsymbol{C}}^i \in \mathbb{R}^{H \times W \times D}$。

3. **深度估计**: 对细化后的代价体沿深度维softmax归一化，然后对所有深度候选加权平均：

$$\boldsymbol{V}^i = \text{softmax}(\hat{\boldsymbol{C}}^i) \boldsymbol{G}$$

其中$\boldsymbol{G} = [d_1, \ldots, d_D]$是深度候选值。另有轻量U-Net做深度残差细化。

4. **Gaussian参数预测**: (a) **中心$\mu$**: 直接将深度图反投影到3D世界坐标，多视角点云简单union；(b) **opacity $\alpha$**: softmax匹配分布的最大值表示匹配置信度，通过两层卷积映射为opacity；(c) **covariance和color**: 从拼接的图像特征+代价体+原始图像用两层卷积预测。每个像素预测1个Gaussian（pixelSplat为3个），总Gaussian数为$H \times W \times K$。

### 损失函数 / 训练策略

$\ell_2$ + 0.05 × LPIPS 的线性组合。不需要任何深度GT监督。在单个A100上训练300K iterations。代价体采样128个深度候选。使用Swin Transformer的local window attention提高效率。

## 实验关键数据

### 主实验

**RealEstate10K + ACID 新视角合成**:

| 方法 | 参数(M) | 时间(s) | RE10K PSNR↑ | RE10K LPIPS↓ | ACID PSNR↑ | ACID LPIPS↓ |
|------|---------|---------|-------------|-------------|------------|-------------|
| pixelNeRF | 28.2 | 5.299 | 20.43 | 0.550 | 20.97 | 0.533 |
| MuRF | 5.3 | 0.186 | 26.10 | 0.143 | 28.09 | 0.155 |
| pixelSplat | 125.4 | 0.104 | 25.89 | 0.142 | 28.14 | 0.150 |
| **MVSplat** | **12.0** | **0.044** | **26.39** | **0.128** | **28.25** | **0.144** |

MVSplat用**1/10参数量**、**2倍以上速度**超越pixelSplat。

### 消融实验

**跨数据集泛化（RE10K训练→ACID/DTU测试）**:

| 方法 | ACID PSNR↑ | ACID LPIPS↓ | DTU PSNR↑ | DTU LPIPS↓ |
|------|------------|-------------|-----------|-----------|
| pixelSplat | 27.64 | 0.160 | 12.89 | 0.560 |
| **MVSplat** | **28.15** | **0.147** | **13.94** | **0.385** |

在源域到DTU的大域差场景下，MVSplat的LPIPS改善31%，证明代价体的特征不变性带来的泛化优势。

### 关键发现

- pixelSplat的底层3D结构包含大量浮动Gaussians，尽管其2D渲染看起来合理；MVSplat的Guarantee中心质量远高于pixelSplat
- 使用代价体后不需要额外的深度正则化微调即可获得高质量几何
- 代价体捕捉的是相对相关性→特征分布改变时（跨数据集）仍然有效
- 每像素仅1个Gaussian（vs pixelSplat的3个），渲染也更快

## 亮点与洞察

- **从回归到匹配的范式转变**: 将深度估计从"数据驱动回归"转为"基于匹配的推断"，本质上降低了学习难度
- **经典MVS智慧的回归**: 在神经网络时代重新证明了代价体这一经典立体视觉工具的价值，且无需GT深度监督
- **极致高效**: 12M参数实现SOTA，22fps推理，实用性极强
- **设计的一致性**: opacity从匹配置信度推导，Gaussian中心从匹配深度推导，所有关键量都来自同一个代价体

## 局限性 / 可改进方向

- 代价体要求已知相机内外参，无法处理unknown camera setting
- $256 \times 256$分辨率限制，高分辨率时代价体内存占用增大
- 仅验证2-3个输入视角，更多视角时代价体计算的扩展性需评估
- 无纹理区域的代价体仍有歧义，虽然U-Net可部分修正但存在上限
- 未探索时序一致性，难以直接应用于视频场景

## 相关工作与启发

- **pixelSplat**: 最直接的对比方法，证明了数据驱动深度回归的局限性
- **MVSNet系列**: 代价体构建的思想源自经典MVS，但MVSplat不需要GT深度监督
- **UniMatch/GMFlow**: Transformer+代价体在光流/立体匹配中的成功，启发了本文的设计

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将MVS代价体引入前馈式3DGS，思路清晰且有效
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三个数据集+跨数据集泛化+几何可视化+详细消融
- **写作质量**: ⭐⭐⭐⭐⭐ — 论文结构精炼，对比分析非常清楚
- **价值**: ⭐⭐⭐⭐⭐ — 显著推进了前馈式3DGS的实用性，代码开源
