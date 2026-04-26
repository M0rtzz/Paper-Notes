---
title: >-
  [论文解读] ReFlex: Text-Guided Editing of Real Images in Rectified Flow via Mid-Step Feature Extraction and Attention Adaptation
description: >-
  [ICCV 2025][图像生成][图像编辑] 针对 Rectified Flow（ReFlow）模型的真实图像编辑难题，通过系统分析 MM-DiT 的中间表示，识别出三个关键特征（I2I-SA、I2T-CA、残差特征），并提出中间步特征提取（mid-step feature extraction）和两种注意力适配技术，在 FLUX 模型上实现了无需训练、无需用户掩码的高质量真实图像编辑，人类评估中 68.2% 优选率远超其他方法。
tags:
  - ICCV 2025
  - 图像生成
  - 图像编辑
  - Rectified Flow
  - FLUX
  - 特征注入
  - 注意力适配
---

# ReFlex: Text-Guided Editing of Real Images in Rectified Flow via Mid-Step Feature Extraction and Attention Adaptation

**会议**: ICCV 2025  
**arXiv**: [2507.01496](https://arxiv.org/abs/2507.01496)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 图像编辑, Rectified Flow, FLUX, 特征注入, 注意力适配

## 一句话总结

针对 Rectified Flow（ReFlow）模型的真实图像编辑难题，通过系统分析 MM-DiT 的中间表示，识别出三个关键特征（I2I-SA、I2T-CA、残差特征），并提出中间步特征提取（mid-step feature extraction）和两种注意力适配技术，在 FLUX 模型上实现了无需训练、无需用户掩码的高质量真实图像编辑，人类评估中 68.2% 优选率远超其他方法。

## 研究背景与动机

文本引导的真实图像编辑是图像生成的重要应用。基于 U-Net 的扩散模型（如 SD）已有成熟的编辑方法（P2P、PnP），它们利用交叉注意力和自注意力特征进行结构保持。然而，更新一代的 Rectified Flow 模型（如 FLUX）采用了完全不同的架构和训练方式，现有方法难以直接迁移。

**现有 ReFlow 编辑方法的痛点**：
- **RF-Inversion**：通过源图像和噪声的插值构建受控 ODE，过度依赖源图像，编辑灵活性差
- **RF-Edit / FireFlow**：使用二阶 Taylor 展开减小重建误差，但利用的是 value 特征，容易导致外观泄漏（appearance leakage）
- **FlowEdit**：构建源-目标直接 ODE，最小化编辑变化，但对大幅修改（如颜色变化）能力不足
- **共同问题1**：MM-DiT 将文本和图像 token 在联合自注意力中纠缠处理，与 U-Net 的交叉/自注意力分离架构完全不同，哪些特征有效尚不清楚
- **共同问题2**：ReFlow 模型从完全反转的 latent 重建图像时往往无法准确恢复原图结构

**核心突破**：
1. 系统分析 MM-DiT 的联合自注意力图，将其分解为 I2I-SA、I2T-CA、T2I-CA、T2T-SA 四个组件，发现只有前两个对编辑有效
2. 发现从中间步（mid-step）而非完全反转的 latent 中提取特征，可以大幅提高结构保持能力

## 方法详解

### 整体框架

ReFlex 的流程分为三步：
1. **反转阶段**：将真实图像通过 ReFlow 前向过程反转到 $z_T$（完全噪声）和 $z_{t'=T/2}$（中间步 latent）
2. **特征提取**：从中间步 latent $z_{t'}$ 出发进行一步推理，提取三个关键特征
3. **目标生成**：从完全反转的 $z_T$ 出发，用目标提示生成目标图像，在前期时间步注入提取的特征并应用注意力适配

### 关键设计

1. **MM-DiT 关键特征识别**：

    - 功能：确定 MM-DiT 中哪些中间表示对图像编辑有效
    - 核心思路：将联合自注意力矩阵 $Q \cdot K^T$ 按 query-key 的模态分解为四个组件。PCA 可视化（Fig. 4a）显示 I2I-SA（$Q_{image} \cdot K_{image}^T$）编码结构信息，I2T-CA（$Q_{image} \cdot K_{text}^T$）编码文本-图像关系。注入实验证实只有 I2I-SA 和 I2T-CA 能保持源图结构，因为只有它们直接影响图像 token 嵌入
    - 对于残差连接，分析了 residual feature $f(x)_{image}$ 和 identity feature $x_{image}$：后者保留了过多的外观细节导致编辑受限，仅选择 residual feature
    - 设计动机：T2I-CA 和 T2T-SA 直接影响的是文本 token 嵌入，而最终只有图像 token 嵌入被送到解码器生成图像，因此它们对图像结构保持无效

2. **中间步特征提取（Mid-Step Feature Extraction）**：

    - 功能：从中间步 latent 而非完全反转的 latent 中提取特征，确保结构准确性
    - 核心思路：不从完全噪声 $z_T$ 开始重建（重建误差大），而是从仅反转到中间步的 $z_{t'=T/2}$ 出发提取特征。Fig. 5 清晰展示了不同时间步 latent 的重建质量：$t=T/2$ 可以近乎完美恢复原图，而 $t=T$ 则丢失大量结构
    - 设计动机：ReFlow 模型的反转-重建过程存在显著误差累积。从中间步 latent 提取的特征更忠实于源图像。虽然这会降低可编辑性（因过度保持源信息），但可通过后续的注意力适配来恢复

3. **I2T-CA 适配**：

    - 功能：在注入 I2T-CA 时增强对目标文本中新词汇的响应
    - 核心思路：定义源-目标提示间的 token 映射函数 $f$。对于有对应的 token，直接注入源的 I2T-CA；对于无对应的新 token（如 "goat" → "horse" 中的 "horse"），用放大因子 $\alpha > 1$ 缩放目标的 I2T-CA：$CA'[:,i] = \alpha \times CA_T[:,i]$（当 $f(i) = \emptyset$）
    - 设计动机：直接注入源的 I2T-CA 会抑制新 token 的语义表达。过放大新 token 的注意力权重可以增强编辑效果，恢复因中间步提取导致的可编辑性损失

4. **I2I-SA 适配**：

    - 功能：防止源的 I2I-SA 过度保持局部结构
    - 核心思路：可视化发现（Fig. 6）I2I-SA 中每个 query 过度集中在自身附近的少数像素上。将 top-k 注意力值替换为目标的 I2I-SA 对应值（经归一化），保持全局结构的同时允许局部修改
    - 设计动机：在线性尺度上 I2I-SA 过度集中于局部，但在对数尺度上能看到它有效捕获了全局结构。替换 top-k 最高值可以释放局部编辑空间而不破坏全局结构

### 损失函数 / 训练策略

- 完全无需训练（training-free），只需推理时操作特征
- 超参数设置：采样步数 $T=28$，特征提取步 $t'=14$，I2T-CA 放大因子 $\alpha=4$，I2I-SA 的 $k=20$
- 有源提示时：I2T-CA 注入前 0.4T 步，I2I-SA 注入前 0.25T 步，残差特征注入前 0.15T 步
- 无源提示时：省略 I2T-CA 注入，I2I-SA 注入延长到 0.4T 步
- 可选的掩码生成：从 I2T-CA 中提取编辑区域掩码，配合 latent blending 实现局部编辑

## 实验关键数据

### 主实验（文本对齐与结构保持指标）

| 方法 | 基础模型 | PIE-Bench CLIP-T↑ | Wild-TI2I CLIP-T↑ | 用户偏好% |
|------|---------|------------------|-------------------|----------|
| RF-Inversion | FLUX | 低于 ReFlex 1.69% | 低于 ReFlex 3.21% | 7.8% |
| RF-Edit | FLUX | — | — | 7.3% |
| FireFlow | FLUX | — | — | 5.5% |
| FlowEdit | FLUX | — | — | 11.2% |
| DI+PnP | SD | — | — | 11.4% |
| DDIM+PnP | SD | — | — | 7.5% |
| SDEdit | SD | — | — | 9.4% |
| P2P-Zero | SD | — | — | 10.6% |
| **ReFlex** | **FLUX** | **最优** | **最优(+3.21~16.46%)** | **68.2% / 61.0%** |

### 消融实验（Wild-TI2I-Real 上各技术贡献）

| 配置 | 文本对齐 | 结构保持(IoU) | 说明 |
|------|---------|-------------|------|
| 无 mid-step 提取（传统全反转） | 较高 | 显著下降 | 源结构大量丢失 |
| 无 I2T-CA 适配 | 下降 | 较高 | 编辑不能完全遵循目标提示 |
| 无 I2I-SA 适配 | 下降 | 较高 | 大幅修改（如颜色）困难 |
| **完整 ReFlex** | **最优** | **Pareto 最优** | **平衡编辑性与保持性** |

### 关键发现
- ReFlex 在人类评估中以 68.2%（vs FLUX 方法）和 61.0%（vs SD 方法）的压倒性优势获得用户偏好
- 中间步特征提取是最关键的组件：移除后 IoU 大幅下降
- $t' = T/2 = 14$ 是最优提取点（Fig. 10）：太近纯噪声端会丢结构，太近图像端会降低图像质量
- $k=20$ 是 I2I-SA 适配的良好平衡点（Fig. 11）：太大会丢失源结构
- 无需源提示也能工作（Wild-TI2I-Real），展现了实用性

## 亮点与洞察
- 对 MM-DiT 特征的系统分析是本文最大贡献，为 ReFlow 模型编辑奠定了理论基础
- 中间步特征提取的想法极其简洁但效果显著，利用了 ReFlow 模型"中间步 latent 重建能力远优于全反转 latent"的特性
- I2I-SA 适配中"替换 top-k 最高值"的设计巧妙利用了注意力分布在线性/对数尺度下的不同表现
- 整体方法完全无需训练，直接插入 FLUX 推理流程，实际应用价值高

## 局限与展望
- 编辑区域与主体重叠时可能意外改变主体的其他特征（如移除眼镜时改变发型，Fig. 12a）
- 从 I2T-CA 生成的编辑掩码不够精确，可能引入伪影（Fig. 12b）
- 编辑结果存在一定随机性变化
- 需要 blended word 来定义编辑区域（目前依赖手动选择或预定义）
- 未探索在其他 ReFlow 模型（如 SD3.5）上的效果

## 相关工作与启发
- P2P 和 PnP 是 DM 编辑的经典方法，ReFlex 将其核心思想（注意力/特征注入）迁移到了 MM-DiT 架构
- "从中间步而非全噪声开始"的思路与 SDEdit 有相似之处，但 ReFlex 是用于特征提取而非直接去噪
- 对 ReFlow 模型反转误差的分析值得关注：这解释了为什么简单套用 DM 编辑方法在 FLUX 上效果不好

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)
- [\[ICML 2025\] Taming Rectified Flow for Inversion and Editing](../../ICML2025/image_generation/taming_rectified_flow_for_inversion_and_editing.md)
- [\[ICCV 2025\] Straighten Viscous Rectified Flow via Noise Optimization](straighten_viscous_rectified_flow_via_noise_optimization.md)
- [\[NeurIPS 2025\] GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer](../../NeurIPS2025/image_generation/guideflow3d_optimization-guided_rectified_flow_for_appearance_transfer.md)
- [\[CVPR 2025\] SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion](../../CVPR2025/image_generation/swiftedit_lightning_fast_text-guided_image_editing_via_one-step_diffusion.md)

<!-- RELATED:END -->
