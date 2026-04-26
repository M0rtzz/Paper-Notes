---
title: >-
  [论文解读] ThermalGen: Style-Disentangled Flow-Based Generative Models for RGB-to-Thermal Image Translation
description: >-
  [NeurIPS 2025][图像生成][RGB-热红外翻译] 提出 ThermalGen，一种基于 Flow 的自适应生成模型，通过 RGB 图像条件化架构和风格解耦机制，首次实现了跨视角、跨传感器、跨环境条件的高保真 RGB-to-Thermal 图像翻译，并发布了三个新的大规模卫星-航拍 RGB-T 配对数据集。
tags:
  - NeurIPS 2025
  - 图像生成
  - RGB-热红外翻译
  - Flow-based 生成模型
  - 风格解耦
  - 多数据集联合训练
  - 热成像
---

# ThermalGen: Style-Disentangled Flow-Based Generative Models for RGB-to-Thermal Image Translation

**会议**: NeurIPS 2025  
**arXiv**: [2509.24878](https://arxiv.org/abs/2509.24878)  
**代码**: [项目页面](https://xjh19971.github.io/ThermalGen)  
**领域**: 图像生成 / 跨模态翻译  
**关键词**: RGB-热红外翻译, Flow-based 生成模型, 风格解耦, 多数据集联合训练, 热成像

## 一句话总结

提出 ThermalGen，一种基于 Flow 的自适应生成模型，通过 RGB 图像条件化架构和风格解耦机制，首次实现了跨视角、跨传感器、跨环境条件的高保真 RGB-to-Thermal 图像翻译，并发布了三个新的大规模卫星-航拍 RGB-T 配对数据集。

## 研究背景与动机

视觉-热红外传感器融合在低光照、恶劣天气等挑战条件下至关重要，但配对 RGB-T 数据的稀缺严重制约了相关研究。RGB-to-Thermal 图像翻译可以从丰富的 RGB 数据合成热图像，具有三大优势：

**完美对齐**：合成热图像与原始 RGB 图像具有像素级对应，适合稠密特征匹配等精细任务

**规模化扩展**：可利用海量公开 RGB 数据，远超硬件采集的 RGB-T 配对规模

**多样性模拟**：从单张 RGB 输入模拟不同热特性和环境条件，增强下游模型鲁棒性

然而，现有方法面临严峻挑战：

- **训练数据窄**：GAN 方法多在单一小数据集上训练，泛化能力差
- **RGB 缺乏热信息**：模型需要从语义内容推断热线索
- **域差异巨大**：不同热传感器、摄像机视角、环境条件之间存在显著分布差异

ThermalGen 的关键创新在于通过**风格解耦机制**将数据集特定的 RGB-T 映射关系编码为可学习的风格嵌入，使一个模型能同时处理多种 RGB-T 风格。

## 方法详解

### 整体框架

ThermalGen 基于 SiT（Scalable Interpolant Transformer）架构，在潜空间中进行 Flow-based 生成。给定 RGB 图像和数据集特定的风格嵌入，模型预测热图像潜变量的速度场，经过 $T=50$ 步去噪后通过热图像解码器重建热图像。

### 关键设计

1. **热图像编解码器**

   采用潜扩散框架，热图像编码器 $E_T$ 将热图像压缩为潜变量 $\mathbf{z}_T \in \mathbb{R}^{\frac{H}{f}\times\frac{W}{f}\times C}$，解码器 $D_T$ 重建热图像。RGB 编码器使用预训练的 KL-VAE 编码器 $E_{\text{RGB}}$ 提取 RGB 潜表示 $\mathbf{z}_{\text{RGB}}$。

   Flow-based 生成在潜空间中操作：

   $$\mathbf{z}_t = \alpha_t \mathbf{z}_0 + \sigma_t \boldsymbol{\epsilon}, \quad \alpha_t = 1-t, \quad \sigma_t = t$$

   速度函数的训练目标：

   $$\mathcal{L}_{\text{flow}} = \mathbb{E}_{\mathbf{z}_t, t}\left[\|v_\theta(\mathbf{z}_t, t) - v(\mathbf{z}_t, t)\|^2\right]$$

2. **风格解耦机制（Style-Disentangled Mechanism）**

   定义一组可学习的风格嵌入 $Y = \{\mathbf{y}_0, \mathbf{y}_1, \ldots, \mathbf{y}_n, \mathbf{y}_{\text{un}}\}$，其中 $n$ 为数据集数量，$\mathbf{y}_{\text{un}}$ 为无条件风格嵌入（维度 1024）。

   风格嵌入通过 **adaLN-Zero 条件化**注入模型：给定风格嵌入 $\mathbf{y}_i$ 和时间步 $t$，生成条件嵌入 $\mathbf{c}_{\mathbf{y}_i, t}$，调制自适应层归一化的 scale 和 shift 参数。

   训练时随机选择数据集特定嵌入或无条件嵌入，支持 Classifier-Free Guidance（CFG）。新增数据集只需追加新风格嵌入，无需重新训练。

   设计动机：受 AdaIN 启发——修改归一化参数可有效实现风格迁移。不同 RGB-T 数据集之间的映射关系差异巨大（传感器、视角、时间），将这种"风格"从模型参数中解耦出来，使一个模型适应多种场景。

3. **RGB 图像条件化架构**

   探索两种变体：
   - **多头交叉注意力（Cross-Attn）**：$\mathbf{z}_{\text{RGB}}$ 作为 query，$\hat{\mathbf{z}}_{t,T}$ 作为 key 和 value
   - **拼接（Concatenation）**：直接拼接 $\hat{\mathbf{z}}_{t,T}$ 和 $\mathbf{z}_{\text{RGB}}$ 作为 SiT 输入

   实验表明拼接方式整体 FID 更优，且便于从预训练 SiT 权重微调。

### 损失函数 / 训练策略

- 使用标准 flow matching 损失训练
- 联合训练时从所有训练集随机采样 batch
- 训练图像随机裁剪到 256×256，评估时 resize 到 256×256
- 总计约 200K 训练样本来自 11+ 数据集
- 推理使用 50 步去噪

## 实验关键数据

### 主实验（卫星-航拍数据集）

| 方法 | 类型 | Boson-night FID↓ | Bosonplus-day FID↓ | Bosonplus-night FID↓ |
|------|------|-----------------|-------------------|---------------------|
| pix2pix | GAN | 149.55 | 170.45 | 137.74 |
| pix2pixHD | GAN | 106.33 | 157.65 | 89.26 |
| VQGAN | GAN | 207.12 | 185.41 | 286.74 |
| DiffV2IR | Diffusion | 150.11 | 215.20 | 96.42 |
| **ThermalGen-L/2** | **Flow** | **161.22** | **76.91** | **75.80** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| SiT-B vs SiT-L vs SiT-XL | FID递减 | 更大 Transformer = 更好生成质量 |
| Patch size 2 vs 4 vs 8 | patch=2 最优 | 更细粒度 patch 提升图像质量 |
| Cross-Attn vs Concatenation | Concatenation FID更优 | 拼接方式整体更好 |
| Unconditional vs Conditional vs CFG | CFG最优（独特风格数据集） | 风格嵌入显著影响有独特风格的数据集 |
| Boson-night CFG scale 1→8 | FID: 161.22→116.46 | CFG 调节可大幅改善低对比度场景 |
| FLIR CFG scale 1→4 | FID: 70.09→63.43 | 极端光照条件下 CFG 同样有效 |

### 关键发现

1. ThermalGen 在大多数数据集上实现了最优或接近最优的感知质量（FID、LPIPS），尤其在 Bosonplus 和 NII-CU 等数据集上显著领先
2. 风格嵌入对具有独特 RGB-T 风格的数据集效果显著；通用数据集（M3FD、MSRS）上改善较小，可能因为风格已编码在模型参数中
3. GAN 方法普遍产生失真或网格伪影；DiffV2IR 倾向于生成过于锐利的边界
4. DDIM 基线倾向于生成接近训练分布的随机样本而非条件化输出，凸显了 ThermalGen 的 RGB 条件化能力
5. 在 LLVIP 等数据集上性能不佳主要因训练/测试分布差异（t-SNE 验证），扩展数据集可解决

## 亮点与洞察

- **首个跨视角、跨传感器、跨环境的通用 RGB-T 翻译模型**，覆盖卫星-航拍、航拍、地面三大类别
- 风格解耦设计优雅且实用——新数据集只需追加嵌入而非重训模型
- 三个新数据集（DJI-day、Bosonplus-day、Bosonplus-night）扩展了 RGB-T 研究的数据基础
- CFG scale 作为推理时调节手段可有效缓解特定数据集的性能问题

## 局限与展望

- 在 Boson-night（低对比度）、LLVIP（分布偏移）和 FLIR（极端光照）上表现欠佳
- 模型假设 RGB 和热图像空间分辨率一致，未处理分辨率不匹配场景
- 风格嵌入是数据集级别而非场景级别，同一数据集内的风格变化未被建模
- 仅在 256×256 分辨率上评估，高分辨率场景未验证

## 相关工作与启发

- adaLN-Zero 风格条件化可推广到其他多域图像翻译任务
- RGB-T 翻译的思路可扩展到其他跨模态任务（如 RGB-深度、RGB-SAR）
- 大规模异构数据集联合训练是提升泛化能力的关键手段

## 评分

- **新颖性**: ⭐⭐⭐⭐ 风格解耦 + Flow-based RGB-T 翻译是新颖组合
- **实验充分度**: ⭐⭐⭐⭐⭐ 跨 11+ 数据集的全面评估，充分的消融和可视化
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，数据详实
- **价值**: ⭐⭐⭐⭐ 为 RGB-T 跨模态翻译提供了实用且可扩展的解决方案

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](../../ICCV2025/image_generation/scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [\[NeurIPS 2025\] Gradient Variance Reveals Failure Modes in Flow-Based Generative Models](gradient_variance_reveals_failure_modes_in_flow-based_generative_models.md)
- [\[NeurIPS 2025\] Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints](physics-constrained_flow_matching_sampling_generative_models_with_hard_constrain.md)
- [\[ICCV 2025\] Deeply Supervised Flow-Based Generative Models](../../ICCV2025/image_generation/deeply_supervised_flow-based_generative_models.md)
- [\[NeurIPS 2025\] EditInfinity: Image Editing with Binary-Quantized Generative Models](editinfinity_image_editing_with_binary-quantized_generative_models.md)

<!-- RELATED:END -->
