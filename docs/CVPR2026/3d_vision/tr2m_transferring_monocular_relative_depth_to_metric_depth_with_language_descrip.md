---
title: >-
  [论文解读] TR2M: Transferring Monocular Relative Depth to Metric Depth with Language Descriptions and Dual-Level Scale-Oriented Contrast
description: >-
  [CVPR2026][3D视觉][单目深度估计] 提出 TR2M 框架，利用图像和文本描述预测像素级的 scale/shift 映射图，将泛化性强但无尺度的相对深度转换为度量深度，仅用 19M 可训练参数和 102K 训练图像即可实现跨域零样本度量深度估计。
tags:
  - CVPR2026
  - 3D视觉
  - 单目深度估计
  - 相对深度转度量深度
  - 语言描述
  - 跨模态注意力
  - 对比学习
  - 像素级缩放
---

# TR2M: Transferring Monocular Relative Depth to Metric Depth with Language Descriptions and Dual-Level Scale-Oriented Contrast

**会议**: CVPR2026  
**arXiv**: [2506.13387](https://arxiv.org/abs/2506.13387)  
**代码**: [GitHub](https://github.com/BeileiCui/TR2M)  
**机构**: 香港中文大学
**领域**: 3d_vision  
**关键词**: 单目深度估计, 相对深度转度量深度, 语言描述, 跨模态注意力, 对比学习, 像素级缩放

## 一句话总结

提出 TR2M 框架，利用图像和文本描述预测像素级的 scale/shift 映射图，将泛化性强但无尺度的相对深度转换为度量深度，仅用 19M 可训练参数和 102K 训练图像即可实现跨域零样本度量深度估计。

## 背景与动机

单目深度估计(MDE)分为两大流派：

- **度量深度估计(MMDE)**：输出真实尺度(米)，但通常局限于特定域，跨域泛化差；使用相机内参或大量数据缓解但代价高。
- **相对深度估计(MRDE)**：通过 affine-invariant loss 训练，跨域泛化好，但输出缺少绝对尺度，下游应用(机器人导航、三维重建)受限。

现有的相对→度量转换方法存在两个核心问题：

1. **单因子缩放的局限**：之前方法(如 RSA)仅估计一个全局 scale 和 shift 因子，无法修正相对深度中局部错误区域，甚至会放大误差。
2. **语义描述歧义**：同一类别物体在不同分布和比例下产生不同深度，但文本描述可能相似，导致尺度估计不准确。

## 核心问题

如何高效地消除相对深度的尺度不确定性，将其转换为度量深度，同时保持 MRDE 的跨域泛化能力？关键挑战在于实现**像素级**的精细缩放而非全局缩放，并在特征层面建立尺度一致性。

## 方法详解

### 整体框架

TR2M 的输入为 RGB 图像 $I \in \mathbb{R}^{H \times W \times 3}$ 和文本描述 $L$（由 LLaVA 自动生成），目标是预测两个像素级映射图：scale map $A \in \mathbb{R}^{H \times W}$ 和 shift map $B \in \mathbb{R}^{H \times W}$，将冻结的相对深度模型(Depth Anything-Small)输出的相对深度 $D_r$ 转换为度量深度：

$$\hat{D}_m = \frac{1}{A \odot D_r + B}$$

其中 $\odot$ 为逐元素乘法。这种像素级变换相比全局单因子缩放可以修正相对深度中的局部错误区域。

### 特征提取与跨模态注意力

- **图像编码器**：冻结的 DINOv2 ViT-L，提取图像特征 $F_I \in \mathbb{R}^{HW \times D}$
- **文本编码器**：冻结的 CLIP ViT-L/14，提取文本特征 $F_L \in \mathbb{R}^{1 \times D}$

跨模态注意力模块将两种特征融合。以图像特征为 Query，分别对自身(self-attention)和文本特征(cross-attention)进行注意力计算：

$$\text{Attn}_{cm}^{i}(Q_I, K_i, V_i) = \text{softmax}\left(\frac{Q_I K_i^T}{\sqrt{d}}\right) \cdot V_i, \quad i \in \{I, L\}$$

最终融合特征：$F_{out} = F_I + \text{Attn}_{cm}^I + \text{Attn}_{cm}^L$

这种设计的关键在于：图像特征保持像素级空间分辨率作为 Query，文本特征作为全局尺度先验通过 cross-attention 注入到每个像素位置。

### 解码器

使用两个轻量级 DPT 风格解码器头分别从融合特征 $F_f$ 中生成 scale map 和 shift map：

- $A = \text{ScaleHead}(F_f)$
- $B = \text{ShiftHead}(F_f)$

### 伪度量深度与阈值筛选

由于 ground truth 度量深度通常是稀疏的（部分像素没有标注），作者通过最小二乘回归将相对深度与 GT 对齐，生成伪度量深度：

$$(\tilde{\alpha}, \tilde{\beta}) = \arg\min_{\tilde{\alpha}, \tilde{\beta}} \sum_{i=1}^{HW} (\tilde{\alpha} D_r(i) + \tilde{\beta} - D_m^{gt}(i))^2$$

得到 $D_m^{pseudo} = \tilde{\alpha} D_r + \tilde{\beta}$。

关键创新点在于**质量筛选**：使用阈值精度 $\delta_1$（$\max(D_m^{gt}/D_m^{pseudo}, D_m^{pseudo}/D_m^{gt}) < 1.25$ 的比例）来判断伪深度是否可信。只有 $\delta_1 > \rho$ 时才加入监督：

$$\mathcal{L}_{tp\text{-}si} = \mathbf{1}(\delta_1 > \rho) \cdot \mathcal{L}_{si}(\hat{D}_m, D_m^{pseudo})$$

这使得稀疏 GT 区域外的像素也能获得合理的监督信号，提升泛化能力。

### 双层尺度导向对比学习(Dual-Level Scale-Oriented Contrast)

这是本文最具创新性的模块，从两个粒度强制特征与深度尺度分布一致：

**1) 图像级粗对比(Image-Level Coarse Contrast)**

在每个训练 batch 中，对图像特征做 average pooling 获得图像级嵌入 $\tilde{F}_f$，根据伪 scale 因子 $\tilde{\alpha}$ 的大小排序。尺度相近的图像特征为正样本对，尺度差异大的为负样本对：

- 正样本：$|i - j| < t$（尺度距离小于阈值 $t$）
- 负样本：$|i - j| \geq t$

通过 InfoNCE 风格的对比损失，使具有相似尺度的场景在特征空间中更接近。

**2) 像素级细对比(Pixel-Level Fine Contrast)**

利用 GT 深度图构造像素级对比。将深度值按分布量化为 $|\mathcal{C}|$ 个离散类别：

$$Y = \text{round}\left(\frac{d_{max} - D}{d_{max} - d_{min}} \times |\mathcal{C}|\right)$$

采用 EMA 双分支结构（类似 MoCo）：Query 和 Key 分支分别处理不同样本，生成像素级特征。对于 Query 像素 $i$，在 Key 特征图中找到相同深度类别的像素作为正样本，不同类别的作为负样本，最大化正样本相似度、最小化负样本相似度。

最终对比损失：$\mathcal{L}_{soc} = \mathcal{L}_{coarse} + \mathcal{L}_{fine}$

### 总损失函数

$$\mathcal{L} = \lambda_1 \mathcal{L}_{si} + \lambda_2 \mathcal{L}_{tp\text{-}si} + \lambda_3 \mathcal{L}_{soc} + \lambda_4 \mathcal{L}_{es}$$

其中 $\mathcal{L}_{es}$ 为边缘感知平滑损失，约束 scale/shift map 的平滑性。超参数设置为 $\lambda_1=1, \lambda_2=0.5, \lambda_3=0.1, \lambda_4=0.01$。

### 实现细节

| 配置项 | 设置 |
|--------|------|
| GPU | NVIDIA RTX 4090 |
| 优化器 | AdamW |
| Batch size | 8 |
| 学习率 | $1 \times 10^{-5}$，每 epoch 衰减 0.9 |
| 训练 epoch | 20 |
| 可学习 Scale Embedding 数 | 256 |
| 相对深度模型 | Depth Anything-Small（冻结） |
| 图像编码器 | DINOv2 ViT-L（冻结） |
| 文本编码器 | CLIP ViT-L/14（冻结） |
| 可训练参数量 | 19M |
| 训练数据量 | 102K 图像 |
| 训练数据集 | NYUv2 + KITTI + VOID + C3VD |
| 文本生成 | LLaVA v1.6 Vicuna & Mistral |

## 实验关键数据

### NYUv2 室内基准(Table 1)

| 方法 | 类型 | 可训练参数 | $\delta_1\uparrow$ | AbsRel$\downarrow$ | RMSE$\downarrow$ |
|------|------|-----------|---------|---------|--------|
| DA V2 | 直接度量 | 25M | 0.969 | 0.073 | 0.261 |
| UniDepth | 零样本度量 | 347M | 0.981 | 0.072 | 0.229 |
| Metric3Dv2 | 零样本度量 | 1011M | 0.980 | 0.067 | 0.260 |
| RSA | 语言+缩放因子 | 4.7M | 0.752 | 0.156 | 0.528 |
| ScaleDepth | 语言+域自适应 | 109M | 0.913 | 0.099 | 0.329 |
| WorDepth | 语言+域自适应 | 137M | 0.926 | 0.090 | 0.330 |
| **TR2M** | **语言+缩放图** | **19M** | **0.954** | **0.082** | **0.293** |

### 零样本跨域泛化(Table 3, 5个未见数据集)

| 方法 | Backbone | 训练图像 | SUN δ₁ | iBims δ₁ | HyperSim δ₁ | DIODE δ₁ | SimCol δ₁ | Avg Rank |
|------|----------|----------|--------|----------|-------------|----------|-----------|----------|
| ZoeDepth | BeiT384-L | - | 0.545 | 0.656 | 0.302 | 0.237 | 0.438 | 4.70 |
| DA Single | ViT-L | - | 0.660 | 0.714 | 0.361 | 0.288 | 0.553 | 2.80 |
| UniDepth | ViT-L | 3M | 0.443 | 0.217 | 0.545 | 0.635 | - | 4.00 |
| RSA | ViT-S | 102K | 0.527 | 0.450 | 0.230 | 0.244 | 0.162 | 5.80 |
| **TR2M** | **ViT-S** | **102K** | **0.591** | **0.736** | **0.361** | **0.274** | **0.445** | **2.40** |

TR2M 以 ViT-Small 骨干和 102K 训练图像达到最佳平均排名(2.40)，远超使用更大模型和更多数据的方法。

### 消融实验

| Rescale Maps | $\mathcal{L}_{tp\text{-}si}$ | $\mathcal{L}_{soc}$ | NYUv2 AbsRel | iBims δ₁ | DIODE δ₁ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ✕ | ✕ | ✕ | 0.118 | 0.566 | 0.225 |
| ✓ | ✕ | ✕ | 0.084 | 0.657 | 0.237 |
| ✓ | ✓ | ✕ | 0.085 | 0.708 | 0.243 |
| ✓ | ✓ | ✓ | **0.082** | **0.736** | **0.274** |

- 从单因子→像素级映射图：AbsRel 从 0.118 降到 0.084，最大提升
- 加伪深度监督：零样本场景显著提升(iBims δ₁ +0.051)
- 加尺度对比学习：零样本进一步提升(DIODE δ₁ +0.031)

## 亮点

1. **像素级 rescale map**替代全局单因子缩放是核心创新，能修正相对深度中的局部错误区域，这是之前方法(如 RSA)无法实现的
2. **极高的参数效率**：仅 19M 可训练参数（编码器全部冻结），比 UniDepth(347M)、Metric3Dv2(1011M) 小 1-2 个数量级，却达到可比性能
3. **双层对比学习**设计精巧：粗粒度保证全局尺度一致性，细粒度保证局部深度分布一致性，两者互补
4. **伪深度+阈值筛选**策略简洁有效地解决了稀疏 GT 问题，提升零样本泛化
5. 文本描述作为"免费"的辅助先验，无需相机参数或传感器信息

## 局限与展望

1. **文本描述依赖 LLaVA 生成**：文本质量受限于 VLM 能力，极端场景(医学内窥镜等)的描述可能不够准确
2. **相对深度模型的上限**：框架性能受限于冻结的相对深度模型(Depth Anything-Small)的输出质量
3. **零样本在部分域仍有差距**：SUN RGB-D 和 DIODE Outdoor 上 AbsRel 仍偏高(0.451/0.673)，说明跨域巨大的尺度差距仍有挑战
4. **对比学习计算开销**：像素级对比需要 EMA 双分支结构，增加训练时间和内存
5. 可尝试更强的相对深度骨干（如 Depth Anything V2）来进一步提升上限

## 与相关工作的对比

| 维度 | RSA | ScaleDepth/WorDepth | UniDepth/Metric3Dv2 | **TR2M** |
|------|-----|---------------------|---------------------|----------|
| 输入 | 文本 | 文本+图像 | 图像(+相机参数) | 文本+图像+相对深度 |
| 缩放方式 | 全局双因子 | 隐式解码 | 端到端度量头 | **像素级映射图** |
| 可训练参数 | 4.7M | 109-137M | 347-1011M | **19M** |
| 训练数据 | 102K | 单域 | 3-16M | **102K** |
| 跨域能力 | 差 | 单域 | 较好 | **强** |
| 核心局限 | 单因子不够精细 | 域自适应受限 | 需大数据+大模型 | 依赖相对深度质量 |

## 启发与关联

- 相对深度→度量深度的范式类似于 NLP 中的"预训练+轻量适配"，冻结大模型+少参数微调的高效思路值得借鉴
- 像素级 rescale map 的思想可推广到其他需要尺度恢复的任务（如单目法线估计、光流尺度恢复）
- 双层对比学习的设计（全局+局部）可迁移到其他密集预测任务中做特征正则化
- 文本辅助深度估计这条路线如果结合更强的 MLLM（如 GPT-4V），可能进一步提升场景理解和尺度推理能力

## 评分

- 新颖性: ⭐⭐⭐⭐ (像素级 rescale map + 双层对比学习组合新颖)
- 实验充分度: ⭐⭐⭐⭐ (4个训练集 + 5个零样本测试集 + 充分消融)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，动机阐述充分)
- 价值: ⭐⭐⭐⭐ (高效的相对→度量深度转换方案，实用性强)
