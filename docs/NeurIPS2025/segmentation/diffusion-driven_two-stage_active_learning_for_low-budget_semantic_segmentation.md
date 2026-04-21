---
title: >-
  [论文解读] Diffusion-Driven Two-Stage Active Learning for Low-Budget Semantic Segmentation
description: >-
  [NeurIPS 2025][图像分割][主动学习] 提出两阶段主动学习流程（覆盖性→不确定性），利用预训练扩散模型的多尺度特征实现极低标注预算下的高效语义分割。
tags:
  - NeurIPS 2025
  - 图像分割
  - 主动学习
  - 语义分割
  - 扩散模型
  - 不确定性采样
  - 低预算标注
---

# Diffusion-Driven Two-Stage Active Learning for Low-Budget Semantic Segmentation

**会议**: NeurIPS 2025  
**arXiv**: [2510.22229](https://arxiv.org/abs/2510.22229)  
**代码**: [有](https://github.com/jn-kim/two-stage-edald)  
**领域**: 语义分割 / 主动学习  
**关键词**: 主动学习, 语义分割, 扩散模型, 不确定性采样, 低预算标注

## 一句话总结

提出两阶段主动学习流程（覆盖性→不确定性），利用预训练扩散模型的多尺度特征实现极低标注预算下的高效语义分割。

## 研究背景与动机

语义分割需要密集的像素级标注，标注成本极高。主动学习（AL）通过策略性选择最有价值的数据进行标注来降低成本，但现有方法在极低预算场景下表现不佳：

- **不确定性方法**（Entropy、Margin等）：倾向选择冗余像素，高不确定性像素往往聚集在物体边界附近
- **表示性方法**（Core-set等）：避免冗余但容易遗漏信息量大的边界像素
- **现有像素级AL**（PixelPick等）：在极低预算下效果有限

本文定义了一个极端低预算场景：每轮仅标注 $b = 0.1N$ 个像素（$N$ 为图像数量），10轮后总共仅标注约 $N$ 个像素——相当于每张图平均一个像素标签。

**关键观察**：扩散模型的逆过程在不同时间步生成从全局结构到局部细节的特征，这些多时间步特征具有类似集成（ensemble）的性质，可用于不确定性估计。

## 方法详解

### 整体框架

提出两阶段像素选择流程：
1. **阶段1（覆盖性）**：基于表示的层次化候选像素选择（MaxHerding），建立多样化候选池
2. **阶段2（不确定性）**：在候选池上用熵增强的分歧度评分（eDALD）选出最终标注像素

### 关键设计

**1. 扩散特征提取**

使用ImageNet预训练的扩散模型提取多尺度特征：
- 采样 $T=3$ 个时间步（$t_1=50, t_2=150, t_3=250$）
- 每个时间步提取 $L=4$ 层特征（$l_1=5, l_2=8, l_3=12, l_4=17$）
- 上采样并拼接得到每个像素的丰富表示 $z_x$

**2. 阶段1：层次化MaxHerding**

- **图像内选择**：对每张图像的所有像素应用MaxHerding，选出 $K=50$ 个代表性像素
- **跨图像精炼**：将所有图像的候选像素合并，再次执行MaxHerding得到全局多样化候选池 $\mathcal{M}$

这种局部→全局的两步策略确保候选池既在每张图像内有代表性，又在全局范围内具多样性。

**3. 阶段2：eDALD不确定性评分**

核心思想：利用扩散模型的随机噪声注入产生多组特征，计算互信息衡量认知不确定性。

**DALD（扩散主动学习分歧度）**基于BALD框架：
$$I(\hat{Y}; Z | x) = H(\hat{Y} | x) - \mathbb{E}_{z}[H(\hat{Y} | Z=z, x)]$$

- 非条件熵 $H(\hat{Y}|x)$：对 $M$ 个噪声样本的平均预测求熵
- 条件熵：每个噪声样本预测的熵的均值

**eDALD（熵增强版）**额外加入单样本熵项：
$$\text{eDALD}(x) = I(\hat{Y}; Z | x) + H(\hat{Y} | z^{(0)}, x)$$

额外的熵项捕捉预测置信度，弥补纯分歧度方法对低置信区域不敏感的问题。

### 损失函数 / 训练策略

分割头（MLP）使用标准交叉熵损失训练：
$$\theta^* = \arg\min_\theta -\frac{1}{|\mathcal{L}|}\sum_{(x,y)\in\mathcal{L}} \log \hat{p}_\theta(y | x, f)$$

训练细节：Adam优化器，学习率 $10^{-3}$，batch size 5，若50次迭代损失无改善且像素精度>95%则早停。

## 实验关键数据

### 主实验

**表2：低预算AL方法mIoU(%)比较（10轮后）**

| 骨干网络 | 方法 | CamVid | ADE-Bed | Cityscapes | Pascal-C | 平均 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepLabV3 | PixelPick | 29.93 | 8.35 | 26.82 | 26.28 | 22.85 |
| DeepLabV3 | Didari et al. | 22.47 | 8.66 | 19.85 | 28.15 | 19.78 |
| DDPM | Random | 25.91 | 17.83 | 27.13 | 41.70 | 28.14 |
| DDPM | Margin | 31.27 | 30.03 | 32.23 | 45.11 | 34.66 |
| DDPM | eDALD (单阶段) | 25.14 | 23.06 | 29.44 | 43.05 | 30.17 |
| DDPM | **2-Stage eDALD** | **36.12** | **31.12** | **33.34** | **47.98** | **37.14** |

关键发现：
- DDPM骨干比DeepLabV3平均提升 **11.81** mIoU
- 两阶段eDALD比最强单阶段方法（Margin）平均高 **2.48** mIoU
- 比PixelPick平均高 **14.29** mIoU

**表1：覆盖性过滤对不确定性采样的影响（CamVid）**

| 不确定性方法 | 单阶段 | Herding→不确定性 | 增益 |
|:---:|:---:|:---:|:---:|
| Entropy | 25.26 | 30.77 | +5.51 (+21.8%) |
| eBALD | 25.96 | 32.12 | +6.16 (+23.7%) |
| eDALD | 25.14 | **36.12** | **+10.98 (+43.7%)** |
| BALD | 24.59 | 22.79 | -1.80 (-7.3%) |
| DALD | 23.81 | 21.05 | -2.76 (-11.6%) |

### 消融实验

核心消融：两阶段 vs 单阶段
- 纯分歧度方法（BALD、DALD）在加入MaxHerding后反而下降，说明纯分歧度在已保证多样性后会过度强调噪声区域
- 熵增强版（eBALD、eDALD）获得巨大提升，说明分歧度+置信度互补效果显著
- eDALD的两阶段增益（+43.7%）远超其他方法，证明扩散特征的分歧度与表示性过滤的互补性最强

### 关键发现

1. **接近全监督性能**：两阶段eDALD仅用 0.003%~0.007% 的像素标注，就能在21-47轮内达到全监督90%的mIoU
2. **扩散骨干显著优于传统骨干**：选择DDPM特征而非DeepLabV3特征，平均mIoU提升11.81
3. **覆盖性→不确定性的解耦优于混合策略**：先保证多样性再做不确定性筛选，效果远好于反过来或混合

## 亮点与洞察

1. **极低预算设定的实际意义**：每张图平均仅1个像素标签，更贴近真实标注预算约束
2. **扩散模型的新用途**：不用于生成，而是利用其多时间步特征做不确定性估计，思路新颖
3. **两阶段解耦设计巧妙**：覆盖性和不确定性各司其职，1+1>2的效果
4. **eDALD中额外熵项的作用**：仅需一个独立噪声样本，计算开销极小但效果显著

## 局限与展望

1. DALD基于扩散骨干的随机噪声注入，仅适用于扩散模型，不具通用性
2. MaxHerding阶段对每张图像的所有像素计算成对相似度，计算成本较高
3. 实验分辨率限制为256×256，高分辨率场景的适用性未验证
4. 仅在像素级AL上评估，未与区域级或图像级方法做效率-精度权衡比较

## 相关工作与启发

- **PixelPick** [Shin et al., 2021]：像素级AL先驱，使用Margin采样+DeepLabV3，本文主要基线
- **MaxHerding** [Bae et al., 2024]：广义覆盖方法，本文在阶段1中使用
- **BALD** [Houlsby et al., 2011]：贝叶斯主动学习分歧度，eDALD的理论基础
- **Baranchuk et al., 2022**：证明扩散模型的多时间步特征有利于半监督分割

## 评分

- **新颖性**: ★★★★☆ — 两阶段流程和eDALD设计新颖，扩散特征做AL的思路别出心裁
- **技术深度**: ★★★★☆ — 信息论框架清晰，各设计选择有理论支撑
- **实验充分性**: ★★★★★ — 4个数据集全面评估，消融实验详尽，定性分析丰富
- **写作质量**: ★★★★☆ — 结构清晰，图表质量高
- **实用性**: ★★★★☆ — 代码开源，极低预算场景有实际应用价值

<!-- RELATED:START -->

## 相关论文

- [OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation](omnisegmentor_a_flexible_multi-modal_learning_framework_for_semantic_segmentatio.md)
- [Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective](towards_robust_pseudo-label_learning_in_semantic_segmentation_an_encoding_perspe.md)
- [Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers](seg4diff_unveiling_open-vocabulary_segmentation_in_text-to-image_diffusion_trans.md)
- [DDB: Diffusion Driven Balancing to Address Spurious Correlations](../../ICCV2025/segmentation/ddb_diffusion_driven_balancing_to_address_spurious_correlations.md)
- [A²LC: Active and Automated Label Correction for Semantic Segmentation](../../AAAI2026/segmentation/a2lc_active_and_automated_label_correction_for_semantic_segm.md)

<!-- RELATED:END -->
