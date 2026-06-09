---
title: >-
  [论文解读] AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation
description: >-
  [ICLR 2026][语义分割][图像分割] 提出对齐感知遮蔽学习(AML)策略，通过量化视觉-语言 patch 级对齐度并过滤低对齐像素，让 RIS 模型在训练时聚焦可靠区域，无需架构改动即在 RefCOCO 全部 8 个 split 上达到 SOTA。
tags:
  - "ICLR 2026"
  - "语义分割"
  - "图像分割"
  - "视觉语言"
  - "masked learning"
  - "跨模态"
---

# AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation

**会议**: ICLR 2026  
**arXiv**: [2602.22740](https://arxiv.org/abs/2602.22740)  
**代码**: [GitHub](https://github.com/pipashu1/AMLRIS)  
**领域**: 图像分割  
**关键词**: referring image segmentation, vision-language alignment, masked learning, cross-modal similarity  

## 一句话总结
提出对齐感知遮蔽学习(AML)策略，通过量化视觉-语言 patch 级对齐度并过滤低对齐像素，让 RIS 模型在训练时聚焦可靠区域，无需架构改动即在 RefCOCO 全部 8 个 split 上达到 SOTA。

## 背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：1. 指称图像分割(RIS)需要根据自然语言表达精准分割图像中的目标对象，依赖跨模态精细对齐
2. RIS 训练中每个样本通常仅有一个标注目标，监督信号稀疏
3. 理解"离人最近的长颈鹿"等表达需要依赖视觉上下文中其他物体的空间关系
4. 现有方法(LAVT/CARIS/DETRIS)通过复杂融合模块增强对齐，但对全部像素施加损失会引入不可靠梯度
5. 在密集损失下，模型容易过拟合到与表达无关的区域
6. 数据增强方法(翻转/颜色抖动)易破坏指称表达的语义一致性

## 方法详解

### 整体框架

AMLRIS 不改动任何 RIS 模型结构，只在训练时插入一个"对齐感知遮蔽"前置步骤：第一阶段把图像和指称表达喂给模型，前向计算出 patch 级的视觉-语言对齐图，并据此生成一张遮蔽；第二阶段把弱对齐区域被 zero-out 后的遮蔽图像再喂回同一套共享参数的模型，用标准分割损失正常训练。换言之，模型始终只在"看得懂、对得上"的像素上回传梯度，把与表达无关的误导性区域从监督信号里剔除。

### 关键设计

**1. PatchMax 匹配评估 (PMME)：用随机投影量化每个 patch 与表达的对齐度。** RIS 的核心难点是缺少 patch 级的对齐标签，无法直接知道哪些像素真正回应了指称表达。PMME 把视觉特征 $V$ 和文本特征 $T$ 先做 $\ell_2$ 归一化，再用两个随机高斯矩阵 $W_i, W_t$ 把它们投影到同一个 $D_a=2048$ 维公共空间。这里用随机投影而非可学习层是关键——由 Johnson-Lindenstrauss 引理保证随机投影能近似保持跨模态内积，因此对齐度量有数学依据、无需额外训练且不引入偏置。投影后计算相似度并按列归一化 $S_{norm} = \text{SoftMax}(V'T'^{\top})$，每个 patch 不取平均而是取它与最强匹配 token 的最大相似度（即"PatchMax"），因为一个像素只要与表达中某个词高度对应就应被判为强对齐，取最大比取均值更能反映这种局部命中。

**2. 对齐感知过滤遮蔽 (AFM)：把弱对齐 patch 翻译成图像层面的遮蔽。** 有了 patch 级对齐分数后，需要把它落到像素并决定遮蔽哪些区域。AFM 先把 patch 级相似度双线性上采样到像素级，再用阈值 $\tau=0.4$ 把低于阈值的像素标记为弱对齐；为避免过度过滤把潜在有用区域全删掉，弱对齐像素并非全部遮蔽，而是随机保留 $1-\rho$（$\rho=0.25$）比例。最后以 $32\times32$ 的 block 为单位聚合：只要块内任一像素弱对齐就整块遮蔽，并对输入图像直接 zero-out。按块而非按像素遮蔽是为了让被屏蔽区域成片、避免破碎的椒盐式遮蔽干扰卷积感受野，但这也带来一个权衡——粗粒度块在小目标场景下可能误覆盖目标本身。整个 AML 仅在训练时启用，推理阶段遮蔽步骤被完全跳过，因此不增加任何部署成本。

### 损失函数 / 训练策略

损失仍是标准交叉熵分割损失 $\mathcal{L}_{seg}$，不引入任何额外正则或对齐损失项——AML 的全部作用都体现在"在哪些像素上算这个损失"，而非改写损失本身。两阶段前向共享同一套参数，代价是约 $17.2\%$ 的训练时间和 $4.9\%$ 的显存增加，换来推理时零开销。

## 实验关键数据


### 主实验

| 方法 | RefCOCO val | RefCOCO+ val | RefCOCOg val | Avg mIoU |
|------|-------------|--------------|--------------|----------|
| CARIS* | 76.77 | 69.33 | 68.87 | 71.8 |
| MagNet | 77.43 | 70.10 | 68.53 | 72.1 |
| **AMLRIS** | **77.89** | **71.33** | **69.24** | **72.9** |

- oIoU 指标：RefCOCO val 75.45(+0.80 vs CARIS），RefCOCO+ val 67.37（+1.83)
- 全部 8 个 split 均 SOTA
- 跨数据集鲁棒性：仅在 RefCOCO+ 训练，在 7 种扰动场景下均优于 baseline
- 额外开销：仅增加 4.9% 显存和 17.2% 训练时间，推理无开销

## 实验关键数据

### 主实验（mIoU）

| 方法 | RefCOCO val | testA | testB | RefCOCO+ val | testA | testB | RefCOCOg val | test | Avg |
|------|-------------|-------|-------|--------------|-------|-------|--------------|------|-----|
| LAVT | 74.46 | 76.89 | 70.94 | 65.81 | 70.97 | 59.23 | 63.34 | 63.62 | 68.0 |
| CGFormer | 76.93 | 78.70 | 73.32 | 68.56 | 73.76 | 61.72 | 67.57 | 67.83 | 71.1 |
| CARIS* | 76.77 | 79.03 | 74.56 | 69.33 | 74.51 | 62.69 | 68.87 | 68.51 | 71.8 |
| MagNet | 77.43 | 79.43 | 74.11 | 70.10 | 74.50 | 63.59 | 68.53 | 69.15 | 72.1 |
| **AMLRIS** | **77.89** | **79.53** | **74.99** | **71.33** | **75.61** | **64.61** | **69.24** | **69.73** | **72.9** |

### 消融实验

| 配置 | RefCOCO val mIoU | 说明 |
|------|-----------------|------|
| CARIS 基线 | 76.77 | 无遮蔽 |
| +随机遮蔽（Random Mask） | 76.92 | 随机遮蔽效果微弱 |
| +PMME+AFM (完整 AML) | **77.89** | 对齐感知遮蔽有效 |
| AML 集成到 DETRIS | 75.64→76.12 | 跨架构一致提升 |
| AML 集成到 ReLA | +0.5-1.0 | 同样有效 |

### 跨数据集鲁棒性

| 扰动场景 | CARIS baseline | AMLRIS |
|---------|---------------|--------|
| 标准评估 | 69.33 | **71.33** |
| 遮挡 | 65.1 | **68.4** |
| 噪声 | 64.8 | **67.9** |
| 模糊 | 66.2 | **69.1** |
| 色彩变换 | 67.5 | **70.2** |

### 关键发现
- 全部 8 个 split 均达到 SOTA，平均 mIoU 72.9（+0.8 vs MagNet）
- oIoU 指标同样全面最优，RefCOCO+ val 达 67.37（+1.83 vs CARIS）
- 随机遮蔽几乎无效（+0.15），证明对齐感知的遮蔽选择是关键
- 在遮挡/噪声等扰动场景下优势更加明显（+3.1-3.3），表明模型学到了更鲁棒的对齐特征
- 额外开销很小：仅增加 4.9% 显存和 17.2% 训练时间，推理时完全无开销（遮蔽阶段被跳过）
- 可无缝集成到 DETRIS/CARIS/ReLA 等多种 RIS 框架

## 亮点与洞察
- **即插即用训练策略**：不修改模型架构、不增加推理成本——纯训练阶段的改进，部署零代价
- **理论保证**：用 Johnson-Lindenstrauss 引理严格证明随机投影保持跨模态内积，对齐度量有数学依据
- **反直觉有效性**：训练时从未见过完整图像（总有部分被遮蔽），但推理时在完整图像上表现更好——说明过滤弱对齐区域确实消除了误导性梯度
- **PatchMax 匹配策略**：每个 patch 取与最强匹配 token 的相似度，比平均匹配更能反映局部对齐质量

## 局限与展望
- 阈值 $\tau=0.4$ 和 dropout 比例 $\rho=0.25$ 需要手动调节，不同数据集可能需要不同设置
- 随机投影的对齐度量基于初始特征相似度，可能遗漏深层语义对齐（训练后期特征空间变化）
- 两阶段前向带来 17.2% 训练时间增加，在大规模数据上可能成为瓶颈
- 仅在 RefCOCO 系列评估，未验证在开放词汇/大规模/更复杂场景下的泛化性
- Block 粒度遮蔽（32×32）可能在小目标场景下误覆盖目标区域

## 相关工作与启发
- **vs CARIS/LAVT/DETRIS**：本文 baseline 和对比方法，通过融合架构提升对齐，但都用全像素损失——AML 从优化信号角度创新
- **vs MaskRIS/NeMo/MagNet**：数据增强路线的 RIS 改进，但仍对全像素施加损失；AML 直接抑制低质量梯度
- **vs CRIS**：基于 CLIP 的像素级适配方法，在预训练特征空间做对齐；AML 可在任意 backbone 上使用

## 评分
- 新颖性: ⭐⭐⭐⭐ 对齐感知遮蔽思路简洁新颖，PatchMax + JL 投影有理论支撑
- 实验充分度: ⭐⭐⭐⭐ 全 split SOTA + 鲁棒性评估 + 跨架构验证 + 消融完整
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，算法伪代码完整
- 价值: ⭐⭐⭐⭐ 通用训练策略，可即插即用到现有 RIS 方法中

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Phrase-Instance Alignment for Generalized Referring Segmentation](../../CVPR2026/segmentation/phrase-instance_alignment_for_generalized_referring_segmentation.md)
- [\[CVPR 2026\] Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation](../../CVPR2026/segmentation/spatio-semantic_expert_routing_architecture_with_mixture-of-experts_for_referrin.md)
- [\[CVPR 2026\] SARMAE: Masked Autoencoder for SAR Representation Learning](../../CVPR2026/segmentation/sarmae_masked_autoencoder_for_sar_representation_learning.md)
- [\[ICLR 2026\] Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval](efficient-sam2_accelerating_sam2_with_object-aware_visual_encoding_and_memory_re.md)
- [\[AAAI 2026\] RS2-SAM2: Customized SAM2 for Referring Remote Sensing Image Segmentation](../../AAAI2026/segmentation/rs2-sam2_customized_sam2_for_referring_remote_sensing_image_segmentation.md)

</div>

<!-- RELATED:END -->
