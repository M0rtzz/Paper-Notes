---
title: >-
  [论文解读] Target Refocusing via Attention Redistribution for Open-Vocabulary Semantic Segmentation: An Explainability Perspective
description: >-
  [AAAI 2026][图像分割][开放词汇语义分割] 从可解释性角度系统研究CLIP内部机制，发现"分心"现象（distraction）——CLIP在深层将大量注意力资源分配给与目标无关的token，提出免训练的RF-CLIP方法通过注意力重分配将被分散的资源重新聚焦到目标区域，在8个基准上达到SOTA性能并保持推理高效。
tags:
  - AAAI 2026
  - 图像分割
  - 开放词汇语义分割
  - CLIP
  - 注意力重分配
  - 分心现象
  - 免训练
---

# Target Refocusing via Attention Redistribution for Open-Vocabulary Semantic Segmentation: An Explainability Perspective

**会议**: AAAI 2026  
**arXiv**: [2511.16170](https://arxiv.org/abs/2511.16170)  
**代码**: [github.com/liblacklucy/RF-CLIP](https://github.com/liblacklucy/RF-CLIP)  
**领域**: 分割  
**关键词**: 开放词汇语义分割, CLIP, 注意力重分配, 分心现象, 免训练

## 一句话总结

从可解释性角度系统研究CLIP内部机制，发现"分心"现象（distraction）——CLIP在深层将大量注意力资源分配给与目标无关的token，提出免训练的RF-CLIP方法通过注意力重分配将被分散的资源重新聚焦到目标区域，在8个基准上达到SOTA性能并保持推理高效。

## 研究背景与动机

开放词汇语义分割（OVSS）通过像素级视觉-语言对齐将类别提示与对应像素关联。现有方法分为三大范式：

**联合微调**：同时微调CLIP和分割组件

**预微调**：通过细粒度对比学习重新训练CLIP

**免训练适配**：仅调制CLIP最后残差注意力层或集成视觉基础模型（VFM）

然而，这些方法**很少从可解释性角度**研究CLIP在密集预测中的性能边界，也未探索其固有的层间空间不对齐的根源。

作者的系统分析揭示了一个关键现象——**"分心"现象**：

**浅层**（1-2层）：注意力主要集中于查询相关的token，空间一致性强

**深层**（7-12层）：出现大量与目标查询无关的高注意力token（分心token），逐步削弱目标区域的显著性
3. 这些分心token在不同查询点间**占据相同空间位置**，表明其与所有查询都有虚假的高相关性
4. 在自注意力图中表现为**明显的垂直条纹**

进一步分析发现，分心token源于**特定维度的过度激活**——CLIP在某些通道（如维度4, 162, 474等）固有地产生巨大嵌入权重，这是一种与数据无关的固有属性。filtrate这些token可显著改善OVSS性能。

## 方法详解

### 整体框架

RF-CLIP是一个免训练的注意力调制方法，模拟人类"分心→重新聚焦"行为，逐层校正CLIP的空间不对齐。每层校正包含三个步骤：
1. **分心定位**（Distractor Localization）：识别消耗大量注意力的分心token
2. **失焦定位**（Defocus Localization）：检测注意力不足的目标token
3. **权重重分配**（Weight Redistribution）：将注意力从分心token转移到失焦目标token

### 关键设计

#### 1. **分心维度与分心token的发现与定位**

通过计算所有层视觉密集嵌入的平均值 $\bar{f} = \frac{1}{L}\sum_{l=1}^{L}\frac{f^l}{\sum_{j=1}^d f^l[:,j]}$，发现三个大规模OVSS基准数据集**在相同维度上呈现一致的权重分布峰值**（如ViT-B/16的维度4, 162, 474等），定义为分心维度 $\mathcal{D}_{dis}$。

定位分心token：对第 $l$ 层的第 $i$ 个token，计算其在分心维度上的最大嵌入权重：

$$\phi_i^l = \max_{j \in \mathcal{D}_{dis}} \frac{f_i^l[j]}{\sum_{k=1}^d f_i^l[k]}$$

满足 $\phi_i^l > \tau$ 的token被识别为分心token，阈值 $\tau = 5/d$。

设计动机：实验证实在分心维度上嵌入权重巨大的token，在自注意力计算中不可避免地发展为分心token。分心token的注意力权重随 $\phi_i$ 增长呈**指数增长**关系。

#### 2. **失焦token定位**

将失焦token视为前景实例，将定位问题形式化为二分图割问题。使用key-key注意力 $\text{Attn}_{kk}^l$ 作为相似度矩阵进行谱聚类，最小化归一化割能量：

$$\bm{y}_1^l = \arg\min_{\bm{y}^{l\top}\bm{D1}=0} \frac{\bm{y}^{l\top}(\bm{D}^l - \text{Attn}_{kk}^l)\bm{y}^l}{\bm{y}^{l\top}\bm{D}^l\bm{y}^l}$$

其中 $\bm{y}_1^l$ 为Fiedler向量（广义特征系统的第二小特征值对应的特征向量），满足 $\bm{y}_1^l[i] > \frac{1}{N}\sum_{j=1}^N \bm{y}_1^l[j]$ 的token为失焦token。

设计动机：图割能自然地将图像分为前景和背景两组，对各种场景都具有鲁棒性，无需额外标注或训练。

#### 3. **权重重分配**

分为两个互补机制：

**注意力权重重分配**：先缩减分心token的注意力权重，将减量保留为重分配预算 $\Omega$：

$$\text{Attn}_{qk}^{l,h}[i,j] \leftarrow (1-\beta) \cdot \text{Attn}_{qk}^{l,h}[i,j], \quad \forall j \in \mathcal{T}_{dis}$$

$$\Omega[i] = \beta \cdot \sum_{j \in \mathcal{T}_{dis}} \text{Attn}_{qk}^{l,h}[i,j]$$

然后按原始注意力权重的比例分配给失焦token：

$$\text{Attn}_{qk}^{l,h}[i,j] \leftarrow \text{Attn}_{qk}^{l,h}[i,j] + \Omega[i] \cdot \rho[i,j], \quad \forall j \in \mathcal{T}_{def}$$

其中 $\beta = 0.7$ 为衰减因子。此过程保持列归一化，保留原始注意力分布，有效防止模型崩溃。

**嵌入权重重分配**：对分心token在分心维度上使用3×3邻域平均替换：

$$f_i^l[j] = \frac{1}{8} \cdot \sum_{\hat{i} \in \mathcal{O}_i} f_{\hat{i}}^l[j], \quad \forall j \in \mathcal{D}_{dis}, i \in \mathcal{T}_{dis}$$

仅调整分心维度的嵌入，不破坏正常维度的分布。

**密集预测**：校正后用层平均注意力 $\overline{\text{Attn}}_{kk} = \frac{1}{L}\sum_{l=1}^L \text{Attn}_{kk}^l$ 替换最后一层的 $\text{Attn}_{qk}^L$。

### 损失函数 / 训练策略

RF-CLIP是**完全免训练**的方法，无需任何训练或微调。所有操作直接在CLIP的推理过程中执行，对注意力机制进行逐层调制。

## 实验关键数据

### 主实验

基于 CLIP ViT-B/16，在8个标准基准上的mIoU (%)：

| 方法 | 额外VFM | VOC21 | Context60 | COCO-Obj | VOC20 | Context59 | COCO-Stuff | Cityscapes | ADE20K | 平均 |
|------|---------|-------|-----------|----------|-------|-----------|------------|------------|--------|------|
| ProxyCLIP | DINO | 59.1 | 35.2 | 36.2 | 78.2 | 38.8 | 26.2 | 38.1 | 19.6 | 41.4 |
| CASS | DINO | 65.8 | 36.7 | 37.8 | 87.8 | 40.2 | 26.7 | 39.4 | 20.4 | 44.4 |
| SC-CLIP | ✗ | 64.6 | 36.8 | 37.7 | 84.3 | 40.1 | 26.6 | 41.0 | 20.1 | 43.9 |
| **RF-CLIP** | **✗** | **64.8** | 36.4 | **37.9** | **87.0** | 39.8 | 26.3 | **41.3** | **20.4** | **44.2** |
| RF-CLIP+PAMR | ✗ | **67.2** | **37.9** | **39.1** | 87.0 | **41.4** | **27.5** | **43.0** | **21.0** | **45.5** |

RF-CLIP不使用任何额外VFM即超越使用DINO的ProxyCLIP (+2.8mIoU) 和CASS，平均mIoU与同基线方法相比提升1.6%。

### 消融实验

| 配置 | VOC21 | COCO-Stuff | Cityscapes | ADE20K | 平均 | 说明 |
|------|-------|------------|------------|--------|------|------|
| 基线 | 59.1 | 23.6 | 32.1 | 16.9 | 32.9 | 层平均kk注意力 |
| +随机均值滤波 | 58.8 | 21.4 | 31.6 | 14.7 | 31.6 | 随机token滤波，性能下降 |
| +分心定位+均值滤波 | 60.3 | 24.4 | 33.6 | 17.5 | 34.0 | 分心感知滤波，+1.1% |
| +注意力重分配 | 61.5 | 24.8 | 35.3 | 18.3 | 35.0 | +2.1% |
| +嵌入重分配 | 62.1 | 25.2 | 36.7 | 18.9 | 35.7 | +2.8% |
| +两种重分配 | 63.2 | 25.4 | 38.5 | 19.3 | 36.6 | +3.7% |
| **+失焦定位** | **64.8** | **26.3** | **41.3** | **20.4** | **38.2** | **+5.3%** |

**效率分析**（VOC21基准）：

| 模型 | FLOPs(G) | Params(M) | Speed(FPS) | mIoU(%) |
|------|----------|-----------|------------|---------|
| 基线 | 16.7 | 149.6 | 12.7 | 58.1 |
| ProxyCLIP | 34.1 | 235.4 | 6.1 | 59.1 |
| **RF-CLIP** | **17.1** | **149.6** | **12.0** | **64.8** |

RF-CLIP推理速度是ProxyCLIP的2倍，mIoU高出5.7%。

**抑制策略比较**：

| 策略 | VOC21 | COCO-Stuff | Cityscapes | ADE20K |
|------|-------|------------|------------|--------|
| 基线 | 58.1 | 23.0 | 31.1 | 16.3 |
| -∞掩码 | 3.5 | 0.1 | 2.0 | 0.1 |
| 低通滤波 | 7.9 | 1.1 | 6.2 | 1.4 |
| 均值滤波 | 59.3 | 24.0 | 35.4 | 18.2 |
| 中值滤波 | 58.6 | 23.7 | 34.5 | 17.6 |

### 关键发现

1. 直接消除分心token（-∞掩码、低通滤波）导致性能崩溃，因为破坏了CLIP高维空间的拓扑结构
2. 分心token应保持与相邻区域的空间一致性，因此均值/中值滤波有效
3. 将注意力资源分配给失焦token比分配给所有非分心token或[CLS] token都更有效
4. 3×3邻域在嵌入重分配中最优，更大邻域反而降低性能，说明分心token集中在高频区域
5. 阈值 $\tau = 5/d$ 在所有基准上取得最优，低阈值（高误报率）的性能下降远大于高阈值

## 亮点与洞察

1. **可解释性驱动的方法设计**：从CLIP内部机制的系统分析出发，发现"分心"现象，再设计针对性解决方案。这种"先理解后设计"的范式极具启发性
2. **免训练即达SOTA**：不引入任何额外模型或训练，仅通过调制CLIP自身注意力机制就超越使用DINO等额外VFM的方法
3. **分心维度的数据无关性**：相同的分心维度在不同数据集上一致出现，说明这是CLIP预训练过程的固有特性
4. **精细的实验控制**：随机token滤波 vs 分心token滤波的对照实验设计精巧，令人信服地证明了分心感知处理的重要性
5. **注意力资源的"守恒"设计**：重分配保持列归一化，按原始比例分配，兼顾性能提升和防止崩溃

## 局限与展望

1. 阈值和分心维度需按CLIP架构（ViT-B/16 vs ViT-L/14）分别设定，泛化性有限
2. 谱聚类的特征值分解增加额外计算，虽然整体仍比引入VFM高效
3. ViT-L/14中分心token的识别需额外引入注意力权重条件，比ViT-B/16复杂
4. 对极度复杂场景（多目标重叠）的二分图割可能过于简化

## 相关工作与启发

- **Registers（ICLR 2024）**：也发现ViT特征图中的高范数token伪影，但归因于低信息背景区域
- **CLIPtrase / DeCLIP**：认为分心token是[CLS]的代理，但本文通过实验证明资源不仅来自[CLS]还来自前景token
- **ProxyCLIP / CASS**：用DINO替换CLIP注意力，而RF-CLIP证明直接修复CLIP自身更高效
- **SCLIP / ClearCLIP / NACLIP**：仅修改最后一层注意力矩阵，忽略中间层空间不对齐
- 分心现象的发现可能对CLIP在其他密集预测任务（深度估计、实例分割）中的应用有启示

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （从可解释性角度发现分心现象并提出注意力重分配，非常original）
- 实验充分度: ⭐⭐⭐⭐⭐ （8个基准，详尽消融，效率分析，多种对照实验）
- 写作质量: ⭐⭐⭐⭐⭐ （逻辑清晰，从现象发现到方法设计层层递进，图表丰富）
- 价值: ⭐⭐⭐⭐⭐ （免训练达SOTA，揭示CLIP内部机制的新洞察）

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] InfoCLIP: Bridging Vision-Language Pretraining and Open-Vocabulary Semantic Segmentation via Information-Theoretic Alignment Transfer](infoclip_bridging_vision-language_pretraining_and_open-vocab.md)
- [\[CVPR 2026\] PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)
- [\[CVPR 2026\] Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)
- [\[CVPR 2025\] Effective SAM Combination for Open-Vocabulary Semantic Segmentation](../../CVPR2025/segmentation/effective_sam_combination_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation](../../CVPR2025/segmentation/semantic_library_adaptation_lora_retrieval_and_fusion_for_open-vocabulary_semant.md)

<!-- RELATED:END -->
