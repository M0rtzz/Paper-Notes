---
title: >-
  [论文解读] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation
description: >-
  [ICCV 2025][图像分割][开放词汇语义分割] 提出 FreeCP，一种无需训练的类别净化框架，通过冗余净化和歧义净化两阶段策略，解决开放词汇语义分割中因过完备词汇表导致的类别冗余和视觉-语言歧义问题，作为即插即用模块在八个基准上显著提升现有方法性能。
tags:
  - ICCV 2025
  - 图像分割
  - 开放词汇语义分割
  - 训练免费
  - 类别净化
  - CLIP
  - 类别冗余
  - 视觉-语言歧义
---

# Training-Free Class Purification for Open-Vocabulary Semantic Segmentation

**会议**: ICCV 2025  
**arXiv**: [2508.00557](https://arxiv.org/abs/2508.00557)  
**代码**: [GitHub](https://github.com/chenqi1126/FreeCP)  
**领域**: segmentation  
**关键词**: 开放词汇语义分割, 训练免费, 类别净化, CLIP, 类别冗余, 视觉-语言歧义

## 一句话总结

提出 FreeCP，一种无需训练的类别净化框架，通过冗余净化和歧义净化两阶段策略，解决开放词汇语义分割中因过完备词汇表导致的类别冗余和视觉-语言歧义问题，作为即插即用模块在八个基准上显著提升现有方法性能。

## 研究背景与动机

开放词汇语义分割（OVSS）旨在根据任意文本类别描述对图像进行像素级分割。大规模视觉-语言模型（如 CLIP）因其出色的新类别识别能力被广泛应用于该任务。现有方法分为需要训练和免训练两大类，免训练方法因零计算开销受到关注。

然而，现有免训练方法面临两个被忽视的核心问题：

**类别冗余（Class Redundancy）**：推理时需要提供包含大量潜在类别的词汇表，但当前测试图像中实际只存在少数类别。不存在的类别会产生假阳性激活，干扰分割结果。例如在一张只有天空、草地和房屋的图像中，"海洋"和"门"等不存在的类别也会产生响应

**视觉-语言歧义（Visual-Language Ambiguity）**：词汇表中存在语义相近的类别（如"树叶/灌木/树"、"河流/水"），它们在同一视觉区域产生高度重叠的激活图，导致分类混淆

作者通过一组关键实验验证了这些问题的严重性：当仅使用真实存在的类别（GT 类别）进行预测时，分割精度显著提升——VOC21 从 59.4% 提升到 72.0%，ADE 从 15.6% 提升到 37.5%。这说明冗余和歧义类别是性能瓶颈的关键因素。

进一步分析发现，利用 CLIP 自注意力矩阵对 CAM 进行 affinity-based refinement 后，真正存在的类别激活图在空间上保持高度一致，而冗余类别则发生显著形变；同时歧义类别之间的 refined 激活图呈现高度空间重叠。这一发现为后续的类别净化提供了理论依据。

## 方法详解

### 整体框架

FreeCP 基于 CLIP ViT 模型构建，包括三个阶段：
1. 利用 CLIP 图像编码器和文本编码器分别提取 patch tokens $\mathbf{F}^p$ 和文本表示 $\mathbf{T}$
2. 计算 image-text affinity（CAM）和 image self-affinity（自注意力矩阵），生成类别激活图 $\mathbf{M}$ 及其 refined 版本 $\tilde{\mathbf{M}}$
3. 依次执行冗余净化和歧义净化，最终用 argmax 生成分割预测

### 关键设计一：CAM 与 Affinity Refinement

类别激活图通过图像-文本余弦相似度的 softmax 归一化计算：

$$\mathbf{M}_j = \text{Reshape}\left(\frac{\exp(\text{Sim}(\mathbf{F}^p, \mathbf{T}_j))}{\sum_j \exp(\text{Sim}(\mathbf{F}^p, \mathbf{T}_j))}\right)$$

图像自亲和矩阵 $SA$ 通过多层自注意力矩阵均值获得：

$$SA = \frac{1}{L}\sum_{l}^{L}\psi(A_l)$$

refined 激活图为：$\tilde{\mathbf{M}}_i = \mathbf{M}_i \times SA$

### 关键设计二：空间一致性度量

引入 IoU 作为空间一致性（Spatial Consistency, SC）的粗粒度度量：

$$\text{SC}(\mathbf{X}, \mathbf{Y}) = \frac{\sum[\mathbf{X} \cdot \mathbf{Y}]}{\sum[\mathbf{X} + \mathbf{Y} - \mathbf{X} \cdot \mathbf{Y}]}$$

这一指标同时用于冗余判断（intra-class SC）和歧义检测（inter-class SC）。

### 关键设计三：冗余净化（Redundancy Purification）

比较每个类别 refinement 前后激活图的 intra-class SC：

$$S_i = \text{SC}(\mathbf{M}_i, \tilde{\mathbf{M}}_i)$$

若 $S_i < T_{rp}$（预设阈值），则认为该类别是冗余的，从候选集中移除。直觉是：真正存在的类别在 refinement 后空间分布保持一致，而冗余类别由于缺乏真实视觉对应，refinement 会引入大量不相关的错误响应。

### 关键设计四：歧义净化（Ambiguity Purification）

在冗余净化后的类别集合 $K'$ 上，计算所有类别对之间的 inter-class SC：

$$P_{i,j} = \text{SC}(\tilde{\mathbf{M}}_i, \tilde{\mathbf{M}}_j)$$

通过阈值 $T_{ap}$ 二值化后，使用深度优先搜索（DFS）提取连通类别组，形成歧义组。对每个歧义组：
1. 平均该组类别的激活图，定位高响应区域并提取 bounding box
2. 从原图裁剪歧义区域，输入 CLIP 图像编码器提取视觉特征 $\hat{\mathbf{F}}^c$
3. 利用 LLM 预先生成各类别的细粒度文本描述 $\hat{\mathbf{T}}_k$
4. 通过余弦相似度确定局部区域的最终类别：$k^* = \arg\max_k \text{Sim}(\hat{\mathbf{F}}^c, \hat{\mathbf{T}}_k)$

### 损失函数

FreeCP 是纯推理阶段的方法，无需训练，因此不涉及损失函数设计。

## 实验

### 主实验

在 8 个基准上与现有免训练 OVSS 方法对比，FreeCP 作为即插即用模块显著提升已有方法：

| 方法 | VOC21 | PC60 | Object | VOC20 | City | PC59 | ADE | Stuff | Avg. |
|------|-------|------|--------|-------|------|------|-----|-------|------|
| SCLIP | 59.1 | 30.4 | 30.5 | 80.4 | 32.2 | 34.2 | 16.1 | 22.4 | 38.2 |
| SCLIP + FreeCP | **65.8** | **35.3** | **37.2** | **84.3** | **33.3** | **38.0** | **18.4** | **24.9** | **42.1** |
| ClearCLIP | 51.8 | 32.6 | 33.0 | 80.9 | 30.0 | 35.9 | 16.7 | 23.9 | 38.1 |
| ClearCLIP + FreeCP | **64.5** | **35.7** | **36.9** | **81.5** | **34.4** | **39.3** | **18.9** | **26.1** | **42.2** |
| MaskCLIP | 43.4 | 23.2 | 20.6 | 74.9 | 24.9 | 26.4 | 11.9 | 16.7 | 30.3 |
| MaskCLIP + FreeCP | **64.4** | **34.7** | **36.2** | **84.1** | **32.5** | **36.6** | **17.6** | **23.3** | **41.2** |

FreeCP 将 MaskCLIP 的平均 mIoU 提升了 **+10.9%**，SCLIP 提升 **+3.9%**，ClearCLIP 提升 **+4.1%**。

### 消融实验：类别净化策略

| 方法 | VOC21 | PC60 | Object | City | ADE | Stuff |
|------|-------|------|--------|------|-----|-------|
| Baseline | 59.8 | 31.6 | 34.5 | 32.0 | 17.2 | 23.2 |
| + Refine（无净化） | 27.5 | 21.1 | 11.9 | 26.0 | 9.1 | 14.3 |
| + RP（仅冗余净化） | 65.8 | 35.1 | 37.2 | 33.2 | 17.8 | 24.1 |
| + AP（仅歧义净化） | 37.7 | 26.1 | 13.6 | 24.0 | 10.8 | 15.0 |
| + RP-AP（FreeCP） | **65.8** | **35.3** | **37.2** | **33.3** | **18.4** | **24.9** |

关键发现：
- 直接 refinement 而不净化会导致性能**大幅下降**（VOC21: 59.8→27.5）
- RP 是性能提升的主要来源，AP 在 RP 之后进一步提升（尤其在细粒度类别多的数据集上）
- AP 必须在 RP 之后执行，AP-RP 顺序会遗留冗余类别的干扰

### 关键发现

1. **普适性强**：FreeCP 对不同初始性能水平的方法均有效，MaskCLIP 初始性能最低但提升最大
2. **文本描述的影响**：不同 LLM 生成的细粒度描述对结果有轻微影响，Vicuna-13b 略优于 GPT-3.5，但 FreeCP 在所有描述选择下均优于 baseline
3. **无需后处理**：FreeCP 不依赖 denseCRF 或 PAMR 等后处理方法即可取得 SOTA

## 亮点与洞察

1. **新颖的问题视角**：首次系统性地分析了开放词汇分割中类别冗余和视觉-语言歧义对性能的影响，通过 GT vocabulary 对比实验提供了令人信服的证据
2. **优雅的解决方案**：利用 CAM refinement 前后的空间一致性变化来区分真实类别和冗余/歧义类别，直觉清晰
3. **即插即用设计**：无需训练，可直接集成到任何基于 CLIP 的免训练 OVSS 方法中
4. **阈值自适应**：根据数据集语义复杂度调整阈值，体现了对不同场景的灵活性

## 局限性

1. 阈值 $T_{rp}$ 和 $T_{ap}$ 需要基于数据集先验知识手动设置，缺乏自适应策略
2. 歧义消解依赖 LLM 生成细粒度描述的质量，对 LLM 有隐式依赖
3. 在类别数极少的简单场景下提升有限（如 Cityscapes 仅 +1.1%）
4. CAM 和 affinity refinement 增加了推理计算开销

## 相关工作

- **免训练 OVSS**：MaskCLIP、SCLIP、GEM、ClearCLIP 等通过修改 CLIP 自注意力实现像素级分割；CaR 渐进过滤无关文本
- **基于原型的方法**：ReCo、OVDiff、FreeDA 利用生成模型合成视觉参考
- **弱监督 CAM**：AffinityNet 等利用亲和矩阵增强 CAM

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 类别净化角度新颖，但冗余过滤和歧义消解的具体设计较直接
- **技术质量**: ⭐⭐⭐⭐ — 实验全面，消融充分，但阈值设置依赖先验
- **实用性**: ⭐⭐⭐⭐⭐ — 训练免费+即插即用，实用价值很高
- **写作质量**: ⭐⭐⭐⭐ — 动机和方法逻辑清晰，可视化有说服力

<!-- RELATED:START -->

## 相关论文

- [FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation](floss_free_lunch_in_openvocabulary_semantic_segmentation.md)
- [PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)
- [Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/looking_beyond_the_window_global-local_aligned_clip_for_training-free_open-vocab.md)
- [Personalized OVSS: Understanding Personal Concept in Open-Vocabulary Semantic Segmentation](understanding_personal_concept_in_open-vocabulary_semantic_segmentation.md)
- [Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)

<!-- RELATED:END -->
