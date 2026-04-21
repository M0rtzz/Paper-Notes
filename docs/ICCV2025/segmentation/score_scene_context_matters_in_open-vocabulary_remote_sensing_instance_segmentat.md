---
title: >-
  [论文解读] SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation
description: >-
  [ICCV 2025][图像分割][开放词汇分割] 提出 SCORE 框架，利用多粒度场景上下文（区域上下文+全局上下文）增强开放词汇遥感实例分割，通过 Region-Aware Integration 和 Global Context Adaptation 两个模块分别强化视觉和文本表示。
tags:
  - ICCV 2025
  - 图像分割
  - 开放词汇分割
  - 遥感
  - 场景上下文
  - CLIP
  - 实例分割
---

# SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation

**会议**: ICCV 2025  
**arXiv**: [2507.12857](https://arxiv.org/abs/2507.12857)  
**代码**: [https://github.com/HuangShiqi128/SCORE](https://github.com/HuangShiqi128/SCORE)  
**领域**: 遥感图像实例分割 / 开放词汇  
**关键词**: 开放词汇分割, 遥感, 场景上下文, CLIP, 实例分割

## 一句话总结

提出 SCORE 框架，利用多粒度场景上下文（区域上下文+全局上下文）增强开放词汇遥感实例分割，通过 Region-Aware Integration 和 Global Context Adaptation 两个模块分别强化视觉和文本表示。

## 研究背景与动机

- 现有遥感实例分割方法多为闭集预测，无法识别新类别或跨数据集泛化
- 自然图像领域的开放词汇（OV）分割模型直接应用于遥感面临挑战：地貌多样、季节变化、小目标/模糊目标多
- 遥感中的关键观察：**目标与其周围环境高度相关**——船出现在水域附近，车出现在停车场，飞机在机场旁
- 通用 CLIP 的冻结文本嵌入缺乏遥感领域适应性，难以捕获遥感中的高类内变异和分辨率差异
- 遥感领域的 OV 实例分割尚未被研究（已有工作仅限语义分割）

## 方法详解

### 整体框架

SCORE 由三个分支构成：(1) 上下文分支（蓝）——使用遥感 CLIP（RemoteCLIP）提取多粒度场景上下文；(2) 语义分支（黄）——冻结 CLIP 文本编码器生成文本嵌入作为分类器；(3) 实例分支（橙）——冻结通用 CLIP 图像编码器提取特征，通过 Mask2Former 生成类嵌入和掩码提案。三分支通过 RAI 和 GCA 模块交互。

### 关键设计

1. **场景上下文提取（Scene Context Extraction）**: 使用冻结的 RemoteCLIP ViT-L/14 编码输入图像，获取两种粒度的上下文：

    - 全局上下文 $\mathbf{F}^{final}_{CLS} \in \mathbb{R}^{1 \times C}$：[CLS] token 编码全局图像语义
    - 区域上下文 $\mathbf{F}^{final}_{HW} \in \mathbb{R}^{\frac{H}{14} \times \frac{W}{14} \times C}$：patch 嵌入提供空间密集特征

2. **区域感知集成（Region-Aware Integration, RAI）**: 利用目标周围环境信息增强类嵌入

    - **自适应区域形成**：通过可学习膨胀因子 $\delta$ 扩展预测掩码，膨胀核大小 $k = 3 + \text{clamp}(\delta, 0, 10)$，使用 max-pooling 实现
    - **区域上下文提取**：在扩展掩码内对 RemoteCLIP patch 嵌入做加权池化得到 $F_{region}$
    - **区域上下文集成**：通过 $l$ 层 Transformer 将区域上下文注入类嵌入：$\mathbf{V}_{i+1} = \text{TransLayer}_i(\mathbf{V}_i, \lambda \cdot \mathbf{F}_{region})$

3. **全局上下文适配（Global Context Adaptation, GCA）**: 将遥感领域特定的全局视觉上下文注入文本嵌入

    - 以全局上下文 $\mathbf{F}^{final}_{CLS}$ 作为 Query，文本嵌入 $\mathbf{T}$ 作为 Key/Value
    - 通过多头交叉注意力：$\hat{\mathbf{T}} = \text{MHA}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
    - 使用可学习线性投影矩阵 $\mathbf{w}_Q, \mathbf{w}_K, \mathbf{w}_V$ 缓解 RS CLIP 与通用 CLIP 间的对齐问题

### 损失函数 / 训练策略

- 基于 Mask2Former 的标准分割损失
- 训练 50 epochs，batch size 2，单卡 L40S GPU
- 学习率 $1.25 \times 10^{-5}$，AdamW 优化器
- 输入图像 resize 至 $512 \times 512$
- 300 个 object query
- 推理时采用集成策略结合 in-vocabulary 和 out-vocabulary 分类

## 实验关键数据

### 主实验 (表格)

**在 iSAID 上训练，跨数据集评估（mAP %）：**

| 方法 | NWPU | SOTA | FAST | SIOR | 平均 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ODISE | 36.40 | 13.91 | 4.65 | 13.68 | 17.16 |
| FC-CLIP | 60.67 | 33.62 | 11.88 | 26.79 | 33.24 |
| MAFT+ | 35.32 | 6.63 | 5.52 | 9.84 | 14.33 |
| ZoRI | 62.06 | 30.02 | 12.65 | 26.27 | 32.75 |
| **SCORE** | **67.59** | **42.57** | **13.67** | **30.90** | **38.68** |

**在 SIOR 上训练，跨数据集评估（mAP %）：**

| 方法 | NWPU | SOTA | FAST | iSAID | 平均 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| FC-CLIP | 60.69 | 19.84 | 8.67 | 22.24 | 27.86 |
| ZoRI | 59.77 | 20.26 | 9.58 | 23.46 | 28.27 |
| **SCORE** | **69.17** | **23.68** | **10.33** | **27.15** | **32.59** |

### 消融实验 (表格)

**组件消融（iSAID 训练，平均 mAP）：**

| RAI | GCA | NWPU | SOTA | FAST | SIOR | 平均 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ✗ | ✗ | 58.59 | 36.44 | 11.56 | 26.43 | 33.25 |
| ✓ | ✗ | 66.32 | 39.55 | 12.85 | 28.91 | 36.91 |
| ✗ | ✓ | 67.21 | 38.14 | 12.37 | 28.96 | 36.67 |
| **✓** | **✓** | **67.59** | **42.57** | **13.67** | **30.90** | **38.68** |

**VLM 选择消融（iSAID 训练）：**

| VLM | NWPU | SOTA | FAST | SIOR |
|:---:|:---:|:---:|:---:|:---:|
| CLIP (通用) | 64.03 | 38.74 | 11.42 | 28.11 |
| SkyCLIP | 65.04 | 33.57 | 12.43 | 28.96 |
| GeoRSCLIP | 64.72 | 39.33 | 12.56 | 28.13 |
| **RemoteCLIP** | **67.59** | **42.57** | **13.67** | **30.90** |

### 关键发现

- SCORE 平均 mAP 超越最佳 SOTA 方法 **5.53%**（iSAID 训练）和 **4.32%**（SIOR 训练）
- RAI 和 GCA 功能互补：单独使用各提升约 3.5%，联合使用提升 5.4%
- **遥感特定 CLIP 比通用 CLIP 更适合场景上下文提取**：RemoteCLIP 在所有数据集上均优于通用 CLIP
- **区域上下文 > 全局上下文 > patch 嵌入**：证明自适应局部上下文比全局 [CLS] token 更有效
- **GCA 注入方法的选择很关键**：简单加法/拼接会破坏预训练对齐，多头交叉注意力效果最好
- 遥感 CLIP 在 out-vocabulary 分类上仍不如通用 CLIP，因此推理时使用通用 CLIP 做 out-vocabulary 分类

## 亮点与洞察

- 首次提出并构建了**开放词汇遥感实例分割**的基准和评价体系
- "目标与周围环境相关"的观察虽直觉但有效：船在水边、车在停车场——通过可学习膨胀掩码捕获这种先验
- 双模态增强策略（视觉侧 RAI + 文本侧 GCA）比单侧增强效果好
- 利用领域特定 CLIP 注入先验知识的范式可推广到其他专业领域

## 局限与展望

- FAST 数据集上提升有限（+1.02%），37 个细粒度类别的分割仍是挑战
- 训练仅在单卡 L40S 上进行，大规模训练可能进一步提升
- 区域上下文依赖掩码质量，若初始掩码预测偏差大可能引入噪声上下文
- 当前 RemoteCLIP 的out-vocabulary 能力仍弱于通用 CLIP，混合使用增加了框架复杂度
- 未探索遥感特定的文本提示模板对性能的影响

## 相关工作与启发

- FC-CLIP 的冻结 CNN CLIP backbone + Mask2Former 方案是有效的 baseline
- 遥感 VLM（RemoteCLIP、SkyCLIP、GeoRSCLIP）的域特定知识可作为即插即用的上下文增强模块
- 该工作展示了**领域特定 CLIP + 通用 CLIP 协作**的范式价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次建立 OV 遥感实例分割任务和基准，RAI/GCA 设计合理但非突破性
- **实验充分度**: ⭐⭐⭐⭐⭐ 多数据集训练/评估、详尽消融（组件/VLM选择/上下文类型/注入方式/OV分类）
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，图 1 的船/车区分示例直观有力
- **价值**: ⭐⭐⭐⭐ 为遥感 OV 分割奠定基础，但受限于遥感 CLIP 泛化能力

<!-- RELATED:START -->

## 相关论文

- [LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity](legrad_an_explainability_method_for_vision_transformers_via_feature_formation_se.md)
- [Stepping Out of Similar Semantic Space for Open-Vocabulary Segmentation](stepping_out_of_similar_semantic_space_for_open-vocabulary_segmentation.md)
- [Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [CAVIS: Context-Aware Video Instance Segmentation](cavis_context-aware_video_instance_segmentation.md)
- [SPADE: Spatial-Aware Denoising Network for Open-vocabulary Panoptic Scene Graph Generation](spade_spatial-aware_denoising_network_for_open-vocabulary_panoptic_scene_graph_g.md)

<!-- RELATED:END -->
