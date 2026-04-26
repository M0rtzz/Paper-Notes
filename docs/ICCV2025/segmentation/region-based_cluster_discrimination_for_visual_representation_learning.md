---
title: >-
  [论文解读] Region-based Cluster Discrimination for Visual Representation Learning
description: >-
  [ICCV 2025][图像分割][区域表示学习] 提出 RICE（Region-Aware Cluster Discrimination），通过构建十亿级区域数据集、设计 Region Transformer 层和统一区域聚类判别损失，联合优化目标感知和 OCR 能力，显著提升视觉编码器在分割、检测和 MLLM 多任务上的表现。
tags:
  - ICCV 2025
  - 图像分割
  - 区域表示学习
  - 聚类判别
  - OCR
  - 视觉编码器
  - 多模态大语言模型
---

# Region-based Cluster Discrimination for Visual Representation Learning

**会议**: ICCV 2025  
**arXiv**: [2507.20025](https://arxiv.org/abs/2507.20025)  
**代码**: [GitHub](https://github.com/deepglint/MVT)  
**领域**: segmentation  
**关键词**: 区域表示学习, 聚类判别, OCR感知, 视觉编码器, 多模态大语言模型

## 一句话总结

提出 RICE（Region-Aware Cluster Discrimination），通过构建十亿级区域数据集、设计 Region Transformer 层和统一区域聚类判别损失，联合优化目标感知和 OCR 能力，显著提升视觉编码器在分割、检测和 MLLM 多任务上的表现。

## 研究背景与动机

CLIP、SigLIP 等视觉-语言对比模型通过大规模图像-文本对齐学习了强大的全局视觉表示。然而，这些模型在**密集预测任务**（分割、定位、OCR）上表现受限，原因是：

**实例判别的语义不足**：正负样本对的划分忽略了语义相似性，来自不同实例但语义相近的样本被视为负对

**OCR 对高层语义的干扰**：视觉-语言对比学习中包含 OCR pair 时，视觉编码器倾向于关注文本识别而非物体语义

**全局表示对局部建模的局限**：现有聚类判别方法（如 UNICOM、MLCD）为每张图像分配一个或多个伪标签，无法学习局部区域级表示

现有区域级方法如 RegionCLIP 和 CLIM 依赖区域文本描述，扩展性受限。**核心动机**：用聚类中心替代文本描述作为区域监督信号，实现大规模可扩展的区域级表示学习。

## 方法详解

### 整体框架

RICE 采用 ViT 架构，在标准 Transformer 层之后接入 Region Transformer 层，单次前向传播同时提取全局语义和区域级语义。训练监督来自两个分支：

- **Object Region Loss**：基于聚类中心的单标签分类
- **OCR Region Loss**：基于 token 嵌入的多标签分类

### 关键设计一：区域数据构建

**目标区域数据**：从 LAION-2B、COYO-700M、SAM-1B 中采样，对 LAION 和 COYO 使用 SAM 生成精细掩码区域，保留最短边 ≥128px 的候选框，获得 4 亿图像、20 亿候选区域。使用 CLIP 提取区域特征后，通过 k-means 聚类为 100 万个语义中心：

$$\mathbf{y}_{i,j}^{object} = \arg\min_{k \in [1,K]} \|\mathbf{f}_{i,j} - \mathbf{c}_k\|_2$$

聚类使用 Faiss GPU 的层次化 k-means，在 64 张 GPU 上约 10 小时完成。

**OCR 区域数据**：使用 PaddleOCR 从 LAION-2B 和 COYO-700M 提取文本（置信度 >0.7），获得 5000 万图像、4 亿候选区域。对提取文本进行 tokenize 得到 OCR 标签。

### 关键设计二：Region Transformer 层

**区域采样**：对每张图像标准化区域数量为 $N$。若实际区域数超过 $N$ 则随机采样，不足则重复采样。

**区域注意力**：
- 各区域 token 数量因空间大小不同而不一致，直接 batch 处理困难
- 引入**区域可见性掩码** $\mathcal{M}$：区域内 token 设为 0，区域外设为 $-\infty$
- 区域注意力计算：

$$\mathcal{R}_{\text{batch}} = \sigma\left(\frac{\mathbf{Q}_{\text{batch}} \mathbf{K}_{\text{batch}}^\top}{\sqrt{d_k}} + \mathcal{M}\right) \mathbf{V}_{\text{batch}}$$

- 单次前向传播即可提取所有区域的固定长度嵌入

### 关键设计三：区域聚类判别损失

**目标区域损失**（单标签分类）：

$$\mathcal{L}_{object} = \log(1 + \exp(-sim(\hat{\mathbf{y}}_{i,j}, \mathbf{y}_{i,j}^{object}))) + \log(1 + \sum_{j \in \Omega_n^{object}} \exp(sim(\hat{\mathbf{y}}_{i,j}, \mathbf{y}_{i,j}^{object})))$$

**OCR 区域损失**（多标签分类）：每个 OCR 区域有多个 token 嵌入作为正类：

$$\mathcal{L}_{ocr} = \log(1 + \sum_{j \in \Omega_p^{ocr}} \exp(-sim)) + \log(1 + \sum_{j \in \Omega_n^{ocr}} \exp(sim))$$

**负类采样策略**：从完整类别集中按采样率 $\rho=0.1$ 均匀采样负类中心，减少语义冲突梯度，提升训练稳定性。

## 实验

### MLLM 多模态理解（LLaVA-NeXT 框架）

| Vision Tower | LLM | DocVQA | OCRBench | InfoVQA | MM-Bench | 其他 Avg |
|-------------|-----|:------:|:--------:|:-------:|:--------:|:-------:|
| CLIP ViT-L-336px | Qwen2.5-7B | 75.21 | 525 | 38.88 | 74.57 | 69.83 |
| SigLIP SO400M-384px | Qwen2.5-7B | 76.71 | 554 | 41.38 | 76.98 | 70.62 |
| AIMv2 ViT-L-336px | Qwen2.5-7B | 77.19 | 572 | 35.44 | 78.61 | 70.58 |
| **RICE ViT-L-336px** | Qwen2.5-7B | **79.19** | **575** | **45.23** | 76.55 | **73.03** |

RICE-336px 在 OCR 任务上全面领先：比 CLIP 的 OCRBench 提升 +50 分，DocVQA 提升 +3.98%。

### 指代图像分割（LLaVA-NeXT + LISA）

| Vision Tower | LLM | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|-------------|-----|:----------:|:----------:|:-----------:|
| CLIP | Qwen2.5-7B | 81.8 | 76.6 | 77.3 |
| MLCD | Qwen2.5-7B | 82.8 | 77.4 | 78.5 |
| **RICE** | Qwen2.5-7B | **83.5** | **79.4** | **79.8** |

RICE 在所有 RefCOCO 基准上均超越 CLIP 和 MLCD，平均 IoU 提升 +2.45（vs CLIP）和 +1.30（vs MLCD）。

### 关键发现

1. 区域聚类判别在 OCR 密集任务上优势突出（InfoVQA +9.79 vs AIMv2），因为联合训练目标语义和 OCR 避免了两者的冲突
2. t-SNE 可视化显示 RICE 的物体特征聚类远优于 DINOv2、MLCD 和 SigLIP
3. 随机负类采样率 $\rho=0.1$ 是最优选择，过高会引入语义冲突梯度
4. Region Transformer 层的位置在最后数层最优，兼顾全局上下文和区域精度

## 亮点与洞察

- **数据工程驱动**：20 亿区域 + 100 万聚类中心的数据构建规模宏大，通过聚类中心替代文本描述实现了无文本的区域监督
- **统一框架**：目标识别和 OCR 任务在同一分类框架下联合训练，避免多任务冲突
- **即插即用**：RICE 视觉编码器可直接替换 CLIP 接入 LLaVA 等框架，无需架构修改

## 局限性

- 数据构建依赖 SAM 和 PaddleOCR 的质量，标注噪声可能传播到聚类中心
- 100 万聚类中心的存储和更新成本较高
- 仅验证了 ViT-L 和 ViT-B 规模，更大模型（如 ViT-G）的效果未知

## 相关工作

- 实例判别：CLIP、SigLIP、DINOv2 等视觉表示学习方法
- 聚类判别：DeepCluster、SwAV、UNICOM、MLCD 等基于聚类的自监督方法
- 区域表示：RegionCLIP、CLIM、GLIP 等区域-语言对齐方法

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 区域聚类判别的思路新颖，统一 Object+OCR 的设计有意义
- **技术深度**: ⭐⭐⭐⭐ — Region Transformer 和损失函数设计完整，数据工程扎实
- **实验**: ⭐⭐⭐⭐⭐ — 覆盖 MLLM/分割/检测/OCR 多维度，消融充分
- **写作**: ⭐⭐⭐⭐ — 动机清晰，框架图直观

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Implicit Counterfactual Learning for Audio-Visual Segmentation](implicit_counterfactual_learning_for_audio-visual_segmentation.md)
- [\[ICCV 2025\] On the Generalization of Representation Uncertainty in Earth Observation](on_the_generalization_of_representation_uncertainty_in_earth_observation.md)
- [\[ICCV 2025\] Hierarchical Visual Prompt Learning for Continual Video Instance Segmentation](hierarchical_visual_prompt_learning_for_continual_video_instance_segmentation.md)
- [\[CVPR 2026\] AFRO: Bootstrap Dynamic-Aware 3D Visual Representation for Scalable Robot Learning](../../CVPR2026/segmentation/bootstrap_dynamic-aware_3d_visual_representation_for_scalable_robot_learning.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)

<!-- RELATED:END -->
