---
title: >-
  [论文解读] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention
description: >-
  [医学图像] FaNe 提出了一个语义增强的医学视觉-语言预训练框架，通过语义感知正样本挖掘、文本条件稀疏注意力池化和难负例感知对比损失，解决医学 VLP 中的假阴性问题和粗粒度对齐不足问题。 医学视觉-语言预训练 (VLP) 通过利用配对的影像-报告数据来推进医学图像理解。现有 CLIP 风格的方法存在两个核心问题： 假阴…
tags:
  - "医学图像"
---

# FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention

## 论文信息

- **会议**: AAAI 2026
- **arXiv**: [2511.12215](https://arxiv.org/abs/2511.12215)
- **代码**: [https://github.com/Aventador8/FaNe](https://github.com/Aventador8/FaNe)
- **领域**: 医学图像
- **关键词**: 视觉-语言预训练, 假阴性消除, 稀疏注意力, 细粒度对齐, 对比学习, 医学影像

## 一句话总结

FaNe 提出了一个语义增强的医学视觉-语言预训练框架，通过语义感知正样本挖掘、文本条件稀疏注意力池化和难负例感知对比损失，解决医学 VLP 中的假阴性问题和粗粒度对齐不足问题。

## 研究背景与动机

医学视觉-语言预训练 (VLP) 通过利用配对的影像-报告数据来推进医学图像理解。现有 CLIP 风格的方法存在两个核心问题：

**假阴性 (False Negative) 问题**：在标准 VLP 训练中，每张图像仅与其对应报告构成正样本对，其余均视为负样本。然而在临床实践中，不同患者可能存在相同疾病或病灶，导致报告描述高度相似甚至相同。将这些语义相似但来自不同报告的样本错误地当作负样本，会产生错误的对齐信号。

**细粒度对齐不足**：CLIP 仅进行全局图像-文本对齐，无法捕捉详细的视觉特征。医学报告中每个句子通常对应图像中特定区域的发现，需要句子级别的局部对齐。现有方法（如 FLAIR）虽尝试文本条件注意力池化，但交叉注意力本身缺乏强制精确空间聚焦的能力。

## 方法详解

### 整体框架

FaNe 包含四个核心组件：语义类别划分 (Semantic Class Division)、多正样本全局对齐 (Multi-Positive Global Alignment)、文本条件细粒度对齐 (Text-Conditioned Fine-Grained Alignment)、难负例模态内对比 (Hard-Negative Intra-Modal Contrast)。

### 关键设计

#### 1. 语义类别划分 (Semantic Class Division)

使用预训练知识提取器 BioClinicalBERT 编码报告为全局和局部表示。为消除临床叙述中的语义冗余并稳定跨批次相似度计算，引入**语义增强自适应归一化**：

- 计算批次原型 $p_b$（批内所有文本全局表示的均值）
- 计算基础语义相似度 $\hat{o}_t^*$（每个报告与原型的平均余弦相似度）
- 使用 EMA 平滑（$\alpha=0.05$）防止跨批次突变
- 进行中心偏移归一化：$\widetilde{S} = \frac{S - o_t^*}{1 - o_t^* + \epsilon}$
- 通过阈值 $\kappa$ 构建相似类矩阵 $H$，将样本分为正例和负例

#### 2. 多正样本全局对齐

标准 CLIP 的 InfoNCE 损失仅支持单个正样本对，无法处理多正样本场景。FaNe 采用 SigLIP 的 sigmoid 对比损失，天然支持批内多正样本对齐：

$$\mathcal{L}_{mp} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N}\log\frac{1}{1+e^{h_{ij}(-\langle v_i^g, t_j^g\rangle/\tau_1 + b)}}$$

其中 $h_{ij}$ 为标签矩阵 $H$ 中的条目，$b$ 为可学习偏置项。

#### 3. 文本条件稀疏注意力池化

核心创新之一。设计流程：

- **图像/文本表示提取**：独立编码器提取局部和全局特征 $v^l \in \mathbb{R}^{I \times D}$, $t^l \in \mathbb{R}^{P \times L \times D}$
- **可学习稀疏注意力掩码**：通过 MLP 和 sigmoid 激活生成 $M \in \mathbb{R}^{L \times I}$，并施加 L1 稀疏约束 $\mathcal{L}_{spa}$
- **跨注意力细粒度对齐**：句子级文本嵌入作为 query，聚合局部图像 patch 嵌入，乘以稀疏掩码 $M$，生成文本条件视觉表示 $v^{tc,u}$
- 负样本仅从同一报告内的不同句子中选取，避免跨报告假阴性

细粒度对齐损失 $\mathcal{L}_{tc}$ 为文本→图像和图像→文本两个方向 InfoNCE 的均值。

#### 4. 难负例模态内对比损失

通过自适应重加权机制强调语义相似的负样本，增强模态内区分能力：

$$\alpha_{ij} = \frac{y_{ij} \cdot v_i^g \cdot (v_j^g)^T / \tau_3}{\sum_{k \neq i} y_{ik} \cdot v_i^g \cdot (v_k^g)^T / \tau_3}$$

权重 $\alpha_{ij}$ 和 $\beta_{ij}$ 使得语义相似度更高的难负例对获得更大权重，迫使模型学习细粒度语义区分。

### 损失函数

总损失为四项加权和：

$$\mathcal{L} = \mathcal{L}_{mp} + \lambda_1 \mathcal{L}_{tc} + \lambda_2 \mathcal{L}_{hn} + \lambda_3 \mathcal{L}_{spa}$$

实验中 $\lambda_1 = \lambda_2 = \lambda_3 = 1$。

## 实验

### 预训练设置

- 数据集：MIMIC-CXR v2（筛选后 182,475 对高质量图像-报告对）
- 文本编码器：BioClinicalBERT；图像编码器：ResNet50 / ViT-B/16
- 训练：2× RTX 4090，batch size 98，50 epochs
- 温度参数 $\tau_1=0.1$, $\tau_2=0.07$, $\tau_3=0.07$

### 主实验表格

**语义分割（Dice）+ 目标检测（mAP）**：

| 方法 | RSNA 1%/10%/100% | SIIM 1%/10%/100% | RSNA Det 1%/10%/100% |
|------|------------------|------------------|---------------------|
| MLIP | 67.7/68.8/73.5 | 51.6/60.8/68.1 | 17.2/19.1/25.8 |
| IMITATE | 70.5/71.4/73.8 | 53.9/61.7/64.5 | 15.3/19.7/26.4 |
| **FaNe** | **69.5/72.4/74.1** | **54.1/62.3/68.8** | **16.4/20.6/27.2** |

**图像分类（AUC/ACC）**：

| 方法 | CheXpert 1%/10%/100% | RSNA 1%/10%/100% | COVIDx 1%/10%/100% |
|------|---------------------|------------------|-------------------|
| FaNe (ResNet-50) | 88.2/89.1/89.9 | 88.9/89.8/92.6 | 78.2/89.1/94.0 |
| FaNe (ViT-B/16) | **89.7/90.4/90.8** | **89.3/90.2/93.1** | **79.5/90.7/95.5** |

### 消融实验

1. **稀疏注意力掩码**：加入可学习掩码 + 稀疏正则化带来显著提升（RSNA Dice 100%: 71.2→72.6→74.1）
2. **语义自适应归一化**：开启后 RSNA Dice 1% 从 67.1 提升至 69.5
3. **各损失项贡献**：$\mathcal{L}_{mp}$ 上加 $\mathcal{L}_{hn}$ 和 $\mathcal{L}_{tc}+\mathcal{L}_{spa}$ 均带来递增收益
4. **阈值 $\kappa$ 敏感性**：$\kappa=0.95$ 时效果最佳

### 关键发现

- ViT-B/16 编码器在分类任务上显著优于 ResNet-50，表明 Transformer 架构更适合医学 VLP
- 稀疏注意力可视化显示模型能准确定位文本描述对应的图像区域（如胸椎侧凸、心脏位置）
- FaNe 在低数据比例（1%）场景下优势尤为明显，体现了细粒度预训练的数据效率

## 亮点与洞察

1. **假阴性问题的系统解决**：从语义相似度计算、自适应归一化到多正样本对齐，形成完整的假阴性消除流水线
2. **稀疏注意力设计巧妙**：医学报告中每个句子通常仅对应图像局部区域，稀疏约束与领域特点完美匹配
3. **模态内对比是重要补充**：不仅跨模态对齐，还显式增强模态内区分能力，有助于区分相似但不同的临床发现
4. **句内负采样策略**：细粒度对齐的负样本仅从同一报告内选取，有效避免了跨报告假阴性

## 局限性

- 仅在胸部 X 光数据集 MIMIC-CXR 上预训练和评估，未验证在其他模态（CT、MRI）上的泛化性
- 阈值 $\kappa$ 需要预定义，虽然消融实验显示 0.95 效果最佳，但最优值可能因数据集而异
- 稀疏注意力掩码的稀疏程度由 $\lambda_3$ 控制，可能需要针对不同任务调优
- ResNet-50 上未超越所有方法（如 IMITATE 在 RSNA 分割 1% 处为 70.5 vs FaNe 的 69.5），但 ViT 版本总体最优

## 相关工作

- **视觉-语言预训练**：CLIP, SigLIP, FLAIR (文本条件注意力池化), MGCA (多粒度对齐)
- **假阴性问题**：MedCLIP (语义匹配损失), MLIP (知识引导类级对比), SAT (语义三元组划分)
- **医学 VLP 方法**：GLoRIA, PRIOR, M-FLAG, MedKLIP, IMITATE

## 评分

⭐⭐⭐⭐ (4/5)

- 问题定义清晰，四个组件环环相扣，在医学 VLP 领域贡献扎实
- 实验全面覆盖分类/分割/检测三类下游任务，五个基准数据集
- 稀疏注意力可视化增强了可解释性
- 扣分点：评估局限于胸部 X 光，方法通用性有待验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] GuideGen: A Text-Guided Framework for Paired Full-Torso Anatomy and CT Volume Generation](guidegen_a_text-guided_framework_for_paired_full-torso_anatomy_and_ct_volume_gen.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[CVPR 2025\] Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](../../CVPR2025/medical_imaging/multi-resolution_pathology-language_pre-training_model_with_text-guided_visual_r.md)
- [\[ICLR 2026\] SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](../../ICLR2026/medical_imaging/seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)
- [\[ICLR 2026\] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](../../ICLR2026/medical_imaging/compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)

</div>

<!-- RELATED:END -->
