---
title: >-
  [论文解读] DiN: Diffusion Model for Robust Medical VQA with Semantic Noisy Labels
description: >-
  [CVPR 2025][医学影像][VQA] 本文提出DiN框架，首次将扩散模型引入医学VQA任务处理语义噪声标签问题，通过Answer Diffuser、Answer Condition Generator和Noisy Label Refinement三个模块协同工作，在VQA-RAD和PathVQA数据集上显著超越现有方法。
tags:
  - CVPR 2025
  - 医学影像
  - 视觉问答
  - 扩散模型
  - 噪声标签
  - 多模态学习
  - 标签去噪
---

# DiN: Diffusion Model for Robust Medical VQA with Semantic Noisy Labels

**会议**: CVPR 2025  
**arXiv**: [2503.18536](https://arxiv.org/abs/2503.18536)  
**代码**: [Erjian96/DiN](https://github.com/Erjian96/DiN)  
**领域**: 医学影像  
**关键词**: 医学VQA, 扩散模型, 语义噪声标签, 标签去噪, 多模态融合

## 一句话总结

本文提出DiN框架，首次将扩散模型应用于医学VQA的噪声标签场景（NM-VQA），通过扩散式答案分类器从生成视角进行粗到细的答案筛选，配合噪声标签精炼模块动态修正标签，在10%语义噪声下VQA-RAD准确率达74.24%，超越SNLC的69.65%。

## 研究背景与动机

医学VQA任务旨在通过整合医学图像和文本信息回答临床问题。当前面临的挑战：

1. **标签噪声被忽视**：医学标注需要专业知识，标注者间一致性低，但现有Med-VQA方法（MMBERT、Q2ATransformer）均假设训练集标签干净
2. **传统噪声模型不适用**：对称/非对称噪声不符合医学场景——医生的错误标注往往是"语义相近的错误"而非随机错误（如将"pneumonia"误标为"pneumonitis"）
3. **现有去噪方法的局限**：
    - NTM（噪声转换矩阵）：Med-VQA类别多时估计困难
    - Co-teaching：两分支网络使参数翻倍，在已包含图像+文本编码器的Med-VQA中开销更大
    - SNLC（自然VQA噪声方法）：仅用鲁棒对比损失，对医学语义噪声处理粗糙

核心矛盾：分类方法受限于预定义类别但对噪声鲁棒；生成方法灵活但可能产生不存在的答案。本文用扩散模型桥接两者。

**切入角度**：用扩散模型做分类——正向过程给答案分布加噪，反向过程从噪声分布逐步精炼回正确答案。这既限制在预定义答案空间内（分类性），又通过逐步精炼实现灵活的分布调整（生成性）。

**核心 idea**：用条件扩散模型对答案概率分布做"加噪→去噪"处理实现分类，配合基于 BERT 最近邻的语义噪声模拟和鲁棒焦点损失+动态伪标签做噪声标签精炼。

## 方法详解

### 整体框架

DiN由三个核心模块组成：
1. **Answer Condition Generator (ACG)**：生成答案感知的条件特征
2. **Noisy Label Refinement (NLR)**：精炼噪声标签为伪标签（仅训练时使用）
3. **Answer Diffuser (AD)**：基于条件扩散模型的答案分类器

### 关键设计

**1. 语义噪声基准构建**

- 对closed-end问题（Yes/No）：使用对称噪声随机翻转答案
- 对open-end问题：使用预训练BERT模型建立答案语义空间，将ground truth替换为语义最近的噪声标签
- 评估设定：10%和20%语义噪声比例

这种语义噪声比随机噪声更难处理——相同20%噪声率下，SNLC在随机噪声上达62%准确率但在语义噪声上降到58%。

**2. Answer Condition Generator (ACG)**

- 视觉编码器（Swin Transformer）提取图像特征 $f_k^v$，文本编码器（BERT）提取问题特征 $f_k^q$
- 融合后生成Key和Value嵌入
- 引入 $L$ 个可学习的候选答案嵌入（Answer Condition Embedding），通过self-attention建立答案间关系
- Cross-attention将答案嵌入与图像-问题特征交互，生成条件特征 $f_k^c$

**3. Noisy Label Refinement (NLR)**

包含两个子策略：

**a) 鲁棒焦点损失 (RFL)**：将对称交叉熵与焦点损失结合，同时抗噪声和抗类别不平衡：
$$\mathcal{L}_{RFL} = -\sum_{l=1}^{L} a_k(1-\hat{p}_k)^\gamma \log\hat{p}_k - \sum_{l=1}^{L} \hat{p}_k \log a_k$$

**b) 答案动态调整 (AA)**：
- 若辅助分类器的预测 $\hat{p}_k$ = 原始标签 $a_k$ → 信任原标签
- 否则生成软伪标签：$\bar{y}_k = w_t \hat{p}_k + (1-w_t) a_k$
- 权重 $w_t$ 通过EMA（$\tau=0.99$）追踪batch平均置信度，随训练逐渐增加对模型预测的信任

### 损失函数

$$\mathcal{L}_{total} = \mathcal{L}_{dif} + \alpha \mathcal{L}_{RFL}$$

- $\mathcal{L}_{dif}$：MSE损失，监督AD输出接近伪标签分布
- $\mathcal{L}_{RFL}$：鲁棒焦点损失，训练辅助分类器
- $\alpha = 0.5$

## 实验关键数据

### VQA-RAD数据集

| 噪声类型 | 方法 | Open | Close | Overall |
|----------|------|------|-------|---------|
| 10%-语义 | Baseline | 67.86 | 68.85 | 68.25 |
| 10%-语义 | MMBERT | 63.36 | 68.21 | 66.42 |
| 10%-语义 | CoDis | 68.02 | 70.54 | 69.53 |
| 10%-语义 | SNLC | 67.83 | 71.35 | 69.65 |
| **10%-语义** | **DiN (Ours)** | **72.68** | **75.81** | **74.24** |
| 20%-语义 | Baseline | 54.23 | 57.14 | 56.01 |
| 20%-语义 | SNLC | 56.45 | 61.30 | 58.88 |
| **20%-语义** | **DiN (Ours)** | **58.06** | **64.52** | **63.17** |

- 10%语义噪声下超越SNLC **+4.59%**，超越CoDis **+4.71%**
- Clean Label上限79.13%，DiN在10%噪声下达74.24%，仅差4.89个百分点
- 语义噪声比随机噪声更难（20%下63.17 vs 63.93），验证了语义噪声模拟的价值

### 消融实验关键发现

| 配置 | 效果 |
|------|------|
| 去掉AD模块 | 扩散分类是性能提升的核心来源 |
| 去掉RFL | 性能下降显著，噪声鲁棒性降低 |
| 去掉AA策略 | 伪标签质量下降 |
| ACG条件质量 | 直接影响扩散精度 |

## 亮点与洞察

1. **问题定义的前瞻性**：首次系统性研究Med-VQA中的噪声标签问题，构建语义噪声基准，更贴近真实临床标注错误
2. **扩散模型的新用途**：将扩散模型从生成任务迁移到分类任务，利用逐步精炼特性处理标签噪声——概念优雅且有效
3. **NLR的自适应权重设计**：EMA追踪置信度使训练初期信任原标签、后期更信任模型预测，避免早期错误累积
4. **推理效率**：推理时丢弃NLR模块，不增加额外参数开销

## 局限性

1. 扩散模型的推理需要多步去噪（T步），增加推理时间，不适合实时临床系统
2. 语义噪声对的构建依赖预训练BERT模型，对专业医学术语的覆盖可能不足
3. 仅在VQA-RAD和PathVQA两个相对小规模数据集上验证
4. 候选答案集需预定义，无法处理完全开放式回答

## 相关工作

- **Med-VQA分类方法**：MMBERT → Q2ATransformer → MMQ
- **噪声标签学习**：SimT (NTM) → CoDis (Co-teaching) → SNLC (自然VQA噪声) → DivideMix
- **扩散分类器**：CARD (条件标签生成) → DiffusionDet (检测) — DiN首次将其应用于VQA
- **标准Med-VQA**：假设干净标签，DiN在20%噪声下仍达到接近干净标签的效果

## 评分

- **新颖性**：5/5 — 扩散分类+语义噪声+Med-VQA三者结合首创
- **有效性**：4/5 — 在两个数据集多个噪声水平上一致优于对比方法
- **清晰度**：4/5 — 三模块协同的描述清晰，噪声类型分析有说服力
- **意义**：4/5 — 填补了Med-VQA噪声标签研究的空白

<!-- RELATED:START -->

## 相关论文

- [Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](../../AAAI2026/medical_imaging/hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)
- [BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](../../NeurIPS2025/medical_imaging/semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)
- [Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis](latent_drifting_in_diffusion_models_for_counterfactual_medical_image_synthesis.md)

<!-- RELATED:END -->
