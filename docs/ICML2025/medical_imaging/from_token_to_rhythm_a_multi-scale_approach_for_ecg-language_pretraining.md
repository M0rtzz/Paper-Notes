---
title: >-
  [论文解读] From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining
description: >-
  [ICML 2025][医学图像][ECG预训练] MELP 提出了一种多尺度 ECG-语言预训练模型，通过 Token/Beat/Rhythm 三个层次的跨模态监督信号，结合心脏学专业语言模型预训练，在零样本分类、线性探测和迁移学习中全面超越现有 ECG 自监督和多模态方法。
tags:
  - ICML 2025
  - 医学图像
  - ECG预训练
  - 多模态学习
  - 多尺度表征
  - 对比学习
  - ECG-文本对齐
  - 零样本分类
  - 自监督学习
---

# From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining

**会议**: ICML 2025  
**arXiv**: [2506.21803](https://arxiv.org/abs/2506.21803)  
**代码**: [https://github.com/HKU-MedAI/MELP](https://github.com/HKU-MedAI/MELP)  
**领域**: 医学AI / 心电图分析  
**关键词**: ECG预训练, 多模态学习, 多尺度表征, 对比学习, ECG-文本对齐, 零样本分类, 自监督学习

## 一句话总结

MELP 提出了一种多尺度 ECG-语言预训练模型，通过 Token/Beat/Rhythm 三个层次的跨模态监督信号，结合心脏学专业语言模型预训练，在零样本分类、线性探测和迁移学习中全面超越现有 ECG 自监督和多模态方法。

## 研究背景与动机

心电图（ECG）是心血管疾病诊断的核心工具。深度学习虽然显著提升了ECG分析能力，但面临大规模人工标注成本高昂的问题。自监督学习（SSL）因此成为有前景的替代方案。

现有方法的不足：

**单模态SSL局限**：对比式（SimCLR、CLOCS等）和生成式（ST-MEM、HeartLang）方法仅利用ECG信号，忽略了临床文本中的丰富语义信息。

**全局对齐不足**：少数ECG-语言对齐方法（如MERL）仅关注全局ECG到文本的对齐，忽视了ECG信号的多尺度结构。

**层次结构遗漏**：心脏病学家以层次化方式解读ECG——从波形成分（token级）到心跳周期（beat级）再到整体节律（rhythm级），现有模型未能捕捉这种多尺度特性。

核心动机：模仿临床医生的多尺度解读方式，在token、beat、rhythm三个层次上建立ECG与文本之间的跨模态监督。

## 方法详解

### 整体框架

MELP（Multi-scale ECG-Language Pretraining）由以下部分组成（Figure 2）：

1. **心脏学语言模型预训练**（Stage 1）
2. **多模态预训练**（Stage 2）——三层次监督

### 关键设计

#### 1. 心脏学语言预训练

基于MedCPT查询编码器，使用三个来源的心脏学语料进行MLM预训练：
- PubMed心脏学相关文献
- Wikipedia心脏学词条
- MIMIC-IV-ECG训练集临床报告

目的是让文本编码器具备丰富的心脏学领域知识，为后续跨模态对齐提供高质量语义表征。

#### 2. Token级：ECG描述生成

采用编码器-解码器架构，ECG编码器产生token级嵌入 $E \in \mathbb{R}^{L_t \times D}$，通过128个可学习query token的注意力池化得到 $\tilde{E} \in \mathbb{R}^{128 \times D}$。文本解码器以GPT风格自回归生成配对报告：

$\mathcal{L}_{\mathrm{LM}}(\zeta) = -\sum_{i=1}^{N} \log p(w_i | w_{0:i-1}, \tilde{E})$

通过报告生成任务，强迫模型捕捉细粒度波形特征（如P波缺失、QRS时长等），隐式学习这些局部指标与临床诊断的关系。

#### 3. Beat级：心跳-句子对齐

引入10个可学习token，通过注意力池化将token级特征聚合为beat级表征 $B \in \mathbb{R}^{N_B \times D}$。文本端将word token按句子平均生成句子嵌入 $S \in \mathbb{R}^{S \times D}$。

注意力加权beat嵌入：

$\hat{B}(l) = \sum_{l=1}^{N_B} \alpha_l S(l), \quad \alpha(l) = \frac{\exp(\langle S(l), B(l) \rangle / \tau_1)}{\sum_j \exp(\langle S(l), B(j) \rangle / \tau_1)}$

局部对比损失：

$\mathcal{L}_{\mathrm{Local}} = \frac{1}{2}(\mathcal{L}_{\mathrm{Local}}^{e \to t} + \mathcal{L}_{\mathrm{Local}}^{t \to e})$

通过beat-句子匹配机制，捕捉短暂异常心跳与特定描述语句的对应关系。

#### 4. Rhythm级：全局ECG-报告对齐

ECG全局嵌入 $X_i^g$ 为所有beat嵌入的平均值，文本全局嵌入 $T_i^g$ 为[CLS] token表示。使用标准InfoNCE对比损失：

$\mathcal{L}_g^{e \to t} = -\frac{1}{B}\log \frac{\exp(\langle X_i^g, T_i^g \rangle / \tau)}{\sum_{j=1}^{B} \exp(\langle X_i^g, T_j^g \rangle / \tau)}$

#### 5. 总体损失

$\mathcal{L} = \mathcal{L}_g + \lambda_1 \mathcal{L}_{\mathrm{LM}} + \lambda_2 \mathcal{L}_{\mathrm{Local}}$

其中 $\lambda_1 = 2$，$\lambda_2 = 0.2$。

## 实验关键数据

### 线性探测（AUC%，100%训练数据）

| 方法 | PTBXL-Rhythm | PTBXL-Super | CPSC2018 | CSN |
|------|-------------|-------------|----------|-----|
| SimCLR | 77.73 | 73.53 | 76.54 | 73.20 |
| Wav2Vec2+CMSC+RLM | 92.05 | 85.53 | **92.61** | 87.87 |
| MERL | 92.31 | 88.01 | 92.48 | 87.39 |
| **MELP** | **93.66** | **89.44** | 92.49 | **90.25** |

### 线性探测（AUC%，1%训练数据）

| 方法 | PTBXL-Rhythm | PTBXL-Super | CPSC2018 | CSN |
|------|-------------|-------------|----------|-----|
| SimCLR | 51.41 | 63.41 | 59.78 | 59.02 |
| MERL | 84.85 | 84.46 | 81.49 | 73.23 |
| **MELP** | **87.72** | **87.63** | **82.05** | **77.65** |

关键发现：
- 在极低标注（1%）场景下MELP优势最为突出，在多个数据集上超越MERL 2-4个百分点
- 100%数据下MELP在6个评估集上平均仍保持SOTA

### 零样本分类

MELP在零样本ECG分类中取得最佳性能，证明了多尺度监督对跨模态对齐质量的提升。

### 消融实验

- 去除Token级监督 → 性能下降最大
- 去除Rhythm级监督 → 零样本能力丧失
- 去除Beat级监督 → 中等程度下降
- 三者缺一不可，验证了多尺度监督的互补性

## 亮点与洞察

1. **临床启发的设计理念**：模仿心脏病学家的多尺度解读方式（波形→心跳→节律）设计监督信号，生物学合理性强
2. **心脏学专用语言模型**：先在领域语料上预训练文本编码器再进行跨模态对齐，显著优于直接使用通用语言模型
3. **生成式+对比式的有机结合**：Token级captioning（生成式）与Beat/Rhythm级对比学习（判别式）互补，避免了单一范式的局限
4. **灵活的下游适配**：生成式预训练使模型可自然扩展到ECG报告生成和ECG问答等任务

## 局限性

1. Beat级固定使用10个可学习token，未根据实际心率自适应调整
2. 预训练数据（MIMIC-IV-ECG）来自单一数据源，跨机构泛化有待验证
3. 文本解码器随机初始化，可能需要更多数据才能充分训练
4. Token级captioning的计算开销较大

## 相关工作

- **ECG表征学习**：SimCLR、CLOCS、Wav2Vec 2.0、ST-MEM、HeartLang
- **ECG-语言预训练**：MERL、C-MELT、ECG-Chat、ESI
- **通用多模态对比学习**：CLIP、CoCa

## 评分

⭐⭐⭐⭐ — 多尺度设计理念优雅且有临床基础，实验覆盖面广（6个数据集，多种评估协议）。心脏学语言预训练的引入是一个重要的实践洞察。代码开源利于复现和后续研究。Beat级token数的固定和单数据源预训练是主要不足。

<!-- RELATED:START -->

## 相关论文

- [EEG-Language Pretraining for Highly Label-Efficient Clinical Phenotyping](eeg-language_pretraining_for_highly_label-efficient_clinical_phenotyping.md)
- [A Novel Approach to Classification of ECG Arrhythmia Types with Latent ODEs](../../NeurIPS2025/medical_imaging/a_novel_approach_to_classification_of_ecg_arrhythmia_types_with_latent_odes.md)
- [Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners (D-BETA)](boosting_masked_ecg-text_auto-encoders_as_discriminative_learners.md)
- [PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation](../../NeurIPS2025/medical_imaging/physiowave_a_multi-scale_wavelet-transformer_for_physiological_signal_representa.md)
- [ZoomLDM: Latent Diffusion Model for Multi-Scale Image Generation](../../CVPR2025/medical_imaging/zoomldm_latent_diffusion_model_for_multi-scale_image_generation.md)

<!-- RELATED:END -->
