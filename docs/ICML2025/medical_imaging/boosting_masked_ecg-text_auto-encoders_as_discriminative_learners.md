---
title: >-
  [论文解读] Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners (D-BETA)
description: >-
  [ICML 2025][医学图像][ECG信号解读] D-BETA 提出了一种融合生成式掩码自编码器与增强判别能力的对比学习框架，通过 ECG-Text Sigmoid (ETS) 损失和最近邻负采样策略 (N3S)，在 ECG-文本跨模态表征学习中显著超越现有方法，在仅用 1% 训练数据的线性探测中平均 AUC 提升 15%，零样本性能提升 2%。
tags:
  - ICML 2025
  - 医学图像
  - ECG信号解读
  - 多模态表征学习
  - 对比掩码自编码器
  - 零样本分类
  - 心血管诊断
---

# Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners (D-BETA)

**会议**: ICML 2025  
**arXiv**: [2410.02131](https://arxiv.org/abs/2410.02131)  
**代码**: [https://github.com/manhph2211/D-BETA](https://github.com/manhph2211/D-BETA)  
**领域**: 医学图像  
**关键词**: ECG信号解读, 多模态表征学习, 对比掩码自编码器, 零样本分类, 心血管诊断

## 一句话总结

D-BETA 提出了一种融合生成式掩码自编码器与增强判别能力的对比学习框架，通过 ECG-Text Sigmoid (ETS) 损失和最近邻负采样策略 (N3S)，在 ECG-文本跨模态表征学习中显著超越现有方法，在仅用 1% 训练数据的线性探测中平均 AUC 提升 15%，零样本性能提升 2%。

## 研究背景与动机

心电图 (ECG) 是心血管疾病诊断的核心工具，标准 12 导联 ECG 在心律失常等疾病诊断中起关键作用。深度学习在 ECG 自动解读方面取得了进展，但面临两大瓶颈：

**标注数据稀缺**：有监督方法依赖大量专家标注数据，获取成本高昂。自监督学习 (SSL) 可从无标注数据中学习鲁棒表征，但现有方法分为对比式和生成式两条路线，缺乏有效整合。

**跨模态信息未充分利用**：临床文本报告包含丰富的诊断线索，但现有 ECG SSL 方法大多忽视了文本信息。少数尝试跨模态学习的方法（如 MERL）主要采用 ResNet + BERT 的标准架构，且仅依赖对比学习，存在负样本选择困难和跨模态关系建模不足的问题。

**对比学习的固有局限**：在医学数据集中，随机负采样容易产生假阴性（如 MIMIC-IV ECG 中 80 万条记录仅有约 18 万条唯一文本），严重影响对比学习效果。

核心动机：将生成式（掩码建模）和判别式（对比学习）两种范式统一到一个框架中，同时解决负样本质量问题，实现更强的 ECG-文本跨模态表征。

## 方法详解

### 整体框架

D-BETA 框架包含两个主分支和四个学习目标：

- **ECG 编码器 (ℱ_x)**：基于 Transformer 的 ECG 信号编码器，输出 H_x ∈ ℝ^{L_x × d}
- **文本编码器 (ℱ_t)**：使用预训练 Flan-T5-base 模型，输出 H_t ∈ ℝ^{L_t × d}
- **融合模块**：通过交叉注意力机制融合两种模态，输出 H_f ∈ ℝ^{(L_x+L_t) × d}
- **三个任务头**：MLM 解码器（文本重建）、MEM 解码器（ECG 重建）、ETM 头（跨模态匹配）
- **两个投影头**：g_x 和 g_t，配合 ETS 损失学习判别性表征

总损失为四项之和：ℒ = ℒ_MLM + ℒ_MEM + ℒ_ETM + ℒ_ETS

### 关键设计

#### 1. ECG 编码器

- 输入：X ∈ ℝ^{L × C}（L 为信号长度，C 为通道数）
- 预处理：随机导联掩码（p=0.5）+ 输入 Dropout（p=0.1）实现掩码建模
- 特征提取：多层卷积 + GELU 激活 + Group Normalization → 768 维投影
- 位置编码：卷积位置编码保持时序信息
- 主干网络：8 层 Transformer 编码器层（多头自注意力）

#### 2. 文本编码器

- 采用 Flan-T5-base（首次应用于 ECG 领域），输出 768 维嵌入
- Flan-T5 在大规模多任务数据上预训练，具备强大的文本理解能力
- 在预训练阶段进行微调

#### 3. 融合模块

- 线性投影将两个编码器输出映射到 768 维空间
- 添加模态特定嵌入以区分 ECG 和文本数据
- **交叉注意力机制**：允许每种模态关注另一种模态的相关特征，充分利用互补信息

#### 4. ETS 损失函数（核心创新）

传统掩码自编码器架构中的 ETM 损失基于融合特征的二分类任务，无法直接增强单个编码器的判别能力。D-BETA 提出 ETS 损失，受 SigLIP 启发：

$$\mathcal{L}_{ETS} = -\frac{1}{\mathcal{B}} \sum_{i=1}^{\mathcal{B}} \sum_{j=1}^{\mathcal{B}} \log \frac{1}{1 + e^{-y_{ij} \cdot \mathbf{x}'_i{}^\top \mathbf{t}'_j}}$$

其中 y_{ij}=1 表示匹配对，y_{ij}=-1 表示不匹配对。

ETS 的关键优势：
- 基于 Sigmoid 而非 Softmax，避免全局归一化的高计算成本
- 对每个 ECG-文本对独立计算，提高内存效率和可扩展性
- 通过独立的投影头（Pooling → Tanh → Dense → 768 维）直接增强编码器的判别性

#### 5. 最近邻负采样策略 (N3S)

针对医学数据集中大量重复/相似文本导致随机负采样产生假阴性的问题：

- 使用预训练 Flan-T5 (small) 为每条文本生成 512 维向量表示 v_t
- 训练时，对给定正样本对 (x_k, t_k^+)，从余弦距离最大的 top-64 报告中选择负样本 t_k^-
- 仅对 batch 中一半的样本应用 N3S
- 利用 FAISS 库实现高效向量检索，支持大规模数据集

N3S 的效果：ETM 准确率从无 N3S 时的 ~75% 提升至 >96%。

### 损失函数 / 训练策略

**四个损失函数**：

| 损失函数 | 类型 | 公式 | 作用 |
|---------|------|------|------|
| ℒ_MLM | 交叉熵 | 预测被掩码的文本 token | 学习上下文化词嵌入 |
| ℒ_MEM | MSE | 重建被掩码的 ECG 信号 | 捕获 ECG 时序结构 |
| ℒ_ETM | 二元交叉熵 | ECG-文本对匹配分类 | 融合特征空间对齐 |
| ℒ_ETS | Sigmoid 对比 | 直接对齐 ECG/文本编码器输出 | 增强编码器判别能力 |

**训练配置**：

- 预训练数据集：MIMIC-IV-ECG v1.0（~78 万对 ECG-文本样本，来自 161,352 名受试者）
- 优化器：Adam (lr=5×10⁻⁵, β₁=0.9, β₂=0.98, ε=10⁻⁶, weight decay=0.01)
- 学习率调度：三阶段调度器（比例 0.1:0.4:0.5）
- 训练步数：300,000 步，batch size=128
- 硬件：单卡 NVIDIA H100-80GB

## 实验关键数据

### 主实验

**实验一：全量微调（PhysioNet 2021）**

| 方法 | 12-lead (Dx.) | 1-lead (Dx.) | 12-lead (Id.) |
|------|:---:|:---:|:---:|
| W2V+CMSC+RLM | 73.2 | 55.4 | 57.7 |
| **D-BETA** | **85.7** | **76.5** | **65.4** |
| 提升 | +12.5 | +21.1 | +7.7 |

D-BETA 仅用 1 个导联 (76.5%) 即超过之前 SOTA 使用全部 12 导联的成绩 (73.2%)。

**实验二：线性探测（冻结编码器）**

| 数据集 | 数据比例 | D-BETA | MERL | 提升 |
|--------|:---:|:---:|:---:|:---:|
| PTBXL-Rhythm | 1% | 86.61 | 53.33 | +33.28 |
| CSN | 1% | 70.10 | 58.26 | +11.84 |
| CPSC2018 | 1% | 85.46 | 70.33 | +15.13 |
| PTBXL-Rhythm | 100% | 96.71 | 88.34 | +8.37 |
| CPSC2018 | 100% | 94.92 | 90.57 | +4.35 |

**实验三：零样本分类**

| 数据集 | D-BETA | MERL | 提升 |
|--------|:---:|:---:|:---:|
| PTBXL-Super | 76.2 | 74.2 | +2.0 |
| PTBXL-Sub | 75.9 | 75.7 | +0.2 |
| PTBXL-Form | 66.1 | 65.9 | +0.2 |
| CSN | 88.6 | 78.5 | +10.1 |
| CODE-test | 80.1 | 82.8 | -2.7 |
| 平均 | **77.1** | 75.3 | **+1.8** |

**亮点：D-BETA 零样本在 CODE-test 数据集上达到 96.79% AUC，超越人类心脏专科医生（92-94%）和有监督 DNN（96.59%）。**

### 消融实验

**组件消融（Table 6）**：

| 配置 | Fine-Tuning (Dx.) | Linear Probe (1%) | Zero-Shot | 说明 |
|------|:---:|:---:|:---:|------|
| Baseline (Bert, 无 ETS, 无 N3S) | 76.81 | 63.50 | – | 基线模型 |
| + ETS | 78.29 | 67.19 | – | ETS 提升 ~4% |
| + N3S | 80.93 | 78.29 | 70.61 | N3S 使零样本可行 |
| + Flan-T5 (完整 D-BETA) | **85.70** | **80.93** | **72.82** | Flan-T5 再提 4%+ |

**文本编码器消融（Table 7）**：

| 编码器 | Fine-Tuning | Linear Probe (1%) | Zero-Shot |
|--------|:---:|:---:|:---:|
| Bert | 78.08 | 77.58 | 69.14 |
| Deberta | 79.23 | 78.24 | 70.67 |
| Med-CPT | 81.02 | 79.57 | 71.81 |
| **Flan-T5** | **85.70** | **80.93** | **72.82** |

### 关键发现

1. **ETS 损失是最大贡献因子**：带来约 15% 的性能提升，证实在掩码自编码器中额外引入判别性损失的必要性。
2. **N3S 对零样本分类至关重要**：ETM 准确率从 ~75% 提升至 >96%，有效解决了医学数据集中文本高度重复的问题。
3. **Flan-T5 显著优于 BERT 系列**：在所有实验设置中均领先，说明更强的预训练语言模型对 ECG 领域同样有效。
4. **导联组合实验**：3 导联 (I, II, V2) 即可达到接近 12 导联的性能（仅差 1.5%），对临床场景有重要价值。
5. **零样本超越人类专家**：在 CODE-test 上 D-BETA 零样本 (96.79%) 超越心脏专科医生 (90.5-93.6%) 和有监督模型 (96.59%)。

## 亮点与洞察

1. **生成+判别的统一框架**：D-BETA 优雅地解决了掩码自编码器（偏向重建）与对比学习（偏向判别）之间的张力，通过独立的投影头和 ETS 损失让两者互补。
2. **Sigmoid vs Softmax 对比损失**：借鉴 SigLIP 的设计，避免 Softmax 全局归一化的开销，特别适合大规模预训练。
3. **N3S 的域感知负采样**：利用 Flan-T5 特征空间和 FAISS 索引实现高效负样本选择，比随机采样在医学高重复数据场景下大幅提升效果。
4. **GPT-4o 增强零样本**：用简洁的 prompt 让 GPT-4o 生成临床描述来增强类别文本编码，比 MERL 的数据库检索方式更可控。
5. **极端低数据表现优异**：仅用 1% 训练数据即可获得显著优势，对标注匮乏的医学场景极具实用价值。

## 局限与展望

1. **预训练数据单一**：仅使用 MIMIC-IV-ECG 数据集预训练，可能存在数据分布偏差，扩展到多中心、多种族数据集值得探索。
2. **文本质量依赖**：预训练效果受原始临床报告质量影响，报告的简短性和重复性可能限制学习上界。
3. **计算成本**：虽然 ETS 比 Softmax 对比损失更高效，但 30 万步的预训练仍需要 H100 GPU，限制了资源有限的科研者复现。
4. **零样本依赖 GPT-4o**：最优零样本结果需要 GPT-4o 生成类别描述，增加了对外部 LLM 的依赖。
5. **ECG 特有信号处理**：当前对 ECG 的预处理较为简单（掩码+dropout），可探索更多生理信号特有的增强策略。
6. **未探索更大模型**：仅使用 Flan-T5 base 版本，更大版本可能带来进一步提升。

## 相关工作与启发

- **SigLIP (Zhai et al., 2023)**：ETS 损失的直接灵感来源，将 Sigmoid 对比损失从视觉-语言扩展到 ECG-文本。
- **MERL (Liu et al., 2024b)**：最强基线，使用 ResNet+BERT+跨模态对齐+测试时知识增强。D-BETA 在架构和训练策略上均有改进。
- **M3AE (Chen et al., 2022)**：多模态掩码自编码器的先驱工作，D-BETA 在其基础上增强了判别能力。
- **CLIP (Radford et al., 2021)**：跨模态对比学习的经典范式，D-BETA 结合了 CLIP 的对比思想和 MAE 的生成思想。
- **对后续工作的启发**：该框架可推广到其他医学信号（如 EEG、EMG）与文本报告的跨模态学习。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|:---:|------|
| 新颖性 | 4 | 生成+判别的统一以及 N3S 负采样是实质性创新 |
| 技术深度 | 4 | 多个精心设计的组件，消融实验充分 |
| 实验充分性 | 5 | 5 个数据集、3 种评估方式、全面消融 |
| 实用价值 | 4 | 极低数据和零样本对临床有重要价值 |
| 写作质量 | 4 | 结构清晰，动机充分 |
| **总分** | **4.2** | 扎实的多模态医学表征学习工作 |

<!-- RELATED:START -->

## 相关论文

- [From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining](from_token_to_rhythm_a_multi-scale_approach_for_ecg-language_pretraining.md)
- [A Novel Approach to Classification of ECG Arrhythmia Types with Latent ODEs](../../NeurIPS2025/medical_imaging/a_novel_approach_to_classification_of_ecg_arrhythmia_types_with_latent_odes.md)
- [Benchmarking ECG FMs: A Reality Check Across Clinical Tasks](../../ICLR2026/medical_imaging/benchmarking_ecg_fms_a_reality_check_across_clinical_tasks.md)
- [How to Make the Most of Your Masked Language Model for Protein Engineering](../../ICLR2026/medical_imaging/how_to_make_the_most_of_your_masked_language_model_for_protein_engineering.md)
- [RadGPT: Constructing 3D Image-Text Tumor Datasets](../../ICCV2025/medical_imaging/radgpt_constructing_3d_image-text_tumor_datasets.md)

<!-- RELATED:END -->
