---
title: >-
  [论文解读] PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation
description: >-
  [NeurIPS 2025][医学图像][小波变换] 提出 PhysioWave，一种基于可学习小波分解和频率引导掩码的多尺度 Transformer 架构，首次为 EMG 和 ECG 构建大规模预训练基础模型，并通过多模态融合框架在单模态和多模态生理信号任务上取得 SOTA 性能。
tags:
  - NeurIPS 2025
  - 医学图像
  - 小波变换
  - 生理信号
  - 自监督学习
  - 多模态融合
  - 基础模型
---

# PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation

**会议**: NeurIPS 2025  
**arXiv**: [2506.10351](https://arxiv.org/abs/2506.10351)  
**代码**: [有](https://github.com/ForeverBlue816/PhysioWave)  
**领域**: Medical Imaging / Biosignal Processing  
**关键词**: 小波变换, 生理信号, 自监督学习, 多模态融合, 基础模型

## 一句话总结

提出 PhysioWave，一种基于可学习小波分解和频率引导掩码的多尺度 Transformer 架构，首次为 EMG 和 ECG 构建大规模预训练基础模型，并通过多模态融合框架在单模态和多模态生理信号任务上取得 SOTA 性能。

## 研究背景与动机

生理信号（EEG、EMG、ECG）是健康监测、临床诊断和脑机接口的核心数据源，但面临三大挑战：(1) **低信噪比**：运动伪影、基线漂移等干扰严重；(2) **强非平稳性**：信号包含尖峰和突变，传统时域或固定窗口傅里叶方法无法有效捕获；(3) **跨模态异质性**：不同模态采样率、维度差异巨大。

虽然 EEG 领域已有 LaBraM、EEGPT 等预训练模型，但 EMG 和 ECG 的基础模型仍然空白。现有的基于 NLP 启发的自监督方法（如随机 token 掩码）不适合生理信号——原始信号片段不像单词那样对应有意义的单元，随机丢弃可能移除关键事件或遮盖冗余部分。

## 方法详解

### 整体框架

PhysioWave 的预训练流程包含四个阶段：(1) 可学习小波分解将原始多通道信号分解为多尺度频带表示；(2) 频率引导掩码（FgM）基于 FFT 能量选择性遮蔽高信息量 patch；(3) Transformer 编码器处理 token 序列；(4) 轻量级解码器重建被掩码的 patch。

### 关键设计

1. **自适应小波选择器 (Adaptive Wavelet Selector)**：维护 $M$ 个候选小波基 $\{(k_w^{\text{low}}, k_w^{\text{high}})\}_{w=1}^M$，通过 MLP + Softmax 对输入信号计算选择权重 $\alpha = \text{Softmax}(\text{MLP}(\text{AvgPool}(x)))$，自适应组合滤波器 $k^{\text{low}} = \sum_w \alpha_w k_w^{\text{low}}$。设计动机是不同信号特性需要不同的小波基，传统手工选择无法适应多样化信号。

2. **软门控多分辨率分析 (Soft-Gated Analysis)**：每层分解后上采样回原始长度，通过 multi-head attention 估计自适应门控 $G_c^{(\ell)} \in [0,1]$，动态加权当前层信号和上采样信号：$\hat{a}_c^{(\ell)}[n] = G_c^{(\ell)} a_c^{(\ell)} + (1-G_c^{(\ell)}) \tilde{a}_c^{(\ell+1)}$。相比 U-Net 的硬跳连，软门控可逐通道调节频率内容侧重，减少混叠和振铃伪影。

3. **跨尺度通道聚合前馈网络 (Cross-Scale CAFFN)**：每层小波特征通过通道聚合和多头注意力进行跨尺度融合，当前层特征作 query、浅层特征作 key/value：$Y^{(\ell)} = U^{(\ell)} + \beta \cdot \text{Attention}(U^{(\ell)}, \{Y^{(i)}\}_{i<\ell})$，使细粒度子带特征获得粗分辨率长程模式的信息。

4. **频率引导掩码 (Frequency-guided Masking, FgM)**：对每个 patch 计算 FFT 频谱能量，与随机噪声混合得到重要性分数 $s_n = \alpha \cdot e_n + (1-\alpha) \cdot z_n$，优先掩码高能量 patch。这迫使模型从上下文推断关键信息，相比随机时域掩码能产生更丰富的判别性特征。

### 损失函数 / 训练策略

- **预训练损失**：仅在被掩码 patch 上计算 Smooth-L1 重建损失：$\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{n \in \mathcal{M}} \text{SmoothL1}(\hat{p}_n, p_n)$
- **单模态下游**：端到端微调，mean pooling 后接两层 MLP 分类
- **多模态下游**：冻结各模态预训练编码器，仅训练分类头和 softmax 约束的融合权重，$z_{\text{fused}} = \sum_{m \in \mathcal{M}} \alpha_m z_m$

## 实验关键数据

### 主实验

**ECG 心律分类 (PTB-XL)**

| 方法 | 参数量 | F1 (%) | AUROC (%) |
|------|--------|--------|-----------|
| ECG-Chat (2024) | 13B | 55.9 | 94.1 |
| MaeFE (2023) | 9M | 64.7 | 88.6 |
| **PhysioWave-Large** | **37M** | **66.7** | **94.6** |

**EMG 手势识别 (EPN-612)**

| 方法 | 参数量 | Acc (%) | F1 (%) |
|------|--------|---------|--------|
| Moment (2024) | 385M | 90.87 | 90.16 |
| OTiS (2024) | 45M | 87.55 | 88.03 |
| **PhysioWave-Large** | **37M** | **94.50** | **94.56** |

### 消融实验

| 配置 | 训练损失 | Acc (%) | F1 (%) |
|------|----------|---------|--------|
| 无 FgM（随机掩码） | 0.24 | 92.48 | 92.85 |
| 无预训练 | 0.27 | 91.67 | 91.57 |
| **完整模型** | **0.22** | **93.12** | **93.67** |

### 关键发现

- PhysioWave-Large 在 PTB-XL 上 F1 达 66.7%，超越 13B 参数的 ECG-Chat
- EMG 模型以不到 Moment 1/10 的参数在三个基准上全面超越
- 多模态融合在 DEAP 情绪识别上带来 +7.3% 准确率提升（81.3%→88.6%）
- FgM 比随机掩码提升 0.8% F1，预训练带来 2.1% F1 增益

## 亮点与洞察

- **首次为 EMG/ECG 构建大规模预训练基础模型**，分别用 823GB EMG 和 182GB ECG 数据训练，填补了重要空白
- 可学习小波分解优雅地解决了生理信号的多尺度、非平稳特性，作为通用前端效果显著
- FgM 策略将掩码从"随机丢弃"升级为"信息引导"，契合生理信号不均匀信息分布的特点
- 多模态框架通过冻结编码器 + 轻量融合的设计，避免了小数据集上的过拟合风险

## 局限与展望

- 暂未支持光学生物传感等其他模态
- 多模态融合仅使用简单的加权求和，更复杂的注意力融合可能进一步提升性能
- 预训练数据集偏向西方人群，在不同种族/年龄群体上的泛化能力有待验证
- 实时推理效率未充分讨论，对边缘设备部署场景可能存在挑战

## 相关工作与启发

- **与 LaBraM / EEGPT 的关系**：PhysioWave 将预训练范式从 EEG 扩展到 EMG 和 ECG，并提供了统一的多模态框架
- **与 Moment / OTiS 的对比**：通用时间序列基础模型在生理信号上表现不如专用模型，说明域特异性设计（如小波前端）的重要性
- **启发**：可学习小波分解 + 频率引导掩码的组合可能推广到其他非平稳信号分析（如地震、声学信号）

## 评分

- 新颖性: ⭐⭐⭐⭐ （可学习小波 + FgM 组合新颖，但各组件有先验工作）
- 实验充分度: ⭐⭐⭐⭐⭐ （3种模态、6+数据集、消融全面）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，公式详尽）
- 价值: ⭐⭐⭐⭐⭐ （填补 EMG/ECG 基础模型空白，实际应用价值高）

<!-- RELATED:START -->

## 相关论文

- [Unpaired Image-to-Image Translation for Segmentation and Signal Unmixing](unpaired_image-to-image_translation_for_segmentation_and_signal_unmixing.md)
- [STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology](starc-9_a_large-scale_dataset_for_multi-class_tissue_classification_for_crc_hist.md)
- [LoMix: Learnable Weighted Multi-Scale Logits Mixing for Medical Image Segmentation](lomix_learnable_weighted_multi-scale_logits_mixing_for_medical_image_segmentatio.md)
- [From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining](../../ICML2025/medical_imaging/from_token_to_rhythm_a_multi-scale_approach_for_ecg-language_pretraining.md)
- [MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](../../CVPR2026/medical_imaging/must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)

<!-- RELATED:END -->
