---
title: >-
  [论文解读] Towards All-in-One Medical Image Re-Identification
description: >-
  [CVPR 2025][医学图像][医学图像重识别] 提出 MaMI，首个全模态统一的医学图像重识别模型，通过连续模态参数适配器 (ComPA) 动态生成模态特定参数，并利用医学基础模型的差异特征对齐传递医学先验，在 11 个数据集上超越 25 个基础模型和 8 个大语言模型。
tags:
  - "CVPR 2025"
  - "医学图像"
  - "医学图像重识别"
  - "模态自适应"
  - "参数适配器"
  - "医学先验"
  - "隐私保护"
---

# Towards All-in-One Medical Image Re-Identification

**会议**: CVPR 2025  
**arXiv**: [2503.08173](https://arxiv.org/abs/2503.08173)  
**代码**: [GitHub](https://github.com/tianyuan168326/All-in-One-MedReID-Pytorch)  
**领域**: 医学图像  
**关键词**: 医学图像重识别, 模态自适应, 参数适配器, 医学先验, 隐私保护

## 一句话总结

提出 MaMI，首个全模态统一的医学图像重识别模型，通过连续模态参数适配器 (ComPA) 动态生成模态特定参数，并利用医学基础模型的差异特征对齐传递医学先验，在 11 个数据集上超越 25 个基础模型和 8 个大语言模型。

## 研究背景与动机

医学图像重识别（MedReID）在个性化医疗和隐私保护中至关重要，但研究严重不足：

1. **现有方法局限于单一模态**：如 Packhäuser 等仅处理胸部 X 光，无法跨模态共享知识
2. **统一模型的挑战**：朴素多模态联合训练对某些模态有益（如眼底图像 76.88%→82.48%），但对另一些反而有害（如 X 光 94.21%→92.30%），因为忽略了模态特定知识
3. **缺乏医学先验**：模型可能偏向学习机器噪声等浅层纹理，而非解剖学特征
4. **实际应用驱动**：历史病历检索和医学图像隐私保护都需要 MedReID 能力

本文定义并构建了 MedReID 的完整 benchmark。

## 方法详解

### 整体框架

MaMI 基于 ViT-Base（CLIP 预训练），两个核心模块：(1) ComPA 连续模态参数适配器根据输入图像动态调整网络参数；(2) 医学先验转移模块通过差异特征对齐从医学基础模型继承解剖学知识。

### 关键设计

**1. 连续模态参数适配器 (ComPA)**

- **功能**：将模态无关模型在运行时动态调整为模态特定模型
- **核心思路**：将每个 $16 \times 16$ patch 通过 MLP 转为局部模态上下文，平均后得到全局模态上下文。通过另一个 MLP 生成模态概率向量 $\mathbf{w} \in \mathbb{R}^L$（$L=32$ 个伪模态），加权求和可学习模态基 $\Omega \in \mathbb{R}^{L \times 768}$ 得到连续模态特征 $\mathcal{M}_i$。Att-PNet 和 FFN-PNet 从 $\mathcal{M}_i$ 生成低秩参数（LoRA 格式），合并到 ViT 各层
- **设计动机**：$L=32$ 远超实际模态数，因为同一模态内不同设备/参数会产生多种成像风格。连续表示比离散标签更灵活，且对训练域外的图像泛化更好

**2. 基于差异特征的医学先验对齐**

- **功能**：将预训练医学基础模型（MFMs）的解剖学知识迁移到 ReID 任务
- **核心思路**：从模态特征 $\mathcal{M}_i$ 生成 $N$ 个模态特定 query token，通过注意力机制从特征图中提取关键特征 $\mathbf{P}_i^n$。对于图像对 $(x_i, x_j)$，计算差异 $\mathbf{u}^n = \mathbf{P}_i^n - \mathbf{P}_j^n$ 和 MFM 差异 $\mathbf{v}^n = \mathbf{Q}_i^n - \mathbf{Q}_j^n$，用对比损失对齐
- **设计动机**：单图特征对齐与 ReID 任务不一致（ReID 需要区分多张图像），差异特征对齐更贴合任务本质。Query token 依赖模态特征，可关注不同模态的关键结构

**3. 模态特定 Query Token**

- **功能**：根据模态自动关注不同的关键解剖结构
- **核心思路**：从 $\mathcal{M}_i$ 通过 3 层 MLP 生成 $N$ 个 query token $\mathbf{O}_i$，通过注意力 $\mathbf{A}_i^n = \text{Softmax}(\frac{\mathbf{O}_i^n \cdot \text{Linear}(\mathbf{f}_i)}{\sqrt{d}})$ 池化得到关键特征
- **设计动机**：X 光的身份线索在肋骨/心脏形状，眼底的线索在视盘/血管，不同模态需要关注不同区域

### 损失函数

$$\mathcal{L} = \mathcal{L}_{id\text{-}classify} + \mathcal{L}_{tri} + \lambda \mathcal{L}_{med\text{-}align}$$

$\mathcal{L}_{id\text{-}classify}$ 为身份分类交叉熵损失，$\mathcal{L}_{tri}$ 为软 margin 三元组损失，$\mathcal{L}_{med\text{-}align}$ 为差异特征对比损失，$\lambda = 0.01$。

## 实验关键数据

### 与基础模型和大语言模型的对比（CMC-R1 %）

| 方法 | MIMIC-X | CCII-CT | EyePACS | Chest-X | OASIS-MRI | 平均 |
|------|---------|---------|---------|---------|-----------|------|
| CLIP | 33.10 | 58.82 | 41.14 | - | - | - |
| BiomedCLIP | 42.30 | 68.91 | 53.48 | - | - | - |
| GPT-4V | 12.50 | 15.00 | 10.20 | - | - | - |
| Single-modality | 94.21 | - | 76.88 | - | - | - |
| **MaMI (Ours)** | **96.89** | **95.59** | **85.71** | **91.23** | **80.00** | - |

### 消融实验

| 方法 | X-ray (%) | Fundus (%) |
|------|-----------|-----------|
| CLIP baseline | 33.10 | 41.14 |
| Single-modality | 94.21 | 76.88 |
| Multiple-modality | 92.30 | 82.48 |
| **Continuous-modality (Ours)** | **96.89** | **85.71** |

### 关键发现

- MaMI 单模型同时超越单模态专用模型（X-ray +2.68%，Fundus +8.83%）
- ComPA 连续模态策略在两个模态上都优于朴素多模态联合训练
- 差异特征对齐比单图特征对齐提升 ~2% CMC-R1
- 25 个基础模型中表现最好的 BiomedCLIP 也仅 42.30%（X-ray），说明通用模型在 MedReID 上严重不足

## 亮点与洞察

1. **问题定义贡献大**：首次系统定义 MedReID 问题并建立完整 benchmark
2. **ComPA 设计优雅**：连续模态表示 + 动态 LoRA 生成，既灵活又高效
3. **差异特征对齐巧妙**：将 MFM 先验适配到 ReID 任务的方式值得借鉴
4. **实际应用价值**：展示了在个性化诊断和隐私保护中的真实部署

## 局限与展望

- $L=32$ 个伪模态的设置可能对模态数量非常多时不够
- 多切片扫描仅用简单平均，丢失了切片间的空间信息
- 未探索 3D 卷积或更高级的切片间建模
- 隐私保护的身份移除后果需更多安全验证

## 相关工作与启发

- **FisherRF**：首次将 ReID 应用于胸部 X 光
- **CLIP/BiomedCLIP**：通用视觉-语言模型在 MedReID 上表现不佳
- **LoRA/MOE-LoRA**：参数高效微调方法，ComPA 通过运行时动态生成超越了静态 LoRA

## 评分

⭐⭐⭐⭐⭐ — 问题定义新颖且有实际价值，方法设计完善，实验全面（11 个数据集、25+ 基线）。ComPA 和差异特征对齐都是高质量贡献。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](../../ICCV2025/llm_safety/oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)
- [\[AAAI 2026\] Federated CLIP for Resource-Efficient Heterogeneous Medical Image Classification](../../AAAI2026/llm_safety/federated_clip_for_resource-efficient_heterogeneous_medical_image_classification.md)
- [\[ICLR 2026\] Train Once, Answer All: Many Pretraining Experiments for the Cost of One](../../ICLR2026/llm_safety/train_once_answer_all_many_pretraining_experiments_for_the_cost_of_one.md)
- [\[ICLR 2026\] Attention Smoothing Is All You Need For Unlearning](../../ICLR2026/llm_safety/attention_smoothing_is_all_you_need_for_unlearning.md)
- [\[ACL 2025\] Core: Robust Factual Precision with Informative Sub-Claim Identification](../../ACL2025/llm_safety/core_robust_factual_precision_with_informative_sub-claim_identification.md)

</div>

<!-- RELATED:END -->
