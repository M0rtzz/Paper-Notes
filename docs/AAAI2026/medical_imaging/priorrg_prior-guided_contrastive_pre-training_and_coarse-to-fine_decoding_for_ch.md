---
title: >-
  [论文解读] PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation
description: >-
  [AAAI 2026][医学图像][胸部X光报告生成] PriorRG 提出了一个两阶段胸部X光报告生成框架，通过先验引导的对比预训练对齐临床语境与时空视觉特征，再通过先验感知的粗到细解码逐步融合临床上下文、疾病进展和多层级视觉线索…
tags:
  - "AAAI 2026"
  - "医学图像"
  - "胸部X光报告生成"
  - "先验知识"
  - "对比预训练"
  - "粗到细解码"
  - "时空融合"
---

# PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation

**会议**: AAAI 2026  
**arXiv**: [2508.05353](https://arxiv.org/abs/2508.05353)  
**代码**: [GitHub](https://github.com/mk-runner/PriorRG)  
**领域**: 医学影像 / 放射学报告生成  
**关键词**: 胸部X光报告生成, 先验知识, 对比预训练, 粗到细解码, 时空融合

## 一句话总结

PriorRG 提出了一个两阶段胸部X光报告生成框架，通过先验引导的对比预训练对齐临床语境与时空视觉特征，再通过先验感知的粗到细解码逐步融合临床上下文、疾病进展和多层级视觉线索，在 MIMIC-CXR 上实现 BLEU-4 提升 3.6%、F1 提升 3.8%。

## 研究背景与动机

放射学报告自动生成（RRG）旨在减轻放射科医生工作负担，通过 AI 自动解读医学影像并生成结构化文本描述。当前大多数方法仅基于单张影像生成报告（见论文 Figure 1(a)），忽略了放射科医生在实际诊断中常规依赖的**患者特定先验知识**，包括：

**临床上下文（Clinical Context, CC）**：即适应症（indication）和病史（medical history），反映患者的诊断意图

**最近一次先验影像（Prior Image, PI）**：用于追踪疾病进展

已有的一些工作尝试利用部分先验信息，但存在明显局限：
- SEI 等方法利用了适应症信息，但忽略纵向（longitudinal）数据，生成的报告在描述疾病进展时容易产生幻觉
- HERGen 等方法引入了先验影像来建模时间变化，但忽略了临床上下文，缺乏个性化推理能力

这引出了核心研究问题：**能否同时建模时间视觉变化和临床上下文，以改善跨模态对齐和报告生成？**

## 方法详解

### 整体框架

PriorRG 采用两阶段训练流水线（见论文 Figure 2），模拟真实临床工作流：
- **视觉编码器**：RAD-DINO（冻结参数）
- **文本编码器**：CXR-BERT（可训练）
- **报告生成器**：DistilGPT2（可训练）

输入包括当前影像 $x_i^{cur}$、先验影像 $x_i^{pri}$（可能缺失）、适应症 $z_i$（可能缺失）和病史 $h_i$（可能缺失）。

### 关键设计

#### 1. Stage 1: 先验引导的对比预训练

**目标**：利用临床上下文引导时空特征提取，增强跨模态对齐。

**视觉特征提取**：使用 RAD-DINO 提取特征后，引入可学习的**视图位置嵌入**（view-position embedding）融合到视觉特征中以处理 AP/PA 等不同投影方式带来的外观差异，最终得到 $\boldsymbol{V} \in \mathbb{R}^{M \times s \times d}$。

**文本特征提取**：使用 CXR-BERT 编码文本，在适应症、病史、报告前分别添加特殊标记 `[INDICATION]`、`[HISTORY]`、`[FINDINGS]`，实现类型感知的统一编码，同时优雅处理缺失字段。

**时空融合网络（STF）**：采用 ViT 风格的跨注意力融合模块建模当前影像与先验影像之间的疾病进展：

$$\boldsymbol{V}^{st}_{ca} = \text{LN}(\boldsymbol{V}^{cur} + \text{CA}(\text{LN}(\boldsymbol{V}^{cur}), \text{LN}(\boldsymbol{V}^{pri})))$$

$$\boldsymbol{V}^{st} = \text{LN}(\boldsymbol{V}^{st}_{ca} + \text{FFN}(\boldsymbol{V}^{st}_{ca}))$$

其中 CA 为跨注意力模块，STF 层数设为 3。若无先验影像，则直接使用当前影像特征。

**实例级跨模态对齐**：模拟临床诊断流程，使用 Perceiver 架构逐步融合临床上下文和时空特征：

$$\boldsymbol{\bar{T}}^c = \text{Perceiver}(\boldsymbol{E}^{lat}, \boldsymbol{T}^c)$$

$$\boldsymbol{\bar{V}}^{st} = \text{Perceiver}(\boldsymbol{\bar{T}}^c, \boldsymbol{V}^{st})$$

通过全局平均池化和 L2 归一化获得全局视觉特征 $\boldsymbol{V}^g$，计算图像-报告相似度，使用支持多正样本对的交叉熵损失 $\mathcal{L}_{align}$ 优化对齐。

#### 2. Stage 2: 先验感知的粗到细解码

**注意力增强的层融合网络（ALF）**：基于 CBAM 为每个编码器层的特征施加通道和空间注意力，突出诊断相关特征，通过 Conv2D 投影器融合得到**多层级视觉表示** $\boldsymbol{V}^{hier}$，弥补了现有方法仅用最后一层隐状态、忽略低层细节（如病灶形态）的不足。

**粗到细解码**：受视觉认知原理启发，渐进式整合先验知识与多层级视觉语义：
- **粗粒度先验**：$\boldsymbol{\bar{T}}^c$（临床上下文）和 $\boldsymbol{\bar{V}}^{st}$（时空特征）提供宏观的临床背景和疾病进展线索
- **细粒度增强**：$\boldsymbol{\bar{V}}^{hier} = \text{Perceiver}(\boldsymbol{\bar{V}}^{st}, \boldsymbol{V}^{hier})$，用时空特征作为查询从多层级特征中提取精细信息
- **最终拼接**：将 $\boldsymbol{\bar{T}}^c$、$\boldsymbol{\bar{V}}^{st}$、$\boldsymbol{\bar{V}}^{hier}$ 沿序列维度拼接后输入 DistilGPT2 生成报告

### 损失函数 / 训练策略

- **Stage 1**：对比对齐损失 $\mathcal{L}_{align}$（交叉熵），支持多视图正样本对
- **Stage 2**：交叉熵损失 $\mathcal{L}_{CE}$，训练自回归报告生成
- 统一特征维度 $d=768$，隐变量数 $N=128$，最大生成长度 $K=100$，beam size 为 3
- 使用 AdamW 优化器、ReduceLROnPlateau 调度器、早停（patience=15）
- MIMIC-CXR: Stage 1 训练 30 epoch，Stage 2 微调 30 epoch

## 实验关键数据

### 主实验

| 数据集 | 指标 | PriorRG | 之前SOTA | 提升 |
|--------|------|---------|----------|------|
| MIMIC-CXR | B-1 | 0.412 | 0.416 (MPO) | -0.4% |
| MIMIC-CXR | B-4 | 0.175 | 0.139 (MPO) | +3.6% |
| MIMIC-CXR | MTR | 0.189 | 0.176 (BioViL-T) | +1.3% |
| MIMIC-CXR | R-L | 0.324 | 0.309 (MPO) | +1.5% |
| MIMIC-CXR | F1 | 0.511 | 0.473 (R2-LLM) | +3.8% |
| MIMIC-ABN | B-1 | 0.326 | 0.267 (SEI) | +5.9% |
| MIMIC-ABN | B-4 | 0.102 | 0.073 (SEI) | +2.9% |
| MIMIC-ABN | F1 | 0.471 | 0.460 (CMN) | +1.1% |

PriorRG 在长 n-gram 匹配（B-4）和临床准确性（F1）上全面领先，在 14 个 CheXpert 观察项中的 13 项 F1 优于 SEI。

### 消融实验

| 配置 | B-4 | F1 | 说明 |
|------|-----|-----|------|
| (a) 无CC无PI无Hidden | 0.108 | 0.472 | 基线：仅 Stage 1 无先验知识 |
| (c) 有CC无PI无Hidden | 0.170 | 0.487 | 加入临床上下文显著提升 |
| (e) 有CC有PI无Hidden | 0.171 | 0.499 | 先验影像进一步提升 |
| (d) 有CC无PI有Hidden | 0.173 | 0.507 | 多层级特征提升临床准确性 |
| (f) 无Stage1 全Stage2 | 0.165 | 0.459 | 缺少预训练导致大幅下降 |
| **PriorRG（完整）** | **0.175** | **0.511** | 所有组件协同最优 |

### 关键发现

1. **临床上下文的影响最大**：CC 从 0%→100% 时，B-2 从 0.139 升至 0.294（表5），说明适应症和病史对报告生成至关重要
2. **先验影像的贡献在 Study-level 检索中更明显**：提升了时间线索建模能力，在 Stu-P@K 指标上有显著提升
3. **粗到细优于细到粗**：PriorRG 在 NLG 指标上优于 Fine2coarse 变体，表明先整合高层语义再融入细节的渐进策略更有效
4. **零样本报告生成能力**：在无监督设置下，PriorRG 达到 B-4=0.178、MTR=0.211，显著优于 R2GenGPT 和 Med-LLM

## 亮点与洞察

1. **完整模拟临床工作流**：从初始临床评估→时空对比→多层级细化，高度还原放射科医生的诊断推理过程
2. **优雅处理缺失输入**：通过特殊标记和 Perceiver 架构，自然处理先验影像、适应症或病史缺失的情况，具有很好的实际部署可行性
3. **疾病进展描述能力**：定性分析显示 PriorRG 能正确描述病灶变化（如"心影增大未变"），减少描述时间变化时的幻觉
4. **GREEN 评分领先**：用预训练 GREEN-RadLlama2-7B 评估，在匹配发现数和 GREEN 综合分上均显著优于所有基线

## 局限与展望

1. 依赖 DistilGPT2 作为报告生成器，能力受限于较小模型规模，未来可探索更强的 LLM（如 LLaMA）
2. 仅在 MIMIC-CXR 数据集上验证，未涉及其他影像模态（如 CT、MRI）的报告生成
3. 未引入器官级别的定位信息，作者在结论中提到未来将探索 organ-aware 诊断框架
4. Stage 1 和 Stage 2 分阶段训练，端到端联合优化可能进一步提升性能

## 相关工作与启发

- **BioViL-T / MLRG**：纵向数据建模的先驱工作，但缺乏临床上下文建模
- **SEI**：利用适应症信息的代表方法，但忽视疾病进展
- **Perceiver 架构**：被巧妙用于跨模态渐进融合，是连接不同粒度信息的有效工具
- 启发：临床先验知识（尤其是适应症）对报告生成的影响远超预期，仅有影像信息是不够的

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 首次完整利用临床上下文+先验影像+多层级视觉的组合 |
| 技术深度 | 4 | 两阶段设计合理，Perceiver 渐进融合有创意 |
| 实验充分性 | 5 | 主实验、消融、检索、定性分析、LLM 评估全面 |
| 实用价值 | 4 | 直接对接临床工作流，能处理缺失输入 |
| 写作质量 | 4 | 结构清晰，图示丰富，动机阐述到位 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)
- [\[CVPR 2025\] Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation](../../CVPR2025/medical_imaging/enhanced_contrastive_learning_with_multi-view_longitudinal_data_for_chest_x-ray_.md)
- [\[AAAI 2026\] Vascular Anatomy-aware Self-supervised Pre-training for X-ray Angiogram Analysis](vascular_anatomy-aware_self-supervised_pre-training_for_x-ray_angiogram_analysis.md)
- [\[ACL 2025\] The Impact of Auxiliary Patient Data on Automated Chest X-Ray Report Generation and How to Incorporate It](../../ACL2025/medical_imaging/auxiliary_patient_data_xray.md)
- [\[CVPR 2026\] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](../../CVPR2026/medical_imaging/cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)

</div>

<!-- RELATED:END -->
