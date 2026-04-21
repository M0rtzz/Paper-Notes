---
title: >-
  [论文解读] Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics
description: >-
  [AAAI 2026][医学图像][空间转录组学] 提出 DKAN，一个双路径知识增强对比对齐网络，通过整合外部基因数据库的语义信息作为跨模态协调器，结合统一的一阶段对比学习范式和自适应加权机制，从病理组织切片图像（H&E WSI）预测空间分辨率的基因表达，在三个公开ST数据集上全面超越SOTA。
tags:
  - AAAI 2026
  - 医学图像
  - 空间转录组学
  - 基因表达预测
  - 对比学习
  - 多模态对齐
  - 知识增强
  - 病理图像
---

# Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics

**会议**: AAAI 2026  
**arXiv**: [2511.17685](https://arxiv.org/abs/2511.17685)  
**代码**: [coffeeNtv/DKAN](https://github.com/coffeeNtv/DKAN)  
**领域**: 医学图像 / 空间转录组学  
**关键词**: 空间转录组学, 基因表达预测, 对比学习, 多模态对齐, 知识增强, 病理图像

## 一句话总结

提出 DKAN，一个双路径知识增强对比对齐网络，通过整合外部基因数据库的语义信息作为跨模态协调器，结合统一的一阶段对比学习范式和自适应加权机制，从病理组织切片图像（H&E WSI）预测空间分辨率的基因表达，在三个公开ST数据集上全面超越SOTA。

## 研究背景与动机

**领域现状**：空间转录组学（Spatial Transcriptomics, ST）可以在保留空间上下文的同时测量组织切片中的基因表达谱，对于理解疾病病因和组织异质性至关重要。然而ST技术成本高昂，分辨率有限，促使研究者探索从低成本的H&E染色全切片图像（WSI）预测空间基因表达。

**现有痛点**：
   - **低级视觉特征依赖**：多数方法主要利用像素强度（颜色分布）和细胞结构（形状纹理）等低级特征，无法捕获基因功能、生物通路、疾病关联等高级语义信息
   - **对样例检索的过度依赖**：基于对比学习和样例引导的流程需要额外构建参考数据集、检索相似patch，增加了不必要的复杂性
   - **异构模态对齐不充分**：现有融合策略直接强制对齐图像与基因表达等异构模态，未能充分保留生物学相关的交互信息

**核心矛盾**：图像特征和基因表达特征属于完全异构的模态，直接对齐困难且容易丢失生物学含义；现有对比学习方法需要额外的检索步骤带来流程冗余。

**本文目标** 在不依赖样例检索的前提下，实现图像与基因表达的有效多模态对齐，同时引入高级生物学语义来提升预测精度。

**切入角度**：引入外部基因数据库知识作为"桥梁"来间接对齐图像和表达两个异构模态——不是直接把苹果和橘子比较，而是通过一个共同的知识中介来建立二者的联系。

**核心 idea**：用基因语义特征作为动态跨模态协调器，在双路径中分别与图像和表达特征交互，实现知识引导的隐式模态对齐。

## 方法详解

### 整体框架

DKAN包含四个核心模块：
1. **基因语义表示模块**（Gene Semantic Representation）：从NCBI基因数据库检索基因知识，用LLM（GPT-4o）生成结构化基因语义文本，再用BioBERT编码为1024维特征，经Transformer处理得到语义特征 $f^{text}$
2. **基因表达嵌入模块**（Gene Expression Embedding）：将 $N_p \times N_g$ 的基因表达矩阵通过线性层+GELU+残差连接+层归一化编码为 $f^{exp}$
3. **多层级图像嵌入模块**（Multi-level Image Embedding）：提取WSI级、区域级（k=25近邻patch）和patch级三个层次的图像特征并通过交叉注意力融合为 $f^{img}$
4. **双路径对比对齐模块**（Dual-Path Contrastive Alignment）：用基因语义特征作为query分别查询图像和表达特征，生成知识增强表示 $e^{ti}$ 和 $e^{te}$，进行对比学习

### 关键设计

**基因语义表示**：
- 从NCBI数据库检索NgN个目标基因的功能、通路、疾病关联等信息
- 设计结构化prompt（包含角色定义、任务要求、输出规范），输入GPT-4o生成规范化的基因语义文本
- 用BioBERT（预训练于大规模生物医学语料）编码为1024维向量
- 经线性投影对齐维度后送入标准Transformer，捕获基因语义嵌入间的全局依赖

**多层级图像嵌入**：
- WSI级和区域级：用UNI（组织病理学预训练基础模型）提取特征（冻结权重），各接多头Transformer
- Patch级：用可训练的ResNet18提取特征（去掉最后的池化和全连接层）
- 融合策略：两个交叉注意力模块——WSI作query分别与区域级和patch级做交叉注意力，结果相加得到最终 $f^{img}$

**双路径对比对齐**：
- 图像路径：基因语义特征作为"功能查询指令"，通过交叉注意力从图像特征中过滤形态学相关区域
- 表达路径：基因语义特征作为"分布校正因子"，通过交叉注意力约束预测基因表达特征
- 每个语义特征独立查询图像和表达特征，生成 $e^{ti}$ 和 $e^{te}$
- 关键优势：不直接强制对齐异构模态，而是各自与语义知识交互后实现隐式对齐

**统一一阶段对比学习**：
- 训练阶段利用全部模态，推理阶段仅用图像和语义模态
- 无需构建参考数据集或进行样例检索
- 正样本：同一基因的 $e^{ti}$ 和 $e^{te}$；负样本：不同基因的表示

### 损失函数

总损失由对比损失和监督损失组成，采用自适应加权：

$$\mathcal{L} = w_{sup}\mathcal{L}_{sup} + w_{cont}\mathcal{L}_{cont}$$

**对比损失**（InfoNCE形式）：
$$\mathcal{L}_{cont} = -\sum_i \log \frac{\exp(sim(e_{ti}^i, e_{te}^i)/\tau)}{\sum_j \exp(sim(e_{ti}^i, e_{te}^j)/\tau)}$$

**监督损失**（含知识蒸馏）：
$$\mathcal{L}_{sup} = \sum_{d \in \mathcal{D}} \mathcal{L}_d + \|\hat{Y} - Y\|^2$$

其中每个中间预测的蒸馏损失：$\mathcal{L}_d = \lambda\|\hat{Y}_d - \hat{Y}\|^2 + (1-\lambda)\|\hat{Y}_d - Y\|^2$

**自适应加权**：权重按损失值的归一化倒数动态调整，使较小的损失获得更高权重，避免某一目标主导训练。

## 实验

### 数据集

| 数据集 | 样本数 | 患者数 | 斑点数 | 基因数/斑点 | 类型 |
|--------|--------|--------|--------|------------|------|
| HER2+ | 36 | 8 | 13,620 | 14,873 | 乳腺癌 |
| STNET | 68 | 23 | 30,612 | 26,949 | 乳腺癌 |
| cSCC | 12 | 4 | 8,671 | 17,047 | 皮肤鳞状细胞癌 |

### 主实验结果（vs. 10个SOTA方法）

**HER2+ 数据集**：

| 方法 | MAE↓ | MSE↓ | PCC-ALL↑ | PCC-HPG↑ | PCC-HEG↑ | PCC-HVG↑ |
|------|------|------|----------|----------|----------|----------|
| TRIPLEX (最强基线) | 0.364 | 0.234 | 0.304 | 0.491 | 0.271 | 0.260 |
| **DKAN (Ours)** | **0.361** | **0.224** | **0.330** | **0.531** | **0.317** | **0.304** |

**STNET 数据集**：DKAN 在 MAE（0.322）、MSE（0.179）和所有PCC指标上全面领先。

**cSCC 数据集**：DKAN 在 MAE（0.383）、MSE（0.239）和 PCC-ALL（0.407）上均显著超越所有基线，提升幅度最大（PCC-ALL从0.363提升至0.407）。

### 消融实验

| 消融项 | PCC-ALL变化 | 说明 |
|--------|------------|------|
| 去掉多尺度上下文 | 0.219→0.117 | 多层级图像特征至关重要 |
| 去掉基因语义文本 | 0.219→0.210 | 语义信息提供有效生物学先验 |
| 去掉对比学习 | 0.219→0.209 | 对比学习提升跨模态对齐质量 |
| 文本作KV | 0.219→0.216 | 文本作Query效果更好 |

**编码器选择**：BioBERT > BioGPT > PLIP > Conch（文本编码器）；UNI > Conch > ResNet18 > ResNet50 > PLIP（图像编码器）

**LLM比较**：GPT-4o > DeepSeek-v3 > LLaMA2 > DeepSeek-R1

**融合策略**：交叉注意力 > Sum+Transformer > Concat+Transformer > Sum > Concat

### 关键发现

1. 基因语义知识的引入在所有数据集上都带来一致提升，验证了高级生物学先验对于基因表达预测的重要性
2. 一阶段统一对比学习消除了对样例检索的依赖，不仅简化了流程还提升了性能
3. 双路径设计比直接对齐异构模态更有效——通过语义知识中介实现的隐式对齐质量更高
4. 癌症biomarker基因（FN1、HSPB1）的可视化显示DKAN能准确捕获空间表达模式

## 亮点与洞察

1. **知识增强范式的创新**：首次将外部基因数据库知识系统性地整合进空间转录组学预测，用LLM生成的结构化基因语义文本为模型提供生物学先验
2. **桥梁式对齐思想**：不直接对齐异构模态，而是通过共享的语义知识空间实现隐式对齐——这一思路对其他异构多模态任务有广泛启发
3. **一阶段范式简化**：将对比学习与监督学习统一到端到端训练中，消除了样例检索的冗余步骤
4. **自适应权重平衡**：动态调整对比损失和监督损失的权重，避免收敛速度差异导致的优化失衡

## 局限性

1. 基因语义文本的质量依赖GPT-4o和NCBI数据库的当前知识，可能存在偏差或不完整
2. 实验仅在三个公开ST数据集（两个乳腺癌+一个皮肤癌）上验证，其他组织类型和疾病的泛化性有待测试
3. WSI级和区域级图像编码器（UNI）冻结权重，可能无法充分适应特定的组织类型
4. 250个空间可变基因的选择可能限制了方法对其他基因集合的适用性
5. 需要预先获取GPT-4o生成的基因描述文本，增加了预处理成本

## 相关工作

- **局部方法**：ST-Net (DenseNet-121), EGN/EGGN (样例检索+图卷积), SEPAL (邻域图+GNN), BLEEP (CLIP式对比学习), mclSTExp
- **全局方法**：HisToGene (ViT+位置编码), HE2RNA (超级瓦片聚合), THItoGene (动态卷积+胶囊模块+ViT+GATv2)
- **多尺度方法**：Hist2ST, TRIPLEX (多视图特征组合), M2OST (多对一预测), ST-Align (niche级聚类+三目标对齐)

## 评分与推荐

⭐⭐⭐⭐ (4/5)

- 创新性: ⭐⭐⭐⭐ — 知识增强+双路径隐式对齐思路新颖
- 实验: ⭐⭐⭐⭐ — 三个数据集全面超越10个基线，消融充分
- 写作: ⭐⭐⭐⭐ — 结构清晰，图示信息量大
- 实用性: ⭐⭐⭐⭐ — 提供代码，流程可复现

<!-- RELATED:START -->

## 相关论文

- [NutriScreener: Retrieval-Augmented Multi-Pose Graph Attention Network for Malnourishment Screening](nutriscreener_retrieval-augmented_multi-pose_graph_attention_network_for_malnour.md)
- [Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](../../CVPR2026/medical_imaging/cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)
- [CD-DPE: Dual-Prompt Expert Network Based on Convolutional Dictionary Feature Decoupling for Multi-Contrast MRI Super-Resolution](cd-dpe_dual-prompt_expert_network_based_on_convolutional_dictionary_feature_deco.md)
- [SpaCRD: Multimodal Deep Fusion of Histology and Spatial Transcriptomics for Cancer Region Detection](spacrd_multimodal_deep_fusion_of_histology_and_spatial_transcriptomics_for_cance.md)
- [HiFusion: Hierarchical Intra-Spot Alignment and Regional Context Fusion for Spatial Gene Expression Prediction from Histopathology](hifusion_hierarchical_intra-spot_alignment_and_regional_context_fusion_for_spati.md)

<!-- RELATED:END -->
