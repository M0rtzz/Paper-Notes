---
title: >-
  [论文解读] DermaCon-IN: A Multi-concept Annotated Dermatological Image Dataset of Indian Skin Disorders
description: >-
  [NeurIPS 2025][医学图像][皮肤病数据集] 构建了 DermaCon-IN——首个以印度肤色为主的密集标注皮肤病图像数据集（5,450 张 / 3,002 患者 / 245 种诊断），提供三级层次诊断标签、47 个病灶描述符和 49 个解剖位置标注，并用 CNN/ViT/概念瓶颈模型进行基准评测。
tags:
  - NeurIPS 2025
  - 医学图像
  - 皮肤病数据集
  - 印度肤色
  - 概念瓶颈模型
  - 层次分类
  - 可解释AI
---

# DermaCon-IN: A Multi-concept Annotated Dermatological Image Dataset of Indian Skin Disorders

**会议**: NeurIPS 2025  
**arXiv**: [2506.06099](https://arxiv.org/abs/2506.06099)  
**代码**: [GitHub](https://github.com/) / [Harvard Dataverse](https://dataverse.harvard.edu/)  
**领域**: 医学图像  
**关键词**: 皮肤病数据集, 印度肤色, 概念瓶颈模型, 层次分类, 可解释AI

## 一句话总结
构建了 DermaCon-IN——首个以印度肤色为主的密集标注皮肤病图像数据集（5,450 张 / 3,002 患者 / 245 种诊断），提供三级层次诊断标签、47 个病灶描述符和 49 个解剖位置标注，并用 CNN/ViT/概念瓶颈模型进行基准评测。

## 研究背景与动机

**领域现状**：皮肤病是全球第四大非致命疾病负担。AI 辅助诊断被视为解决皮肤科资源不足的重要手段，但现有模型主要在欧美数据集上训练和评估。

**现有痛点**：
   - 数据集偏倚严重：ISIC、HAM10000 等聚焦黑色素瘤等肿瘤性病变，忽略真菌感染、疥疮等热带高发病种
   - 肤色代表性不足：Fitzpatrick17k 中 75%+ 为 Type I-III（浅肤色），深色皮肤（Type IV-VI）严重欠缺
   - 标注维度单一：多数数据集只有诊断标签，缺少解剖位置和病灶形态学描述符
   - 数据来源偏差：SD-198、Fitzpatrick17k 来自教学图谱而非真实临床采集

**核心矛盾**：现有数据集的地域/肤色/病种偏倚导致 AI 模型对深肤色人群准确率下降 30-40%（DDI 数据集实验），无法公平服务全球人口

**切入角度**：从印度门诊前瞻性采集，构建兼具疾病覆盖广度、肤色代表性和多维标注的数据集

**核心 idea**：提供首个 South Asian 肤色为主、同时包含诊断层次 + 解剖位置 + 病灶描述符三重标注的皮肤病数据集

## 方法详解

### 整体框架
数据集构建遵循"采集→标注→质控→基准评测"pipeline。临床图像从南印度 3 家三级医院采集，4 位皮肤科认证医师采用 Rook 分类体系进行三级层次标注 + 概念标注，最终用多种架构进行分类与可解释性基准评测。

### 关键设计

1. **三级层次诊断标签**

    - 功能：按 Rook 皮肤病学教科书构建 8 个主类 → 19 个亚类 → 245 种具体疾病的层次分类
    - 核心思路：主类基于病因学（感染性、炎症性、色素性、角化性等），亚类处理混合/共存状态（如炎症+真菌感染），具体标签对应 ICD-11
    - 设计动机：反映临床诊断工作流（先判断大类再细化），支持粗粒度和细粒度两级建模。疾病标签呈长尾分布（log-normal 指数 1.8），与真实门诊频率一致

2. **双维度概念标注（96 个概念）**

    - **47 个病灶描述符**：鳞屑、红斑、水疱、色素沉着等，遵循临床皮肤科词汇标准
    - **49 个解剖位置**：头皮、手掌、足底、躯干等，保留全视野解剖上下文
    - 设计动机：皮肤科诊断依赖病灶形态 + 解剖位置的组合推理（如头皮鳞屑→银屑病 vs 足部鳞屑→足癣），独立标注两类概念可支持概念瓶颈模型的可解释性研究
    - Pearson 相关分析验证了概念-疾病关联的临床合理性（如白斑↔色素障碍 r=+0.71）

3. **概念瓶颈模型（CBM）基准**

    - 功能：在 Swin-B 骨干网络上构建 CBM，通过概念层进行可解释性分类
    - 架构：图像 → Swin Encoder → 概念 logits $c^\ell \in \mathbb{R}^{B+D}$ → sigmoid → 概念向量 $c$ → 分类器 → 诊断
    - 探索了两种层次 CBM 设计：
        - Type 1（级联）：概念 → 亚类预测 → 主类预测，保证分类一致性
        - Type 2（并行）：概念同时独立预测亚类和主类，多任务学习正则化
    - Type 2 在所有指标上优于 Type 1

### 损失函数 / 训练策略
- 概念层用 BCE 损失监督 96 个概念的二分类
- 分类层用交叉熵 + 加权采样处理类不平衡
- ImageNet-22k 预训练，input resize 到 512×512，subject-wise 80:20 分层切分

## 实验关键数据

### 主实验：8-主类分类

| 模型 | 预训练 | Accuracy | Balanced Acc. | F1 Score |
|------|--------|----------|--------------|----------|
| ResNet50 | - | 47.45 | 23.93 | 46.43 |
| ResNet50 | ImageNet | 64.31 | 38.77 | 63.31 |
| DenseNet121 | ImageNet | 65.20 | 37.31 | 64.37 |
| ViT-B/16-384 | ImageNet | 66.95 | 35.78 | 65.78 |
| **Swin-B/4W12-384** | **ImageNet** | **70.41** | **45.06** | **69.69** |

Swin Transformer 在所有指标上一致最优。

### CBM 消融实验

| 配置 | Concepts | Accuracy | Macro AUC |
|------|----------|----------|-----------|
| 直接 MC 分类（无 CBM） | - | 70.41 | 78.51 |
| CBM-D（仅描述符） | 47 | 68.57 | 85.18 |
| CBM-B（仅解剖位置） | 49 | 68.38 | 84.96 |
| CBM 全概念 | 96 | 68.12 | 82.78 |
| Type 2 层次 CBM - MC | 96 | **69.90** | 77.01 |

### 关键发现
- CBM 引入概念瓶颈后准确率略降（~2%），但 Macro AUC 显著提升（78.51→85.18），说明概念层提供了更好的类间判别能力
- 单流概念（仅描述符或仅位置）性能相当，但合并两流时出现竞争效应——模型倾向只激活一组概念而抑制另一组
- Grad-CAM 验证概念激活确实定位到语义正确的解剖区域
- 色素障碍和角化障碍类的概念-权重对齐（Spearman 相关）良好（p<0.05），但肿瘤类对齐差，可能因样本量不足

## 亮点与洞察
- **首个印度肤色密集标注数据集**：同时覆盖 Fitzpatrick IV-VI 和 MST 4-9 肤色，填补了全球皮肤病 AI 公平性的关键空白
- **三重标注体系巧妙**：诊断层次 + 病灶描述符 + 解剖位置的组合，首次在数据集层面系统化了皮肤科的临床推理路径（形态+位置→诊断）
- **概念竞争现象**的发现（双流 CBM 中一组概念被抑制）指出了多概念学习中的表示瓶颈问题，值得后续研究
- Cohen's Kappa = 0.84 的标注一致性保证了数据质量

## 局限与展望
- 面部图像因隐私脱敏（遮挡眼部/裁切）影响面部疾病的建模能力
- 长尾分布中稀有疾病样本极少，需要 few-shot/long-tail 学习策略
- 缺少像素级分割标注，无法支持病灶定位和分割任务
- 数据来源集中于南印度卡纳塔克邦，未覆盖北印度和东南亚其他地区
- CBM 中概念竞争问题需要新的正则化或注意力机制来解决

## 相关工作与启发
- **vs Fitzpatrick17k**: 后者来自图谱而非临床，75% 浅肤色，缺少真菌/病毒感染；DermaCon-IN 是前瞻性临床采集，覆盖感染性疾病
- **vs SkinCon**: SkinCon 事后对已有数据集添加描述符；DermaCon-IN 在源头同时采集病灶+位置概念
- **vs DDI**: DDI 仅 656 张 / 78 类，DermaCon-IN 规模更大（5,450 / 245 类）且标注更密集
- **vs PASSION**: PASSION 聚焦非洲儿童 4 种病，DermaCon-IN 覆盖 245 种全谱系成人疾病

## 评分
- 新颖性: ⭐⭐⭐⭐ 填补印度肤色数据空白+三重标注体系设计新颖
- 实验充分度: ⭐⭐⭐⭐ 多架构对比+CBM 探索+Grad-CAM 定性分析，全面
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，相关工作对比表格详尽
- 价值: ⭐⭐⭐⭐⭐ 数据集贡献对公平性研究和全球适用性有重要意义

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Interactive Medical Image Analysis with Concept-based Similarity Reasoning](../../CVPR2025/medical_imaging/interactive_medical_image_analysis_with_concept-based_similarity_reasoning.md)
- [\[NeurIPS 2025\] RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis](ram-w600_a_multi-task_wrist_dataset_and_benchmark_for_rheumatoid_arthritis.md)
- [\[ACL 2025\] Concept Bottleneck Language Models For Protein Design](../../ACL2025/medical_imaging/concept_bottleneck_language_models_for_protein_design.md)
- [\[NeurIPS 2025\] STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology](starc-9_a_large-scale_dataset_for_multi-class_tissue_classification_for_crc_hist.md)
- [\[NeurIPS 2025\] Care-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment](care-pd_a_multi-site_anonymized_clinical_dataset_for_parkinsons_disease_gait_ass.md)

<!-- RELATED:END -->
