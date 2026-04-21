---
title: >-
  [论文解读] LLM-Assisted Emergency Triage Benchmark: Bridging Hospital-Rich and MCI-Like Field Simulation
description: >-
  [医学图像] 基于MIMIC-IV-ED构建了一个开放的、LLM辅助策划的急诊分诊基准数据集，定义了医院丰富资源和大规模伤亡事件(MCI)模拟两种场景，提供基线模型和SHAP可解释性分析，推动分诊预测研究的可复现性和普及化。
tags:
  - 医学图像
---

# LLM-Assisted Emergency Triage Benchmark: Bridging Hospital-Rich and MCI-Like Field Simulation

## 元信息
- **会议**: NeurIPS 2025
- **arXiv**: [2509.26351](https://arxiv.org/abs/2509.26351)
- **代码**: 暂无
- **领域**: 医学图像
- **关键词**: 急诊分诊, 大语言模型, 基准数据集, 恶化预测, MIMIC-IV

## 一句话总结
基于MIMIC-IV-ED构建了一个开放的、LLM辅助策划的急诊分诊基准数据集，定义了医院丰富资源和大规模伤亡事件(MCI)模拟两种场景，提供基线模型和SHAP可解释性分析，推动分诊预测研究的可复现性和普及化。

## 研究背景与动机

急诊科面临巨大压力，需要在大量患者中快速识别恶化风险（如意外ICU转入或院内死亡），尤其是在资源有限的大规模伤亡事件(MCI)中。然而，现有的分诊研究存在三个核心瓶颈：

**缺乏可复现的基准**：尽管MIMIC-IV-ED公开可用，但将其转化为分诊导向的基准需要大量预处理、特征协调和模式对齐，技术门槛极高

**场景覆盖不全**：现有工作通常聚焦于医院丰富资源场景，缺乏对MCI现场有限条件的模拟

**传统评分系统局限**：NEWS2、AVPU、START等评分依赖固定阈值和狭窄输入，跨人群表现不稳定

**核心动机**：利用LLM辅助数据策划，降低技术壁垒，构建覆盖医院和现场两种场景的开放分诊基准，实现"数据集民主化"。

## 方法详解

### 整体框架

基于MIMIC-IV v3.1和MIMIC-IV-ED v2.2构建确定性预处理流水线，生成两种特征体制下的分诊基准数据集，并提供基线模型和可解释性分析。

### 关键设计

1. **数据构建流水线**：从ED就诊记录出发，通过(subject_id, hadm_id)等临床有意义的键进行记录链接，避免跨入院泄漏。限制生命体征和实验室数据在到达后1小时内。对生理不合理值进行规则过滤，连续特征z-score标准化，缺失值用均值/未知类别填补。所有预处理参数仅在训练折上学习。

2. **双场景特征体制**：

    - **医院丰富(Hospital-rich)**：人口统计学 + 初始ED生命体征 + 主诉 + 分诊观察（疼痛、敏锐度）+ 早期实验室（血红蛋白、BUN、钠、钾、肌酐）+ 意识/呼吸代理指标
    - **MCI现场模拟(MCI-like)**：仅保留人口统计学 + 生命体征 + 主诉 + 分诊观察 + AVPU/氧气标志

3. **LLM辅助策划**：LLM用于数据整理而非预测建模。具体包括：

    - GCS言语反应到AVPU的一致映射和one-hot编码
    - 氧气支持设备标准化（室内空气/鼻导管/面罩/CPAP等 + 二元标志）
    - 呼吸记录噪声过滤（如"clear"/"regular"等模糊值）
    - 主诉自由文本的关键词提取（含同义词扩展和简单否定处理）
    - 表合并策略（连接键和去重规则）

4. **衍生特征**：包括从GCS言语分推导的AVPU编码、氧气支持分级向量、休克指数(HR/SBP)等。

### 预测任务定义

二分类任务：预测ED到达后24小时内意外ICU转入或同次住院期间院内死亡。正类(标签=1)为上述复合结局，其余为负类。

### 基线模型

选用四种可解释性强的模型：Logistic Regression、Random Forest、XGBoost、LightGBM。通过5折交叉验证网格搜索调参，患者级别70/30分层划分。

## 实验关键数据

### 主实验：两种场景基线性能

| 模型 | AUROC(Hospital) | Acc(Hospital) | AP(Hospital) | F1(Hospital) | AUROC(MCI) | Acc(MCI) | AP(MCI) | F1(MCI) |
|------|----------------|--------------|-------------|-------------|-----------|---------|--------|--------|
| Logistic Regression | 0.40 | 0.43 | 0.27 | 0.15 | 0.703 | 0.761 | 0.575 | 0.429 |
| Random Forest | 0.73 | 0.72 | 0.38 | 0.35 | 0.783 | 0.851 | 0.721 | 0.643 |
| XGBoost | 0.56 | 0.65 | 0.33 | 0.36 | 0.734 | 0.746 | 0.599 | 0.452 |
| LightGBM | 0.39 | 0.60 | 0.30 | 0.20 | 0.794 | 0.791 | 0.690 | 0.563 |

MCI场景下Random Forest表现最优（AUROC=0.783, F1=0.643），且MCI场景整体优于Hospital-rich场景——可能因demo子集中早期实验室数据噪声导致Hospital-rich退化。

### 消融实验：特征组贡献

| 最佳模型 | 特征集 | AUROC | Accuracy | AP | F1 |
|---------|-------|-------|---------|-----|-----|
| Logistic Regression | 观察指标 | 0.74 | 0.79 | 0.43 | 0.42 |
| LightGBM | 生命体征 | 0.80 | 0.79 | 0.49 | 0.63 |
| LightGBM | 实验室 | 0.71 | 0.72 | 0.36 | 0.46 |
| Random Forest | 体征+观察 | 0.76 | 0.82 | 0.60 | 0.63 |
| Random Forest | 体征+观察+实验室 | 0.82 | 0.81 | 0.67 | 0.61 |

### 关键发现

1. **生命体征是最稳健的信号源**：单独使用即可达AUROC=0.80，实验室数据在1小时窗口内信号弱且噪声大
2. **观察指标高性价比**：敏锐度、AVPU等简单床旁评估具有显著预测价值，累积添加后一致提升F1和AP
3. **SHAP全局分析**：分诊敏锐度和呼吸指标（呼吸率、血氧、血压）是所有场景下最重要的驱动因素，与临床直觉高度吻合
4. **MCI场景可行性**：即使只有生命体征和简单观察，模型仍具有强预测能力，支持资源受限场景下的应用

## 亮点与洞察

1. **LLM作为数据策划加速器**：不用于预测，而用于解决数据清洗中最耗时的特征协调问题，是LLM在医疗AI中一种"安全且高效"的应用范式
2. **双场景设计**：首次在同一基准中桥接医院和现场分诊，为MCI研究提供了标准化评估框架
3. **反直觉发现**：在demo子集上MCI场景性能优于Hospital-rich，提示早期实验室数据可能引入噪声——需要在全量数据上验证
4. **数据集民主化理念**：通过释放预处理代码、特征字典和划分索引，降低分诊研究的技术准入门槛

## 局限与展望

- 仅在MIMIC demo子集（64患者、222次就诊）上验证，样本量极小
- 未纳入波形数据和叙述性临床笔记
- LLM辅助策划目前为交互式验证，未完全自动化
- 院内死亡率在demo子集中为0%，限制了结局事件的代表性
- 后续应扩展至完整MIMIC-IV，并对比深度序列模型（如RETAIN）

## 相关工作与启发

- **传统分诊评分**: NEWS2, AVPU, START, SALT — 简单但固定阈值限制适用性
- **ML恶化预测**: Gradient Boosting优于传统量表，文本增强模型接近医师水平
- **LLM数据策划**: DALL-M特征增强、检索增强LLM用于模式对齐
- **可解释性**: SHAP在ICU和MCI模型中广泛使用，透明化分诊决策
- **启发**: LLM辅助数据策划的范式可推广到其他EHR基准构建

## 评分
- 新颖性：⭐⭐⭐☆☆ — 数据集构建工作，方法创新有限但场景设计独特
- 实验充分度：⭐⭐☆☆☆ — demo子集过小，结论泛化性存疑
- 写作质量：⭐⭐⭐⭐☆ — 流程清晰，动机充分
- 价值：⭐⭐⭐⭐☆ — 推动分诊AI可复现研究，扩展至全量数据后价值更大

<!-- RELATED:START -->

## 相关论文

- [Scaling Laws and Pathologies of Single-Layer PINNs: Network Width and PDE Nonlinearity](scaling_laws_and_pathologies_of_single-layer_pinns_network_width_and_pde_nonline.md)
- [Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis](scalable_diffusion_transformer_for_conditional_4d_fmri_synthesis.md)
- [RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification](raxss_retrieval-augmented_sparse_sampling_for_explainable_variable-length_medica.md)
- [Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)
- [Protein Structure Tokenization: Benchmarking and New Recipe](../../ICML2025/medical_imaging/protein_structure_tokenization_benchmarking_and_new_recipe.md)

<!-- RELATED:END -->
