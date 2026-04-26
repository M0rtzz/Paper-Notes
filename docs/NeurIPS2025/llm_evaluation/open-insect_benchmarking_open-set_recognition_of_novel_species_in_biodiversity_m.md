---
title: >-
  [论文解读] Open-Insect: Benchmarking Open-Set Recognition of Novel Species in Biodiversity Monitoring
description: >-
  [NeurIPS 2025][开放集识别] 提出Open-Insect——首个面向昆虫物种发现的大规模细粒度开放集识别基准数据集，涵盖三个地理区域和三类开放集划分，系统评测38种OSR算法，发现简单的后验方法（如MSP）在细粒度场景中仍是强基线，同时验证了领域相关辅助数据对提升OSR性能的关键作用。
tags:
  - NeurIPS 2025
  - 开放集识别
  - 细粒度分类
  - 昆虫物种发现
  - OOD检测
  - 生物多样性监测
---

# Open-Insect: Benchmarking Open-Set Recognition of Novel Species in Biodiversity Monitoring

**会议**: NeurIPS 2025  
**arXiv**: [2503.01691](https://arxiv.org/abs/2503.01691)  
**代码**: [GitHub](https://github.com/RolnickLab/Open-Insect)  
**领域**: 开放集识别 · 生物多样性  
**关键词**: 开放集识别, 细粒度分类, 昆虫物种发现, OOD检测, 生物多样性监测

## 一句话总结

提出Open-Insect——首个面向昆虫物种发现的大规模细粒度开放集识别基准数据集，涵盖三个地理区域和三类开放集划分，系统评测38种OSR算法，发现简单的后验方法（如MSP）在细粒度场景中仍是强基线，同时验证了领域相关辅助数据对提升OSR性能的关键作用。

## 研究背景与动机

机器学习在物种识别和生物多样性监测中日益重要，但现有分类器通常假设封闭世界——推理阶段只会遇到训练时见过的类别。这一假设在生物多样性领域严重不成立，原因有三：

**大量未描述物种**：地球约86%的陆地物种和91%的海洋物种尚未被科学描述

**数据覆盖不全**：只有约65%的已描述物种在GBIF数据库中有记录

**区域模型局限**：基于特定地区物种清单训练的模型无法正确处理超出范围的物种

因此，开放集识别（OSR）——同时准确分类已知类别并检测未知类别——对于自动化物种发现和入侵物种检测至关重要。然而现有OSR基准多来自ImageNet，无法反映生物多样性数据的细粒度特性和长尾分布。少数包含生物类别的基准规模较小，且主要关注研究较充分的脊椎动物。昆虫占所有动物物种的三分之二以上，其中超过80%尚未被描述，是理想的研究对象。

## 方法详解

### 整体框架

Open-Insect基于AMI数据集构建，包含5,364种蛾类和12类非蛾节肢动物的图像，覆盖三个地理区域：北美东北部（NE-America）、西欧（W-Europe）和中美洲（C-America）。核心设计思路是通过地理元数据研究不同语义偏移下的物种检测难度。

### 关键设计

1. **三类开放集划分**：

    - **本地蛾类（O-L）**：对每个区域在经纬度上稍作扩展，选取扩展区域中有记录但不在封闭集中的物种，模拟"当地未记录但可能存在的新物种"。基于Tobler地理学第一定律，这些物种与封闭集在分类学上更接近，因此检测难度最大
    - **非本地蛾类（O-NL）**：取澳大利亚的全部蛾类物种（地理隔离确保物种差异性），模拟入侵/引入物种
    - **非蛾类（O-NM）**：从AMI-GBIF数据集随机抽取35,000张非蛾节肢动物图像，模拟监测系统中遇到的非目标物种

2. **辅助数据集构建**：
   为公平比较需要辅助数据的OSR方法，为每个区域显式构建辅助训练集。排除所有出现在封闭集、本地开放集或非本地开放集中的物种，同时排除属级别重叠的物种以防止捷径学习。NE-America和W-Europe各选8,000种、每种20张；C-America选4,000种、每种20张

3. **BCI真实验证集（O-B）**：
   纳入197张来自巴拿马Barro Colorado Island的实际采集图像，包含133个开放集物种，其中59种通过DNA条形码比对（>1.5%序列差异）可能是科学上全新的物种，为基准提供了极为真实的测试场景

### 提出的基线方法

作者提出两种利用辅助数据的简单方法：

- **NovelBranch**：保留原始$C$维分类头，额外训练一个$C+1$维分类头，将所有辅助物种视为一个"新类"
- **Extended**：额外训练一个$C+A$维分类头，将辅助集中每个物种视为独立类别

两种方法在测试时均使用封闭集分类头计算OSR分数，因此不影响ID准确率。

### 评估体系

- 开放集检测：AUROC（开放集为正，封闭集为负）
- 封闭集分类：Top-1准确率
- BCI真实场景：TPR@FPR=5%

## 实验关键数据

### 主实验：38种OSR方法基准评测

| 方法类别 | 代表方法 | NE-A (L/NL/NM) | W-E (L/NL/NM) | C-A (L/NL/NM) |
|---------|---------|-----------------|-----------------|-----------------|
| 后验方法 | MSP | 86.7/93.6/94.3 | 86.0/93.2/94.5 | 86.7/88.0/89.0 |
| 后验方法 | TempScale | 86.8/93.7/94.1 | 86.0/93.5/94.5 | 85.0/86.5/89.3 |
| 训练正则化 | LogitNorm | 80.6/87.3/95.3 | 80.8/87.7/95.6 | 87.6/89.5/90.5 |
| 辅助数据 | Energy | **87.4/95.1**/92.5 | 84.6/**94.8**/89.5 | **90.0/93.8**/91.2 |
| 辅助数据 | NovelBranch | 85.5/94.1/90.0 | 83.9/93.8/91.7 | 87.8/89.7/91.1 |
| 辅助数据 | MixOE | 86.2/92.5/94.4 | 85.2/91.9/94.1 | 86.2/87.9/90.1 |

### 消融实验

| 配置 | L AUROC | NL AUROC | NM AUROC | 说明 |
|------|---------|----------|----------|------|
| Energy + OI-CA辅助 | 90.07 | 93.70 | 91.33 | 领域相关辅助数据 |
| Energy + TinyImageNet | 85.25 | 88.16 | 93.85 | 通用辅助数据，L降4.82pp |
| 原始图像(MSP) | 80.01 | - | - | 正常推理 |
| 遮盖昆虫 | 46.57 | - | - | 近随机，说明模型依赖昆虫特征 |
| 遮盖背景 | 76.01 | - | - | 性能略降，昆虫特征为主 |

### 关键发现

1. **本地物种最难检测**：三个区域中，本地蛾类开放集的AUROC始终最低，验证了"地理越近、分类学越近、越难区分"的假设
2. **MSP及其变体仍是强基线**：在细粒度OSR场景中，复杂的后验方法并未一致性地超越简单的MSP基线，部分方法（如GradNorm、DICE）表现极差
3. **辅助数据质量至关重要**：使用领域相关辅助数据（同目不同种的蛾类）比通用数据（TinyImageNet）提升显著，且物种多样性比每种图像数量更重要
4. **预训练权重有帮助**：ImageNet预训练初始化可提升小规模数据集的OSR性能，BioCLIP权重在ViT-B-16上整体优于ImageNet-1K权重
5. **基准反映真实场景**：在标准开放集上AUROC高的方法在BCI真实未描述物种数据上同样表现较好

## 亮点与洞察

1. **数据集设计精巧**：利用地理信息构建难度递进的开放集类型，同时提供统一辅助数据，解决了现有基准缺乏公平比较条件的问题
2. **实际物种发现验证**：纳入可能全新物种的BCI数据进行端到端验证，弥合了基准性能和实际应用间的鸿沟
3. **可解释性实验**：通过遮盖实验证明模型确实使用了昆虫本身的形态特征而非背景线索
4. **开放科学**：数据集和代码均公开可用

## 局限与展望

- 仅评估高分辨率图像，未包含相机陷阱的低分辨率场景
- 训练正则化和辅助数据方法由于计算资源限制只做了单次训练
- 仅关注蛾类，其他昆虫类群的适用性有待验证
- 物种识别模型存在被滥用于非法野生动物贸易的风险

## 相关工作与启发

- **OSR/OOD基准**: OpenOOD, SSB, COOD, iNat21-OSR
- **生物多样性ML**: BioCLIP, BIOSCAN, iNaturalist
- **OSR方法**: MSP, ODIN, Energy, MixOE
- 启发：在其他细粒度领域（如医学影像、工业缺陷检测），辅助数据的质量和领域相关性可能同样关键

## 评分

- 新颖性：⭐⭐⭐⭐ — 首个面向昆虫的大规模细粒度OSR基准，填补重要空白
- 实验充分度：⭐⭐⭐⭐⭐ — 38种方法系统评测，覆盖消融、可解释性和真实场景验证
- 写作质量：⭐⭐⭐⭐⭐ — 动机清晰，数据集构建逻辑严密，实验设计有深度
- 价值：⭐⭐⭐⭐ — 对生物多样性监测和OSR方法评估均有直接推动作用

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Unlocking Transfer Learning for Open-World Few-Shot Recognition](unlocking_transfer_learning_for_open-world_few-shot_recognition.md)
- [\[ACL 2025\] ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities](../../ACL2025/llm_evaluation/onebench_to_test_them_all_sample-level_benchmarking_over_open-ended_capabilities.md)
- [\[ACL 2025\] KRISTEVA: Close Reading as a Novel Task for Benchmarking Interpretive Reasoning](../../ACL2025/llm_evaluation/kristeva_close_reading_as_a_novel_task_for_benchmarking_interpretive_reasoning.md)
- [\[NeurIPS 2025\] Benchmarking is Broken — Don't Let AI be its Own Judge](benchmarking_is_broken_--_dont_let_ai_be_its_own_judge.md)
- [\[NeurIPS 2025\] Merlin L48 Spectrogram Dataset](merlin_l48_spectrogram_dataset.md)

<!-- RELATED:END -->
