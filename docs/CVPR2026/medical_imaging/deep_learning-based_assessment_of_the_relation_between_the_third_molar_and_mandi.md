---
title: >-
  [论文解读] Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning
description: >-
  [CVPR 2026][医学图像][第三磨牙] 在全景X光片上比较本地学习(LL)、联邦学习(FL)和集中学习(CL)三种范式对第三磨牙与下颌管重叠关系的二分类性能，发现集中学习最优(AUC 0.831)，联邦学习作为隐私保护替代方案(AUC 0.757)显著优于本地学习(AUC均值 0.672)。
tags:
  - CVPR 2026
  - 医学图像
  - 第三磨牙
  - 下颌管
  - 全景X光片
  - 联邦学习
  - 深度学习分类
---

# Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning

**会议**: CVPR 2026  
**arXiv**: [2603.11850](https://arxiv.org/abs/2603.11850)  
**作者**: Johan Andreas Balle Rubak, Sara Haghighat, Sanyam Jain, Mostafa Aldesoki, Akhilanand Chaurasia, Sarah Sadat Ehsani, Faezeh Dehghan Ghanatkaman, Ahmad Badruddin Ghazali, Julien Issa, Basel Khalil, Rishi Ramani, Ruben Pauwels
**领域**: medical_imaging  
**关键词**: 第三磨牙, 下颌管, 全景X光片, 联邦学习, 深度学习分类

## 一句话总结

在全景X光片上比较本地学习(LL)、联邦学习(FL)和集中学习(CL)三种范式对第三磨牙与下颌管重叠关系的二分类性能，发现集中学习最优(AUC 0.831)，联邦学习作为隐私保护替代方案(AUC 0.757)显著优于本地学习(AUC均值 0.672)。

## 研究背景与动机

### 临床问题
下颌第三磨牙（智齿）阻生是口腔外科最常见的问题之一。当第三磨牙与下颌管（内含下牙槽神经）距离过近时，拔牙手术存在下牙槽神经损伤的风险，可能导致下唇和下颌区域永久性感觉异常。

### 现有诊断流程
- **全景X光片（OPG）** 是评估第三磨牙与下颌管空间关系的常规筛查手段
- 当全景片提示重叠时，通常需要进一步做 **CBCT（锥形束CT）** 确认三维关系
- 然而大量 CBCT 转诊是不必要的，增加了医疗成本和患者辐射暴露

### 自动化分类的需求
- 人工判读全景片存在观察者间变异，且耗时
- 深度学习可以自动化重叠/非重叠的二分类，辅助临床分诊决策
- 但医学影像数据分散在多个医疗机构，数据共享面临严格的隐私法规（如 GDPR、HIPAA）

### 联邦学习的引入动机
- **联邦学习（FL）** 允许多中心协作训练模型而无需共享原始患者数据
- 本文旨在系统比较三种学习范式在该临床任务上的表现差异，为实际部署提供指导

## 方法详解

### 数据与标注
- 全景X光片数据集，裁剪为第三磨牙与下颌管区域的局部图像
- 标注任务为**二分类**：重叠（overlap）vs. 非重叠（no-overlap）
- 数据分布在 **8 个独立标注者**（模拟 8 个临床中心/客户端）之间
- 每个客户端拥有各自独立标注的数据子集

### 模型架构
- 采用 **预训练 ResNet-34** 作为骨干网络
- 在 ImageNet 预训练权重基础上进行微调
- 输出层为二分类（sigmoid），预测重叠概率

### 三种学习范式

**1. 本地学习 (Local Learning, LL)**
- 每个客户端仅在自己的本地数据上独立训练模型
- 不进行任何跨客户端的模型交互或数据共享
- 得到 8 个独立的本地模型

**2. 集中学习 (Centralized Learning, CL)**
- 将所有 8 个客户端的数据汇集到一起
- 在整个数据集上统一训练单一模型
- 代表理论上限（忽略隐私约束）

**3. 联邦学习 (Federated Learning, FL)**
- 各客户端在本地数据上训练，仅上传模型参数到中央服务器
- 服务器聚合参数后下发更新模型给各客户端
- 迭代多轮直至收敛，全程不共享原始数据

### 评估策略
- **逐客户端评估**：为每个客户端在其验证集上优化最佳分类阈值，评估本地性能
- **汇总测试评估**：使用全局阈值在汇总的测试集上评估整体泛化性能
- **评估指标**：AUC（ROC曲线下面积）、准确率、灵敏度、特异度等阈值依赖指标
- **Grad-CAM 可视化**：分析模型关注的解剖区域，验证特征学习的合理性
- **训练动态分析**：监控训练/验证曲线，检测过拟合现象
- **服务端聚合监控**：FL 框架中监控服务端的聚合信号

## 实验关键数据

### Table 1: 三种学习范式在汇总测试集上的整体性能

| 学习范式 | AUC | Accuracy | 备注 |
|---|---|---|---|
| Centralized Learning (CL) | **0.831** | **0.782** | 性能最优，数据集中训练 |
| Federated Learning (FL) | 0.757 | 0.703 | 隐私保护，性能次优 |
| Local Learning (LL) 均值 | 0.672 | — | 跨客户端平均 |
| Local Learning (LL) 范围 | 0.619–0.734 | — | 客户端间差异显著 |

**关键观察**：
- CL 比 FL 高 **7.4 个 AUC 百分点**，比 LL 均值高 **15.9 个百分点**
- FL 相对 LL 提升 **8.5 个 AUC 百分点**，证明跨客户端参数聚合的价值
- LL 的 AUC 范围（0.619–0.734）跨度达 11.5%，反映数据异质性对本地模型的影响

### Table 2: 各学习范式的训练特性与模型行为对比

| 分析维度 | CL | FL | LL |
|---|---|---|---|
| 过拟合程度 | 轻微 | 中等 | 严重 |
| Grad-CAM 关注区域 | 解剖结构聚焦 | 解剖结构聚焦 | 分散/不一致 |
| 跨客户端泛化性 | 最强 | 较强 | 最弱 |
| 数据隐私 | 无保护（需数据共享） | 保护（仅共享参数） | 保护（完全隔离） |
| 训练数据量 | 全部数据 | 等效全部（参数聚合） | 仅本地子集 |
| 临床部署可行性 | 需数据集中，受限 | 可多中心部署 | 仅限单中心 |

**关键观察**：
- LL 模型过拟合最严重，因为每个客户端数据量有限，模型容易记忆训练数据
- CL 和 FL 的 Grad-CAM 热图都聚焦在第三磨牙根尖和下颌管走行区域，说明学到了有临床意义的特征
- LL 的 Grad-CAM 热图分散且不一致，表明模型可能学到了虚假相关性

## 亮点与洞察

- **系统性范式比较**：首次在牙科全景片第三磨牙-下颌管关系分类任务上系统比较 LL/FL/CL 三种学习范式，为该领域的多中心协作提供直接证据
- **联邦学习的实用价值**：FL 在不共享数据的前提下，AUC 比 LL 均值提升 12.6%（0.672→0.757），证明联邦学习在口腔医学影像领域的可行性
- **Grad-CAM 验证临床合理性**：通过可解释性分析确认 CL 和 FL 模型关注正确的解剖结构，增强了模型的临床可信度
- **过拟合风险量化**：明确揭示小样本本地训练的过拟合风险，为数据稀缺场景下的模型训练策略选择提供参考
- **减少不必要 CBCT 转诊**：自动化分类可辅助临床分诊，减少不必要的 CBCT 检查，降低辐射暴露和医疗成本

## 局限性

- **仅限二分类**：重叠/非重叠的粗粒度分类可能不足以反映临床中更细致的风险分级（如接触、穿通等）
- **数据规模与多样性**：8 个标注者的数据分布未必能代表真实多中心场景中的数据异质性（设备差异、人群差异等）
- **单一骨干网络**：仅使用 ResNet-34，未探索其他架构（如 DenseNet、EfficientNet、Vision Transformer）的表现
- **FL 聚合策略单一**：未比较不同的联邦聚合算法（如 FedProx、FedBN、SCAFFOLD 等），可能存在更优的FL策略
- **缺乏与临床专家的对比**：未报告放射科医师的诊断性能作为人类基线
- **阈值优化的局限**：全局阈值与局部最优阈值的差异影响部署策略，但文中未详细讨论阈值鲁棒性

## 相关工作

- **牙科AI**：近年来深度学习在牙科影像中的应用涵盖龋齿检测、牙周病分级和种植体识别，但针对第三磨牙与下颌管关系的自动评估研究相对有限
- **联邦学习在医学影像中的应用**：FedAvg 等方法已在胸部X光、病理切片等任务上验证，但在口腔医学影像领域的探索尚处于早期阶段
- **全景X光片分析**：传统方法多依赖手工特征或简单CNN，本文使用预训练 ResNet-34 并系统比较学习范式，在方法论上更为完整
- **可解释性分析**：Grad-CAM 在医学影像中广泛用于验证模型关注区域的合理性，本文将其作为跨范式比较的辅助工具

## 评分

- 新颖性: ⭐⭐⭐ — 方法本身（ResNet-34 + FedAvg）较成熟，贡献主要在临床场景和范式比较上
- 实验充分度: ⭐⭐⭐ — 有三种范式的定量比较和 Grad-CAM 分析，但缺少多架构对比和人类基线
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，问题定义明确，实验设计合理
- 价值: ⭐⭐⭐⭐ — 为牙科影像的多中心隐私保护协作提供了有价值的实证参考

<!-- RELATED:START -->

## 相关论文

- [OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modalityspecific_encoders_and_partially.md)
- [Unlocking Multi-Site Clinical Data: A Federated Approach to Privacy-First Child Autism Behavior Analysis](unlocking_multi-site_clinical_data_a_federated_approach_to_privacy-first_child_a.md)
- [Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [FedVG: Gradient-Guided Aggregation for Enhanced Federated Learning](fedvg_gradient-guided_aggregation_for_enhanced_federated_learning.md)

<!-- RELATED:END -->
