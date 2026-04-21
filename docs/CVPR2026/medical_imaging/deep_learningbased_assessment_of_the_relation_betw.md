---
title: >-
  [论文解读] Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning
description: >-
  [CVPR 2026][医学图像][联邦学习] 在按8个独立标注者划分的全景口腔X光裁剪片上，系统对比本地学习（LL）、联邦学习（FL）和集中学习（CL）三种训练范式在第三磨牙-下颌管重叠二分类任务上的表现，验证了CL > FL > LL的性能排序（AUC分别为0.831、0.757和0.672），证明FL在保护数据隐私的前提下显著优于各站点独立训练。
tags:
  - CVPR 2026
  - 医学图像
  - 联邦学习
  - 全景X光片
  - 第三磨牙
  - 下颌管
  - 隐私保护
---

# Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning

**会议**: CVPR 2026  
**arXiv**: [2603.11850](https://arxiv.org/abs/2603.11850)  
**代码**: 无  
**领域**: 医学图像分析 / 联邦学习 / 口腔影像  
**关键词**: 联邦学习, 全景X光片, 第三磨牙, 下颌管, 隐私保护

## 一句话总结
在按8个独立标注者划分的全景口腔X光裁剪片上，系统对比本地学习（LL）、联邦学习（FL）和集中学习（CL）三种训练范式在第三磨牙-下颌管重叠二分类任务上的表现，验证了CL > FL > LL的性能排序（AUC分别为0.831、0.757和0.672），证明FL在保护数据隐私的前提下显著优于各站点独立训练。

## 研究背景与动机

**领域现状**：阻生下颌第三磨牙（智齿）是口腔外科中最常见的手术之一。当智齿与下颌管（容纳下牙槽神经的骨性管道）存在密切空间关系时，拔牙手术中有损伤下牙槽神经的风险，可能导致下唇和下颌皮肤永久性感觉异常。全景X光片（OPG）是术前评估磨牙与下颌管空间关系的常规手段。

**现有痛点**：全景X光片上磨牙与下颌管的重叠判断依赖放射科医生的主观经验，不同标注者之间存在显著的判断差异。自动化分类可以辅助临床分诊并减少不必要的CBCT转诊（CBCT辐射剂量更高、费用更贵）。然而，口腔影像数据分散在不同的临床机构和标注团队中，受隐私法规（如GDPR、HIPAA）限制，各中心的数据无法直接汇聚进行集中训练。

**核心矛盾**：集中训练（CL）需要数据汇聚但违反隐私规定；本地训练（LL）保护隐私但各站点数据量小，模型泛化能力差。联邦学习（FL）作为折中方案，理论上可以在不共享原始数据的前提下实现多中心协作，但其在真实口腔影像场景中的性能-隐私权衡尚未被系统验证。

**本文目标**：在存在真实标注者差异（Non-IID特性）的口腔影像数据上，量化LL、FL、CL三种范式的性能差距，回答"FL能否作为CL的隐私保护替代方案"。

**切入角度**：选择语义清晰、临床意义明确的二分类任务（磨牙-下颌管重叠 vs 无重叠），以预训练ResNet-34为统一backbone，将8个独立标注者的数据视为8个FL客户端，系统对比三种训练范式。

**核心 idea**：联邦学习在口腔影像的多标注者设置下可以作为集中训练的可行替代，性能显著优于各站点独立训练。

## 方法详解

### 整体框架
完整流程包括四个阶段：(1) 从全景X光片中裁剪出包含第三磨牙和下颌管区域的ROI；(2) 按8个独立标注者将数据划分为8个客户端，模拟多中心设置；(3) 分别在LL、FL、CL三种范式下使用预训练ResNet-34进行二分类训练（overlap vs no-overlap）；(4) 通过per-client（本地最优阈值）和pooled（全局阈值）两种评估方式分析模型性能，辅以训练动态分析和Grad-CAM可视化。

### 关键设计
1. **三范式公平对比框架**:

    - 功能：在相同数据划分和相同backbone下对比LL、FL、CL三种训练范式
    - 核心思路：LL——每个标注者的数据独立训练一个ResNet-34模型，共8个独立模型；FL——使用FedAvg算法在服务端聚合模型参数$w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_t^k$，各客户端数据不出本地，仅上传模型更新；CL——将所有8个客户端的数据集中到一处训练一个统一模型。三者使用相同的预训练ResNet-34初始化和超参数设置
    - 设计动机：控制变量是公平对比的前提。8个标注者的自然数据划分比人工随机划分更接近真实临床场景中的数据异质性

2. **双轨评估协议**:

    - 功能：从本地部署和跨中心泛化两个角度评估模型性能
    - 核心思路：per-client评估——在每个客户端的验证集上独立优化分类阈值，用各自最优阈值在本地测试集上评估，反映模型在本地部署时的最佳表现；pooled评估——使用全局统一阈值在合并测试集上评估，反映模型在跨中心部署时的泛化能力。主要指标包括AUC和基于阈值的准确率、敏感性、特异性
    - 设计动机：FL的实际部署可能是"各中心使用全局模型+本地阈值"或"全局模型+全局阈值"，两种评估方式对应了不同的部署策略

3. **可解释性与训练动态分析**:

    - 功能：通过Grad-CAM和训练曲线分析三种范式的行为差异
    - 核心思路：Grad-CAM可视化三种范式训练出的模型的注意力热力图，观察模型是否关注解剖学上正确的区域（第三磨牙根尖和下颌管走行处）。训练曲线监控各范式的收敛行为和过拟合程度，FL端还监控server-side聚合信号以追踪全局模型的训练状态
    - 设计动机：不仅要知道"谁更好"，还要理解"为什么好"。Grad-CAM可以验证模型是否基于正确的解剖学特征做出判断（而非利用数据伪影），训练动态分析揭示LL过拟合的原因

### 损失函数 / 训练策略
- 使用ImageNet预训练的ResNet-34作为backbone，最后全连接层替换为二分类头
- 二分类交叉熵损失，标签为overlap（第三磨牙与下颌管重叠）和no-overlap
- FL采用标准FedAvg聚合策略：各客户端在本地数据上训练若干epoch后上传模型权重，服务端加权平均后分发回各客户端
- 同时监控server-side的聚合信号（如全局验证损失、客户端更新幅度的一致性）以追踪训练稳定性

## 实验关键数据

### 主实验

| 训练范式 | AUC | 准确率(%) | 备注 |
|---------|-----|----------|------|
| CL（集中学习） | **0.831** | **78.2** | 最高性能，所有数据集中训练 |
| FL（联邦学习） | 0.757 | 70.3 | 中间水平，隐私保护 |
| LL（本地学习） | 0.619-0.734 (均值0.672) | - | 最低且方差大 |

FL相比LL均值AUC提升+0.085（从0.672到0.757），CL相比FL进一步提升+0.074（从0.757到0.831）。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| LL 8个客户端 | AUC 0.619-0.734 | 跨度>0.1，说明标注者数据分布差异巨大 |
| FL vs LL best | AUC 0.757 vs 0.734 | FL超越最强本地模型 |
| CL vs FL gap | AUC 0.831 vs 0.757 | 差距0.074，数据集中仍有不可忽视的优势 |
| 训练曲线过拟合 | LL > FL > CL | LL过拟合最严重 |

### 关键发现
- CL > FL > LL的性能排序在AUC和准确率上一致成立，证实数据共享的价值
- FL作为CL的隐私保护替代方案是有效的，但与CL之间仍存在约0.074 AUC的性能差距
- LL模型的过拟合最为严重——各客户端数据量小，单独训练难以学到泛化特征
- Grad-CAM显示CL和FL模型的注意力更集中在解剖学上与磨牙-管关系相关的区域（如根尖和管道走行处），而LL模型的注意力更分散，可能利用了非解剖学伪影
- 8个客户端之间的LL性能跨度超过0.1 AUC，反映了真实标注者差异对模型的显著影响

## 亮点与洞察
- **首次在牙科AI中系统验证FL**：联邦学习在口腔影像领域的应用此前几乎空白，本文提供了首个基准测试
- **真实标注者划分比随机划分更有价值**：8个标注者作为8个FL客户端，自然引入了标注标准差异带来的Non-IID特性，比人工制造的异质性更接近临床现实
- **双轨评估严谨**：区分per-client和pooled两种评估方式，对应不同的实际部署策略
- **Grad-CAM验证模型可信度**：CL和FL模型关注正确解剖区域的发现增加了临床可信度
- **清晰的性能排序**：CL > FL > LL的一致结论为临床决策者提供了明确的参考

## 局限与展望
- 二分类任务（overlap vs no-overlap）过于粗糙，临床实际需要更细粒度的风险分级（如Winter分类、Pell-Gregory分类）
- FL仅使用标准FedAvg，未探索FedProx、SCAFFOLD、FedBN等处理数据异质性的先进算法，这些方法可能缩小FL与CL之间的性能差距
- 8个标注者的规模较小，大规模多中心（如20+机构）的验证更有说服力
- 未详细报告标注者间一致性（inter-rater agreement），如Kappa系数或Fleiss' Kappa，无法量化标注噪声对训练的影响
- ResNet-34是较旧的backbone，未尝试Vision Transformer等更现代的架构
- 仅使用裁剪的ROI区域，未探索全景片端到端检测+分类的流程

## 相关工作与启发
- **vs 通用医学FL研究（如FedAvg在X-ray/CT上的应用）**：本文聚焦口腔特定任务，数据按标注者而非机构划分是独特设置，更接近标注标准差异驱动的Non-IID
- **vs 传统dental AI工作**：大多依赖单中心集中训练，本文首次在磨牙-下颌管关系评估中引入FL范式
- **vs 更先进FL算法（FedProx/FedBN/SCAFFOLD）**：本文仅用FedAvg作为基线，明确指出了改进空间，但基线结果本身已经证明了FL的可行性
- **vs 同场景CBCT研究**：全景片获取成本和辐射剂量远低于CBCT，自动分类可以筛选出真正需要CBCT的病例

## 评分
- 新颖性: ⭐⭐⭐ FL在口腔影像中首次系统验证有一定新意，但方法层面（FedAvg+ResNet-34）无创新
- 实验充分度: ⭐⭐⭐ 三范式对比合理、双轨评估加分，但FL算法探索单一、数据规模有限、缺少inter-rater分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，摘要信息密度高，实验设计的逻辑表述连贯
- 价值: ⭐⭐⭐ 为口腔AI的隐私保护训练提供了有价值的基线参考，但实用性受限于二分类任务的粗糙度

<!-- RELATED:START -->

## 相关论文

- [Unlocking Multi-Site Clinical Data: A Federated Approach to Privacy-First Child Autism Behavior Analysis](unlocking_multi-site_clinical_data_a_federated_approach_to_privacy-first_child_a.md)
- [OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modalityspecific_encoders_and_partially.md)
- [Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [FedVG: Gradient-Guided Aggregation for Enhanced Federated Learning](fedvg_gradient-guided_aggregation_for_enhanced_federated_learning.md)

<!-- RELATED:END -->
