---
title: >-
  [论文解读] Forgetting Through Transforming: Enabling Federated Unlearning via Class-Aware Representation Transformation
description: >-
  [ICCV2025][LLM安全][federated unlearning] 提出 FUCRT 方法，通过类感知表征变换实现联邦遗忘：将遗忘类的表征“变换”到语义最近的保留类，而非直接消除，配合双重对比学习对齐跨客户端的变换一致性，在四个数据集上实现 100% 遗忘保障的同时保持甚至提升剩余类性能。
tags:
  - "ICCV2025"
  - "LLM安全"
  - "federated unlearning"
  - "class-aware representation"
  - "对比学习"
  - "transformation alignment"
  - "Non-IID"
---

# Forgetting Through Transforming: Enabling Federated Unlearning via Class-Aware Representation Transformation

**会议**: ICCV2025  
**arXiv**: [2410.06848](https://arxiv.org/abs/2410.06848)  
**代码**: 待确认  
**领域**: 联邦学习 / 机器遗忘 / 隐私保护  
**关键词**: federated unlearning, class-aware representation, contrastive learning, transformation alignment, Non-IID

## 一句话总结

提出 FUCRT 方法，通过类感知表征变换实现联邦遗忘：将遗忘类的表征“变换”到语义最近的保留类，而非直接消除，配合双重对比学习对齐跨客户端的变换一致性，在四个数据集上实现 100% 遗忘保障的同时保持甚至提升剩余类性能。

## 研究背景与动机

联邦学习（FL）允许多个客户端在不共享原始数据的情况下协作训练全局模型。随着 GDPR、CCPA 等隐私法规的出台，客户端享有“被遗忘的权利”，要求从已训练模型中消除特定类别数据的影响。联邦遗忘（Federated Unlearning, FU）即为应对此需求而提出。

### 现有方法的不足

现有 FU 方法主要分三类：

**梯度上升法**（如 Halimi et al.）：通过反向梯度上升收紧泛化边界，但极易导致模型性能严重退化（保留类准确率降至约10%）

**通道剪枝法**（FUDP）：利用 TF-IDF 剪枝与遗忘类最相关的通道，但当遗忘类增多时剪枝过度导致保留类性能下降

**动量退化法**（FUMD）：使用同构随机初始化的退化模型擦除遗忘类信息，保留类性能仍不理想

这些方法均将遗忘数据视为需要消除的对立面，忽略了遗忘数据与保留数据之间潜在的表征关系。

### 关键观察

作者在模型表征空间（用 t-SNE 可视化 CIFAR10）中发现了两个重要现象：

- **观察1**：在从零重训模型的表征空间中，遗忘类数据（如“汽车”类）与保留类数据交织在一起失去可分性，但保留类数据仍保持紧凑和可分的聚类特征
- **观察2**：遗忘类数据的分布呈现非随机模式——集中在特定保留类的表征域中（“汽车”类集中到“卡车”和“飞机”类），表明遗忘过程存在优先“遗忘到”特定语义相近保留类的倾向

这两个观察启示：遗忘与保留并非对立，而是可以在表征空间中通过变换来实现。

## 方法详解

### 整体框架：FUCRT

FUCRT（Federated Unlearning via Class-aware Representation Transformation）保持标准的客户端-服务器 FL 架构，引入类感知表征变换来执行遗忘。核心包含两个组件：

1. **变换类选择策略**（Transformation Class Selection）——确定每个遗忘样本应“变换到”哪个保留类
2. **变换对齐技术**（Transformation Alignment）——确保跨客户端变换的一致性

### 变换类选择

#### 生成全局变换类集合

直觉上，在表征空间中，遗忘类的变换类通常是最容易与之混淆的类别。具体步骤：

1. 对每个客户端，在被正确分类的遗忘类样本上计算模型输出的概率向量
2. 提取每个样本的**次高概率输出**及其对应类别（利用原始模型的高性能）
3. 按类别汇总次高概率，选出汇总值超过阈值的类别，形成本地变换类集合
4. 仅当客户端遗忘类样本数超过阈值时才生成本地集合，避免数据稀疏引起的偏差
5. 服务器聚合所有客户端的本地集合，通过众数确定集合大小、频率投票确定具体类别，生成全局变换类集合

#### 为每个遗忘样本分配变换类

在本地遗忘训练中，对每个遗忘样本，从全局变换类集合中选择概率输出最高的类作为变换目标。此策略既尊重全局共识，又利用模型对每个样本的当前知识。

### 变换对齐：双重类感知对比学习

将遗忘数据标签替换为变换类后用交叉熵微调即可实现基本的表征变换。但 Non-IID 数据分布导致不同客户端表征空间存在差异，必须解决变换不一致问题。

**局部类感知对比损失**：基于批内样本的表征，拉近同类样本、推远异类样本，优化本地表征空间结构。

**全局类感知对比损失**：引入服务器聚合的全局类原型，将每个样本的表征拉向对应的全局原型，实现本地-全局对齐。

**总体本地遗忘训练损失**：交叉熵损失 + 局部对比损失 + 全局对比损失，由超参数控制各项权重。

### 遗忘流程

1. 各客户端基于原始全局模型计算本地变换类集合并上传服务器
2. 服务器聚合生成全局变换类集合
3. 迭代 R 轮遗忘训练：服务器下发全局模型和全局类原型 -> 客户端执行标签转换和对比学习训练 -> 上传模型和本地原型 -> 服务器 FedAvg 聚合
4. 仅有保留类数据的客户端只执行变换对齐（不做标签转换）

## 实验结果

### 实验设置

- **数据集**：CIFAR10、CIFAR100、FMNIST、EuroSAT
- **架构**：ResNet18（CIFAR10/FMNIST/EuroSAT）、ResNet50（CIFAR100）
- **基线**：Fine-tune、Gradient-ascent、FUDP、FUMD、From-scratch（上界）
- **遗忘类比例**：10%

### 主要结果（遗忘类 ACC / 保留类 ACC）

| 数据集 | 方法 | 遗忘类 ACC (IID) | 保留类 ACC (IID) | 保留类 ACC (Non-IID) |
|--------|------|-----------------|-----------------|---------------------|
| CIFAR10 | Gradient-ascent | 0.00% | 10.94% | 11.15% |
| CIFAR10 | FUDP | 0.00% | 84.85% | 85.68% |
| CIFAR10 | FUMD | 0.00% | 82.38% | 82.60% |
| CIFAR10 | **FUCRT** | **0.00%** | **90.06%** | **89.82%** |
| CIFAR10 | From-scratch | 0.00% | 87.99% | 85.44% |
| CIFAR100 | FUDP | 0.03% | 72.25% | 69.71% |
| CIFAR100 | **FUCRT** | **0.00%** | **75.25%** | **74.88%** |
| FMNIST | **FUCRT** | **0.00%** | **90.82%** | **90.95%** |
| EuroSAT | **FUCRT** | **0.00%** | **94.92%** | - |

关键发现：

1. **100% 遗忘保障**：FUCRT 在所有数据集遗忘类 ACC = 0%
2. **保留类性能最优**：显著超过所有基线，在 CIFAR10 上甚至超过从零重训（90.06% vs 87.99%）
3. **跨设置鲁棒**：IID 和 Non-IID 下均一致最优
4. **Gradient-ascent 的灾难性退化**：保留类准确率降至约10%，说明对抗式方法的脆弱性

## 优势与局限

### 优势

- 首次从表征空间视角审视联邦遗忘，发现遗忘类与保留类的变换关系
- “变换”策略比“消除”策略更优雅，既保障遗忘效果又保持保留类性能
- 双重对比学习有效解决 Non-IID 下的变换一致性问题
- 保留类性能可超越从零重训，说明表征变换实际上起到了正则化效果

### 局限

- 需额外通信全局变换类集合和类原型，增加通信开销
- 假设服务器和客户端均为良性，未考虑恶意参与者
- 变换类选择依赖原始模型的概率输出质量

## 个人思考

1. 将“遗忘”重新定义为“表征变换”是很有洞察力的视角——利用了遗忘类与保留类的语义邻近性
2. 保留类性能超过从零重训这一结果值得关注：表征变换可能起到了知识蒸馏/正则化的效果
3. 该框架在多遗忘类场景下的扩展性值得进一步研究
4. 全局类原型的传输存在隐私泄露风险（原型可能编码关于数据分布的信息），未来可考虑差分隐私保护

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] SafeConstellations: Mitigating Over-Refusals in LLMs Through Task-Aware Representation Steering](../../ACL2026/llm_safety/safeconstellations_mitigating_over-refusals_in_llms_through_task-aware_represent.md)
- [\[CVPR 2026\] DAMP: Class Unlearning via Depth-Aware Removal of Forget-Specific Directions](../../CVPR2026/llm_safety/damp_class_unlearning_via_depth_aware_removal_of_forget_specific_directions.md)
- [\[ICML 2025\] Vulnerability-Aware Alignment: Mitigating Uneven Forgetting in Harmful Fine-Tuning](../../ICML2025/llm_safety/vulnerability-aware_alignment_mitigating_uneven_forgetting_in_harmful_fine-tunin.md)
- [\[AAAI 2026\] Beyond Superficial Forgetting: Thorough Unlearning through Knowledge Density Estimation and Block Re-insertion](../../AAAI2026/llm_safety/beyond_superficial_forgetting_thorough_unlearning_through_knowledge_density_esti.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](../../ACL2026/llm_safety/representation-guided_parameter-efficient_llm_unlearning.md)

</div>

<!-- RELATED:END -->
