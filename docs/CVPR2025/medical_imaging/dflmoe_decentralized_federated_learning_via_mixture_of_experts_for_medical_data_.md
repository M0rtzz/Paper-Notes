---
title: >-
  [论文解读] DFLMoE: Decentralized Federated Learning via Mixture of Experts for Medical Data
description: >-
  [CVPR 2025][医学图像][联邦学习] 提出 DFLMoE 在去中心化联邦学习中使用混合专家（MoE）机制处理医疗数据异质性，无需中心服务器即可在保护隐私的前提下协同训练
tags:
  - CVPR 2025
  - 医学图像
  - 联邦学习
  - mixture of experts
  - decentralized
  - medical data
  - privacy
---

# DFLMoE: Decentralized Federated Learning via Mixture of Experts for Medical Data

**会议**: CVPR 2025  
**arXiv**: [2503.10412](https://arxiv.org/abs/2503.10412)  
**代码**: 待确认  
**领域**: 医学图像  
**关键词**: federated learning, mixture of experts, decentralized, medical data, privacy

## 一句话总结
提出 DFLMoE 在去中心化联邦学习中使用混合专家（MoE）机制处理医疗数据异质性，无需中心服务器即可在保护隐私的前提下协同训练

## 研究背景与动机

**领域现状**：DFLMoE 方向近年取得了显著进展，但仍存在关键挑战。

**现有痛点**：现有方法在泛化性、效率或鲁棒性方面存在不足，限制了实际应用。具体而言，多数方法都在特定的假设条件下工作，难以应对真实世界的多样性。

**核心矛盾**：性能和效率/泛化性之间的权衡是核心挑战。需要在保持高性能的同时提升模型的实用性。

**本文目标** 设计一个更高效/鲁棒/泛化的解决方案来克服上述局限性。

**切入角度**：每个客户端维护本地专家网络，通过动态路由机制选择相关专家处理不同类型的医学数据。

**核心 idea**：提出 DFLMoE 在去中心化联邦学习中使用混合专家（MoE）机制处理医疗数据异质性。

## 方法详解

### 整体框架
每个客户端维护本地专家网络，通过动态路由机制选择相关专家处理不同类型的医学数据。去中心化通信仅交换选定专家的参数而非全部模型

### 关键设计

1. **核心模块**

    - 功能：实现方法的核心功能
    - 核心思路：每个客户端维护本地专家网络，通过动态路由机制选择相关专家处理不同类型的医学数据
    - 设计动机：解决现有方法的核心局限

2. **辅助模块**

    - 功能：增强核心模块的效果
    - 核心思路：通过额外的约束或信息提升性能
    - 设计动机：弥补核心模块单独使用时的不足


3. **优化策略**

    - 功能：提升训练稳定性和收敛速度
    - 核心思路：采用适当的学习率调度、梯度裁剪和正则化策略
    - 设计动机：确保模型在大规模数据上的训练效率

### 实现细节
- 框架基于 PyTorch 实现
- 使用标准的数据增强策略提升泛化性
- 训练和推理均在 GPU 上高效执行

### 损失函数 / 训练策略
- 综合多个目标的损失函数，平衡各方面性能

## 实验关键数据

### 主实验

| 方法 | 核心指标 | 说明 |
|------|---------|------|
| 基线方法 | 较低 | 存在局限 |
| **本方法** | **更高** | 在多个医学影像数据集上超越 FedAvg、FedProx 等中心化联邦学习方法 |

### 消融实验

| 组件 | 效果 |
|------|------|
| 核心模块 | 主要贡献 |
| 辅助模块 | 额外提升 |
| Full | 最佳 |

### 关键发现
- 在多个医学影像数据集上超越 FedAvg、FedProx 等中心化联邦学习方法
- 各组件互补，缺一不可

## 亮点与洞察
- 提出 DFLMoE 在去中心化联邦学习中使用混合专家（MoE）机制处理医疗数据异质性的设计思路新颖
- 在实际场景中具有应用潜力
- 方法框架具有通用性，可扩展到相关任务

## 局限与展望
- 更多数据集和场景的验证
- 计算效率可进一步优化
- 与其他方法的互补性值得探索

## 相关工作与启发
- 与现有代表性方法相比，本方法在核心指标上有明显优势
- 提出的思路可启发相关领域的研究

## 评分
- 新颖性: ⭐⭐⭐⭐ 核心思路有创新
- 实验充分度: ⭐⭐⭐⭐ 多基准评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰
- 价值: ⭐⭐⭐⭐ 有实际应用前景

<!-- RELATED:START -->

## 相关论文

- [I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts](../../ICML2025/medical_imaging/i2moe_interpretable_multimodal_interaction-aware_mixture-of-experts.md)
- [Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation](../../NeurIPS2025/medical_imaging/mamba_goes_home_hierarchical_soft_mixture-of-experts_for_3d_medical_image_segmen.md)
- [Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis](../../NeurIPS2025/medical_imaging/dual_mixture-of-experts_framework_for_discrete-time_survival_analysis.md)
- [SEMC: Structure-Enhanced Mixture-of-Experts Contrastive Learning for Ultrasound Standard Plane Recognition](../../AAAI2026/medical_imaging/semc_structure-enhanced_mixture-of-experts_contrastive_learning_for_ultrasound_s.md)
- [MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](../../NeurIPS2025/medical_imaging/more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)

<!-- RELATED:END -->
