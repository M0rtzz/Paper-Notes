---
title: >-
  [论文解读] SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport
description: >-
  [CVPR 2026][人体理解][联邦学习] 提出 SubFLOT 框架，在服务器端利用最优传输（Optimal Transport）将全局模型的参数分布与客户端历史模型对齐，实现无需访问原始数据的个性化剪枝，并通过自适应正则化抑制剪枝导致的参数偏移，在多个数据集上大幅超越现有联邦剪枝方法。
tags:
  - CVPR 2026
  - 人体理解
  - 联邦学习
  - 网络剪枝
  - 最优传输
  - 个性化模型
  - 异构系统
---

# SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport

**会议**: CVPR 2026  
**arXiv**: [2604.06631](https://arxiv.org/abs/2604.06631)  
**代码**: 无  
**领域**: 联邦学习 / 模型压缩  
**关键词**: 联邦学习, 网络剪枝, 最优传输, 个性化模型, 异构系统

## 一句话总结

提出 SubFLOT 框架，在服务器端利用最优传输（Optimal Transport）将全局模型的参数分布与客户端历史模型对齐，实现无需访问原始数据的个性化剪枝，并通过自适应正则化抑制剪枝导致的参数偏移，在多个数据集上大幅超越现有联邦剪枝方法。

## 研究背景与动机

**领域现状**：联邦学习（FL）在保护数据隐私的同时进行协作训练，但在实际部署中面临系统异构性（设备资源差异大）和统计异构性（非IID数据分布）的双重挑战。联邦网络剪枝作为应对策略，允许不同客户端训练不同大小的子模型，减少计算和通信开销。

**现有痛点**：联邦剪枝面临两个关键未解问题。第一，剪枝决策的位置存在两难困境：服务器端剪枝（如HeteroFL）采用统一压缩策略，无法个性化；客户端剪枝（训练-剪枝-微调范式）能实现个性化但对资源受限设备计算负担过重。第二，剪枝行为本身会加剧异构性——高剪枝率的子模型权重分布会偏离全局模型（参数漂移），破坏训练稳定性和全局收敛。

**核心矛盾**：如何在服务器端实现个性化剪枝（不访问原始数据），同时解决剪枝引发的参数空间偏移？

**本文目标** (1) 服务器端个性化剪枝——不接触客户端原始数据，为每个客户端生成定制化子模型；(2) 参数偏移抑制——防止不同剪枝率的子模型在训练过程中参数分布过度发散。

**切入角度**：作者将客户端的历史模型参数视为其本地数据分布的代理（proxy），基于这一洞察，将剪枝问题转化为全局模型与历史模型之间的Wasserstein距离最小化问题，通过最优传输计划指导个性化剪枝。

**核心 idea**：用最优传输在参数空间中对齐全局模型与客户端历史模型的神经元，实现无需数据访问的服务器端个性化剪枝。

## 方法详解

### 整体框架

SubFLOT 在每轮联邦通信中包含三个阶段：(1) **服务器端OTP模块**——利用最优传输将全局模型参数与客户端历史模型对齐，生成定制化子模型分发给各客户端；(2) **客户端SAR训练**——客户端使用自适应正则化损失在本地数据上训练子模型，防止参数偏移；(3) **服务器端OTA聚合**——复用OT机制将更新后的异构子模型对齐回全局参数空间后再聚合。

### 关键设计

1. **最优传输增强剪枝（OTP）**:

    - 功能：在服务器端为每个客户端生成个性化的剪枝子模型
    - 核心思路：采用逐层渐进匹配策略。对于客户端 $i$ 的第 $l$ 层，先利用前一层的传输计划 $T_i^{(l-1)}$ 重映射全局模型权重的输入空间 $\hat{W}_G^{(l,l-1)} = W_G^{(l,l-1)} T_i^{(l-1)}$，然后将对齐后的全局权重和客户端历史权重的输出神经元视为两个离散概率分布，计算欧氏距离成本矩阵，求解离散OT问题得到传输计划 $T_i^{(l)}$。最终子模型通过融合对齐后的全局知识和客户端历史参数得到：$\tilde{W}_i = \alpha \cdot W_{aligned} + (1-\alpha) \cdot W_i$，其中 $\alpha=0.5$ 平衡全局知识迁移与本地特化
    - 设计动机：历史模型参数隐式编码了客户端的本地数据分布信息，利用OT可以在不接触原始数据的前提下找到全局模型中与客户端数据最相关的神经元子集，实现真正的数据感知（data-aware）个性化

2. **基于缩放的自适应正则化（SAR）**:

    - 功能：在客户端本地训练时抑制子模型的参数漂移
    - 核心思路：在标准交叉熵损失之外添加正则项 $\mathcal{L}_{SAR}(W_i) = \rho_i \cdot \|W_i - \tilde{W}_i\|_2^2$，其中 $\rho_i$ 是客户端的剪枝率，$\tilde{W}_i$ 是服务器提供的锚模型。完整目标函数为 $\mathcal{L}_i = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{SAR}$，$\lambda=1.0$
    - 设计动机：剪枝率越高的小模型越容易发生参数偏移，因此惩罚强度与 $\rho_i$ 成正比——自适应地给予高剪枝率客户端更强的约束。与HeteroFL的事后修正不同，SAR在训练过程中主动正则化，从源头防止漂移

3. **最优传输增强聚合（OTA）**:

    - 功能：将来自不同客户端的异构子模型对齐后聚合为新的全局模型
    - 核心思路：对每个客户端 $i$ 计算传输映射 $\mathcal{T}_i$，将更新后的客户端模型 $W_i^t$ 对齐回全局模型 $W_G^t$ 的参数空间（与OTP方向相反），然后按加权平均聚合 $W_G^{t+1} = \sum_{i=1}^N p_i \cdot \mathcal{T}_i(W_i^t)$
    - 设计动机：标准FedAvg在异构设置下直接平均可能将语义不同的参数混合（破坏性干扰），OTA通过OT对齐确保功能等价的神经元被正确匹配后再聚合，同时自然地归一化了不同剪枝率导致的参数尺度差异

### 损失函数 / 训练策略

客户端本地损失：$\mathcal{L}_i(W_i) = \mathcal{L}_{CE}(W_i; \mathcal{D}_i) + \lambda \cdot \rho_i \cdot \|W_i - \tilde{W}_i\|_2^2$。训练配置：20个客户端，全参与（join ratio 1.0），200轮通信，每轮5个本地epoch，SGD优化器（lr=0.001），batch size 256。剪枝率从 {0, 1/4, 1/2, 3/4} 中随机采样。收敛分析证明 SubFLOT 以 $1 - \mu\eta_l E/2$ 的线性速率收敛到全局最优的邻域。

## 实验关键数据

### 主实验（Label Skew设置）

| 方法 | CIFAR10 | CIFAR100 | TinyImageNet | AG News | HAR |
|------|---------|----------|-------------|---------|-----|
| HeteroFL | 84.54 | 40.95 | 19.68 | 84.12 | 69.80 |
| FlexFL | 85.13 | 49.21 | 22.23 | 86.02 | 76.24 |
| **SubFLOT** | **86.89** | **58.37** | **29.30** | **87.88** | **79.72** |

### 消融实验（Feature Shift - PACS数据集平均准确率）

| 方法 | Photo | Art | Cartoon | Sketch | Avg |
|------|-------|-----|---------|--------|-----|
| HeteroFL | 16.23 | 13.66 | 20.27 | 26.90 | 19.27 |
| FlexFL | 16.34 | 14.78 | 21.56 | 28.01 | 20.17 |
| **SubFLOT** | **48.23** | **28.73** | **42.55** | **46.83** | **41.58** |

### 关键发现

- CIFAR100上SubFLOT达到58.37%，比次优FlexFL（49.21%）高出9.16个百分点，优势随任务复杂度增加而扩大
- 在PACS特征偏移设置下，SubFLOT平均准确率41.58%，是次优方法（~20%）的两倍以上
- 扩展性实验：客户端从10扩展到100时，SubFLOT性能衰减最小（39.20%→32.61%），而大多数基线急剧下降
- Grad-CAM可视化证实OTP生成的子模型与客户端历史模型关注相同的语义关键区域

## 亮点与洞察

- **范式突破**：首次系统性地解决了服务器端个性化剪枝问题，打破了"服务器端=无个性化"的固有认知，证明了不接触原始数据也能实现数据感知的剪枝
- **OT的双重应用**：同一套最优传输机制被优雅地复用于剪枝（OTP）和聚合（OTA），形成了从分发到回收的完整闭环
- **自适应正则化设计精巧**：SAR以剪枝率为权重的设计简洁但有效，抓住了"剪枝越狠越需要约束"的核心直觉
- **理论保障完备**：提供了严格的收敛分析，证明了线性收敛速率

## 局限与展望

- OT计算虽然采用了逐层分解策略，但在模型层数和神经元数量较大时仍有较高的计算开销
- 实验中剪枝率从固定集合中随机采样，未考虑根据客户端实际资源动态确定最优剪枝率
- 收敛分析依赖于强凸性假设（Assumption 2），在非凸神经网络中的实际意义有限
- 缺少与知识蒸馏等其他模型压缩范式在联邦场景下的对比
- 目前仅评估了结构化剪枝（宽度剪枝），未探索非结构化剪枝或其他压缩形式

## 相关工作与启发

- HeteroFL [Diao et al.] 是本文的直接改进对象，其服务器端均匀剪枝策略是SubFLOT要突破的基线
- FedOTP [Singh et al.] 和 FedAli 将OT用于客户端特征空间对齐，SubFLOT将OT从特征空间转移到参数空间、从客户端转移到服务器端
- SAR的设计思路可以推广到其他异构联邦学习场景，如不同模型架构的联邦学习

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次将OT应用于联邦剪枝，服务器端个性化剪枝的范式是全新的
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖CV/NLP/IoT多领域，label skew/feature shift/real-world多设置，大量扩展性和消融实验
- **写作质量**: ⭐⭐⭐⭐ — 问题定义和方法描述清晰，但公式较多，部分读者可能需要OT背景知识
- **价值**: ⭐⭐⭐⭐ — 对联邦学习在边缘设备上的实际部署有重要意义，方法具有即插即用的潜力

<!-- RELATED:START -->

## 相关论文

- [FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)
- [Optimal Transport-Guided Source-Free Adaptation for Face Anti-Spoofing](../../CVPR2025/human_understanding/optimal_transport-guided_source-free_adaptation_for_face_anti-spoofing.md)
- [An Efficient, Provably Optimal Algorithm for the 0-1 Loss Linear Classification Problem](../../ICLR2026/human_understanding/an_efficient_provably_optimal_algorithm_for_the_0-1_loss_linear_classification_p.md)
- [COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation](cog_confidence-aware_optimal_geometric_correspondence_for_unsupervised_single-re.md)
- [TriLite: Efficient WSOL with Universal Visual Features and Tri-Region Disentanglement](trilite_efficient_weakly_supervised_object_localization_with_universal_visual_fe.md)

<!-- RELATED:END -->
