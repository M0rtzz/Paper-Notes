---
title: >-
  [论文解读] Bridging Domain Generalization to Multimodal Domain Generalization via Unified Representations
description: >-
  [ICCV 2025][机器人][多模态域泛化] 提出URMMDG框架，通过监督对比学习构建跨模态统一表示空间，并利用互信息最小化解耦类别通用信息与模态/域特定信息，将传统单模态域泛化方法（Mixup、JiGen、IBN-Net）有效迁移到多模态域泛化场景，在EPIC-Kitchens和HAC基准上取得SOTA。
tags:
  - ICCV 2025
  - 机器人
  - 多模态域泛化
  - 统一表示
  - 监督对比学习
  - 信息解耦
  - Mixup
---

# Bridging Domain Generalization to Multimodal Domain Generalization via Unified Representations

**会议**: ICCV 2025  
**arXiv**: [2507.03304](https://arxiv.org/abs/2507.03304)  
**代码**: 无  
**领域**: 机器人 / 多模态学习  
**关键词**: 多模态域泛化, 统一表示, 监督对比学习, 信息解耦, Mixup

## 一句话总结

提出URMMDG框架，通过监督对比学习构建跨模态统一表示空间，并利用互信息最小化解耦类别通用信息与模态/域特定信息，将传统单模态域泛化方法（Mixup、JiGen、IBN-Net）有效迁移到多模态域泛化场景，在EPIC-Kitchens和HAC基准上取得SOTA。

## 研究背景与动机

域泛化（DG）旨在让模型从源域训练后在未见目标域上保持鲁棒性能。现有DG方法（数据增强、学习策略、表示学习）在单模态数据上效果显著，但直接迁移到多模态域泛化（MMDG）场景往往效果不佳。

核心问题在于**模态异步性**：不同模态（视频、音频、光流）的数据分布差异巨大。例如对同样两个类别做Mixup，视频模态的插值可能语义偏向"跑步"，而光流模态的插值可能偏向"吃东西"——跨模态的泛化方向不一致，导致联合训练效果劣于单模态独立训练。

作者通过实验量化了这个问题：JiGen在单模态（Video）上提升2.87%，但在三模态联合训练中仅提升1.61%。这说明**模态间的内在差异限制了DG方法在MMDG上的直接迁移**。

## 方法详解

### 整体框架

URMMDG框架分两步：（1）通过监督对比学习+信息解耦构建统一表示空间；（2）在统一表示上应用DG方法（Mixup/JiGen/IBN-Net），实现跨模态同步增强。

### 关键设计

1. **监督对比解耦（Supervised Contrastive Decoupling）**:

    - 对每个模态 $m$，用两个编码器分别提取通用信息 $\mathbf{z}_i^m = \Phi^m(\mathbf{x}_i^m)$ 和特定信息 $\bar{\mathbf{z}}_i^m = \Psi^m(\mathbf{x}_i^m)$
    - 通用信息捕获类别语义（跨模态跨域共享），特定信息捕获域/模态特有特征
    - 通过多模态监督对比损失 $\mathcal{L}_{scl}$ 拉近同类别跨模态样本的通用表示：
    $\mathcal{L}_{scl} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$
    - 设计动机：构建模态无关的统一语义空间，使DG方法能在该空间中同步操作所有模态

2. **互信息最小化（Mutual Information Minimization）**:

    - 使用CLUB方法最小化通用信息 $\mathbf{z}_i^m$ 和特定信息 $\bar{\mathbf{z}}_i^m$ 之间的互信息上界：
    $L_{club} = \frac{1}{N} \sum_{i=1}^{N} [\log q_\theta(\bar{\mathbf{z}}_i^m | \mathbf{z}_i^m) - \frac{1}{N} \sum_{j=1}^{N} \log q_\theta(\bar{\mathbf{z}}_j^m | \mathbf{z}_i^m)]$
    - 同时引入重建损失 $L_{rec} = \|\mathbf{x}_i^m - D(\mathbf{z}_i^m; \bar{\mathbf{z}}_i^m)\|_2^2$ 确保解耦后信息完整性
    - 设计动机：确保通用表示仅包含类别语义，不被域/模态噪声污染

3. **统一表示上的DG方法迁移**:

    - **UR-Mixup**：在通用表示 $\mathbf{z}^m$ 上做Mixup，生成增强样本后与特定信息拼接送入解码器重建特征，用于分类训练
    - **UR-JiGen**：将通用表示切分为片段，**跨模态随机选片**拼接后打乱排列，作为跨模态拼图任务
    - **UR-IBN**：在统一表示上直接应用IBN-a规范化（半通道IN+半通道BN）
    - 设计动机：在统一空间中操作，所有模态同步增强，避免了各模态独立增强导致的泛化方向分歧

### 损失函数 / 训练策略

总损失为多项加权组合：
$$L = \alpha_1 L_{cls} + \alpha_2 L_{scl} + \alpha_3 L_{club} + \alpha_4 L_{rec}$$
UR-JiGen额外加入 $L_{jig}$（权重设为1）。

## 实验关键数据

### 主实验（EPIC-Kitchens，Video+Audio+Flow三模态）

| 方法 | D2,D3→D1 | D1,D3→D2 | D1,D2→D3 | 平均 |
|------|----------|----------|----------|------|
| Base(VAF) | 54.71 | 67.20 | 61.70 | 61.20 |
| SimMMDG | 62.08 | 66.13 | 64.40 | 64.20 |
| CMRF | 61.84 | 70.13 | 70.12 | 67.36 |
| Mixup(VAF) | 57.95 | 67.95 | 64.37 | 63.42 |
| **UR-Mixup** | **61.72** | **70.89** | **70.76** | **67.79** |
| **UR-JiGen** | 62.20 | 71.14 | 67.78 | 67.04 |

### 消融实验（通过对比验证统一表示的价值）

| 配置 | Video | Audio | Flow | V-A-F | 说明 |
|------|-------|-------|------|-------|------|
| Base(V)单模态 | 58.73 | - | - | - | 单模态基线 |
| Base(VAF)多模态 | 57.13 | 37.96 | 56.65 | 61.20 | 多模态联合训练各模态性能下降 |
| JiGen(V) | 61.60 | - | - | - | 单模态DG提升+2.87 |
| JiGen(VAF) | 59.23 | 39.58 | 57.18 | 62.81 | 多模态DG仅提升+1.61 |
| UR-Mixup(VA) | 56.99 | 68.85 | - | 64.77 | 两模态统一表示效果 |
| UR-Mixup(VF) | 64.85 | - | 68.84 | 66.42 | 两模态统一表示效果 |
| UR-Mixup(VAF) | 61.72 | 70.89 | 70.76 | 67.79 | 三模态统一表示最优 |

### 关键发现

- 多模态联合训练时，各单模态性能反而不如独立训练（模态竞争现象），统一表示有效缓解了这一问题
- UR-Mixup和UR-JiGen均大幅超越直接在各模态上应用Mixup/JiGen的结果
- 本方法本质上将MMDG问题转化为统一表示空间中的单模态DG问题，使问题更易处理
- 在HAC数据集上UR-Mixup达到73.40%平均准确率，超过CMRF的72.44%

## 亮点与洞察

- **问题定义清晰**：通过Table 1的实验精确量化了"DG方法直接迁移到MMDG效果打折"这个现象
- **方法论贡献大于技术贡献**：提出的是一个通用范式——先构建统一表示，再在上面应用任意DG方法——具有很好的扩展性
- UR-JiGen的跨模态拼图设计很巧妙：从不同模态随机选片组合，既融合了多模态信息又保持了自监督任务的难度

## 局限与展望

- 仅在视频+音频+光流三模态上验证，缺少图像+文本等更常见的多模态组合实验
- 统一表示的质量高度依赖对比学习的效果，在模态差异极大时可能效果下降
- 缺少与大规模预训练多模态模型（如CLIP）的对比
- 超参数（$\alpha_1$ 到 $\alpha_4$）的敏感性分析不够充分

## 相关工作与启发

- SimMMDG、CMRF是MMDG的先行工作，但未明确提出"统一表示桥接"的范式
- CLUB互信息上界估计技术来自变分推断领域，在此处用于解耦通用/特定信息是合理的选择
- 该方法可启发将更多成熟的单模态技术（如域自适应、数据增强策略）迁移到多模态场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次提出通过统一表示将DG方法系统地桥接到MMDG
- 实验充分度: ⭐⭐⭐⭐ 多种DG方法×多种模态组合，实验设计完整
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，Figure 1很直观
- 价值: ⭐⭐⭐⭐ 提供了MMDG的通用解决范式，有较好的实践指导意义

<!-- RELATED:START -->

## 相关论文

- [Beyond Losses Reweighting: Empowering Multi-Task Learning via the Generalization Perspective](beyond_losses_reweighting_empowering_multi-task_learning_via_the_generalization_.md)
- [UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning](../../NeurIPS2025/robotics/pretraining_a_unified_pddl_domain_from_real-world_demonstrations_for_generalizab.md)
- [On the Generalization Capacities of MLLMs for Spatial Intelligence](../../ICLR2026/robotics/on_the_generalization_capacities_of_mllms_for_spatial_intelligence.md)
- [Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](../../NeurIPS2025/robotics/generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)
- [Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)

<!-- RELATED:END -->
