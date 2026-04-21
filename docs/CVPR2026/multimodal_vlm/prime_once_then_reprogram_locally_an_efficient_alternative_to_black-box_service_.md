---
title: >-
  [论文解读] Prime Once, then Reprogram Locally: An Efficient Alternative to Black-Box Service Model Adaptation
description: >-
  [CVPR 2026][多模态][模型即服务] 本文提出AReS方法，用单次API查询预热本地编码器代替传统零阶优化（ZOO）的持续API调用，在GPT-4o上获得+27.8%提升（ZOO方法几乎无效），同时将API调用量减少99.99%以上，实现了无成本推理。
tags:
  - CVPR 2026
  - 多模态
  - 模型即服务
  - 黑盒适配
  - 视觉重编程
  - 零阶优化
  - API高效利用
---

# Prime Once, then Reprogram Locally: An Efficient Alternative to Black-Box Service Model Adaptation

**会议**: CVPR 2026  
**arXiv**: [2604.01474](https://arxiv.org/abs/2604.01474)  
**代码**: [https://github.com/yunbeizhang/AReS](https://github.com/yunbeizhang/AReS)  
**领域**: 多模态VLM  
**关键词**: 模型即服务, 黑盒适配, 视觉重编程, 零阶优化, API高效利用

## 一句话总结

本文提出AReS方法，用单次API查询预热本地编码器代替传统零阶优化（ZOO）的持续API调用，在GPT-4o上获得+27.8%提升（ZOO方法几乎无效），同时将API调用量减少99.99%以上，实现了无成本推理。

## 研究背景与动机

1. **领域现状**：Model-as-a-Service（MaaS）是部署SOTA模型的主流范式，用户只能通过API获取输入-输出的预测结果。闭盒视觉重编程（如BAR、BlackVIP）通过零阶优化修改输入图像来适配API模型。
2. **现有痛点**：ZOO方法面临三重困境：(1)需要海量API调用（约10^8次），训练和推理成本极高；(2)梯度估计不稳定，优化过程缓慢且不可靠；(3)现代强大API（如GPT-4o）对输入扰动具有鲁棒性，ZOO依赖的微小输入扰动被模型忽略，导致几乎无法获得性能提升。
3. **核心矛盾**：ZOO方法的根本假设是"通过扰动输入可以影响模型输出"，但现代模型越强大，对扰动越鲁棒，使得这一假设逐渐失效。
4. **本文目标** 如何在最严格的闭盒设定下（仅有输入-预测概率访问）高效适配服务模型，尤其当ZOO方法对现代API无效时。
5. **切入角度**：与其在闭盒模型上持续做代价昂贵的零阶优化，不如进行一次性API交互获取知识，在本地模型上做高效的白盒优化。
6. **核心 idea**：单次查询API预热本地编码器，然后在本地完成全部视觉重编程和推理，彻底消除后续API依赖。

## 方法详解

### 整体框架

AReS分两阶段：(1) **Prime Once**（预热阶段）——对每张训练图像仅查询API一次获取预测概率，用这些概率训练一个加在本地编码器上的轻量线性层，使本地模型学会"模仿"服务模型的行为；(2) **Reprogram Locally**（本地重编程阶段）——在已预热的本地模型上用标准梯度下降优化视觉prompt，完全白盒操作，无需再访问API。推理时仅使用本地模型，实现零API成本推理。

### 关键设计

1. **单次API预热（Prime Once）**:

    - 功能：用单次API交互将服务模型的知识转移到本地编码器
    - 核心思路：冻结本地编码器，仅训练其顶层的一个线性层 $\theta \in \mathbb{R}^{K^S \times (d_{enc}+1)}$。对每张训练图 $x_i$ 查询API获取预测概率 $p_S(x_i)$，然后最小化本地模型与服务模型输出的KL散度 $\mathcal{L}_P(p_L, p_S) = -\sum_j p_{S,j} \log p_{L,j}$。关键点：预热不要求标签空间一致——即使API在ImageNet空间输出，目标任务是Flowers，预热仍然有效，因为其目的是"准备"而非"蒸馏"。
    - 设计动机：与传统知识蒸馏不同，预热的目标不是生成最终高性能模型，而是让本地模型对后续重编程更"敏感"、更"可编程"。

2. **本地白盒视觉重编程**:

    - 功能：在已预热的本地模型上高效学习视觉prompt
    - 核心思路：定义可学习视觉prompt $\mathbf{P}$和输入变换函数 $g_{in}$，用标准交叉熵损失和精确梯度优化：$\mathbf{P}^* = \arg\min_{\mathbf{P}} \mathbb{E}_{(x,y)} [\ell(g_{out}(\mathcal{F}_L(g_{in}(x, \mathbf{P}); \theta^*)), y)]$。由于是白盒操作，可使用一阶优化器（如Adam），比ZOO的近似梯度更稳定、更快速。
    - 设计动机：将难以优化的闭盒问题转化为简单的白盒优化问题，利用精确梯度实现更好更快的收敛。

3. **AReS-MS模型选择策略**:

    - 功能：自动判断何时使用本地模型、何时回退到API零样本
    - 核心思路：利用预热阶段作为低成本诊断工具——如果本地模型的预热性能达到API零样本基线的容忍度 $\tau$ 内，则使用高效的本地路径；否则回退到零样本API。这使得AReS不仅是适配方法，更是成本-性能权衡的智能决策框架。
    - 设计动机：在Food101、Cars等数据集上，所有重编程方法（含白盒）都不如CLIP零样本，说明输入级重编程对这些领域有天然局限。AReS-MS自动识别这些场景并做出最优选择。

### 损失函数 / 训练策略

预热阶段：KL散度损失，Adam优化器lr=0.001。重编程阶段：交叉熵损失，Adam优化器lr=0.01，padding-based视觉prompt。对于VM（标准视觉模型），需额外使用贝叶斯标签映射（BLM）来桥接源/目标标签空间差异。

## 实验关键数据

### 主实验（CLIP ViT-B/16作为服务模型, 16-shot）

| 方法 | Flowers | DTD | UCF | Food | GTSRB | EuroSAT | Pets | Cars | SUN | SVHN | Avg | API调用(M) | 时间(h) |
|------|---------|-----|-----|------|-------|---------|------|------|-----|------|-----|----------|--------|
| Zero-shot | 71.3 | 43.9 | 66.9 | 85.9 | 21.0 | 47.9 | 89.1 | 65.2 | 62.6 | 17.9 | 57.2 | 0.12 | 0 |
| BAR | 71.0 | 46.8 | 64.2 | 84.4 | 21.5 | 77.3 | 88.4 | 63.0 | 62.4 | 34.6 | 61.4 | 612.8 | 185.6 |
| BlackVIP | 70.6 | 45.3 | 68.7 | 85.9 | 21.3 | 73.3 | 89.1 | 65.4 | 64.5 | 44.4 | 62.9 | 754.2 | 197.5 |
| **AReS** | **86.6** | 48.2 | 67.1 | 68.8 | **39.4** | **85.7** | 88.9 | 43.2 | 62.8 | **63.2** | **65.4** | **0.02** | **3.7** |
| **AReS-MS** | **86.6** | 48.2 | 67.1 | **85.9** | **39.4** | **85.7** | 88.9 | **65.2** | 62.8 | **63.2** | **69.3** | 0.06 | 3.7 |

### 消融实验（真实API评估，EuroSAT 16-shot）

| 方法 | LLaVA Acc | GPT-4o Acc | GPT-4o 总费用($) | Clarifai Acc | Clarifai 总费用($) |
|------|-----------|-----------|----------------|-------------|-------------------|
| Zero-shot | 40.1 | 59.4 | 14.6 | - | - |
| BAR | 34.1 | 59.1 | 72.2 | 68.1 | 48.1 |
| BlackVIP | 39.4 | 60.1 | 101.0 | 72.1 | 67.3 |
| **AReS** | **73.1** | **87.2** | **0.3** | **83.2** | **0.2** |

### 关键发现

- **GPT-4o上的巨大优势**：AReS在GPT-4o上提升+27.8%（59.4→87.2），而BlackVIP仅提升+0.7%。这直接证实了ZOO方法在强鲁棒API面前失效的论断。
- **API调用减少99.99%以上**：AReS仅需0.02M次API调用（vs BlackVIP的754M），训练时间从197小时降至3.7小时。
- **组件分析**：仅预热（45.6%）< 仅本地VR（70.6%）< 仅本地LP（73.8%）< 本地VR+LP（80.1%）< AReS完整方案（85.7%），预热+重编程的协同效应显著。
- **额外无标签数据的利用**：预热阶段可利用无标签下游数据进一步提升性能，这是ZOO方法无法实现的优势。

## 亮点与洞察

- **范式转换**：从"在闭盒模型上持续优化"转向"一次交互+本地优化"，这是一个根本性的思路转变。关键洞察是：与其花大代价直接优化闭盒模型，不如用少量代价让本地模型变得更可编程。
- **预热≠蒸馏**：预热与知识蒸馏在机制上相似，但目的完全不同。蒸馏追求最终性能，需要标签空间对齐；预热只是"准备"，即使标签空间完全不同也能工作。这个区分非常巧妙。
- **理论保证**：通过$\epsilon$-忠实预热假设建立了性能界限 $\mathcal{R}_L(\mathcal{D}^T, \mathbf{P}^*) - \epsilon \leq \mathcal{R}_S(\mathcal{D}^T, \mathbf{Q}^*) \leq \mathcal{R}_L(\mathcal{D}^T, \mathbf{P}^*)$，将ZOO的不稳定优化转化为本地的稳定优化问题。

## 局限与展望

- 在Food101和Cars上，AReS（以及所有重编程方法）不如零样本CLIP，说明输入级视觉重编程对某些数据域有天然局限
- 预热阶段仍需对所有训练样本做一次API查询，在大规模数据集上可能成本不低
- 仅验证了图像分类任务，对检测、分割等需要密集预测的任务的适用性未知
- 本地编码器的选择对性能有较大影响（ViT-B/16 vs RN50差距明显），如何自动选择最佳本地编码器值得探索

## 相关工作与启发

- **vs BlackVIP**：BlackVIP用SPSA-GC估计梯度并引入Coordinator网络，但本质仍是ZOO。AReS通过预热彻底避开了闭盒优化。两者都假设有相同的本地编码器可用。
- **vs BAR**：BAR用随机无梯度(RGF)优化，API调用量更大（612M vs 754M BlackVIP），且在强API上效果更差。
- **vs 知识蒸馏**: 传统蒸馏需要标签空间对齐，目标是学生模型独立达到高性能。AReS的预热允许标签空间不匹配，且后续还有重编程步骤来弥补差距。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 范式转换级别的创新，从持续闭盒优化到一次预热+本地优化
- 实验充分度: ⭐⭐⭐⭐⭐ 10个数据集+多种服务模型（CLIP/ViT/LLaVA/GPT-4o/Clarifai），真实费用对比
- 写作质量: ⭐⭐⭐⭐ 故事叙述流畅，实验设计精巧，但符号记法偶有重载
- 价值: ⭐⭐⭐⭐⭐ 解决了实际痛点（API成本），且在GPT-4o时代尤为重要

<!-- RELATED:START -->

## 相关论文

- [Locate-then-Sparsify: Attribution Guided Sparse Strategy for Visual Hallucination Mitigation](locate-then-sparsify_attribution_guided_sparse_strategy_for_visual_hallucination.md)
- [Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model](../../ICCV2025/multimodal_vlm/interpretable_zero-shot_learning_with_locally-aligned_vision-language_model.md)
- [Evolving Prompt Adaptation for Vision-Language Models](evolving_prompt_adaptation_for_visionlanguage_mode.md)
- [Disentangle-then-Align: Non-Iterative Hybrid Multimodal Image Registration via Cross-Scale Feature Disentanglement](disentangle-then-align_non-iterative_hybrid_multimodal_image_registration_via_cr.md)
- [A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1](../../NeurIPS2025/multimodal_vlm/a_frustratingly_simple_yet_highly_effective_attack_baseline.md)

<!-- RELATED:END -->
