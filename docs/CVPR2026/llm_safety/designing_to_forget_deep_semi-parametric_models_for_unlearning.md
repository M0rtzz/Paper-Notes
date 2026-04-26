---
title: >-
  [论文解读] Designing to Forget: Deep Semi-parametric Models for Unlearning
description: >-
  [CVPR 2026][机器遗忘] 提出"Designing to Forget"理念，设计了一族深度半参数模型 (SPM)，在推理时通过简单删除训练样本即可实现遗忘（无需修改模型参数），在 ImageNet 分类上将与重训基线的预测差距减少 11%，遗忘速度提升 10 倍以上。
tags:
  - CVPR 2026
  - 机器遗忘
  - 半参数模型
  - 测试时删除
  - 数据隐私
  - 扩散模型
---

# Designing to Forget: Deep Semi-parametric Models for Unlearning

**会议**: CVPR 2026  
**arXiv**: [2603.22870](https://arxiv.org/abs/2603.22870)  
**代码**: [github.com/amberyzheng/spm_unlearning](https://github.com/amberyzheng/spm_unlearning) (有)  
**领域**: Others (Machine Unlearning / AI Safety)  
**关键词**: 机器遗忘, 半参数模型, 测试时删除, 数据隐私, 扩散模型

## 一句话总结
提出"Designing to Forget"理念，设计了一族深度半参数模型 (SPM)，在推理时通过简单删除训练样本即可实现遗忘（无需修改模型参数），在 ImageNet 分类上将与重训基线的预测差距减少 11%，遗忘速度提升 10 倍以上。

## 研究背景与动机
**领域现状**: 机器遗忘 (MU) 受 GDPR 等隐私法规驱动，要求从训练好的模型中移除特定样本的影响。现有方法主要通过微调模型参数近似"好像从未用过该数据训练"的效果。

**现有痛点**: 深度学习的黑箱特性使得很难解耦单个训练样本对参数的贡献；现有 MU 算法需要额外的微调步骤，在频繁遗忘场景下开销显著。

**核心矛盾**: 参数化模型将所有训练数据信息压缩到参数中，导致遗忘必须修改参数；而非参数模型（如 KNN）天然支持删除但性能不够。

**本文目标**: 设计一种"天生适合遗忘"的神经网络架构，而非为已有架构设计遗忘算法。

**切入角度**: 从 KNN 的"删除即遗忘"特性出发，设计同时具备参数模型性能和非参数模型遗忘便利性的半参数模型。

**核心 idea**: 半参数模型 $\hat{y} = G_{\theta^*}(x, \mathcal{T})$ 在推理时显式依赖训练集 $\mathcal{T}$，遗忘只需 $G_{\theta^*}(x, \mathcal{T} \setminus \mathcal{U})$——删除数据而不修改参数。

## 方法详解

### 整体框架
SPM 由三种模块组成：(1) Fusion 模块 $g$：融合参数和非参数分支；(2) Non-parametric 模块 $h$：将训练集转换为 instance embeddings；(3) Parametric 模块 $f$：标准深度网络层。两个分支交替作用：参数分支处理输入特征，非参数分支维护训练集的 instance 表示，通过 fusion 模块在每层融合。

### 关键设计
1. **Fusion 模块 (加权聚合)**：$g(z, \mathcal{S}) = \sum_{s_i \in \mathcal{S} \setminus \{s_z\}} \alpha(z, s_i) \cdot s_i$，其中 $\alpha$ 是注意力权重，关键地排除了当前样本自身的 instance embedding $s_z$。设计动机：排除自身强制模型学习其他数据点的相对关系，类似非参数方法的精神。如果不排除，模型可能退化为参数模型（直接使用自身的 embedding）。

2. **Non-parametric 模块 (置换等变)**：$\mathcal{S}^{(l)} = \{[h^{(l)}(s_i^{(l)}), y_i]\}_{i=1}^{|\mathcal{T}|}$，使用共享的实例级变换处理每个训练样本，保持置换等变性。设计动机：置换等变确保模型行为不依赖于数据插入顺序，且支持聚类/检索来降低集合大小。

3. **标签置换增强 (Label-permutation Augmentation)**：训练时随机打乱 one-hot 标签向量的索引。设计动机：防止模型忽略 $x$ 而仅使用 one-hot 标签作为"偏置项"来绕过非参数分支。这是确保模型真正依赖训练数据内容的关键技巧。

4. **效率优化**：通过 (R)etrieval（近邻检索）或 (C)lustering（按类平均）将 $\mathcal{S}$ 缩减到固定大小。对于生成任务，SPM 基于 UNet 架构，将 mid block 替换为 fusion 模块。

### 损失函数 / 训练策略
- 分类：交叉熵损失 + label-permutation augmentation
- 生成：标准 DDPM 扩散损失，fusion 在 patch 级别操作
- 预训练 ResNet 可通过添加非参数分支被适配为 SPM

## 实验关键数据

### 主实验（分类性能）

| 模型 | CIFAR-10 Acc↑ | ImageNet Acc↑ |
|------|-------------|--------------|
| ResNet18 | 94.9 | 68.93 |
| ResNet18-KNN (100%) | 94.5 | 66.9 |
| SPM-C (100%) | 94.5 | 67.1 |
| SPM-R (100%) | 94.1 | 59.9 |

**生成性能（CIFAR-10 FID↓）**

| 方法 | FID | 运行时间 |
|------|-----|---------|
| DDPM | 7.28 | 42s |
| SPM (|T|=100) | 7.09 | 173s |
| SPM (|T|=1024) | 7.04 | 1486s |

### 消融实验（CIFAR-10 分类遗忘）

| 方法 | PG_H↓ | PG_S↓ | 遗忘时间↓ |
|------|-------|-------|----------|
| Retrain (Oracle) | 0.00 | 0.00 | 2317.6s |
| GA | 18.48 | 0.99 | 8.9s |
| FT | 13.11 | 0.48 | 148.7s |
| **SPM-C (Ours)** | **0.43** | **0.08** | **0.7s** |

### 关键发现
- SPM-C 在分类任务上几乎与 ResNet18 持平（CIFAR-10: 94.5 vs 94.9，ImageNet: 67.1 vs 68.93）
- **遗忘效果接近完美**：与重训基线的预测差距 (PG_H) 仅 0.43（最佳 MU 算法 GA 为 18.48）
- **遗忘速度极快**：0.7s vs 重训的 2317.6s（3300x 加速）vs 最快 MU 算法 GA 的 8.9s（12.7x 加速）
- 生成 SPM（基于 DDPM）的 FID 与标准 DDPM 接近（7.04 vs 7.28），但推理速度因维护集合而显著增加
- 在 ImageNet 上，SPM 比参数模型的遗忘差距减少 11%

## 亮点与洞察
- **设计理念的转变**：从"如何遗忘"（算法导向）到"如何设计易于遗忘的模型"（架构导向），是 MU 领域的范式创新。
- **KNN 启发的 fusion 设计**：排除自身 embedding + 注意力加权聚合，优雅地在深度网络中实现了非参数行为。
- **Label-permutation augmentation** 是防止模型绕过非参数分支的关键，体现了 SPM 设计中的细致考虑。
- 可以将预训练的参数模型（ResNet）改造为 SPM，降低了从头训练的成本。

## 局限与展望
- **推理时间增加**：SPM 在推理时需要维护和检索训练集，ImageNet 上增加约 20% 开销（聚类模式）
- **ImageNet 准确性差距**：SPM-C (67.1) vs ResNet18 (68.93) 还有约 2% 的差距
- **生成 SPM 的运行时间代价**：|T|=1024 时推理时间增加 35 倍，限制了实际应用规模
- 未在更大规模模型（如 ViT, DiT）上验证
- 目前仅验证了分类和无条件生成，文生图等更复杂的生成任务待探索

## 相关工作与启发
- 与 SISA (Bourtoule et al.) 的区别：SISA 将数据分片训练多个模型，删除整个模型实现遗忘；SPM 保持单一模型但在推理时删除数据。
- 半参数模型在 NLP（检索增强生成）和视觉（检索增强生成）中已有应用，但本文首次将其用于遗忘。
- 与差分隐私的互补：DP 提供训练时隐私保证，SPM 提供部署后的样本删除能力。
- Label-permutation augmentation 类似 dropout 的正则化——强迫模型不走捷径

## 技术细节补充
- **Fusion 注意力**: $\alpha(z, s_i) = \frac{\exp((W_q z)^\top (W_k s_i))}{\sum_j \exp((W_q z)^\top (W_k s_j))}$
- **Patch-level 生成融合**: 基于 UNet 的 mid block 替换为融合模块，Bahdanau 注意力
- **SPM-C (聚类模式)**: 按类平均 instance embeddings→集合大小=类数→运行时间与参数模型持平
- **GNN 增强**: class-aware GNN + 多头图注意力可将 CIFAR-10 从 94.1% 提升到 94.4%
- **PG_H/PG_S**: 硬/软预测差距，衡量与重训 oracle 的距离
- **5 类遗忘 (50%)**: SPM-C 的 PG_H = 0.02 vs GA = 32.62，接近完美的遗忘效果

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从架构设计角度解决遗忘问题，范式创新
- 实验充分度: ⭐⭐⭐⭐ 分类和生成双任务验证 + 多种 MU 算法对比
- 写作质量: ⭐⭐⭐⭐ 概念解释清晰，图示直观
- 价值: ⭐⭐⭐⭐ 在隐私和安全合规场景中有重要意义，但推理开销限制应用

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] DAMP: Class Unlearning via Depth-Aware Removal of Forget-Specific Directions](damp_class_unlearning_via_depth_aware_removal_of_forget_specific_directions.md)
- [\[ACL 2025\] Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning](../../ACL2025/llm_safety/answer_when_needed_forget_when_not_language_models_pretend_to_forget_via_in-cont.md)
- [\[CVPR 2026\] ⊘ Source Models Leak What They Shouldn't ↛: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization](oslash_source_models_leak_what_they_shouldnt_nrightarrow_unlearning_zero-shot_tr.md)
- [\[CVPR 2026\] SineProject: Machine Unlearning for Stable Vision–Language Alignment](sineproject_machine_unlearning_for_stable_vision_language_alignment.md)
- [\[AAAI 2026\] Designing Truthful Mechanisms for Asymptotic Fair Division](../../AAAI2026/llm_safety/designing_truthful_mechanisms_for_asymptotic_fair_division.md)

<!-- RELATED:END -->
