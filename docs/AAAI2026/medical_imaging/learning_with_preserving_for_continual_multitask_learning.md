---
title: >-
  [论文解读] Learning with Preserving for Continual Multitask Learning
description: >-
  [AAAI 2026][医学图像][持续多任务学习] 提出 Learning with Preserving（LwP）框架，通过动态加权距离保持（DWDP）损失函数维护共享表示空间的几何结构，在无需回放缓冲的条件下解决持续多任务学习（CMTL）中的灾难性遗忘问题…
tags:
  - "AAAI 2026"
  - "医学图像"
  - "持续多任务学习"
  - "表示空间保持"
  - "距离保持损失"
  - "灾难性遗忘"
  - "无回放缓冲"
---

# Learning with Preserving for Continual Multitask Learning

**会议**: AAAI 2026  
**arXiv**: [2511.11676](https://arxiv.org/abs/2511.11676)  
**代码**: [LwP](https://github.com/AICPS-Lab/lwp)  
**领域**: 持续学习 / 多任务学习  
**关键词**: 持续多任务学习, 表示空间保持, 距离保持损失, 灾难性遗忘, 无回放缓冲

## 一句话总结

提出 Learning with Preserving（LwP）框架，通过动态加权距离保持（DWDP）损失函数维护共享表示空间的几何结构，在无需回放缓冲的条件下解决持续多任务学习（CMTL）中的灾难性遗忘问题，在 BDD100k、CelebA、PhysiQ 等基准上显著超越现有持续学习方法，并且是唯一超越单任务学习基线的方法。

## 研究背景与动机

**领域现状**：自动驾驶、医学影像等关键应用中，AI 系统需要在共享数据流上持续学习新任务（如先学交通标志检测，再学行人分类）。这构成了持续多任务学习（CMTL）范式——在同一输入域上顺序学习多个任务。

**现有痛点**：传统持续学习（CL）方法（EWC、SI、ER 等）主要为任务增量学习设计，依赖参数隔离或回放缓冲来防止遗忘。但在 CMTL 设置下，这些方法学到的是碎片化的、任务特定的特征，导致：（1）任务间特征互相干扰；（2）无法建立有利于多任务的统一表示；（3）性能甚至低于独立训练的单任务基线。

**核心矛盾**：CL 的隔离策略与 MTL 的统一表示需求根本冲突——保护旧任务知识的手段（如参数冻结、回放）恰恰阻碍了跨任务共享表示的形成。

**本文切入角度**：从保护"任务输出"转向保护"表示空间的几何结构"。关键洞察是：如果潜在空间中数据点的成对距离被保持，那么在该空间上定义的任何学习问题都能保持其最优解（通过 RKHS 等价性证明）。

**核心 idea**：用动态加权的成对距离保持损失正则化共享表示空间，使新任务学习不破坏旧任务编码的几何关系。

## 方法详解

### 整体框架

LwP 采用共享特征提取器 + 任务特定头的架构。学习新任务 $\mathcal{T}_t$ 时：（1）复制前一步模型并冻结为教师；（2）添加新任务头 $g_{\theta_t}$；（3）用三部分复合损失 $\mathcal{L}_{lwp} = \lambda_c \mathcal{L}_{cur} + \lambda_o \mathcal{L}_{old} + \lambda_d \mathcal{L}_{DWDP}$ 训练当前模型——当前任务监督损失、旧任务蒸馏损失、以及核心的 DWDP 几何保持损失。

### 关键设计

1. **表示空间几何保持（Preservation Loss）**:

    - 功能：确保当前模型潜在空间中数据点的成对欧氏距离与冻结教师模型一致
    - 核心思路：基础保持损失 $\mathcal{L}_{pres} = \frac{1}{N^2}\sum_{i,j}(d(\mathbf{z}_i, \mathbf{z}_j) - d(\mathbf{z}'_i, \mathbf{z}'_j))^2$，其中 $d$ 为平方欧氏距离。论文证明：保持成对距离等价于保持高斯核 Gram 矩阵，即在 RKHS 中保持等距映射 $\phi(\mathbf{z}'_i) = T(\phi(\mathbf{z}_i))$。由此，旧表示上的任何学习问题都有等价最优解可迁移到新表示上
    - 设计动机：比直接保持参数（如 EWC）或任务输出（如 LwF）更根本——保持了表示的功能等价性，允许参数自由调整同时不丧失信息

2. **动态加权距离保持（DWDP Loss）**:

    - 功能：为成对距离保持引入动态掩码，仅保持同类样本对的距离，避免与分类目标冲突
    - 核心思路：引入掩码 $m_{ij} = \mathbb{1}[y^{[t]}_i = y^{[t]}_j]$，DWDP损失 $\mathcal{L}_{DWDP} = \frac{1}{N^2}\sum_{i,j} m_{ij}(\Delta d_{ij})^2$。仅对当前任务标签相同的样本对保持距离，不同类样本对的距离可以自由调整
    - 设计动机：无掩码时保持所有成对距离会限制模型学习新的判别特征；动态掩码在保持类内结构的同时允许类间分离，实验验证了掩码重要性

3. **教师-学生蒸馏**:

    - 功能：冻结前一步模型作为教师，用伪标签监督旧任务头
    - 核心思路：教师模型为旧任务 $o < t$ 生成伪标签 $\tilde{y}_o$，当前模型被训练匹配这些输出
    - 设计动机：蒸馏损失保持显式任务知识，DWDP 保持隐式结构知识，两者互补

### 损失函数 / 训练策略

总损失 $\mathcal{L}_{lwp} = \lambda_c \mathcal{L}_{cur} + \lambda_o \mathcal{L}_{old} + \lambda_d \mathcal{L}_{DWDP}$。其中 $\mathcal{L}_{cur}$ 为当前任务交叉熵，$\mathcal{L}_{old}$ 为旧任务蒸馏 MSE，$\mathcal{L}_{DWDP}$ 为核心几何保持。无需回放缓冲区，适用于隐私敏感场景。

## 实验关键数据

### 主实验（无分布偏移）

| 方法 | BDD100k (3任务) | CelebA (10任务) | PhysiQ (3任务) | FairFace (3任务) |
|------|----------------|----------------|---------------|-----------------|
| STL (单任务) | 75.12 | 72.23 | 87.17 | 64.44 |
| LwF | 76.65 | 64.63 | 69.95 | 61.03 |
| oEWC | 74.87 | 69.67 | 82.64 | 63.60 |
| DER++ | 76.68 | 67.69 | 82.84 | 63.81 |
| OBC | 76.99 | 70.83 | 84.00 | 63.87 |
| **LwP (Ours)** | **78.30** | **73.48** | **88.24** | **66.48** |

### 分布偏移下（BDD100k）

| 方法 | Weather Shift | Scene Shift | Time-of-Day | Combined |
|------|-------------|------------|-------------|----------|
| STL | 76.76 | 76.79 | 76.42 | 76.75 |
| LwF | 76.79 | 77.50 | 76.03 | 76.94 |
| SI | 75.85 | 77.89 | 74.82 | 74.57 |
| **LwP** | **77.94** | **78.20** | **77.64** | **77.77** |

### 关键发现

- LwP 是唯一在所有数据集和设置下**超越单任务基线（STL）**的方法，说明 CMTL 中的知识共享是可能的
- 传统 CL 方法（EWC、ER、SI）在 CMTL 下全部低于 STL 基线，验证了它们的隔离策略不适合 CMTL
- 在分布偏移场景中 LwP 的鲁棒性特别突出，因为几何保持使表示对输入分布变化更稳定
- 消融实验表明：平方欧氏距离优于 RBF 核距离（消融 Section 4.6），动态掩码贡献约 1-2% 的性能提升

## 亮点与洞察

- **保持几何结构的深刻洞察**：通过 RKHS 等价性证明，保持成对距离即保持了表示空间的全部功能性，这比保持参数或输出更根本。这个洞察超越了 CMTL，可能启发更广泛的迁移学习和联邦学习研究
- **超越 STL 基线的意义**：证明了在 CMTL 中任务间的知识确实可以正向迁移，而非仅仅"防止遗忘"
- **无回放设计**：不需要存储历史数据，适合医学影像等隐私敏感领域，而性能还优于使用回放的方法（如 ER、DER++）

## 局限与展望

- **DWDP 的计算复杂度**：成对距离计算为 $O(N^2)$，对大 batch size 可能显著增加训练开销
- **任务数量的扩展性**：实验最多 10 个任务（CelebA），超大规模任务序列（如 50+）的性能未验证
- **掩码仅使用当前任务标签**：DWDP 掩码基于当前任务标签 $y^{[t]}$，可能无法最优地保护旧任务的类间关系
- **理论与实践的 gap**：RKHS 等价性理论假设精确的距离保持，但实践中是近似优化

## 相关工作与启发

- **vs EWC/SI**: 这些方法通过参数正则化防遗忘，但隔离了任务特定知识。LwP 保护的是共享表示结构，允许参数自由变化
- **vs PODNet**: PODNet 均匀保持空间特征的成对距离，LwP 用动态掩码区分类内/类间对，避免优化冲突
- **vs RKD**: RKD 保持所有成对距离用于知识蒸馏，不区分类别。LwP 的动态掩码是关键差异
- CMTL 设置的正式定义和实验证据表明，这是一个值得独立研究的方向，而非简单的 Task-IL 子问题

## 评分

- 新颖性: ⭐⭐⭐⭐ DWDP 损失和几何保持的理论处理很有深度，CMTL 形式化有启发性
- 实验充分度: ⭐⭐⭐⭐ 4个数据集 + 分布偏移 + 完整消融 + 与12种方法对比
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，问题动机清晰，RKHS 证明优雅
- 价值: ⭐⭐⭐⭐ 无回放+超STL的持续学习对实际部署有直接意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning](../../ICML2026/medical_imaging/dp-kfc_data-free_preconditioning_for_privacy-preserving_deep_learning.md)
- [\[CVPR 2026\] Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](../../CVPR2026/medical_imaging/residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)
- [\[NeurIPS 2025\] EWC-Guided Diffusion Replay for Exemplar-Free Continual Learning in Medical Imaging](../../NeurIPS2025/medical_imaging/ewc-guided_diffusion_replay_for_exemplar-free_continual_learning_in_medical_imag.md)
- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](../../CVPR2026/medical_imaging/forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)
- [\[AAAI 2026\] Radiation-Preserving Selective Imaging for Pediatric Hip Dysplasia: A Cross-Modal Approach](radiation-preserving_selective_imaging_for_pediatric_hip_dysplasia_a_cross-modal.md)

</div>

<!-- RELATED:END -->
