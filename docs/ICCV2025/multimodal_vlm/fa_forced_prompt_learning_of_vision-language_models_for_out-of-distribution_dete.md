---
title: >-
  [论文解读] FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection
description: >-
  [ICCV 2025][多模态][OOD检测] 提出FA（Forced prompt leArning），通过引入一个可学习的"强制提示"并迫使其产生比冻结原始提示更高的ID类别匹配度，使提示学到超越标签文本语义的丰富ID类别描述，在无需外部辅助数据和额外参数的条件下显著提升基于CLIP的少样本OOD检测性能。
tags:
  - ICCV 2025
  - 多模态
  - OOD检测
  - CLIP提示学习
  - 少样本学习
  - 分布外检测
  - 强制提示
---

# FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection

**会议**: ICCV 2025  
**arXiv**: [2507.04511](https://arxiv.org/abs/2507.04511)  
**代码**: [https://github.com/0xFAFA/FA](https://github.com/0xFAFA/FA)  
**领域**: 多模态VLM  
**关键词**: OOD检测, CLIP提示学习, 少样本学习, 分布外检测, 强制提示

## 一句话总结
提出FA（Forced prompt leArning），通过引入一个可学习的"强制提示"并迫使其产生比冻结原始提示更高的ID类别匹配度，使提示学到超越标签文本语义的丰富ID类别描述，在无需外部辅助数据和额外参数的条件下显著提升基于CLIP的少样本OOD检测性能。

## 研究背景与动机
OOD检测旨在判断测试样本是否属于训练分布（ID），对AI可靠性至关重要。基于CLIP的方法因其强大的零样本泛化能力而在少样本OOD检测中表现优异，但现有方法存在根本性路线分歧：

**现有主流路线——学习OOD相关知识**：
- LoCoOp/SCT：最大化ID无关区域的熵来推远OOD嵌入
- ID-like：从ID样本邻域构造OOD样本来学习精细差异
- NegPrompt/LSN：学习负提示来表示ID类别的"反面"
- 依赖外部大规模辅助数据集的方法

这些方法的共同问题是：OOD数据在实际中是**无限且不可知**的，试图显式建模OOD知识有天然局限性。从暴露的特定OOD特征难以泛化到未知的OOD分布。

**本文的新路线——充分利用ID知识**：
受开放集识别研究的启发（提升闭集精度通常也能改善开放集识别），作者认为**与其费力学习无限的OOD模式，不如让模型更深入地理解ID类别**。如果模型对ID类别有足够丰富和精细的描述，OOD样本自然难以与之匹配。

核心idea：学习一个"强制提示"（forced prompt），使其产生的文本特征与ID图像的相似度**超过**原始手工提示，迫使其学到标签名称之外更丰富的ID类别描述信息。

## 方法详解

### 整体框架
FA框架包含两个提示——冻结的原始提示（参考基准）和可学习的强制提示（学习目标），两者初始化相同（"a photo of a [class-c]"）。训练时冻结原始提示，仅优化强制提示，使ID图像与强制提示的相似度高于与原始提示的相似度。

### 关键设计
1. **强制提示（Forced Prompt）**:

    - 功能：引入一个与原始提示初始化相同但可学习的提示副本
    - 核心思路：提示格式 $\mathbf{u}_c = [\mathbf{v}_1, \cdots, \mathbf{v}_L, \mathbf{w}_c]$，其中 $\mathbf{v}_i$ 是所有类共享的可学习向量（与CoOp一致），$\mathbf{w}_c$ 是冻结的类名嵌入
    - 设计动机：
        - 使用手工模板初始化（而非随机初始化），保留CLIP先验知识中的语义信息，提升泛化能力
        - 共享可学习向量（而非类独立），在少样本场景下避免参数过多导致的过拟合
        - 冻结原始提示作为参考基线，确保强制提示必须学到"更多"而非偏离

2. **强制交叉熵损失（FCE Loss）**:

    - 功能：修改标准交叉熵损失的分母，将原始提示的得分也纳入归一化
    - 核心公式：
    $\mathcal{L}_{FCE} = \mathbb{E}_{(\mathbf{x}, y_c)} \left[-\log \frac{e^{s_c^f/\tau}}{\sum_{j=1}^C e^{s_j^f/\tau} + \sum_{j=1}^C e^{s_j^o/\tau}}\right]$
      其中 $s_j^f = \cos(\mathbf{z}, \mathbf{t}_j^f)$ 和 $s_j^o = \cos(\mathbf{z}, \mathbf{t}_j^o)$ 分别是图像与强制/原始提示的相似度
    - 设计动机：分母中包含原始提示的相似度，意味着仅靠与原始提示一样好是不够的——强制提示必须产生**更高**的相似度才能最小化损失。这迫使强制提示学习到更丰富、更具区分性的ID类别描述

3. **强制系数 $K$**:

    - 功能：控制原始提示在分母中的权重，调节"强制"的强度
    - 核心公式：
    $\mathcal{L}_{FCE-K} = \mathbb{E}_{(\mathbf{x}, y_c)} \left[-\log \frac{e^{s_c^f/\tau}}{\sum_{j=1}^C e^{s_j^f/\tau} + K\sum_{j=1}^C e^{s_j^o/\tau}}\right]$
    - 设计动机：$K=0$ 时退化为CoOp（无强制约束），$K$ 越大则强制提示被迫学到越全面的ID描述。实验发现 $K=1$ 即可带来显著提升，最优值与数据集相关

4. **推理阶段的得分函数**:

    - MCM得分：$S_{MCM}(\mathbf{x}) = \max_c \frac{e^{\cos(\mathbf{z}^g, \mathbf{t}_c^a)/\tau_0}}{\sum_j e^{\cos(\mathbf{z}^g, \mathbf{t}_j^f)/\tau_0} + K e^{\cos(\mathbf{z}^g, \mathbf{t}_j^o)/\tau_0}}$
    - GL-MCM得分：同时考虑全局和局部特征的匹配
    - $\mathbf{t}_c^a$ 是强制提示和原始提示特征的拼接

### 损失函数 / 训练策略
- 仅训练强制提示的共享token向量，参数量与CoOp完全一致
- 冻结原始提示和类名嵌入，保留CLIP的泛化能力
- 不使用任何外部OOD数据或辅助数据集

## 实验关键数据

### 主实验（ImageNet-1k作为ID，1-shot设置）

| 方法 | iNaturalist FPR95↓ | SUN FPR95↓ | Places FPR95↓ | Textures FPR95↓ | 平均FPR95↓ | 平均AUROC↑ |
|------|-------------------|-----------|--------------|----------------|-----------|-----------|
| SCT_GL (SOTA) | 20.57 | 24.56 | 33.27 | 48.12 | 31.62 | 92.01 |
| LoCoOp_GL | 21.97 | 24.95 | 34.14 | 49.04 | 32.53 | 92.17 |
| IDLike | 17.73 | 48.17 | 50.43 | 29.12 | 36.36 | 91.93 |
| **FA_GL (Ours)** | **14.12** | **29.99** | **32.48** | **34.66** | **27.81** | **93.26** |

### 消融实验 / 更多OOD基准

| 配置 | 挑战性OOD (16-shot) 平均FPR95↓ | 平均AUROC↑ | 说明 |
|------|-------------------------------|-----------|------|
| CoOp_GL | 57.21 | 81.99 | 基线 |
| SCT_GL | 58.25 | 82.24 | 不比CoOp好多少 |
| LoCoOp_GL | 59.03 | 82.23 | 类似 |
| **FA_GL (Ours)** | **53.61** | **83.93** | 显著改善 |

| ID精度 (ImageNet Top-1) | 1-shot | 4-shot | 16-shot |
|------------------------|--------|--------|---------|
| CoOp | 67.44 | 69.71 | 70.99 |
| LoCoOp | 67.40 | 69.55 | 71.53 |
| SCT | 68.63 | 69.93 | 71.78 |
| **FA (Ours)** | **68.67** | **69.96** | 71.02 |

### 关键发现
- **FA在不使用任何外部数据的情况下超越所有方法**：1-shot FPR95从31.62%降到27.81%，AUROC从92.01%提升到93.26%
- **ID分类精度不降反升**：强制提示学到的丰富描述反而帮助了分类
- **与MCM/GL-MCM得分函数兼容**：FA可灵活组合不同OOD检测得分
- **挑战性OOD数据集上优势更大**：在OpenImage-O、NINCO、ImageNet-O等更干净的数据集上，FA_GL的FPR95比SCT_GL低4.64个百分点
- **参数量零增加**：与CoOp保持完全相同的可学习参数量
- 学习OOD知识的方法（如NegPrompt的FPR95高达62.08%）在某些设置下反而表现很差

## 亮点与洞察
1. **思路的逆向突破**：不去学复杂的OOD模式，而是让模型更好地理解ID——"知己"而非"知彼"
2. **极简设计实现强效果**：仅修改损失函数分母和引入一个标量系数 $K$，无任何额外模块或参数，却带来显著提升
3. **强制机制的优雅性**：通过将冻结原始提示作为分母中的"竞争对手"，自然迫使可学习提示超越语义标签的信息量
4. **理论直觉**：ID样本与强制提示的高匹配度意味着更紧凑的ID分布表示，OOD样本与两种提示的匹配度差异更小，从而更容易被区分

## 局限与展望
1. 强制系数 $K$ 需要调优，虽然作者声称不敏感但最优值仍随数据集变化
2. $K=0$ 退化为CoOp时的表现已经不错，说明FA的提升部分来自强制机制，部分来自初始化策略——两者的贡献未彻底解耦
3. 仅在ViT-B/16上实验，未测试更大backbone或其他VLM架构
4. 16-shot时FA的ID精度略低于LoCoOp和SCT，暗示强制约束在数据充足时可能不如直接学习
5. 未探索将FA与现有OOD知识学习方法结合——两种路线是否互补？

## 相关工作与启发
- **与CoOp的关系**：FA可以视为CoOp的改良版——同样的参数预算，通过引入冻结参考和强制损失学到更好的提示
- **与LoCoOp/SCT的关系**：后者通过熵最大化推远OOD，FA通过拉近ID——方向相反但互补
- **与NegPrompt的关系**：NegPrompt显式学习负提示（OOD知识），FA隐式通过增强ID知识来排斥OOD——后者在少样本场景更稳定
- **启发**：在开集/OOD问题中，**增强已知类别的表示质量**可能比**显式建模未知类别**更高效可靠——这与人类认知中"认识越深确定性越强"的直觉一致

## 评分
- 新颖性: ⭐⭐⭐⭐ 思路逆向而巧妙，强制提示设计简洁优雅
- 实验充分度: ⭐⭐⭐⭐ 多个OOD基准、多shot设置、挑战性数据集，但模型种类单一
- 写作质量: ⭐⭐⭐⭐⭐ 动机阐述极为清晰，方法推导自然流畅
- 价值: ⭐⭐⭐⭐ 为OOD检测提供了新范式，但需更多实验验证泛化性

<!-- RELATED:START -->

## 相关论文

- [Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-Distribution Detection](adaptive_prompt_learning_via_gaussian_outlier_synthesis_for_out-of-distribution_.md)
- [Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](../../NeurIPS2025/multimodal_vlm/revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)
- [PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation](pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)
- [FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)
- [Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)

<!-- RELATED:END -->
