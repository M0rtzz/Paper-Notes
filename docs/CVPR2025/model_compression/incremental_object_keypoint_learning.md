---
title: >-
  [论文解读] Incremental Object Keypoint Learning (KAMP)
description: >-
  [CVPR 2025][模型压缩][增量关键点] 首次定义增量关键点学习（IKL）范式——新任务只标注新关键点、不保留旧数据的增量训练，提出 KAMP 框架通过知识关联网络（KA-Net）建模新旧关键点间的解剖学空间关系，配合关键点导向的空间蒸馏损失，在 4 个数据集上不仅有效防遗忘，甚至实现了对旧关键点的正向迁移提升（MPII AAA 79.93% vs LWF 75.75%）。
tags:
  - CVPR 2025
  - 模型压缩
  - 增量关键点
  - 持续学习
  - 空间蒸馏
  - 知识关联
  - 三角化约束
---

# Incremental Object Keypoint Learning (KAMP)

**会议**: CVPR 2025  
**arXiv**: [2503.20248](https://arxiv.org/abs/2503.20248)  
**代码**: 无  
**领域**: 模型压缩 / 增量学习  
**关键词**: 增量关键点, 持续学习, 空间蒸馏, 知识关联, 三角化约束

## 一句话总结

首次定义增量关键点学习（IKL）范式——新任务只标注新关键点、不保留旧数据的增量训练，提出 KAMP 框架通过知识关联网络（KA-Net）建模新旧关键点间的解剖学空间关系，配合关键点导向的空间蒸馏损失，在 4 个数据集上不仅有效防遗忘，甚至实现了对旧关键点的正向迁移提升（MPII AAA 79.93% vs LWF 75.75%）。

## 研究背景与动机

**领域现状**：关键点检测模型基于固定的预定义关键点集合训练。当下游任务需要新关键点时（如医学影像中从 Downs 分析扩展到 Steiner 分析），只能重新标注所有数据或训练独立模型。

**现有痛点**：（1）重新标注旧关键点不现实——标注昂贵且数据可能不可获取（隐私）；（2）独立训练多个模型——模型数线性增长，且无法捕捉新旧关键点间的结构关系，性能次优；（3）无监督关键点学习（UKL）和类无关姿态估计（CAPE）依赖预训练模型的泛化，迁移性受限。

**核心矛盾**：IKL 的关键挑战是"标签不共现"（Label Non-Co-Occurrence, LNCO）——新数据只有新关键点标注，旧关键点没有标注，模型无法在标签空间中显式学习新旧关键点之间的运动学/解剖学约束。

**切入角度**：关键点之间存在内在的空间关联（如肘关节和腕关节的三角化约束），可以训练辅助网络从新关键点推断旧关键点位置，将这种关联知识注入增量学习过程。

**核心 idea**：KA-Net 利用新→旧关键点的空间三角化关系推断旧位置 + 空间 softmax 蒸馏 = 增量关键点防遗忘+正向迁移。

## 方法详解

### 整体框架

两阶段学习方案 KAMP：Stage-I（知识关联）训练辅助 KA-Net 从新关键点 GT 热图推断相关旧关键点位置；Stage-II（互促学习）冻结 KA-Net 和旧模型作为教师，联合训练新模型学习所有新旧关键点。

### 关键设计

1. **KA-Net（知识关联网络）**：输入两个新关键点的 GT 热图与冻结旧模型提取的视觉特征（元素乘积→拼接→3 层 CNN），输出旧关键点的预测热图。利用解剖学上空间相邻的新旧关键点三元组（如"左膝→左髋"），用旧模型的伪标签监督训练

2. **关键点导向的空间蒸馏（KSD）**：不同于标准 LWF 的跨通道 softmax 蒸馏，对每个关键点的热图在 H 和 W 维度分别做 spatial softmax 后计算 KL 散度——$\ell_{KSD} = \sum_j \sum_d -s^d_{sp}(\hat{y}^{t-1}_{i,j}) \cdot \log s^d_{sp}(\hat{y}^t_{i,j})$。这保留了每个关键点的空间位置信息，比通道蒸馏更适合回归任务

3. **辅助任务创建**：基于标准解剖图定位新旧关键点的空间邻近关系，选择三元组（2 新 + 1 旧）。每步只需创建一个辅助任务，可用 GPT-4o 自动完成。KA-Net 仅用于训练时蒸馏，推理时不使用

### 损失函数 / 训练策略

$$l_{MP} = \ell_{GT} + \alpha(\ell_{KSD} + \ell_{KA})$$

- $\ell_{GT}$：新关键点的 L2 回归损失
- $\ell_{KSD}$：空间蒸馏损失（旧模型→新模型，所有旧关键点）
- $\ell_{KA}$：KA-Net 产出的辅助监督（选定的旧关键点）
- $\alpha$：$10^2$（MPII/Head-2023）或 $10^4$（Chest/ATRW）

骨干网络 HRNet-W32，训练 100 epoch，学习率 2e-3 或 1e-2。

## 实验关键数据

| 数据集 | 指标 | KAMP | LWF | CPR | Finetune | Joint Training |
|--------|------|------|-----|-----|----------|---------------|
| MPII 5-step | AAA₄↑ | **79.93** | 75.75 | 75.52 | 37.41 | 88.50 |
| MPII 5-step | AT₄↑ | **+1.80** | -3.86 | -3.24 | — | — |
| MPII 5-step | MT₄↑ | **+4.23** | +0.41 | +0.75 | — | — |
| Head-2023 5-step | A-MRE₄↓ | **2.32** | 4.31 | 3.71 | 51.3 | 2.12 |
| Chest 2-step | A-MRE₁↓ | **5.67** | 6.35 | 6.17 | 43.1 | 5.43 |
| ATRW 4-step | AAA₃↑ | **93.16** | 87.31 | 89.34 | 13.24 | 94.69 |

### 消融实验

| 方法 | AAA₄↑ | AT₄↑ | MT₄↑ |
|------|-------|------|------|
| LWF | 75.75 | -3.86 | +0.41 |
| KAMP (仅 $\ell_{KSD}$) | 76.93 | -2.24 | +0.65 |
| KAMP (随机 KA-Net) | 77.13 | -0.48 | +1.24 |
| **KAMP (完整)** | **79.93** | **+1.80** | **+4.23** |

### 与 low-shot 方法对比

| 方法 | 1-shot | 5-shot | 10-shot | 50-shot |
|------|--------|--------|---------|---------|
| CC2D | 5.14 | 4.83 | 4.08 | 3.47 |
| EGT | 5.01 | 4.58 | 3.87 | 3.21 |
| **KAMP** | **4.35** | **3.70** | **3.03** | **2.32** |

### 关键发现

- **正向迁移存在**：AT₄=+1.80 表明学习新关键点平均**提升**了旧关键点性能
- **空间蒸馏 >> 通道蒸馏**：仅替换 softmax 维度即提升 1.18% AAA
- **解剖学先验关键**：基于解剖关系的 KA-Net 比随机关联提升 2.80% AAA
- **IKL 与 CAPE 互补**：KAMP + MetaPoint+ = 79.18% PCK（远超各自单独使用）
- **极低标注量即有效**：1-shot 场景下仍优于专用少样本方法

## 亮点与洞察

- **新范式定义**——增量关键点学习（IKL）是一个自然且实用的新问题，首次被正式提出
- **正向迁移超越防遗忘**——不仅不忘旧知识，还能利用新关键点改进旧关键点检测
- **空间 softmax 蒸馏**——简单但关键的改进，将分类任务的蒸馏方法适配到空间回归
- **标注高效**——对比 CAPE 和 UKL，IKL 使用更少的标注但扩展性更好

## 局限与展望

- 需要手工/LLM 定义关键点间的解剖学关联（三元组选择）
- 伪标签累积误差在长序列增量中可能恶化
- 假设新旧关键点有空间关联——完全无关的情况下 KA-Net 效果受限
- 仅探索了单对象类别内的增量，跨类别增量尚未验证
- KA-Net 设计较简单（3 层 CNN），更强的关联网络可能进一步提升

## 相关工作

- **关键点估计**：HRNet, SimpleBaseline, ViTPose——基于固定关键点集合的监督学习
- **增量学习**：LWF, EWC, MAS, CPR——侧重分类任务的持续学习
- **类无关姿态估计**：MetaPoint+, CAPE——依赖预训练模型泛化到新关键点
- **无监督关键点**：Autolink——受限于刚性运动和视频数据

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 新范式+KA-Net+空间蒸馏的有机结合
- 实验充分度: ⭐⭐⭐⭐⭐ 医学/人体/动物三种场景+low-shot+CAPE融合
- 写作质量: ⭐⭐⭐⭐ 清晰，问题阐述有深度
- 价值: ⭐⭐⭐⭐⭐ 为增量标注场景提供了实用方案，正向迁移发现有洞察力

<!-- RELATED:START -->

## 相关论文

- [Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning](tripartite_weight-space_ensemble_for_few-shot_class-incremental_learning.md)
- [Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning](adapter_merging_with_centroid_prototype_mapping_for_scalable_class-incremental_l.md)
- [CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)
- [Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](../../NeurIPS2025/model_compression/mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)
- [Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](../../ICCV2025/model_compression/integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)

<!-- RELATED:END -->
