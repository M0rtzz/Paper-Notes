---
title: >-
  [论文解读] What Makes a Good Dataset for Knowledge Distillation?
description: >-
  [CVPR 2025][模型压缩][知识蒸馏] 本文系统探究了知识蒸馏中"什么数据好用"这个基本问题，发现甚至非自然的 OpenGL shader 合成图像也能有效蒸馏，并总结出好的蒸馏数据集需满足：教师预测类别分布均匀、覆盖足够决策空间、数据多样性高、并包含决策边界信息。 领域现状：知识蒸馏（KD）是模型压缩的主流方法…
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "知识蒸馏"
  - "代理数据集"
  - "合成数据"
  - "决策边界"
  - "数据特性分析"
---

# What Makes a Good Dataset for Knowledge Distillation?

**会议**: CVPR 2025  
**arXiv**: [2411.12817](https://arxiv.org/abs/2411.12817)  
**代码**: [https://github.com/osu-cvl/good-kd-dataset](https://github.com/osu-cvl/good-kd-dataset)  
**领域**: 模型压缩 / 知识蒸馏  
**关键词**: 知识蒸馏, 代理数据集, 合成数据, 决策边界, 数据特性分析

## 一句话总结
本文系统探究了知识蒸馏中"什么数据好用"这个基本问题，发现甚至非自然的 OpenGL shader 合成图像也能有效蒸馏，并总结出好的蒸馏数据集需满足：教师预测类别分布均匀、覆盖足够决策空间、数据多样性高、并包含决策边界信息。

## 研究背景与动机

**领域现状**：知识蒸馏（KD）是模型压缩的主流方法，教师网络的软标签输出为学生网络提供比硬标签更丰富的学习信号。标准 KD 假设教师的原始训练数据可用。

**现有痛点**：在很多实际场景中，教师的原始数据不可用：持续学习中数据流式到达不可回溯；大公司释放模型权重但不公开训练数据（如 CLIP、DINOv2、GPT）。实践者不得不使用替代数据，但什么样的替代数据有效，学界缺乏系统研究。先前工作认为 OOD（域外）数据不可用于 KD，但这是否真的如此？

**核心矛盾**：直觉上认为只有域内的真实数据才适合做蒸馏，但这个假设阻碍了在数据不可用场景下的 KD 应用。问题的本质是——蒸馏到底在"传递"什么？如果理解了传递的机制，就可能找到更灵活的数据选择策略。

**本文目标**：回答"什么特性使得数据集适合知识蒸馏？"这个核心问题，并证明即使是最非常规的合成图像也可以实现有效蒸馏。

**切入角度**：将 KD 视为"函数匹配"（function matching），数据集的作用就是对教师决策函数的充分采样。从这个视角出发，是否域内、是否真实都不是本质，关键是采样是否充分。

**核心 idea**：KD 是一个充分采样问题，好的蒸馏数据集需要均匀覆盖教师的所有类别决策区域，而非必须与原始数据同分布。

## 方法详解

### 整体框架
研究采用标准 KD 框架（Hinton 2015），系统性评估多种替代数据集在蒸馏中的表现。实验在 6 个教师数据集（CIFAR10/100, Tiny ImageNet, FGVC-Aircraft, Pets, EuroSAT）上训练 ResNet50 教师，然后用 12 种不同数据集进行蒸馏到学生网络，包括真实域内/域外数据集和三种合成数据（OpenGL shaders, Leaves, 噪声图像）。

### 关键设计

1. **多源数据系统评测**:

    - 功能：全面评估各类替代数据集在不同教师/学生组合下的蒸馏效果
    - 核心思路：将数据分为三大类——真实域内（ID）、真实域外（OOD）、非自然合成。对 ImageNet 按与各教师数据集的类别重叠拆分为 ID/OOD 子集。合成数据包括：OpenGL shader 图像（由 1089 个 TwiGL 着色器程序渲染，含丰富纹理和结构）、Leaves 图像（简单随机形状组合）、以及按原数据集均值/方差采样的高斯噪声。所有合成数据通过教师网络筛选，保留各类预测均匀的 50K 样本
    - 设计动机：需要从最极端的案例（完全非自然的合成图像）开始，才能真正理解数据特性的底线要求

2. **蒸馏成功因素分析**:

    - 功能：提炼好数据集的关键特性
    - 核心思路：（1）**类别预测直方图的相对熵**：计算教师对数据集的 argmax 预测分布相对于均匀分布的熵比。发现好的蒸馏数据集相对熵接近 1.0（均匀），差的接近 0。如 OpenGL 在 CIFAR10 教师上达到 0.939，而 FGVCA（最差真实数据集）仅 0.116。（2）**数据多样性**：OpenGL > Leaves > 噪声，丰富的纹理和结构导致更好的信息传递。（3）**决策边界信息**：温控软标签（$\tau$ >1）比硬标签好，因为软标签携带类间关系信息。对 OOD 数据尤其重要
    - 设计动机：不仅要证明"能用"，还要理解"为什么能用"，提出可操作的指导原则

3. **决策边界对抗攻击策略**:

    - 功能：强制将数据样本推向教师的决策边界附近，增强蒸馏效果
    - 核心思路：对数据样本 $x_j$ 施加对抗扰动使其越过决策边界到目标类 $t$（$\mathcal{F}_T(x_j) \neq t$），生成"成功前"和"成功后"两个对抗样本对，分别位于决策边界两侧。再添加一步将样本对推入各自类别更深处以增加采样覆盖。使用 Bold Driver 自适应步长提高攻击效率。最终，每个原始样本可生成 4 个对抗样本（边界前后各一对 + 深入各一对）
    - 设计动机：分析发现决策边界附近的信息对 KD 最关键（类似于 SVM 的支持向量）。对于本身不好的数据集（如 FGVCA 用于蒸馏 CIFAR10 教师），对抗攻击可以将其 11.39% 的精度提升至 88.14%

### 损失函数 / 训练策略
标准 KD 损失为 KL 散度 $\mathcal{L}(p_T||p_S) = \sum_{i \in \mathcal{C}} [p_T(i)\log p_T(i) - p_T(i)\log p_S(i)]$，配合温度参数 $\tau$。对通用数据集用 $\tau=2$，细粒度和合成数据用 $\tau=20$。所有实验均使用 mixup 增强。

## 实验关键数据

### 主实验

| 教师/数据集 | 原始数据 | ImageNet-OOD | OpenGL | Leaves | 噪声 |
|------------|---------|-------------|--------|--------|------|
| CIFAR10→RN18 | 95.98 | 94.69 | **94.02** | 92.08 | 69.24 |
| CIFAR100→RN18 | 78.35 | 76.66 | **73.27** | 66.02 | 22.09 |
| Tiny-IN→RN18 | 67.14 | 59.44 | **56.89** | 28.03 | 5.37 |
| EuroSAT→RN18 | 98.60 | 98.55 | **98.45** | 98.05 | 54.55 |
| Pets→RN18 | 86.80 | 85.01 | **72.59** | 42.19 | 3.43 |

### 消融实验

| 实验设置 | CIFAR10 | OpenGL | 说明 |
|---------|---------|--------|------|
| 标准 KD | 95.98 | 94.02 | 基线 |
| One-Hot 标签 | 96.09 | 91.68 | 软标签对 OOD 数据更重要 |
| 20K 长尾采样 | 91.85 | 88.93 | 不均匀采样严重损害效果 |
| 20K 均匀采样+mixup | 95.07 | 92.32 | 均匀+mixup部分弥补 |
| 极端增强+mixup | 94.04 | 93.95 | 强增强使真实数据变OOD |
| 无增强无mixup | 87.33 | 31.84 | 增强对合成数据至关重要 |

### 关键发现
- **数据不需要真实或域内**：OpenGL shader图像在CIFAR10上仅比原始数据低2%，在EuroSAT上几乎持平
- **决策空间均匀采样是核心**：教师预测直方图的熵与蒸馏成功率高度正相关。长尾采样比均匀采样差约 2%
- 数据增强对合成数据至关重要——无增强的OpenGL只有31.84%（几乎不工作），加强增强后达93.95%
- **对抗攻击的戏剧性改善**：对最差数据集（FGVCA蒸馏C10）施加对抗攻击从11.39%提升到88.14%，提升76.8个百分点
- 教师架构越复杂（ViT-S > ConvNeXt-T > RN50），需要更多"耐心"（更长训练）来蒸馏合成数据

## 亮点与洞察
- **信号采样理论类比**：将 KD 与 Nyquist 采样定理类比——数据集的作用就是对教师决策函数进行采样，频率不够（某些类被采样不足）就会"失真"（蒸馏失败）。这个直觉非常精辟且易于理解
- **2D GAP 特征可视化**：在 MNIST 教师上的可视化直接展示了为什么 OpenGL 比 CIFAR10 更适合蒸馏——OpenGL 图像落在教师所有类别的决策区域内，而 CIFAR10 只覆盖少数类
- **对抗攻击增强蒸馏数据**：用对抗样本作为决策边界的"探针"来改善 KD，比 DFKD 方法简单得多（不需要生成器网络），且 48小时 vs 2小时 的计算效率优势明显

## 局限与展望
- 实验规模较小（最大到 Tiny ImageNet 200类），对于 ImageNet-1K 甚至更大规模的教师蒸馏效果未验证
- 当类别数增多或细粒度增强时，合成数据的表现会下降（Pets: 72.59 vs 86.80），因为合成图像很难覆盖细粒度的决策空间
- 对抗攻击策略的计算开销随类别数线性增长，对大类别数场景可能不经济
- 未来可探索用生成模型（如 Stable Diffusion）生成更有针对性的合成蒸馏数据

## 相关工作与启发
- **vs DFKD 方法（CMI, Spaceship）**：DFKD 用生成器网络合成数据来蒸馏，但存在模式坍缩和计算昂贵等问题。本文表明简单的非优化合成图像可以达到相似效果，且只需 1/24 的时间
- **vs Beyer et al.（函数匹配）**：他们提出 KD 是函数匹配并发现 OOD 数据效果差。本文进一步分析发现 OOD 数据效果差是因为采样不均匀而非本质不可用
- **vs 单图蒸馏（Asano & Saeed）**：从单张大图的随机裁剪中蒸馏，思路新颖但在CIFAR100上只有69.34%，不如OpenGL的73.86%

## 评分
- 新颖性: ⭐⭐⭐⭐ 系统性回答了一个重要的开放问题，合成数据蒸馏的发现很有启发
- 实验充分度: ⭐⭐⭐⭐⭐ 大量消融和分析实验，控制变量非常仔细
- 写作质量: ⭐⭐⭐⭐ 问题驱动的叙事方式引人入胜，分析层层深入
- 价值: ⭐⭐⭐⭐ 提出的蒸馏数据集选择准则对实践有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] A Good Teacher Adapts Their Knowledge for Distillation](../../ICCV2025/model_compression/a_good_teacher_adapts_their_knowledge_for_distillation.md)
- [\[CVPR 2025\] Dataset Distillation with Neural Characteristic Function: A Minmax Perspective](dataset_distillation_with_neural_characteristic_function_a_minmax_perspective.md)
- [\[CVPR 2025\] Emphasizing Discriminative Features for Dataset Distillation in Complex Scenarios](emphasizing_discriminative_features_for_dataset_distillation_in_complex_scenario.md)
- [\[CVPR 2025\] Enhancing Dataset Distillation via Non-Critical Region Refinement](enhancing_dataset_distillation_via_non-critical_region_refinement.md)
- [\[CVPR 2025\] Good, Cheap, and Fast: Overfitted Image Compression with Wasserstein Distortion](good_cheap_and_fast_overfitted_image_compression_with_wasserstein_distortion.md)

</div>

<!-- RELATED:END -->
