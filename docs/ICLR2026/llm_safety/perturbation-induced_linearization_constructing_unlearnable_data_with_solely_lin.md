---
title: >-
  [论文解读] Perturbation-Induced Linearization: Constructing Unlearnable Data with Solely Linear Classifiers
description: >-
  [ICLR 2026][不可学习样本] 提出PIL方法，仅使用无偏置线性分类器作为代理模型生成不可学习扰动，通过诱导深度模型线性化来阻止其学习语义特征，比现有方法快100倍以上（CIFAR-10上不到1分钟GPU时间）。
tags:
  - ICLR 2026
  - 不可学习样本
  - 数据保护
  - 线性化
  - 快捷学习
  - 对抗扰动
---

# Perturbation-Induced Linearization: Constructing Unlearnable Data with Solely Linear Classifiers

**会议**: ICLR 2026  
**arXiv**: [2601.19967](https://arxiv.org/abs/2601.19967)  
**代码**: [GitHub](https://github.com/jinlinll/pil)  
**领域**: 其他  
**关键词**: 不可学习样本, 数据保护, 线性化, 快捷学习, 对抗扰动

## 一句话总结

提出PIL方法，仅使用无偏置线性分类器作为代理模型生成不可学习扰动，通过诱导深度模型线性化来阻止其学习语义特征，比现有方法快100倍以上（CIFAR-10上不到1分钟GPU时间）。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：将网络数据用于训练深度学习模型的行为越来越普遍，但许多数据是在未经创作者同意的情况下被爬取的。**不可学习样本（Unlearnable Examples）**通过向数据添加不可察觉的扰动，使在扰动数据上训练的模型无法泛化到干净测试数据，从而保护数据不被未授权使用。

现有方法（如EM、REM）通常用深度网络作为代理模型生成扰动，计算代价极高（REM在CIFAR-10上需15+小时GPU时间）。一个自然的问题是：**是否可以用更简单的模型生成同样有效的扰动？**

更深层的问题是：**不可学习样本的有效性机制是什么？** 本文发现答案是**线性化诱导**——扰动迫使深度模型表现得像线性模型，从而丧失学习复杂语义特征的能力。

## 方法详解

### 整体框架

PIL使用两步流程：(1) 在干净数据上训练一个无偏置线性分类器；(2) 用该线性分类器优化扰动，使扰动同时满足两个目标——语义混淆和快捷学习。最终将扰动减去原始图像作为不可学习数据集。

### 关键设计

1. **语义混淆（Semantic Obfuscation）**:

    - 功能：使原始图像的语义信息在扰动后变得无用
    - 核心思路：优化 $\delta_1$ 使 $f_{lin}(x - \delta_1)$ 输出接近均匀分布（最小化KL散度）
    - 设计动机：当深度模型线性化后，x-δ部分不再携带有用的分类信息

2. **快捷学习（Shortcut Learning）**:

    - 功能：使扰动本身成为强类别信号
    - 核心思路：优化 $\delta_2$ 使线性模型能直接从 $\delta_2$ 准确预测标签（最小化交叉熵）
    - 设计动机：深度模型会"偷懒"学习扰动中的快捷方式而非图像的语义特征

3. **联合优化**:

    - 功能：将两个目标合并为单一优化
    - 核心思路：$L_{total} = \lambda L_{CE}(f_{lin}(\delta), y) + (1-\lambda) L_{KL}(f_{lin}(x-\delta), \text{uniform})$，$\lambda=0.9$偏重快捷学习
    - 设计动机：实际中优化一个扰动δ而非分别优化δ₁和δ₂
    - 关键细节：使用PGD式更新，步长α=8/2550，预训练线性模型后再优化扰动

### 损失函数 / 训练策略

- 先在干净数据上用SGD训练M轮线性模型（捕获数据语义结构）
- 然后N轮PGD更新每个样本的扰动，限制L∞≤8/255
- 从均匀分布[-ε,ε]初始化扰动

## 实验关键数据

### 主实验：不同数据集和模型上的测试精度（越低越好）

| 模型 | SVHN-干净 | SVHN-PIL | CIFAR10-干净 | CIFAR10-PIL | ImageNet100-干净 | ImageNet100-PIL |
|------|-----------|----------|-------------|-------------|-----------------|----------------|
| ResNet-18 | 95.64 | **15.94** | 92.11 | **12.77** | 66.00 | **2.26** |
| VGG-19 | 95.22 | **9.12** | 90.61 | **15.22** | 36.04 | **1.36** |
| MobileNet-V2 | 95.95 | **28.48** | 91.94 | **14.05** | 71.26 | **2.20** |

### 消融实验：数据增强下的鲁棒性（CIFAR-10测试精度↓）

| 方法 | 无增强 | Basic | Rotation | Cutout | CutMix |
|------|--------|-------|----------|--------|--------|
| PIL | **14.70** | **12.87** | **18.15** | 14.62 | **11.05** |
| SEP | 28.43 | 8.94 | 19.68 | 9.74 | 10.48 |
| TAP | 35.90 | 19.11 | 21.18 | 15.09 | 20.30 |

### 关键发现

- PIL在CIFAR-10上仅需不到1分钟GPU时间，而REM需要15+小时，加速超过100倍
- 线性模型生成的扰动能有效降低多种深度架构的泛化能力，证明了架构无关性
- 所有不可学习方法（包括非线性代理的EM、REM等）都会导致训练模型线性度增加，PIL只是把这个机制推到了极致
- 在高分辨率ImageNet-100上测试精度降至1-3%，效果甚至更好
- PIL在JPEG压缩防御下仍保持较强鲁棒性

## 亮点与洞察

- **核心洞察极其优美**：不可学习样本的本质机制是诱导线性化——既然如此，直接用线性模型做代理就够了
- 将复杂的不可学习样本问题简化为线性模型+PGD优化，大幅降低了实现和计算门槛
- 语义混淆+快捷学习的双目标分解直观且有效
- 还揭示了一个部分扰动的基本限制：不可学习样本在仅部分数据被扰动时无法显著降低测试精度

## 局限与展望

- 对抗性训练（adversarial training）作为防御仍可能削弱PIL的效果
- 部分扰动场景下（只有一部分数据被保护），保护效果急剧下降
- 未测试文本/音频等非图像模态
- 线性化机制的理论解释仍是经验性的

## 相关工作与启发

与EM、REM、TAP、NTGA等不可学习样本方法直接对比。与shortcut learning文献紧密关联——说明深度模型容易被简单特征误导。启发：有时候最简单的代理模型反而最有效。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "用线性模型就够了"的发现出人意料且优美
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多架构、多防御手段全面对比
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法简洁
- 价值: ⭐⭐⭐⭐⭐ 既有实用价值（100x加速），也有理论洞察（线性化机制）

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Temporal Unlearnable Examples: Preventing Personal Video Data from Unauthorized Exploitation](../../ICCV2025/llm_safety/temporal_unlearnable_examples_preventing_personal_video_data_from_unauthorized_e.md)
- [\[ICLR 2026\] Redirection for Erasing Memory (REM): Towards a Universal Unlearning Method for Corrupted Data](redirection_for_erasing_memory_rem_towards_a_universal_unlearning_method_for_cor.md)
- [\[ICML 2025\] Improving Continual Learning Performance and Efficiency with Auxiliary Classifiers](../../ICML2025/llm_safety/improving_continual_learning_performance_and_efficiency_with_auxiliary_classifie.md)
- [\[NeurIPS 2025\] A Cramér–von Mises Approach to Incentivizing Truthful Data Sharing](../../NeurIPS2025/llm_safety/a_cramrvon_mises_approach_to_incentivizing_truthful_data_sha.md)
- [\[ICLR 2026\] Understanding Sensitivity of Differential Attention through the Lens of Adversarial Robustness](understanding_sensitivity_of_differential_attention_through_the_lens_of_adversar.md)

<!-- RELATED:END -->
