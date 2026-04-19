---
title: >-
  [论文解读] Grounding and Enhancing Informativeness and Utility in Dataset Distillation
description: >-
  [ICLR 2026][模型压缩][Dataset Distillation] 提出InfoUtil框架，用博弈论Shapley Value最大化样本信息量（找到最重要的patch），用梯度范数最大化样本效用（选择对训练最有价值的样本），在ImageNet-1K上比前SOTA提升6.1%。
tags:
  - ICLR 2026
  - 模型压缩
  - Dataset Distillation
  - Shapley Value
  - 梯度范数
  - 信息量
  - 效用
---

# Grounding and Enhancing Informativeness and Utility in Dataset Distillation

**会议**: ICLR 2026  
**arXiv**: [2601.21296](https://arxiv.org/abs/2601.21296)  
**代码**: 无  
**领域**: 数据集蒸馏  
**关键词**: Dataset Distillation, Shapley Value, 梯度范数, 信息量, 效用

## 一句话总结
提出InfoUtil框架，用博弈论Shapley Value最大化样本信息量（找到最重要的patch），用梯度范数最大化样本效用（选择对训练最有价值的样本），在ImageNet-1K上比前SOTA提升6.1%。

## 研究背景与动机

**领域现状**：数据集蒸馏旨在从大数据集中合成小型数据集，使模型训练效果接近原始数据。主流方法分为匹配类（matching-based, 如梯度匹配/轨迹匹配）和知识蒸馏类。知识蒸馏类方法（如RDED）性能更好但缺乏理论解释。

**现有痛点**：匹配类方法效率与性能难以兼顾（如轨迹匹配需4×A100）；知识蒸馏类方法（如RDED）用随机裁剪+启发式评分选patch，缺乏principled的理论保证——随机选的patch经常错过语义关键区域。

**核心矛盾**：如何在理论可解释的框架下同时解决两个问题：(1) 每个样本中哪些区域最重要（Informativeness）？(2) 哪些样本对训练最有价值（Utility）？

**本文目标**：为数据集蒸馏提供理论基础，定义最优蒸馏，并据此设计算法。

**切入角度**：提出Informativeness（patch级别，衡量信息量）和Utility（样本级别，衡量训练价值）两个概念，数学化定义最优蒸馏，用Shapley Value做信息量归因，用梯度范数做效用评估。

**核心 idea**：Shapley Value选最重要的patch + 梯度范数选最有价值的样本 = 理论有基础的最优蒸馏。

## 方法详解

### 整体框架
两步流水线：Step 1用Shapley Value对每个样本找最informative的patch进行压缩→得到压缩数据集 $\mathcal{D}'$；Step 2用梯度范数评估每个压缩样本的训练价值→选top样本得到蒸馏数据集 $\tilde{\mathcal{D}}$。最后重建为正常大小图像并生成soft label。

### 关键设计

1. **博弈论信息量最大化 (Game-theoretic Informativeness Maximization)**:

    - 功能：用Shapley Value归因找到每个图像中最重要的patch
    - 核心思路：将图像视为博弈游戏，每个patch是一个玩家，$\phi_f(x^{(i)}) = \frac{1}{d}\sum_{s:s_i=0}\binom{d-1}{\mathbf{1}^\top s}(f(x\circ(s+e_i)) - f(x\circ s))$。用KernelShap快速估计。选Shapley值最高的patch保留。
    - 设计动机：Shapley Value是唯一同时满足线性、虚拟、对称、效率四条公理的归因方法，理论基础最扎实。

2. **有原则的效用最大化 (Principled Utility Maximization)**:

    - 功能：用梯度范数评估每个样本对训练的重要性并选top-m
    - 核心思路：Theorem 1证明效用函数 $\mathcal{U}$ 的上界是梯度范数：$\mathcal{U}(x_i, y_i; f_{\theta^{(t)}}) \leq c\|\nabla_{\theta^{(t)}}\ell_t(f_{\theta^{(t)}}(x_i), y_i)\|$
    - 设计动机：直接计算Utility需要对每个样本做"有/无"实验，计算代价太高。梯度范数是可计算的上界，越大说明该样本对训练影响越大。

3. **多样性控制**:

    - 功能：在Shapley归因热力图上加随机噪声引入patch选择的多样性
    - 核心思路：$\phi + \varepsilon$，$\varepsilon \sim \mathcal{N}(0, \sigma^2)$
    - 设计动机：纯Shapley可能总是选同一区域，噪声让不同样本选择不同的informative区域

### 损失函数 / 训练策略
- Shapley Value用KernelShap快速估计（避免 $2^{16}$ 次推理）
- 梯度范数用教师模型的中间检查点计算
- 压缩为1/4分辨率，4张压缩图拼成1张全尺寸图
- Soft label从教师模型中间检查点获取

## 实验关键数据

### 主实验
ResNet-18, IPC=50 (每类50张)：

| 数据集 | 方法 | Top-1 Acc | 提升 |
|--------|------|-----------|------|
| ImageNet-1K | RDED (前SOTA) | 基线 | — |
| ImageNet-1K | **InfoUtil** | 基线+6.1% | +6.1% |
| ImageNet-100 | RDED | 基线 | — |
| ImageNet-100 | **InfoUtil** | 基线+16% | +16% |
| CIFAR-10 IPC50 | RDED | 62.1 | — |
| CIFAR-10 IPC50 | **InfoUtil** | **71.0** | +8.9% |

### 消融实验
| 配置 | ImageNet-1K Acc | 说明 |
|------|----------------|------|
| Full InfoUtil | 最优 | Shapley + 梯度范数 |
| 随机patch (无Shapley) | 显著下降 | 信息量选择很关键 |
| 随机样本选择 (无梯度范数) | 下降 | 效用排序有价值 |
| 无多样性噪声 | 略降 | 多样性有帮助 |

### 关键发现
- Shapley Value选的patch对准了语义关键区域（如动物的头部而非背景），RDED的随机裁剪经常选到无关背景
- 梯度范数作为效用评估指标简单有效——高梯度范数样本确实对训练更重要
- 在ImageNet-1K这样的大规模数据集上提升仍然显著(6.1%)，说明方法可扩展
- 跨架构泛化：用ResNet-18蒸馏的数据在ResNet-101上评估仍有显著提升

## 亮点与洞察
- **理论基础扎实**：从Informativeness和Utility两个概念出发定义最优蒸馏(Definition 4)，再用Shapley和梯度范数分别近似——整个流程有理论支撑而非启发式。
- **Shapley Value做图像归因**用于数据集蒸馏是首次，且效果远超随机裁剪。这个思路可以推广到其他需要"选最重要区域"的场景。
- **效用=梯度范数上界**的定理(Theorem 1)给出直觉清晰的计算替代——梯度大的样本对训练"动量"影响大，应优先保留。

## 局限与展望
- Shapley Value计算即使用KernelShap仍有一定开销，在超大规模(>1M图像)上效率待验证
- 压缩为1/4分辨率是固定设置，自适应压缩率可能更优
- 梯度范数只用了一个检查点，多检查点的集成评估可能更鲁棒
- 仅在分类任务上验证，检测/分割等密集预测任务未探索

## 相关工作与启发
- **vs RDED**: RDED随机裁剪+启发式评分，InfoUtil用Shapley+梯度范数，理论性更强，性能好6.1%-16%
- **vs SRe2L**: SRe2L是另一种知识蒸馏方法，InfoUtil在所有IPC设置上都大幅超越
- **vs 匹配类方法(MTT/DATM)**: 匹配类在小数据集上有竞争力但对大数据集不可扩展，InfoUtil兼顾大小数据集

## 评分
- 新颖性: ⭐⭐⭐⭐ Shapley Value用于蒸馏patch选择是首次，理论框架完整
- 实验充分度: ⭐⭐⭐⭐⭐ 7个数据集、3种架构、多IPC设置、跨架构泛化
- 写作质量: ⭐⭐⭐⭐ 理论定义清晰，定理证明完整
- 价值: ⭐⭐⭐⭐ 为数据集蒸馏提供了有理论基础的新范式

<!-- RELATED:START -->

## 相关论文

- [Understanding Dataset Distillation via Spectral Filtering](understanding_dataset_distillation_via_spectral_filtering.md)
- [Enhancing Dataset Distillation via Non-Critical Region Refinement](../../CVPR2025/model_compression/enhancing_dataset_distillation_via_non-critical_region_refinement.md)
- [Rectified Decoupled Dataset Distillation: A Closer Look for Fair and Comprehensive Evaluation](rectified_decoupled_dataset_distillation_a_closer_look_for_fair_and_comprehensiv.md)
- [Dataset Distillation as Pushforward Optimal Quantization](dataset_distillation_as_pushforward_optimal_quantization.md)
- [Hyperbolic Dataset Distillation](../../NeurIPS2025/model_compression/hyperbolic_dataset_distillation.md)

<!-- RELATED:END -->
