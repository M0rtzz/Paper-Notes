---
title: >-
  [论文解读] Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting
description: >-
  [ICCV 2025][CLIP] 指出现有 CLIP 少样本分类基准因 CLIP 预训练时已见过测试数据集而实际是"部分转导设置"，提出基于 unlearning 的归纳基准评估方案，并设计了一种在新基准下稳定 SOTA 的少样本分类方法。
tags:
  - ICCV 2025
  - CLIP
  - 少样本学习
  - 归纳设置
  - 遗忘学习
  - 基准评估
---

# Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting

**会议**: ICCV 2025  
**arXiv**: [2507.20834](https://arxiv.org/abs/2507.20834)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: CLIP, 少样本学习, 归纳设置, 遗忘学习, 基准评估

## 一句话总结

指出现有 CLIP 少样本分类基准因 CLIP 预训练时已见过测试数据集而实际是"部分转导设置"，提出基于 unlearning 的归纳基准评估方案，并设计了一种在新基准下稳定 SOTA 的少样本分类方法。

## 研究背景与动机

**领域现状**：CLIP 凭借大规模图文预训练展现出强大的零样本和少样本迁移能力。近年来涌现了大量 CLIP 少样本分类方法——CoOp、Tip-Adapter、CLIP-Adapter、TaskRes 等——它们在 ImageNet、Flowers102、EuroSAT 等标准数据集上展示了显著的提升效果。

**现有痛点**：所有这些方法都使用相同的标准少样本数据集做评测，但一个被忽视的关键问题是——CLIP 在预训练时大概率已经"见过"这些数据集中的图像或其变体。这意味着 CLIP 对这些数据集的分类能力并非来自少样本样本的泛化，而是部分来自预训练记忆。这种评测实际上是"部分转导"（partially transductive）设置而非真正的"归纳"（inductive）设置。

**核心矛盾**：评测声称衡量的是"从少量样本归纳泛化的能力"，但基础模型的预训练记忆严重污染了评测结果，导致方法对比不公平，且无法反映方法在真正未见数据上的泛化能力。

**本文目标**：(1) 量化预训练记忆对少样本评测的影响；(2) 提供一种消除这种偏差的评测方案；(3) 在新的归纳设置下重新评估现有方法并提出更强的基线。

**切入角度**：作者引入"机器遗忘"（machine unlearning）技术——在不完全重训 CLIP 的情况下，选择性地让 CLIP "忘记"目标数据集中的知识，从而构建真正的归纳基线。

**核心 idea**：用 unlearning 技术构建"CLIP 从未见过目标数据集"的假设环境，在此环境下重新评估少样本方法，揭示真实泛化能力。

## 方法详解

### 整体框架

整体框架分为两部分：(1) 归纳基准构建——通过 unlearning 让 CLIP 遗忘目标数据集的知识，获得"未见过数据集"的 CLIP 模型；(2) 改进的少样本分类方法——在归纳设置下设计一种结合视觉和文本特征的分类策略。输入为 CLIP 模型 + 少量标注样本，输出为目标数据集的分类预测。

### 关键设计

1. **选择性遗忘机制（Selective Unlearning）**:

    - 功能：让 CLIP 模型选择性遗忘对特定数据集的知识，同时保留通用视觉-语言能力
    - 核心思路：采用梯度上升（gradient ascent）技术：对目标数据集的训练样本计算对比损失，然后反向更新模型参数使该损失增大（即让模型"忘记"如何正确匹配这些样本）。关键在于只对文本编码器（text encoder）执行 unlearning，保持视觉编码器不变——因为视觉特征的通用性更重要，而文本侧的类别名到特征的映射才是"记忆"的主要载体。控制 unlearning 的强度通过梯度上升步数和学习率来调节。
    - 设计动机：完全从头训练 CLIP 既不可行也不现实；选择性 unlearning 提供了一种高效的替代方案，能在保留通用能力的同时消除特定数据集的记忆偏差。

2. **Oracle 验证机制（Oracle Validation）**:

    - 功能：验证 unlearning 的有效性
    - 核心思路：设计两个 oracle 基线——(a) Random Init Oracle：随机初始化 CLIP 最后几层参数再做少样本适配，代表完全无记忆的下界；(b) Full Retrain Oracle：在排除目标数据集的数据上重训 CLIP，代表真正没见过目标数据的上界。通过检查 unlearning 后的模型性能是否落在这两个 oracle 之间来验证 unlearning 是否有效剔除了目标数据集的记忆。
    - 设计动机：没有 ground truth 来衡量"完美遗忘"的效果，因此需要通过 oracle 基线建立置信区间，以校准 unlearning 的程度既不过度也不不足。

3. **增强少样本分类器（Enhanced Few-Shot Classifier）**:

    - 功能：在归纳设置下实现稳健的少样本分类
    - 核心思路：提出一种结合原型网络思想和视觉-文本对齐的分类方法。具体做法：(a) 对每类的 K 个样本取视觉特征均值作为类原型；(b) 将类原型与 CLIP 文本特征通过可学习的权重融合；(c) 加入特征归一化和温度缩放。分类时计算测试样本与融合后类原型的余弦相似度。整个适配过程仅需优化融合权重和温度参数，参数量极小。
    - 设计动机：在归纳设置下，大部分复杂方法（如 prompt tuning）失效是因为它们依赖预训练记忆做初始化。简单而稳健的原型方法反而在真正的少样本场景下更可靠。

### 损失函数 / 训练策略

少样本适配阶段使用交叉熵损失，在 K-shot 支撑集上微调融合权重。Unlearning 阶段使用负对比损失（即标准 CLIP 对比损失取负进行梯度下降，等价于梯度上升）。整个过程不需要大规模计算资源。

## 实验关键数据

### 主实验

在 11 个数据集上对 13 种 CLIP 少样本方法进行标准设置 vs 归纳设置对比（16-shot）：

| 方法 | 标准设置准确率 | 归纳设置准确率 | 性能下降 |
|------|-------------|-------------|---------|
| Zero-shot CLIP | 63.7% | 28.6% | -55.1% |
| CoOp | 71.2% | 33.8% | -52.5% |
| Tip-Adapter-F | 73.1% | 35.2% | -51.8% |
| CLIP-Adapter | 71.8% | 34.1% | -52.5% |
| TaskRes | 73.5% | 36.4% | -50.5% |
| 本文方法 | 73.8% | 41.2% | -44.2% |

### 消融实验

| 配置 | 归纳设置准确率 | 说明 |
|------|-------------|------|
| 本文完整方法 | 41.2% | 融合原型 + 温度缩放 |
| w/o 文本特征融合 | 38.6% | 仅用纯视觉原型 |
| w/o 温度缩放 | 39.8% | 固定温度 $\tau=1$ |
| 用均匀融合替代可学习权重 | 39.3% | 简单平均不如自适应 |
| Unlearning 仅对视觉编码器 | 35.1% | 效果差，说明记忆主要在文本侧 |
| Unlearning 对双编码器 | 30.2% | 过度遗忘损害通用能力 |

### 关键发现

- **性能暴跌**：在归纳设置下，所有 13 种方法平均性能下降 55%，揭示当前基准严重高估了方法的泛化能力
- **排名变化**：某些在标准设置下领先的方法（如 prompt tuning 类）在归纳设置下排名大幅下滑，说明其"性能提升"更多来自利用预训练记忆
- **本文方法最稳健**：在归纳设置下性能下降最少（-44.2% vs 平均 -55%），且在所有数据集上一致 SOTA
- **文本编码器是记忆核心**：仅对文本编码器做 unlearning 就能有效消除数据集偏差，对视觉编码器做 unlearning 反而损害基础能力

## 亮点与洞察

- **揭示根本性评测缺陷**：指出整个 CLIP 少样本领域的评测方法论存在系统性偏差。这不是某种方法的问题，而是整个社区的共识性问题。这种"元研究"视角非常有价值。
- **Unlearning 作为评测工具**：将机器遗忘技术从隐私保护领域迁移到基准评测中，创造性地解决了"如何在不重训大模型的情况下模拟未见数据"这一难题。这个思路可以推广到任何基础模型的下游评测。
- **实验规模**：5880 个实验涵盖多种数据集、shot 数、seed、unlearning 设置，统计上非常扎实，难以被偶然性解释。

## 局限与展望

- Unlearning 是否能完美模拟"从未见过"的状态仍存疑，尤其对早期层编码的低级视觉知识可能无法完全消除
- 实验主要在 ViT-B/16 上进行，更大规模 CLIP（ViT-L、ViT-G）上的结论是否一致未验证
- 作者的改进方法虽然在归纳设置下最好，但绝对性能仍然有限（41.2%），说明真正的少样本学习仍是开放问题
- 可以探索更精细的 unlearning 策略——如层级选择性遗忘、基于影响函数的精确消除
- 建议后续工作建立标准化的归纳少样本基准，推动社区采用更公平的评测方式

## 相关工作与启发

- **vs CoOp/CoCoOp**: 这些 prompt tuning 方法在标准设置下性能强，但在归纳设置下暴跌最多，说明 prompt 优化很大程度上是在"回忆"预训练知识
- **vs Tip-Adapter**: 基于缓存的方法在归纳设置下相对稳健，因为它更直接地利用 few-shot 样本而非依赖预训练编码
- **vs FLYP/WiSE-FT**: 全模型微调类方法也受预训练偏差影响严重，在归纳设置下优势消失
- 这项工作提醒我们在评估任何基础模型的下游适配方法时，都需要考虑预训练数据集与评测数据集的重叠问题

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 揭示了整个领域被忽视的系统性评测问题，贡献独特且影响深远
- 实验充分度: ⭐⭐⭐⭐⭐ 5880个实验、13种基线、11个数据集，极其全面
- 写作质量: ⭐⭐⭐⭐ 论述清晰、逻辑严密
- 价值: ⭐⭐⭐⭐⭐ 可能重塑 CLIP 少样本分类的评测标准

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Feedforward Few-shot Species Range Estimation](../../ICML2025/llm_evaluation/feedforward_few-shot_species_range_estimation.md)
- [\[NeurIPS 2025\] Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](../../NeurIPS2025/llm_evaluation/benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)
- [\[ICCV 2025\] BATCLIP: Bimodal Online Test-Time Adaptation for CLIP](batclip_bimodal_online_test-time_adaptation_for_clip.md)
- [\[ICML 2025\] Random Registers for Cross-Domain Few-Shot Learning](../../ICML2025/llm_evaluation/random_registers_for_cross-domain_few-shot_learning.md)
- [\[NeurIPS 2025\] Unlocking Transfer Learning for Open-World Few-Shot Recognition](../../NeurIPS2025/llm_evaluation/unlocking_transfer_learning_for_open-world_few-shot_recognition.md)

<!-- RELATED:END -->
