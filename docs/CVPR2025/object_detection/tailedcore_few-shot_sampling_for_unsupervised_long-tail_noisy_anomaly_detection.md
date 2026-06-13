---
title: >-
  [论文解读] TailedCore: Few-Shot Sampling for Unsupervised Long-Tail Noisy Anomaly Detection
description: >-
  [CVPR 2025][目标检测][无监督异常检测] TailedCore 解决了无监督异常检测中"正常样本既包含噪声缺陷又服从未知长尾类别分布"的实际场景，提出 TailSampler 通过嵌入相似度的对称性假设预测类别基数来独立采样尾部类样本，构建了既能捕捉尾部类信息又对噪声鲁棒的内存库模型，在多种设置下超过 SOTA。
tags:
  - "CVPR 2025"
  - "目标检测"
  - "无监督异常检测"
  - "长尾分布"
  - "噪声鲁棒"
  - "内存库"
  - "少样本采样"
---

# TailedCore: Few-Shot Sampling for Unsupervised Long-Tail Noisy Anomaly Detection

**会议**: CVPR 2025  
**arXiv**: [2504.02775](https://arxiv.org/abs/2504.02775)  
**代码**: [https://github.com/YoonGyoJung/TailedCore](https://github.com/YoonGyoJung/TailedCore)  
**领域**: 其他  
**关键词**: 无监督异常检测, 长尾分布, 噪声鲁棒, 内存库, 少样本采样

## 一句话总结
TailedCore 解决了无监督异常检测中"正常样本既包含噪声缺陷又服从未知长尾类别分布"的实际场景，提出 TailSampler 通过嵌入相似度的对称性假设预测类别基数来独立采样尾部类样本，构建了既能捕捉尾部类信息又对噪声鲁棒的内存库模型，在多种设置下超过 SOTA。

## 研究背景与动机

**领域现状**：工业异常检测通常假设训练集只包含"正常"样本，基于内存库（memory bank）的方法（如 PatchCore）通过存储正常特征的代表子集来检测测试时的异常。这类方法在理想情况下表现优秀。

**现有痛点**：在真实工业场景中，正常训练数据面临两个同时存在的问题：(1) **噪声污染**——收集到的"正常"数据中不可避免地混入少量有缺陷的样本（像素级噪声）；(2) **长尾分布**——正常产品有多个类别/变体，但类别分布是长尾的且未知。现有方法在同时应对这两个问题时失效。

**核心矛盾**：存在一个"尾部类 vs 噪声"的 trade-off。对噪声鲁棒的方法（如通过过滤稀有特征来去噪）会误删尾部类的稀有正常特征；对尾部类友好的方法（如保留所有稀有特征）则无法过滤噪声。两者不能同时优化。

**本文目标**：打破 tail-vs-noise trade-off，让模型的内存库既包含尾部类的完整信息，又不被噪声特征污染。

**切入角度**：作者提出将尾部类样本和噪声样本的处理解耦——先识别出哪些是尾部类样本，将其单独处理（增强代表性），对非尾部类样本则正常做噪声过滤。这样两个问题不再互相干扰。

**核心 idea**：用 TailSampler 基于嵌入相似度分布的对称性来预测每个样本的类别基数，从而精准采样尾部类样本，将其与主体类样本分别处理，构建分治式内存库。

## 方法详解

### 整体框架
TailedCore 是一个基于内存库的异常检测模型。输入为一组无标注的"正常"图像（可能含噪声和长尾类别），输出为每个测试图像的异常分数和异常位置热力图。流程分为三步：(1) 用预训练特征提取器获取图像/patch 的嵌入；(2) TailSampler 预测每个样本的类别基数，据此将样本分为头部类和尾部类；(3) 对头部类做标准的 coreset 采样并过滤噪声，对尾部类做全量保留或增强采样，合并构建最终内存库。

### 关键设计

1. **TailSampler（尾部类采样器）**:

    - 功能：估计每个样本所属类别的大小（类别基数），从而识别出尾部类样本
    - 核心思路：基于一个关键观察——如果某个样本属于一个大类别，它与其他样本的嵌入相似度分布会在高相似度区域有较多样本；如果属于小类别，高相似度区域的样本会很少。具体做法是：对每个样本计算其与所有其他样本的 cosine 相似度，利用**对称性假设（Symmetric Assumption）**——假设每个类别的嵌入分布关于类中心对称——来估计类别基数。如果样本 $x$ 与前 $k$ 近邻的平均相似度为 $s$，则其类别基数估计为 $\hat{n} = f(s, k)$，其中 $f$ 基于对称分布的特性推导得到
    - 设计动机：直接用聚类方法估计类别需要预设簇数（但这里类别数未知），且聚类对噪声敏感。TailSampler 不需要显式聚类，只通过局部相似度统计量就能估计类别规模

2. **分治式内存库构建（Split Memory Construction）**:

    - 功能：分别处理头部类和尾部类样本，构建既全面又干净的内存库
    - 核心思路：TailSampler 输出每个样本的类别基数估计后，设定阈值 $\tau$ 将样本分为头部类集合 $\mathcal{H}$ 和尾部类集合 $\mathcal{T}$。**对头部类**：样本数量充足，可以安全地做 coreset 采样（如 greedy coreset selection）并过滤掉可能的噪声样本；**对尾部类**：样本本就稀少，采用更保守的策略——增加采样比例，确保尾部类特征在内存库中有充足代表。最终内存库 $\mathcal{M} = \mathcal{M}_H \cup \mathcal{M}_T$
    - 设计动机：这种分治策略正是打破 tail-vs-noise trade-off 的关键。头部类可以放心去噪（删掉少量样本不影响代表性），尾部类则优先保全（宁可保留少量噪声也不丢失稀有正常模式）

3. **噪声鲁棒的 Coreset 采样**:

    - 功能：在头部类样本中进行噪声感知的代表性采样
    - 核心思路：在标准 coreset 采样基础上引入噪声得分。对每个 patch 特征，计算其与局部邻域的一致性——如果一个 patch 特征与其 $k$-NN 的相似度异常低于同类别的平均水平，则标记为疑似噪声。采样时降低疑似噪声 patch 的优先级
    - 设计动机：标准 coreset 采样会刻意选择距已选特征最远的样本来最大化覆盖率，但这恰好容易把噪声样本（它们往往是离群点）选入内存库

### 损失函数 / 训练策略
TailedCore 作为基于内存库的方法不需要额外训练，特征提取使用预训练的 backbone（如 WideResNet-50）。测试时，异常分数为测试 patch 特征到内存库最近邻的距离。

## 实验关键数据

### 主实验

在 MVTec-AD 和 VisA 数据集上，设置不同的长尾比例（imbalance ratio）和噪声比例（noise ratio）进行评估。

| 方法 | MVTec (IR=100, NR=5%) | MVTec (IR=50, NR=10%) | VisA (IR=100, NR=5%) | VisA (IR=50, NR=10%) |
|------|----------------------|----------------------|---------------------|---------------------|
| PatchCore | 87.2 | 83.5 | 82.1 | 78.4 |
| SoftPatch | 89.1 | 86.3 | 84.5 | 81.2 |
| NoisyAD | 88.7 | 87.1 | 83.8 | 80.6 |
| **TailedCore** | **92.4** | **90.8** | **88.3** | **85.7** |

### 消融实验

| 配置 | MVTec (AUROC) | 说明 |
|------|-------------|------|
| Full TailedCore | 92.4 | 完整模型 |
| w/o TailSampler | 88.1 | 不区分头尾类，统一处理，掉4.3% |
| w/o 噪声过滤 | 90.2 | 不做噪声感知采样，掉2.2% |
| w/o 分治内存库 | 88.9 | 统一采样策略，掉3.5% |
| Random sampler 替代 TailSampler | 88.5 | 随机采样尾部类，掉3.9% |

### 关键发现
- TailSampler 是最关键组件，去掉后性能退化最严重（-4.3%），表明准确识别尾部类是解决 tail-vs-noise trade-off 的核心
- 在噪声比例高（10%）的设置下，TailedCore 的优势更加明显，说明分治策略在困难场景下收益更大
- TailSampler 的对称性假设在实际分布中成立：实验验证了工业图像的预训练特征确实近似满足类内对称分布
- 在纯噪声无长尾（IR=1, NR>0）设置下，TailedCore 与 SoftPatch 持平；在纯长尾无噪声（IR>1, NR=0）设置下超过 PatchCore。说明 TailedCore 没有为"分治"付出额外代价

## 亮点与洞察
- **对称性假设来估计类别基数非常优雅**：不需要聚类、不需要密度估计，仅通过 $k$-NN 相似度的统计量就能区分头尾类，计算高效且无超参数敏感性问题
- **问题 formulation 本身是一个重要贡献**：首次将"长尾+噪声"作为异常检测的联合挑战正式提出，并系统化了 tail-vs-noise trade-off 的分析
- **分治思想可迁移到其他内存库方法**：任何基于代表性子集的方法都可以引入类似的头尾分离策略

## 局限与展望
- TailSampler 依赖于预训练特征的质量，如果 backbone 对某些产品类型的区分度不够，对称性假设可能不成立
- 阈值 $\tau$ 的设定仍需经验调整，不同数据集可能需要不同的阈值
- 目前只在工业异常检测场景验证，对自然图像、医学图像中的长尾噪声场景是否有效未做实验
- 没有处理类别间存在渐变/连续分布的情况（如颜色渐变的同一产品），这种场景下"类别"本身定义模糊

## 相关工作与启发
- **vs PatchCore**: PatchCore 是经典内存库方法，假设训练集干净且均衡。TailedCore 的核心优势在于应对非理想数据
- **vs SoftPatch**: SoftPatch 通过软权重处理噪声，但不考虑长尾问题。在纯噪声设置下两者相当，但在长尾+噪声联合设置下 TailedCore 大幅领先
- **vs NoisyAD**: NoisyAD 聚焦噪声鲁棒但忽略类别不均衡，在高长尾比下性能下降

## 评分
- 新颖性: ⭐⭐⭐⭐ 问题定义新颖（长尾+噪声联合），TailSampler 设计巧妙，但整体框架仍在 PatchCore 范式内
- 实验充分度: ⭐⭐⭐⭐ 多种长尾比+噪声比组合的系统评估，消融充分
- 写作质量: ⭐⭐⭐⭐ trade-off 分析清晰，动机阐述有说服力
- 价值: ⭐⭐⭐⭐ 实际工业场景中数据不完美是常态，这项工作直接解决了实际痛点

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval](search_and_detect_training-free_long_tail_object_detection_via_web-image_retriev.md)
- [\[CVPR 2025\] UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection](univad_a_training-free_unified_model_for_few-shot_visual_anomaly_detection.md)
- [\[ICLR 2026\] Dual Distillation for Few-Shot Anomaly Detection](../../ICLR2026/object_detection/dual_distillation_for_few-shot_anomaly_detection.md)
- [\[CVPR 2025\] AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP](aa-clip_enhancing_zero-shot_anomaly_detection_via_anomaly-aware_clip.md)
- [\[AAAI 2026\] Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory](../../AAAI2026/object_detection/commonality_in_few_few-shot_multimodal_anomaly_detection_via_hypergraph-enhanced.md)

</div>

<!-- RELATED:END -->
