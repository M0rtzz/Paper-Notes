---
title: >-
  [论文解读] DMin: Scalable Training Data Influence Estimation for Diffusion Models
description: >-
  [CVPR 2026][图像生成][扩散模型] 提出 DMin，一个可扩展的扩散模型训练数据影响力估计框架，通过高效梯度压缩将存储需求从数百 TB 降至 MB/KB 级别，首次实现对数十亿参数扩散模型的影响力估计，支持亚秒级 top-k 检索。
tags:
  - CVPR 2026
  - 图像生成
  - 扩散模型
  - Influence Estimation
  - Gradient Compression
  - KNN
  - Scalability
---

# DMin: Scalable Training Data Influence Estimation for Diffusion Models

**会议**: CVPR 2026  
**arXiv**: [2412.08637](https://arxiv.org/abs/2412.08637)  
**代码**: 有（将开源 PyTorch 实现，含多进程支持）  
**领域**: Image Generation / Model Interpretability  
**关键词**: Diffusion Models, Influence Estimation, Gradient Compression, KNN, Scalability

## 一句话总结

提出 DMin，一个可扩展的扩散模型训练数据影响力估计框架，通过高效梯度压缩将存储需求从数百 TB 降至 MB/KB 级别，首次实现对数十亿参数扩散模型的影响力估计，支持亚秒级 top-k 检索。

## 研究背景与动机

理解「生成图像受哪些训练数据影响最大」对模型透明度、偏差分析和版权追溯至关重要。现有影响力估计方法面临三大瓶颈：

**模型规模不可扩展**：二阶方法（DataInf、K-FAC）需要 Hessian 逆近似，对大模型内存需求爆炸。以 Stable Diffusion 3 Medium 的 20 亿参数为例，单样本 10 步的梯度缓存就需要 80 GB，1 万样本需要 800 TB。

**投影矩阵过大**：一阶方法（D-TRAK、Journey-TRAK）使用随机投影降维，但 20 亿参数 × 32,768 维的投影矩阵需要 238 TB 存储。

**梯度不稳定性**：深层模型中梯度值可能极大，导致内积计算被异常值主导。

因此，现有方法只能应用于 LoRA 微调或小型扩散模型，无法处理全参数训练的大型模型。

## 方法详解

### 整体框架

DMin 的核心流程（Fig. 2）分为两个阶段：
1. **梯度计算与压缩**：对每个训练样本，在采样的时间步上计算梯度，经归一化和压缩后缓存
2. **影响力估计**：对生成图像同样计算压缩梯度，通过内积或 KNN 检索估计影响力分数

影响力估计的数学基础是一阶 Taylor 展开下的损失变化近似：

$$\mathcal{I}_\theta(X^s, X^i) = e\bar{\eta} \sum_{t=1}^{T} \nabla_\theta \mathcal{L}(f_\theta(z^i_p, z^i_t, t), \epsilon) \cdot \nabla_\theta \mathcal{L}(f_\theta(z^s_p, z^s_t, t), \epsilon)$$

即训练样本和生成样本在各时间步上损失梯度的内积之和。

### 关键设计

1. **高效梯度压缩（四步压缩管线）**：

    - 功能：将 20 亿维的梯度向量压缩到 v 维（v 可低至 2^12=4096）
    - 核心步骤：(1) Padding 到 v 的整数倍 → (2) 随机置换打乱结构 → (3) 逐元素乘随机 ±1 向量投影 → (4) 分组求和降维
    - 设计动机：随机置换+随机投影保证了 Johnson-Lindenstrauss 性质，分组聚合实现极高压缩比。与传统随机投影不同，无需存储巨大的投影矩阵，仅需一个置换向量（4 bytes/元素）和二值投影向量（1 bit/元素）

2. **L2 归一化**：

    - 功能：在压缩前对梯度向量做 L2 归一化
    - 核心思路：消除异常大梯度值的主导效应
    - 设计动机：实验发现不归一化时检测率骤降（SD 1.4 LoRA Flowers Top-5: 0.887 → 0.133），证实深层模型梯度不稳定是影响力估计的核心障碍

3. **KNN 索引加速**：

    - 功能：将各时间步的压缩梯度拼接后构建 HNSW 索引
    - 核心思路：用近似最近邻搜索代替穷举内积
    - 设计动机：实现亚秒级 top-k 检索。有趣的是，KNN 检索在实验中经常优于精确内积计算，可能是因为近似搜索隐式起到了正则化效果

4. **时间步采样**：

    - 对 1000 步的完整扩散过程进行子采样（如取 5-10 步），大幅降低计算和存储负担
    - 类似于扩散模型推理时的步骤调度策略

### 损失函数 / 训练策略

DMin 本身不训练模型，而是对已训练好的扩散模型进行后分析。关键操作：
- 梯度收集：对 LoRA 模型只收集适配器参数的梯度；对全参模型收集所有参数梯度
- 全参 SD3 Medium（20 亿参数）梯度收集成本约 330 GPU 小时
- KNN 索引构建仅需几分钟

## 实验关键数据

### 主实验（条件扩散模型检测率）

在混合数据集（9,288 样本，含 Flowers/Lego/Magic Cards 等子集）上微调 SD1.4 LoRA、SD3 Medium LoRA 和 SD3 Medium Full，对生成图像检索最相关的训练样本：

| 方法 | Flowers Top-5 | Flowers Top-10 | Magic Cards Top-5 | 适用模型 |
|------|--------------|----------------|-------------------|---------|
| Random | 0.000 | 0.000 | 0.200 | 任意 |
| CLIP Similarity | 0.000 | 0.000 | 0.444 | 任意 |
| LiSSA | 0.514 | 0.457 | 0.967 | 小模型/LoRA |
| DataInf | 0.413 | 0.406 | 0.967 | 小模型/LoRA |
| DMin (v=2^16) | **0.862** | **0.823** | **0.978** | 任意规模 |
| DMin (SD3 Full, v=2^16) | **0.959** | **0.931** | **0.996** | 20亿参数 |

在 SD3 Medium Full 上，LiSSA/DataInf/D-TRAK 因需要数百 TB 缓存而**完全无法运行**，DMin 是唯一可行的方法。

### 存储与速度对比

| 方法 | SD3 Full 每样本存储 | 全数据集存储 | 压缩比 |
|------|-------------------|-------------|--------|
| 未压缩梯度 | 37.42 GB | 339.39 TB | 100% |
| DMin (v=2^12) | 80 KB | 726 MB | 0.00017% |
| DMin (v=2^16) | 1.25 MB | 11.34 GB | 0.0028% |

| 方法 | SD3 LoRA 时间/测试样本 | 加速比 |
|------|---------------------|--------|
| LiSSA | 2136.7s | 0.19x |
| DataInf (Hessian) | 932.8s | 0.44x |
| DMin (v=2^12, KNN top-5) | **0.004s** | **101,878x** |

### 关键发现

1. **压缩几乎无损**：v=2^16 的压缩梯度与未压缩梯度在检测率上差异不到 1%
2. **归一化是关键**：不做归一化时性能暴跌，证实了大模型中梯度不稳定是核心问题
3. **KNN 略优于精确计算**：可能因近似搜索具有正则化效果
4. 首次在 20 亿参数全微调的 SD3 上完成影响力估计，其他方法均不可行
5. 在 MNIST 的无条件 DDPM 上，DMin Top-5 检测率 0.80，远超 Journey-TRAK（0.26）和 D-TRAK（0.13）

## 亮点与洞察

- **工程贡献极为突出**：将 339 TB 的存储需求压缩到 726 MB（压缩比 0.00017%），使原本不可能的任务变得可行
- **四步梯度压缩管线**设计巧妙：置换+随机投影+分组求和，既避免了巨大投影矩阵的存储问题，又保持了 JL 引理的距离保持性质
- **L2 归一化**的发现具有普遍意义：揭示了大模型梯度不稳定性对基于梯度的分析方法的根本影响
- KNN 检索优于精确计算的反直觉结论值得进一步研究

## 局限与展望

1. 梯度收集阶段对全参模型仍有较高成本（330 GPU 小时），但属于一次性投入
2. 影响力估计基于一阶近似，忽略了跨时间步的二阶交互
3. 目前主要在相对小规模数据集（~9K 样本）上验证，百万级训练集的实际应用仍需探索
4. 使用固定高斯噪声近似训练过程中的实际噪声，理论上存在偏差
5. 尚未在最新的 FLUX/SORA 等更大模型上验证

## 相关工作与启发

- **TRAK / D-TRAK / Journey-TRAK**：一阶随机投影方法，但投影矩阵过大限制了规模
- **DataInf / K-FAC**：二阶 Hessian 近似方法，需全量梯度加载
- **向量压缩文献**启发了 DMin 的压缩管线设计
- 本文思路可推广到 LLM 的训练数据溯源、数据污染检测等场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 梯度压缩管线设计精巧，但核心思想（梯度内积估计影响力）是已有框架的工程延伸
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三种模型规模、多子集评估、存储/时间/精度全方位对比，消融充分
- **写作质量**: ⭐⭐⭐⭐ — 问题动机清晰，公式推导完整，但部分表格过长影响阅读
- **价值**: ⭐⭐⭐⭐⭐ — 首次将影响力估计扩展到数十亿参数扩散模型，对模型审计和数据版权有重要实际意义

<!-- RELATED:START -->

## 相关论文

- [Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection](../../AAAI2026/image_generation/diffusion_reconstruction-based_data_likelihood_estimation_for_core-set_selection.md)
- [Learning a Distance Measure from the Information-Estimation Geometry of Data](../../ICLR2026/image_generation/learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [Erasing Undesirable Influence in Diffusion Models (EraseDiff)](../../CVPR2025/image_generation/erasing_undesirable_influence_in_diffusion_models.md)
- [HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models](ham_a_training-free_style_transfer_approach_via_heterogeneous_attention_modulati.md)
- [Difficulty Controlled Diffusion Model for Synthesizing Effective Training Data](../../AAAI2026/image_generation/difficulty_controlled_diffusion_model_for_synthesizing_effec.md)

<!-- RELATED:END -->
