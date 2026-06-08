---
title: >-
  [论文解读] DMin: Scalable Training Data Influence Estimation for Diffusion Models
description: >-
  [CVPR 2026][图像生成][扩散模型] 提出 DMin，一个可扩展的扩散模型训练数据影响力估计框架，通过高效梯度压缩将存储需求从数百 TB 降至 MB/KB 级别，首次实现对数十亿参数扩散模型的影响力估计，支持亚秒级 top-k 检索。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "扩散模型"
  - "Influence Estimation"
  - "Gradient Compression"
  - "KNN"
  - "Scalability"
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

DMin 想回答一个很具体的问题：给定一张生成图，训练集里哪几张样本对它影响最大？它的判据延续了影响力估计的经典思路——如果把某个训练样本拿去更新一步模型，对生成样本损失的拉动越大，这个训练样本就越"相关"。在一阶 Taylor 展开下，这种拉动可以写成两个样本损失梯度的内积，对扩散模型还要把这个内积在采样到的时间步上累加起来：

$$\mathcal{I}_\theta(X^s, X^i) = e\bar{\eta} \sum_{t=1}^{T} \nabla_\theta \mathcal{L}(f_\theta(z^i_p, z^i_t, t), \epsilon) \cdot \nabla_\theta \mathcal{L}(f_\theta(z^s_p, z^s_t, t), \epsilon)$$

公式本身不新，难点全在"算得起"。20 亿参数的梯度向量直接缓存要 37 GB 一条、整个数据集 339 TB，根本存不下。所以整条管线（Fig. 2）拆成两段离线/在线流程：**离线**把每个训练样本在采样时间步上的梯度算出来、归一化、压缩进一个几 KB 到几 MB 的向量，建好索引；**在线**对一张生成图做同样的压缩，再用内积或近邻检索挑出 top-k。真正让它从"理论可行"变成"工程可跑"的，是下面几处针对存储、数值稳定和检索速度的设计。

### 关键设计

**1. 高效梯度压缩：在不存投影矩阵的前提下把 20 亿维压到几千维。**

最大的拦路虎是降维本身的代价。传统随机投影要把 20 亿维降到 32,768 维，光投影矩阵就要 238 TB——比要省的梯度还大。DMin 换了一条不显式存矩阵的压缩路线：先把梯度 padding 到目标维度 $v$（可低至 $2^{12}=4096$）的整数倍，再用一个固定随机置换把参数顺序彻底打乱，逐元素乘上一个随机 $\pm1$ 向量做符号翻转，最后按组求和聚合到 $v$ 维。置换加随机符号投影保证了 Johnson–Lindenstrauss 性质（压缩后内积近似保持原始内积），而分组求和把维度直接砍到目标尺寸。关键在于这条管线要重放时只需记住"怎么打乱、怎么翻符号"：一个置换向量（4 bytes/元素）加一个二值符号向量（1 bit/元素）就够了，省掉了那张 238 TB 的稠密矩阵，这才是 339 TB → 几百 MB 压缩比的来源。

**2. 压缩前先做 L2 归一化：堵住异常大梯度对内积的劫持。**

深层扩散模型里个别参数的梯度值可能极端大，内积一旦被这些异常值主导，影响力分数就退化成"谁的梯度范数大谁赢"，跟语义相关性脱钩。DMin 在压缩之前对每个梯度向量先做 L2 归一化，把比较拉回到方向而非幅度上。这一步看着不起眼，却是性能的生死线：消融里关掉归一化后，SD 1.4 LoRA 在 Flowers 上的 Top-5 检测率从 0.887 直接掉到 0.133。这个落差也反过来佐证了"大模型梯度不稳定"是基于梯度的影响力估计长期被忽视的核心障碍。

**3. KNN 近似检索：把每条查询的穷举内积换成亚秒级最近邻。**

即便单个向量压到了几 KB，面对万级、百万级训练集逐条算内积仍然慢。DMin 把各时间步的压缩梯度拼接成一条向量后建 HNSW 近似最近邻索引，查询时用 ANN 直接召回 top-k，而不再扫全库。代价是结果从精确变近似，收益是单测试样本的检索从上千秒降到 0.004 秒。一个反直觉的观察是：KNN 检索在不少设置下反而略优于精确内积，作者猜测近似搜索隐含了某种正则化效果，抹平了压缩噪声带来的个别异常匹配。

**4. 时间步子采样：用 5–10 步代替 1000 步扩散链。**

扩散损失原本要在上千个去噪时间步上累加梯度，全算一遍既慢又把存储再翻上千倍。DMin 借用扩散推理里"少步采样"的思路，只在子采样到的少数时间步（如 5–10 步）上计算并缓存梯度，用它们近似整条链的影响力贡献。这把梯度收集与存储成本压下一个数量级，实测对检测率几乎无损。

### 损失函数 / 训练策略

DMin 不训练任何模型，而是对训练好的扩散模型做事后分析，所以没有损失函数，只有梯度收集的工程取舍：LoRA 模型只收集适配器参数的梯度，全参模型则收集全部参数；全参 SD3 Medium（20 亿参数）这一次性梯度收集约需 330 GPU 小时，而压缩后建 KNN 索引只要几分钟。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection](../../AAAI2026/image_generation/diffusion_reconstruction-based_data_likelihood_estimation_for_core-set_selection.md)
- [\[ICLR 2026\] Learning a Distance Measure from the Information-Estimation Geometry of Data](../../ICLR2026/image_generation/learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [\[CVPR 2025\] Erasing Undesirable Influence in Diffusion Models (EraseDiff)](../../CVPR2025/image_generation/erasing_undesirable_influence_in_diffusion_models.md)
- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](../../ICML2026/image_generation/guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)
- [\[CVPR 2026\] HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models](ham_a_training-free_style_transfer_approach_via_heterogeneous_attention_modulati.md)

</div>

<!-- RELATED:END -->
