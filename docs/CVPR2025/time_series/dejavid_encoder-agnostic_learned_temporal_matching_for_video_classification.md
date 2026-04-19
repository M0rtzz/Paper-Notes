---
title: >-
  [论文解读] DejaVid: Encoder-Agnostic Learned Temporal Matching for Video Classification
description: >-
  [CVPR 2025][时间序列][视频分类] 提出 DejaVid，一种编码器无关的轻量级视频分类增强方法：将视频表示为变长时序嵌入序列 (TSE) 而非单个嵌入，通过学习每个时间步、每个特征维度的重要性权重，结合改进的可微分 DTW 算法做时序对齐分类，仅增加 <1.8% 参数就在 SSV2 达到 77.2%、K400 达到 89.1% 的 SOTA。
tags:
  - CVPR 2025
  - 时间序列
  - 视频分类
  - Dynamic Time Warping
  - 时序嵌入序列
  - 编码器无关
  - 轻量级后处理
---

# DejaVid: Encoder-Agnostic Learned Temporal Matching for Video Classification

**会议**: CVPR 2025  
**arXiv**: [2506.12585](https://arxiv.org/abs/2506.12585)  
**代码**: https://github.com/darrylho/DejaVid  
**领域**: 时间序列 / 视频理解  
**关键词**: 视频分类, Dynamic Time Warping, 时序嵌入序列, 编码器无关, 轻量级后处理

## 一句话总结
提出 DejaVid，一种编码器无关的轻量级视频分类增强方法：将视频表示为变长时序嵌入序列 (TSE) 而非单个嵌入，通过学习每个时间步、每个特征维度的重要性权重，结合改进的可微分 DTW 算法做时序对齐分类，仅增加 <1.8% 参数就在 SSV2 达到 77.2%、K400 达到 89.1% 的 SOTA。

## 研究背景与动机

**领域现状**：大型视频 Transformer（如 VideoMAE V2-g，10 亿参数）在动作识别上取得了很好的结果，但它们处理变长视频的方式简单粗暴——从多个时间片段和空间裁剪中提取嵌入然后平均。

**现有痛点**：平均操作丢失了三种重要的时间信息：(1) 视频时长差异——不同视频长度不同；(2) 事件的时间顺序——开门和关门的动作时间颠倒语义完全不同；(3) 特征重要性的时间变化——投篮视频中早期帧强调球员位置，后期帧关注球是否进框。现有改进方案（如在 Transformer 中插入时序层）需要重新训练大模型，成本极高。

**核心矛盾**：大模型效果好但时序建模差，专门的时序方法需要侵入式修改架构并重新训练，两者难以兼得。

**本文目标** 如何在不修改、不重训大型预训练编码器的前提下，通过轻量级后处理器增强时序建模能力。

**切入角度**：将视频表示为时序嵌入序列 (TSE) 而非单个嵌入向量，用 DTW 算法做时序对齐，并学习时间-特征维度的重要性权重。

**核心 idea**：用滑动窗口将视频编码为变长 TSE，通过加权 DTW 与每类的质心 TSE 做时序对齐匹配来分类。

## 方法详解

### 整体框架
预训练阶段：用滑动窗口对视频提取 $T \times N_f$ 的时序嵌入序列 (TSE)，用 DBA 算法为每个类计算质心 TSE。训练阶段：学习类质心 TSE 和时间-特征权重张量 $U$。测试阶段：计算输入 TSE 到各类质心的加权 DTW 距离，softmin 得到分类概率。

### 关键设计

1. **时序嵌入序列 (TSE)**:

    - 功能：保留视频的时间顺序和变长特性
    - 核心思路：对视频施加滑动窗口，每个窗口送入冻结的编码器得到一个嵌入向量，所有窗口的嵌入拼接成 $T \times N_f$ 的 TSE（$T$ 随视频长度变化）。相比传统的多片段平均，TSE 天然保持时间顺序并支持变长。质心 TSE 通过 DBA（DTW 重心平均）算法初始化，之后作为可学习参数优化
    - 设计动机：单个嵌入无法区分时间顺序不同但帧集合相同的视频（如开门 vs 关门）

2. **时间加权 DTW 距离**:

    - 功能：建模特征重要性的时间变化
    - 核心思路：经典 DTW 用曼哈顿距离计算逐点距离，本文引入可学习权重 $u_{i,k}$，使 $dist_w(u_i, a_i, b_j) = \sum_k u_{i,k} |a_{i,k} - b_{j,k}|$。权重 $U \in \mathbb{R}_{>0}^{N_c \times T_c \times N_f}$ 与质心 TSE 同形状，表示每类在每个时间步上每个特征的重要性。通过存储 log(U) 并前向传播时取 exp 来保证正性。去除 DTW 的对角转移以稳定模型（固定路径长度为 $n+m-1$）
    - 设计动机：投篮视频中"球进框"的特征在后期更重要，DTW 默认的均匀权重无法捕捉这种时变性

3. **DTW 神经网络重构**:

    - 功能：将 DTW 动态规划高效重构为可反向传播的神经网络
    - 核心思路：将 DTW 的 2D 网格按对角线重组——第 $l$ 条对角线在第 $l-1$ 条完成后即可并行执行。每条对角线等价于一个 kernel=2、stride=1 的 min-pooling 层加一个加性 skip-connection。整个 DTW 变成 min-pooling 层的串行堆叠，可直接用标准反向传播优化。自定义 CUDA kernel 使计算速度提升 2 个数量级
    - 设计动机：传统 DTW 的 $O(nm)$ 串行路径太长，重构后关键路径降为 $O(n+m)$

### 损失函数 / 训练策略
对训练 TSE 计算到所有类质心的加权 DTW 距离，softmin 得到概率分布，交叉熵损失。仅优化质心 TSE $C$ 和权重 $U$，编码器完全冻结。AdamW 优化器，36 epoch，余弦学习率调度。每 epoch 冻结当前质心/权重副本用于计算 warping path，用非冻结原始参数反向传播。

## 实验关键数据

### 主实验

| 方法 | 参数量 | K400 Top-1 | SSV2 Top-1 | HMDB51 Top-1 |
|------|--------|-----------|-----------|-------------|
| VideoMAE V2-g (baseline) | 1013M | 88.4% | 76.7% | 88.1% |
| InternVideo2-6B | 6B | 92.1% | 77.5% | - |
| **DejaVid (frozen weights)** | **+5.8M** | **89.1%** | 77.1% | 88.3% |
| **DejaVid (full learning)** | **+11.6M** | 88.9% | **77.2%** | **88.6%** |

相比 VideoMAE V2-g：K400 +0.7%、SSV2 +0.5%、HMDB51 +0.5%。额外参数量不足编码器的 1.8%，训练不到 3 小时。

### 消融实验

| 配置 | SSV2 Top-1 | K400 Top-1 | HMDB51 Top-1 |
|------|-----------|-----------|-------------|
| 仅质心学习 (冻结权重) | 77.1% | 89.1% | 88.3% |
| 仅权重学习 (冻结质心) | 77.0% | 88.4% | 88.5% |
| 全部学习 | **77.2%** | 88.9% | **88.6%** |
| 时序超采样 (无 DejaVid) | 76.2% | 88.1% | 88.0% |

### 关键发现
- 质心学习贡献最大（平均 +0.43%），权重学习在数据稀缺时更有用（HMDB51 +0.4%）
- 单纯时序超采样（增加采样帧数但仍然平均）不带来性能提升，证明 DejaVid 的收益来自时序对齐而非信息量
- 去除 DTW 对角转移对模型稳定性至关重要——固定路径长度消除了不同路径间的不公平比较
- 每 epoch 而非每 batch 更新 warping path 计算的冻结副本也有助于收敛稳定性

## 亮点与洞察
- **编码器无关的即插即用增强**：不改任何架构、不重训任何权重，就能在 10 亿参数的 SOTA 模型上获得显著提升。这种后处理范式可以推广到任何视频编码器
- **DTW 的神经网络重构**：将经典动态规划算法表达为 min-pooling + skip-connection 的堆叠，既提升并行性又自然支持反向传播，这个重构思路可以迁移到其他 DP 算法
- **特征重要性的时间变化建模**：学习每个时间步、每个特征维度的权重是一个有洞察力的设计——不同动作在不同时间阶段关注的信息确实不同

## 局限与展望
- 仅在 VideoMAE V2-g 上验证（因为只有这个 SOTA 发布了预训练权重），编码器无关的声称需更多模型验证
- 权重学习在大数据集上容易过拟合（K400 上 full learning 反而比 frozen weights 差），需要正则化
- TSE 生成需要对视频做多次编码器前向传播（K400 需 33 次），推理成本较高
- 质心 TSE 长度固定为 8，这个选择缺乏理论依据

## 相关工作与启发
- **vs ILA/ATM**: 这些方法在 Transformer 块之间插入时序层，需要重训。DejaVid 完全后处理，零侵入
- **vs Soft-DTW/D3TW**: 之前的可微 DTW 工作没有建模时间-特征维度的重要性变化，DejaVid 通过学习权重扩展了 DTW 到高维场景
- **vs SlowFast**: SlowFast 用双路编码器融合不同帧率，计算成本高。DejaVid 用单个编码器 + 滑动窗口 + DTW 更轻量

## 评分
- 新颖性: ⭐⭐⭐⭐ DTW 的神经网络重构和时间加权很有创意
- 实验充分度: ⭐⭐⭐⭐ 三个数据集验证，但仅一个编码器
- 写作质量: ⭐⭐⭐⭐ 清晰且系统性强
- 价值: ⭐⭐⭐⭐ 即插即用的通用性有实际工程价值

<!-- RELATED:START -->

## 相关论文

- [FLAVC: Learned Video Compression with Feature Level Attention](flavc_learned_video_compression_with_feature_level_attention.md)
- [TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting](../../NeurIPS2025/time_series/timeperceiver_an_encoder-decoder_framework_for_generalized_time-series_forecasti.md)
- [LETS-C: Leveraging Text Embedding for Time Series Classification](../../ACL2025/time_series/lets-c_leveraging_text_embedding_for_time_series_classification.md)
- [Learning Soft Sparse Shapes for Efficient Time-Series Classification](../../ICML2025/time_series/learning_soft_sparse_shapes_for_efficient_time-series_classification.md)
- [SELDON: Supernova Explosions Learned by Deep ODE Networks](../../AAAI2026/time_series/seldon_supernova_explosions_learned_by_deep_ode_networks.md)

<!-- RELATED:END -->
