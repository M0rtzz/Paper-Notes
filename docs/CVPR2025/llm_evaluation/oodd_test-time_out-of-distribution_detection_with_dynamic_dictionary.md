---
title: >-
  [论文解读] OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary
description: >-
  [CVPR 2025][分布外检测] 提出 OODD，通过优先队列维护动态 OOD 字典在测试时实时收集潜在 OOD 样本特征来校准 OOD 分数，在 CIFAR-100 Far OOD 上相比 SOTA 方法 FPR95 降低 26.0%，且无需微调。
tags:
  - CVPR 2025
  - 分布外检测
  - 动态字典
  - 优先队列
  - 测试时自适应
  - KNN
---

# OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary

**会议**: CVPR 2025  
**arXiv**: [2503.10468](https://arxiv.org/abs/2503.10468)  
**代码**: [GitHub](https://github.com/zxk1212/OODD)  
**领域**: others  
**关键词**: 分布外检测, 动态字典, 优先队列, 测试时自适应, KNN

## 一句话总结

提出 OODD，通过优先队列维护动态 OOD 字典在测试时实时收集潜在 OOD 样本特征来校准 OOD 分数，在 CIFAR-100 Far OOD 上相比 SOTA 方法 FPR95 降低 26.0%，且无需微调。

## 研究背景与动机

分布外（OOD）检测是深度学习安全部署的关键问题。现有方法分为三类：(1) 后处理方法（基于 logits/特征设计分数函数）；(2) 微调方法（使用辅助数据集暴露异常值）；(3) 测试时校准方法。

微调方法的核心局限在于训练阶段使用的异常值与测试时遇到的 OOD 样本分布差异大时，性能急剧下降。测试时自适应方法（如 RTL++、AUTO）通过批次级参数更新进行校准，但可能导致灾难性遗忘和检测不稳定。

**核心动机**: 测试时 OOD 检测的关键在于利用潜在的 OOD 特征。如果能在测试过程中实时收集并维护一个 OOD 样本字典，就可以利用这些特征来校准检测分数，而无需任何参数更新。

## 方法详解

### 整体框架

OODD 包含三个核心组件：(1) 信息性内点采样（IIS）——从训练ID数据中筛选高置信度的代表性样本构建 ID 字典；(2) 动态 OOD 字典——通过优先队列在测试时实时收集低 OOD 分数的样本特征；(3) 双重 OOD 稳定化（DOS）——用从 ID 数据生成的异常值初始化优先队列和记忆库。最终 OOD 分数为 ID 字典相似度和 OOD 字典相似度之和：$S(\mathbf{x}^*) = S_{in}(\mathbf{x}^*) + S_{out}(\mathbf{x}^*)$。

### 关键设计1: 信息性内点采样 (IIS)

**功能**: 构建高质量、紧凑的 ID 字典，替代 KNN 中使用全部训练数据。

**核心思路**: 对每个训练样本进行多次随机裁剪，选取置信度最高的裁剪 patch。然后在每个类内按置信度排序，选取前 $\alpha\%$ 的 patch 特征存入 ID 字典 $\mathcal{K}_{n'}^{id}$。测试时计算查询特征与 ID 字典中第 $\mathbb{K}$ 近邻的余弦相似度作为 $S_{in}$。

**设计动机**: 全部训练数据作为 ID 字典包含冗余和噪声样本，降低检测效率和质量。通过 patch 级和类别级双重筛选，保留最具代表性的 ID 特征，同时显著减少字典大小。

### 关键设计2: 优先队列动态 OOD 字典

**功能**: 在测试时实时收集和维护最具代表性的潜在 OOD 样本特征。

**核心思路**: 维护以 OOD 分数为优先级的优先队列。OOD 分数最高的样本在队首。当新测试样本的 OOD 分数低于队首（更可能是 OOD）时入队，队首出队（队满时）。测试时计算查询与 OOD 字典的负第 $\hat{\mathbb{K}}$ 近邻余弦相似度作为 $S_{out}$。字典大小为超参数，与 batch 大小解耦。

**设计动机**: 测试时的 OOD 样本是最直接的校准信号。优先队列确保字典始终保留 OOD 分数最低（最可能是 OOD）的样本特征，避免了批次级参数更新导致的灾难性遗忘问题。字典更新仅需一次矩阵乘法，计算开销极低。

### 关键设计3: 双重 OOD 稳定化 (DOS)

**功能**: 解决测试初期 OOD 字典为空时检测分数不稳定的问题。

**核心思路**: 使用从 ID 数据生成的异常值（C-Out）分为两部分：一部分初始化优先队列，一部分存入固定记忆库 $\mathcal{K}_{mb}^{ood}$。最终 OOD 字典为 $\mathcal{K}_{total}^{ood} = \mathcal{K}_l^{ood} \cup \mathcal{K}_{mb}^{ood}$。C-Out 方法通过随机裁剪生成低置信度 patch 作为异常值，无需外部数据。

**设计动机**: 空 OOD 字典在测试初期导致 $S_{out}$ 不可用，校准分数剧烈波动。固定记忆库提供稳定基准，优先队列逐步用真实 OOD 特征替换初始化特征。

### 损失函数

无训练损失。方法完全在测试时运行，仅涉及余弦相似度计算和优先队列维护。

## 实验关键数据

### 主实验结果 (OpenOOD 基准, CIFAR-100)

| 方法 | Near OOD AUROC↑ | Near OOD FPR95↓ | Far OOD AUROC↑ | Far OOD FPR95↓ |
|------|----------------|----------------|----------------|----------------|
| KNN | — | — | — | — |
| TULIP (SOTA) | — | — | — | 58.17 |
| **OODD** | — | — | — | **24.74** |

### CIFAR-10 基准

| 方法 | Near OOD AUROC↑ | Far OOD AUROC↑ | Far OOD FPR95↓ |
|------|----------------|----------------|----------------|
| KNN | 90.64 | 92.96 | 24.27 |
| ViM | 88.68 | 93.48 | 25.05 |
| TULIP | 89.67 | 92.55 | 24.43 |
| **OODD** | **90.96** | **95.77** | **17.44** |

### 关键发现

1. **Far OOD 检测大幅提升**: CIFAR-100 Far OOD 上 FPR95 从 58.17%→24.74%（TULIP→OODD），降低超过 33 个百分点。
2. **C-Out 足够有效**: 仅从 ID 数据通过随机裁剪生成异常值（不引入外部数据），即可实现强劲性能。
3. **与后处理方法互补**: OODD 可叠加在 KNN、ViM 等方法上进一步提升性能。
4. **KNN 加速**: 用余弦相似度替代欧氏距离，理论证明判别能力等价，实现 3x 加速。
5. **无微调开销**: 不修改任何模型参数，仅需一次矩阵乘法计算与 OOD 字典的相似度。

## 亮点与洞察

- **测试时信号利用**: 将测试时遇到的 OOD 样本视为宝贵资源而非单纯的待检测对象。
- **优先队列的精妙**: 既保证字典质量（保留最可能的 OOD 样本），又避免灾难性遗忘。
- **零外部数据**: C-Out 方法仅需 ID 训练数据，通过"反向"筛选（选低置信度而非高置信度）获得初始化异常值。

## 局限与展望

- **假设 OOD 分数分布**: 依赖 OOD 样本处于 OOD 分数左尾的假设，对 hard OOD（分数与 ID 重叠大）可能效果有限。
- **字典大小敏感性**: 优先队列大小和 $\mathbb{K}$、$\hat{\mathbb{K}}$ 参数需要调优。
- **流式场景顺序依赖**: 早期测试样本中如果 OOD 比例极低，字典质量可能长期不佳。
- 未来可探索自适应字典大小、多尺度特征字典等方向。

## 相关工作与启发

- **KNN OOD 检测**: OODD 本质上是 KNN 方法的双向扩展——同时维护 ID 和 OOD 两个字典。
- **MoCo 动量字典**: 优先队列维护字典的思想与 MoCo 的队列设计有异曲同工之妙。
- **启发**: 测试时可用信号的利用是一个被低估的方向，适用于任何需要适应新分布的场景。

## 评分

⭐⭐⭐⭐ — 方法简洁高效，无需微调的测试时 OOD 检测在实际部署中极具价值。优先队列设计巧妙，FPR95 降低 26% 效果显著。理论分析（余弦/欧氏距离等价性）是加分项。

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries](../../AAAI2026/llm_evaluation/graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)
- [\[ECCV 2024\] Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams](../../ECCV2024/llm_evaluation/distribution_alignment_for_fully_test-time_adaptation_with_dynamic_online_data_s.md)
- [\[ICCV 2025\] DisCoPatch: Taming Adversarially-driven Batch Statistics for Improved Out-of-Distribution Detection](../../ICCV2025/llm_evaluation/discopatch_taming_adversarially-driven_batch_statistics_for_improved_out-of-dist.md)
- [\[NeurIPS 2025\] Test-Time Adaptation by Causal Trimming](../../NeurIPS2025/llm_evaluation/test-time_adaptation_by_causal_trimming.md)
- [\[ICCV 2025\] BATCLIP: Bimodal Online Test-Time Adaptation for CLIP](../../ICCV2025/llm_evaluation/batclip_bimodal_online_test-time_adaptation_for_clip.md)

<!-- RELATED:END -->
