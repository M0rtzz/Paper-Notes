---
title: >-
  [论文解读] Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models
description: >-
  [ICLR 2026][模型剪枝] 将LLM层剪枝建模为合作博弈，利用轻量代理网络近似Shapley值来捕获层间依赖关系，实现比静态启发式方法更优的深度剪枝效果。
tags:
  - ICLR 2026
  - 模型剪枝
  - 合作博弈
  - Shapley值
  - 代理网络
  - 大语言模型
---

# Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models

**会议**: ICLR 2026  
**arXiv**: [2602.07804](https://arxiv.org/abs/2602.07804)  
**代码**: [GitHub](https://github.com/920927/Pruning_As_A_Cooperative_Game)  
**领域**: 强化学习  
**关键词**: 模型剪枝, 合作博弈, Shapley值, 代理网络, 大语言模型

## 一句话总结

将LLM层剪枝建模为合作博弈，利用轻量代理网络近似Shapley值来捕获层间依赖关系，实现比静态启发式方法更优的深度剪枝效果。

## 研究背景与动机

大语言模型（LLM）的部署受限于巨大的计算和内存需求，层剪枝（删除整个Transformer层）是一种有效的压缩方法。现有的层剪枝方法普遍基于**静态启发式规则**（如权重大小、激活范数、敏感度分析），假设每层的重要性是固定且独立的。

然而，作者通过实验发现层重要性具有**上下文依赖性**：在单层剪枝中，中间层的排名波动显著；在多层剪枝中，这种波动进一步放大。更关键的是，按静态重要性逐层剪枝得到的结果（PPL=15.4535）并非全局最优，最优的两层组合（Layer 10+11, PPL=15.4279）可能不包含排名最低的单层。这揭示了层间存在动态依赖关系，静态方法无法捕捉。

本文的核心idea：将层剪枝形式化为**合作博弈**，每层是一个参与者，模型性能是效用函数，通过Shapley值量化每层在考虑层间交互后的真实贡献。

## 方法详解

### 整体框架

两阶段框架：阶段一生成多样的剪枝掩码并评估性能，阶段二训练代理网络近似性能下降，进而高效估计Shapley值。

### 关键设计

1. **分层蒙特卡洛掩码采样（阶段一）**:

    - 功能：生成具有受控Hamming权重的多样化二元剪枝掩码
    - 核心思路：对保留层数 $k$ 进行分层采样，每层 $k_j$ 采样 $N_{k_j}$ 个掩码 $\mathbf{m}^{(k_j,t)} \sim \text{Uniform}\{\mathbf{m} \in \{0,1\}^L : k(\mathbf{m})=k_j\}$。性能分数为 $s(\mathbf{m}) = \text{PPL}_\text{orig} / \text{PPL}(M(\mathbf{m}))$
    - 设计动机：确保不同剪枝比例的均衡覆盖，防止采样偏向特定压缩率

2. **轻量代理网络（阶段二）**:

    - 功能：训练两层前馈网络 $f_\theta(\mathbf{m})$ 预测任意掩码的性能分数
    - 核心思路：使用MSE损失 $\mathcal{L}(\theta) = \frac{1}{N}\sum_{n=1}^{N}(f_\theta(\mathbf{m}_n) - s(\mathbf{m}_n))^2$ 训练代理
    - 设计动机：避免对每个掩码组合进行完整模型推理，将Shapley值估计的计算成本降至可接受水平

3. **近似Shapley值估计**:

    - 功能：利用代理网络高效计算每层的边际贡献
    - 核心思路：$\hat{\phi}_i = \frac{1}{Q}\sum_{q=1}^{Q}(f_\theta(\mathbf{m}^{(k_j,q)} \cup \{i\}) - f_\theta(\mathbf{m}^{(k_j,q)}))$
    - 设计动机：精确Shapley值需要遍历 $2^L$ 个子集，计算不可行；代理网络使大规模采样估计成为可能

### 损失函数 / 训练策略

代理网络采用MSE损失训练，使用阶段一收集的（掩码，性能分数）对作为监督信号。训练完成后，在大量候选掩码上估计Shapley值，按贡献从低到高移除层直到达到目标压缩率。

## 实验关键数据

### 主实验（LLaMA-2-7B，WikiText2 PPL）
| 方法 | 删3层 | 删6层 | 删9层 | 删12层 |
|------|-------|-------|-------|--------|
| SliceGPT | 108.10 | 212.89 | 291.85 | 393.89 |
| SLEB | 14.24 | 19.47 | 27.45 | 58.12 |
| Shortened-LLaMA | 16.65 | 36.37 | 81.96 | 304.52 |
| ShortGPT | 16.65 | 36.37 | 81.96 | 157.99 |
| **本文** | **14.69** | **18.87** | **24.61** | **38.12** |

### 消融实验
| 配置 | PPL(WikiText2) | 说明 |
|------|---------------|------|
| 静态单层删除方案 | 15.45 | 按独立重要性选两层 |
| 重新计算方案 | 15.45 | 删一层后重新评估 |
| **最优组合** | **15.43** | 考虑层间交互的全局最优 |

### 关键发现
- 在激进剪枝（12层）下本文方法优势最大，PPL远低于基线（38.12 vs 58.12-304.52）
- 方法可扩展到非Transformer架构（RWKV-7B、Mamba-2.8B）
- 可与量化技术无缝结合获得额外效率提升
- 在8个零样本基准上也表现出一致的准确率优势

## 亮点与洞察

- **博弈论视角新颖**：将层剪枝从独立评估转向考虑层间协作的全局优化
- **实用性强**：代理网络训练开销很小，且一旦训练好即可快速评估大量掩码组合
- **层重要性的动态性**：Table 1清楚展示了静态排名的局限性——最不重要的两层组合并非最优

## 局限与展望

- 代理网络的泛化能力依赖于阶段一采样的多样性和数量
- 当前仅考虑PPL作为效用函数，未考虑下游任务特定的性能指标
- 分层采样策略中各层预算分配较为均匀，可能存在更优的自适应分配方案

## 相关工作与启发

- GTAP方法（Diaz-Ortiz Jr et al., 2023）首先将博弈论用于神经元重要性，但受限于计算复杂度
- ShortGPT（Men et al., 2024）的Block Influence (BI)指标是本文的重要对比基线
- 代理网络的思路可推广到其他需要大量评估的NAS和超参搜索场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 合作博弈视角对层剪枝问题的建模很有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型、多基准、含消融和非Transformer验证
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，Table 1很有说服力
- 价值: ⭐⭐⭐⭐ 为LLM压缩提供了新的系统性方法论

<!-- RELATED:START -->

## 相关论文

- [Towards Strategic Persuasion with Language Models](towards_strategic_persuasion_with_language_models.md)
- [Efficient Estimation of Kernel Surrogate Models for Task Attribution](efficient_estimation_of_kernel_surrogate_models_for_task_attribution.md)
- [AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](awm_accurate_weight-matrix_fingerprint_for_large_language_models.md)
- [VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [Robust Multi-Objective Controlled Decoding of Large Language Models](robust_multi-objective_controlled_decoding_of_large_language_models.md)

<!-- RELATED:END -->
