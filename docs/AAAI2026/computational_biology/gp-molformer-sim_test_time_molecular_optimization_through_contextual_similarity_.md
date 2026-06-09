---
title: >-
  [论文解读] GP-MoLFormer-Sim: Test Time Molecular Optimization through Contextual Similarity Guidance
description: >-
  [AAAI 2026][计算生物][分子优化] 提出 GP-MoLFormer-Sim，一种无需训练的测试时分子生成引导方法：利用化学语言模型（GP-MoLFormer）自身的上下文嵌入计算与目标分子的相似度，在自回归解码时动态调整logits来引导生成…
tags:
  - "AAAI 2026"
  - "计算生物"
  - "分子优化"
  - "化学语言模型"
  - "测试时引导"
  - "遗传算法"
  - "SMILES"
  - "相似性引导生成"
---

# GP-MoLFormer-Sim: Test Time Molecular Optimization through Contextual Similarity Guidance

**会议**: AAAI 2026  
**arXiv**: [2506.05628](https://arxiv.org/abs/2506.05628)  
**代码**: 无公开链接  
**领域**: AI for Science / 药物发现  
**关键词**: 分子优化, 化学语言模型, 测试时引导, 遗传算法, SMILES, 相似性引导生成

## 一句话总结
提出 GP-MoLFormer-Sim，一种无需训练的测试时分子生成引导方法：利用化学语言模型（GP-MoLFormer）自身的上下文嵌入计算与目标分子的相似度，在自回归解码时动态调整logits来引导生成，结合遗传算法（GP-MoLFormer-Sim+GA）后在PMO基准的23个任务上平均排名第2，且在黑盒oracle设定下优于依赖GPT-4的MOLLEO。

## 研究背景与动机

**领域现状**：分子优化是药物发现和材料设计的核心问题，需要在巨大的分子空间中搜索满足特定性质约束的分子。现有方法包括强化学习、深度生成模型（VAE、扩散模型）、GFlowNet和遗传算法。近年来研究表明传统遗传算法在分子优化中极具竞争力。

**现有痛点**：
   - 将GA与深度学习结合的方法通常需要**重新训练**生成模型以适应特定优化任务
   - MOLLEO等方法依赖GPT-4生成候选分子，但GPT-4的提示中包含了oracle的任务信息（如目标分子名称），破坏了黑盒oracle的公平性，且调用成本高
   - 强化学习方法需要训练reward模型或策略网络

**核心矛盾**：如何在不重新训练的情况下，利用预训练化学语言模型进行有针对性的分子生成？

**切入角度**：利用化学语言模型**自身**的嵌入空间来估计生成分子与目标分子的相似度，在测试时"倾斜"（tilt）解码概率，无需任何额外训练。

## 方法详解

### 整体框架

**GP-MoLFormer-Sim**（引导生成模块）+ **遗传算法**（GA搜索循环）= **GP-MoLFormer-Sim+GA**

### GP-MoLFormer 基座模型
- GPT风格的自回归解码器，使用线性注意力和旋转位置编码
- 约47M参数，在~650M canonicalized SMILES（ZINC+PubChem）上训练
- 无条件采样可探索化学空间

### 核心算法：相似性引导生成（Algorithm 1）

在每一步自回归解码中：

1. **嵌入当前候选**：对词汇表中每个可能的下一个token $i$，计算拼接后序列 $s \oplus i$ 的GP-MoLFormer嵌入 $x_i$
2. **嵌入目标分子**：对每个目标分子 $m_j$，计算其前缀 $m_j[1:t]$ 的嵌入 $y_j$
3. **计算余弦相似度**：$S_{ij} = \langle x_i, y_j \rangle$，对所有目标取平均 $\bar{S}_i$
4. **混合logits**：$u \leftarrow \frac{1}{\tau}((1-\alpha)u + \alpha\bar{S})$

其中 $\alpha \in [0,1]$ 控制引导强度（0=无条件生成，1=纯相似性采样），$\tau$ 控制采样温度。

**理论保证**：该过程等价于求解一个优化问题——最大化对目标分子KDE的相似度，同时通过KL散度约束不过度偏离GP-MoLFormer的原始分布。

**可选增强**：
- 随机傅里叶特征（RFF）：用高斯核密度估计替代余弦相似度，增加局部性控制
- 计算高效：引导生成每个token仅比无条件生成慢4倍（0.049s vs 0.013s）

### GP-MoLFormer-Sim+GA（Algorithm 2）

GA循环过程（见Figure 1）：
1. **选择引导分子**（A）：从当前分子池选Top-G高分分子 + 多样性候选
2. **引导生成**（B）：用GP-MoLFormer-Sim为每个引导分子生成近邻候选（变异）
3. **剪枝/增强**（C）：过滤Tanimoto相似度低于阈值的候选；可选图操作交叉
4. **评分**：用oracle评估新候选，加入分子池
5. **循环**直到oracle预算耗尽

### 理论解释

引导概率分布是如下优化问题的闭式解：
$$\max_{p \in \Delta_V} (1-\alpha) \sum_i p_i \pi_{KDE}^{target}(x_i|c) - KL(p \| \pi_{ref}^\alpha(\cdot|c))$$

即在保持接近GP-MoLFormer原始分布的同时，最大化目标分子KDE的相似度。

## 实验

### 实验1：相似性引导分子生成

目标：生成与5个胰蛋白酶抑制剂目标分子高度相似的分子

| 方法 | top-1 Tsim | top-10 Tsim | top-100 Tsim | top-1000 Tsim |
|------|-----------|------------|-------------|--------------|
| **GPMFS (Ours)** | **1.000** | **0.972** | **0.877** | **0.763** |
| S Model (RL-tuned) | 0.694 | 0.618 | 0.554 | 0.499 |
| Random Gen. | 0.438 | 0.391 | 0.348 | 0.290 |
| Random Search | 0.477 | 0.450 | 0.417 | 0.377 |

- GP-MoLFormer-Sim 在所有k值下显著优于所有基线
- top-1可达完美匹配（Tsim=1.0），top-10000中有132个分子 QED>0.7

### 实验2：PMO基准（23个分子优化任务）

| 方法 | 平均排名 | 平均AUC | 需要训练？ | 需要LLM调用？ |
|------|---------|---------|-----------|-------------|
| MOLLEO (GPT-4) | **1** | **0.777** | 否 | 是（$$$） |
| **GPMFS+GA** | **2** | **0.662** | **否** | **否** |
| Mol-GA | 3 | 0.639 | 否 | 否 |
| Graph-GA | 4 | 0.597 | 否 | 否 |
| STONED SELFIES | 5 | 0.566 | 否 | 否 |
| SynNet | 6 | 0.499 | 否 | 否 |

- GPMFS+GA 在3个任务上（GSK3、JNK3、ranolazine_mpo）**超越所有基线包括MOLLEO**
- 在另外9个任务上排名第2

### 实验3：黑盒Oracle公平性分析（核心贡献）

| 任务 | GPMFS+GA | MOLLEO (隐去分子名) | MOLLEO (含分子名) |
|------|----------|-------------------|-----------------|
| thiothixene重发现 | **0.504** | 0.462 | 0.692 |
| mestranol相似度 | **0.658** | 0.644 | 0.983 |

- 隐去目标分子名称后，MOLLEO性能**下降~33%**，不及GPMFS+GA
- 证明MOLLEO的高性能部分来自GPT-4对分子知识的记忆（如GPT-4知道thiothixene的SMILES）
- 反转实验：给GPMFS+GA提供目标SMILES后，平均排名从2.7升至1.7

### 消融实验

| 配置 | 平均排名 | 平均AUC |
|------|---------|---------|
| 纯引导生成(GG) | 5.0 | 0.603 |
| +交叉(XO) | 3.4 | 0.672 |
| +RFF768 | 5.3 | 0.597 |
| +XO+DIV | 3.0 | 0.678 |
| +RFF768+XO | 2.5 | 0.682 |
| **+RFF768+XO+DIV** | **1.8** | **0.690** |

- 交叉操作(XO) 贡献最大，多样性增强(DIV)进一步提升
- RFF 单独使用无效，但与XO结合有增益

## 亮点与洞察

1. **测试时引导的优雅设计**：不需要训练reward模型、不需要策略梯度、不需要调用外部LLM——仅利用模型自身的嵌入空间作为相似度信号
2. **理论扎实**：算法有严格的优化问题解释，不是简单的启发式
3. **黑盒公平性揭露MOLLEO的"作弊"**：证明MOLLEO利用了GPT-4对已知分子的记忆，这一发现对领域评估标准有重要意义
4. **成本效率**：47M参数的小模型 vs GPT-4的API调用费用，在公平对比下性能更好
5. **域无关性**：方法框架理论上可推广到任何自回归语言模型和任何序列优化任务

## 局限性

1. **仅用GP-MoLFormer一种CLM测试**：方法声称模型无关，但未在其他化学语言模型上验证
2. **仅测试单目标优化**：未扩展到多目标优化（如同时优化结合亲和力、溶解度、药物相似性）
3. **未利用"负样本"引导**：即利用低分分子作为负引导来主动远离不良区域
4. **词汇表迭代计算**：每步需为词汇表中所有token计算嵌入（2362个），序列较长时开销增大
5. **合成可行性未充分讨论**：生成的分子是否可实际合成未做深入分析（仅在附录报告SA分数）

## 相关工作

- **分子优化方法**：RL-based (REINVENT), VAE (JT-VAE), Bayesian Optimization, GFlowNets, 扩散模型
- **GA方法**：Graph-GA, STONED, Mol-GA, GEAM, MOLLEO (GPT-4辅助GA)
- **测试时引导LLM**：Reward-guided Decoding, SASA (自约束采样), Conditional Activation Steering
- **化学语言模型**：GP-MoLFormer, MoLFormer, BioT5, MoleculeSTM

## 评分 ⭐⭐⭐⭐

方法优雅且有理论保证，黑盒公平性分析是重要贡献，无需训练即可实现有竞争力的分子优化。但在完整PMO排行中仅排第2，且未做多目标优化和多CLM验证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design](../../ICML2026/computational_biology/constrained_flow_optimization_via_sequential_fine_tuning_for_molecular_design.md)
- [\[AAAI 2026\] EPO: Diverse and Realistic Protein Ensemble Generation via Energy Preference Optimization](epo_diverse_and_realistic_protein_ensemble_generation_via_energy_preference_opti.md)
- [\[AAAI 2026\] MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging](mergedna_context-aware_genome_modeling_with_dynamic_tokenization_through_token_m.md)
- [\[ICLR 2026\] DriftLite: Lightweight Drift Control for Inference-Time Scaling of Diffusion Models](../../ICLR2026/computational_biology/driftlite_lightweight_drift_control_for_inference-time_scaling_of_diffusion_mode.md)
- [\[ICLR 2026\] A Genetic Algorithm for Navigating Synthesizable Molecular Spaces](../../ICLR2026/computational_biology/a_genetic_algorithm_for_navigating_synthesizable_molecular_spaces.md)

</div>

<!-- RELATED:END -->
