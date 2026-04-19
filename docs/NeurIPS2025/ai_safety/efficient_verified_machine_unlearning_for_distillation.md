---
title: >-
  [论文解读] Efficient Verified Machine Unlearning for Distillation
description: >-
  [NeurIPS 2025][AI安全][machine unlearning] 提出 PURGE 框架，通过教师-学生 constituent mapping 和增量式多教师蒸馏策略，将 SISA 的验证式遗忘扩展到知识蒸馏场景，在教师端遗忘时仅需部分重训学生模型，实现至少 $N\times$ 的加速。
tags:
  - NeurIPS 2025
  - AI安全
  - machine unlearning
  - 知识蒸馏
  - SISA
  - data privacy
  - ensemble learning
---

# Efficient Verified Machine Unlearning for Distillation

**会议**: NeurIPS 2025  
**arXiv**: [2503.22539](https://arxiv.org/abs/2503.22539)  
**代码**: 无  
**领域**: AI Safety / 机器遗忘  
**关键词**: machine unlearning, knowledge distillation, SISA, data privacy, ensemble learning

## 一句话总结
提出 PURGE 框架，通过教师-学生 constituent mapping 和增量式多教师蒸馏策略，将 SISA 的验证式遗忘扩展到知识蒸馏场景，在教师端遗忘时仅需部分重训学生模型，实现至少 $N\times$ 的加速。

## 研究背景与动机

**领域现状**：GDPR/CCPA 等隐私法规赋予用户"被遗忘权"，要求从训练好的模型中移除特定数据的影响。SISA (Sharded, Isolated, Sliced, Aggregated) 是目前最具代表性的验证式遗忘框架，通过数据分片和检查点实现精确遗忘。

**现有痛点**：在知识蒸馏（KD）场景下，教师模型的知识通过 soft label 渗透到整个学生网络。即使教师和学生各自独立使用 SISA，教师端的遗忘仍会迫使学生网络**全量重训**——因为每个学生 constituent 都间接接触了完整教师集合的信息。

**核心矛盾**：SISA 的效率依赖于数据隔离性（data isolation），但标准蒸馏过程破坏了这种隔离——教师集合作为整体提供监督信号，学生的每个分片都受到全部教师数据的影响。

**本文目标**：如何在 KD 流水线中实现高效的验证式遗忘，特别是当遗忘请求针对教师训练数据时？

**切入角度**：如果将教师 constituent 和学生 constituent 之间建立严格的映射关系，每个教师只影响特定学生子集，就能在蒸馏过程中保持数据隔离。

**核心 idea**：通过 constituent mapping 将教师影响限制在特定学生分片内，并用增量式多教师策略替代全教师集合蒸馏，从而在 KD 中恢复 SISA 的遗忘效率。

## 方法详解

### 整体框架
PURGE (Partitioned Unlearning with Retraining Guarantee for Ensembles) 框架包含五个关键步骤：
- **输入**：教师集合 $\{T_1, \dots, T_M\}$（已用 SISA 训练），学生数据 $\mathcal{D}^S$
- **输出**：具备高效遗忘能力的学生集合 $\{S_1, \dots, S_N\}$
- **流程**：分片(Sharding) → 映射(Mapping) → 分块(Chunking) → 增量蒸馏 → 切片训练(Slicing) → 聚合推理

### 关键设计

1. **Constituent Mapping（教师-学生映射）**:

    - 功能：将 $M$ 个教师 constituent 划分为 $N$ 个不相交的子集 $\mathscr{T}_k$，每个学生 $S_k$ 只从自己的教师子集学习
    - 核心思路：$\mathscr{T}_k = \{T_{k,1}, \dots, T_{k,c_k}\}$，满足 $\cap_{k} \mathscr{T}_k = \emptyset$ 且 $\cup_k \mathscr{T}_k = \{T_1, \dots, T_M\}$
    - 设计动机：严格隔离使得教师 $T_{k,i}$ 的遗忘只影响对应学生 $S_k$，不波及其他学生

2. **Incremental Multi-Teacher Distillation（增量式多教师蒸馏）**:

    - 功能：每个学生分片的数据被进一步分成 $c_k$ 个 chunk，第 $l$ 个 chunk 使用前 $l$ 个教师的子集合生成 soft label
    - 核心思路：$Y_{k,l} = \mathscr{T}_{k,l}(\mathcal{D}^S_{k,l})$，其中 $\mathscr{T}_{k,l} = \cup_{i \in [l]} T_{k,i}$
    - 设计动机：限制每个教师的影响范围——教师 $T_{k,i}$ 仅影响 chunk $i$ 及之后的数据。相比使用单一教师（single-teacher ablation），增量集合平滑了监督信号的跳变，避免训练不稳定

3. **Hierarchical Slicing with Checkpointing（层次化切片与检查点）**:

    - 功能：每个 chunk 再细分为多个 slice，训练按 chunk→slice 顺序增量进行，每个 slice 训练后存储检查点
    - 核心思路：学生 $S_k$ 在状态 $S_{k,l,j}$ 上用累积数据 $(\cup_{i=1}^{l-1} \mathcal{D}^\dagger_{k,i}) \cup (\cup_{q=1}^j \mathcal{D}^\dagger_{k,l,q})$ 训练 $e_{l,j}$ 个 epoch
    - 设计动机：分层结构（shard→chunk→slice）提供细粒度检查点，遗忘时只需回退到受影响的最早检查点

### 遗忘过程

- **学生端遗忘**：移除数据点 $d_u \in \mathcal{D}^\dagger_{k,l,j}$ 时，回退到 $S_{k,l,j-1}$，从该 slice 开始部分重训，继承 SISA 效率
- **教师端遗忘**：教师 $T_{k,l}$ 遗忘后需更新 soft label（chunk $l$ 到 $c_k$），学生回退到 $S_{k,l-1}$ 并从 chunk $l$ 开始重训，仅影响 $S_k$ 一个 constituent

### 理论加速分析

在均匀分配条件下（$c = M/N$ chunks/student, $r$ slices/chunk），PURGE 相对 naive SISA 的加速比为：

$$\frac{t_{\text{sisa}}}{t_{\text{PURGE}}} = N \cdot \frac{6c^2r + 6c}{4c^2r + 3cr + 3c - r + 3}$$

第二项对所有正整数 $r, c$ 大于 1，因此加速比至少为 $N\times$。当 $c=1$（$N=M$）时加速最大。

## 实验关键数据

### 主实验：遗忘速度评估（MNIST, $M=32$, 100次教师端遗忘请求）

| 学生数 $N$ | 配置 | 平均重训时间/次 | 加速比 | 理论预测 |
|------------|------|----------------|--------|----------|
| Baseline SISA ($N$=8) | 全量重训 | 737.14±10.08s | 1× | — |
| PURGE $N$=8 | $c$=4, $r$=1 | ~92s | ~8× | ~8× |
| PURGE $N$=16 | $c$=2, $r$=1 | ~46s | ~16× | ~16× |
| PURGE $N$=32 | $c$=1, $r$=1 | 23.17±0.17s | ~32× | 32× |

### 性能评估（准确率对比）

| 数据集 | 方法 | $M$=32, $N$=32 准确率 | $M$=32, $N$=1 准确率 |
|--------|------|---------------------|---------------------|
| MNIST | Teacher | ~98.5% | ~98.5% |
| MNIST | SISA Baseline | 97.08% | 97.30% |
| MNIST | PURGE | 97.16% | 97.16% |
| MNIST | Single-teacher | 95.78% | 95.98% |
| SVHN | SISA Baseline | 83.44% | 83.27% |
| SVHN | PURGE | 83.09% | 83.09% |
| SVHN | Single-teacher | 76.12% | ~76% |

### 关键发现
- PURGE 在各种 $N$ 和 $M$ 配置下准确率与 SISA baseline 几乎持平，最大差异 <0.5%
- Single-teacher ablation 在 $N$=1（$c$=32）时性能严重退化（SVHN 上掉 7.35%），验证了增量多教师策略的必要性
- $r$ 的选择存在权衡：大 $r$ 加速学生端遗忘但减慢教师端遗忘，最优 $r$ 取决于两类遗忘请求的比例
- 实际速度与理论预测高度吻合，$r$=4 时因 ceiling 函数略有偏差

## 亮点与洞察
- **SISA→KD 的桥接思路**非常自然：核心挑战是蒸馏过程破坏数据隔离，而 constituent mapping 以最小代价恢复隔离
- **增量多教师策略**一举两得：既保持隔离（每个教师影响有限 chunk），又平滑学生训练（避免单教师切换导致的信号跳变）
- 理论分析给出了**可操作的配置建议**（$N$, $c$, $r$ 的权衡），对实际部署有参考价值

## 局限与展望
- 假设教师和学生都用 SISA 集合结构，对单一大模型蒸馏场景不适用
- 实验仅覆盖 MNIST/SVHN/CIFAR-100/SST5，缺少大规模语言模型蒸馏的验证
- 增量多教师的 soft label 质量随 chunk 推进而提升，前面 chunk 的标签质量较低
- 未讨论当教师和学生同时收到遗忘请求的复杂交互场景（仅在附录简略提及）

## 相关工作与启发
- **vs SISA**: SISA 在单模型场景高效，但无法处理 KD 中的信息传播问题；PURGE 通过结构化映射解决
- **vs SCRUB**: SCRUB 让学生"违背"教师来近似遗忘，无正式保证；PURGE 提供精确遗忘
- **vs RKLD**: RKLD 需要额外的"干净"参考教师，PURGE 无此要求

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个在 KD 场景中实现验证式遗忘的框架，constituent mapping 思路简洁有效
- 实验充分度: ⭐⭐⭐⭐ 速度和性能评估全面，理论与实验吻合良好，但数据集规模偏小
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，理论推导完整，图示直观
- 价值: ⭐⭐⭐⭐ 填补了 KD+verified unlearning 的空白，对隐私合规下的模型部署有实际意义

<!-- RELATED:START -->

## 相关论文

- [Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions](rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)
- [Position: Bridge the Gaps between Machine Unlearning and AI Regulation](position_bridge_the_gaps_between_machine_unlearning_and_ai_regulation.md)
- [The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples](the_unseen_threat_residual_knowledge_in_machine_unlearning_under_perturbed_sampl.md)
- [Machine Unlearning Doesn't Do What You Think: Lessons for Generative AI Policy and Research](machine_unlearning_doesnt_do_what_you_think_lessons_for_generative_ai_policy_and.md)
- [Self-Refining Language Model Anonymizers via Adversarial Distillation](self-refining_language_model_anonymizers_via_adversarial_distillation.md)

<!-- RELATED:END -->
