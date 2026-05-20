---
title: >-
  [论文解读] MIB: A Mechanistic Interpretability Benchmark
description: >-
  [ICML 2025][可解释性][机械可解释性] 提出 MIB（Mechanistic Interpretability Benchmark），包含电路定位和因果变量定位两个赛道、四个任务、五个模型，通过标准化的反事实干预评估和新指标（CPR/CMD）系统比较 MI 方法…
tags:
  - "ICML 2025"
  - "可解释性"
  - "机械可解释性"
  - "benchmark"
  - "电路定位"
  - "因果变量"
  - "SAE"
  - "attribution patching"
---

# MIB: A Mechanistic Interpretability Benchmark

**会议**: ICML 2025  
**arXiv**: [2504.13151](https://arxiv.org/abs/2504.13151)  
**代码**: [有](https://github.com/)  
**领域**: 可解释性  
**关键词**: 机械可解释性, benchmark, 电路定位, 因果变量, SAE, attribution patching

## 一句话总结

提出 MIB（Mechanistic Interpretability Benchmark），包含电路定位和因果变量定位两个赛道、四个任务、五个模型，通过标准化的反事实干预评估和新指标（CPR/CMD）系统比较 MI 方法，发现 attribution + mask optimization 方法在电路定位中最优，而 SAE 特征在因果变量定位中并不优于原始神经元。

## 研究背景与动机

**现状**：机械可解释性（MI）方法快速增长，用于理解语言模型内部行为的因果路径和关键概念。但不同方法之间缺乏标准化的比较基准。

**痛点**：新方法通常使用临时性评估指标和不同任务进行比较，导致无法判断是否真正取得了进步。faithfulness 指标被用于两个不同目标：(i) 找到对任务有正面贡献的组件 vs (ii) 找到对任务有任何显著影响的组件。

**核心矛盾**：MI 领域缺乏统一的评估标准来跨方法比较，现有 benchmark 要么只比较特定类别的方法，要么只针对特定任务和模型。

**切入点**：构建跨方法、跨任务、跨模型的标准化 benchmark，提供固定的反事实输入和标准化指标。

**核心 idea**：将 faithfulness 拆分为两个互补指标（CPR 关注正面影响、CMD 关注全部影响），在多个电路大小下积分（AUC 思想），消除阈值超参数的影响。

## 方法详解

### 整体框架 (pipeline)

MIB 包含两个赛道：
- **电路定位赛道**：评估方法发现对特定任务最重要的模型组件子图（电路）的能力
- **因果变量定位赛道**：评估方法将隐向量特征化（如 SAE/DAS）并对齐到任务相关因果变量的能力

每个赛道覆盖 4 个任务 × 多个模型，使用标准化反事实输入进行干预评估。

### 关键设计

1. **两个新指标 CPR 和 CMD**：

    - **CPR（Integrated Circuit Performance Ratio）**：faithfulness 曲线关于电路大小的 AUC，衡量方法找到正面贡献组件的能力，越高越好
    - **CMD（Integrated Circuit-Model Distance）**：faithfulness 曲线与 $f=1$ 之间面积，衡量电路行为与完整模型的差距，越小越好
    - 核心思路：$f(\mathcal{C}, \mathcal{N}; m) = \frac{m(\mathcal{C}) - m(\emptyset)}{m(\mathcal{N}) - m(\emptyset)}$，在 10 个不同的电路大小比例下计算 faithfulness，梯形法则积分
    - 设计动机：消除阈值 $\lambda$ 对方法比较的影响，同时捕获 minimality（小电路 → 高 faithfulness）

2. **加权边计数（Weighted Edge Count）**：统一不同粒度电路（边级 vs 神经元级）的大小度量。$|\mathcal{C}| = \sum_{(u,v) \in \mathcal{C}} \frac{|N_u \cap N_\mathcal{C}|}{|N_u|}$。设计动机：包含一个 submodule 的一个神经元等价于包含该 submodule 的 $1/d_{\text{model}}$ 的出边。

3. **四个任务设计**：IOI（间接宾语识别，经典 MI 任务）、Arithmetic（加减法）、MCQA（多选题，合成数据）、ARC（真实科学问题）。前两个是已被广泛研究的任务（验证已有进步），后两个是未被研究的（防止过度刷分）。每个任务都有固定的反事实输入映射。

4. **因果变量定位赛道**：用户提交特征化方法（如 SAE、DAS、PCA）将隐向量映射到新空间，评估交换干预是否能精确操控特定因果变量。使用 IIA（Interchange Intervention Accuracy）指标。

### 损失函数/训练策略

MIB 本身是评估框架，不包含训练。但 benchmark 中的 InterpBench 模型（已知真实电路的合成模型）使用标准训练。UGS 等 mask 方法使用 KL 散度 + L1 稀疏化联合优化。

## 实验关键数据

### 主实验 — 电路定位 CMD 得分（越低越好）

| 方法 | IOI(GPT-2) | IOI(Qwen) | Arithmetic(Llama) | MCQA(Qwen) | ARC-E(Gemma) | ARC-C(Llama) |
|------|-----------|-----------|-------------------|-----------|-------------|-------------|
| Random | 0.75 | 0.72 | 0.74 | 0.73 | 0.68 | 0.74 |
| EAP (CF) | 0.03 | 0.15 | **0.01** | 0.07 | 0.04 | 0.18 |
| EAP-IG-inp (CF) | **0.03** | **0.02** | **0.01** | 0.08 | **0.04** | 0.22 |
| EAP-IG-act (CF) | **0.03** | **0.01** | **0.01** | **0.05** | **0.04** | 0.37 |
| NAP (CF) | 0.38 | 0.33 | 0.29 | 0.30 | 0.33 | 0.69 |
| UGS | **0.03** | **0.03** | - | 0.20 | - | - |
| IFR | 0.42 | 0.69 | 0.83 | 0.60 | 0.66 | 0.76 |

### 消融实验 — 因果变量定位（IIA 得分）

| 方法 | 特征化 | IOI | Arithmetic | 特点 |
|------|--------|-----|-----------|------|
| DAS | 监督旋转 | 最高 | 最高 | 需要标注 |
| SAE | 无监督稀疏 | 中等 | 中等 | 不优于神经元 |
| Neuron (Probe) | 无特征化 | 中等 | 中等 | 基线 |
| PCA | 无监督线性 | 较低 | 较低 | 简单基线 |

### 因果变量定位 — IIA 关键结果

| 方法 | 特征化类型 | ARC-E Gemma | ARC-E Llama | 特点 |
|------|-----------|------------|------------|------|
| DAS | 监督旋转方向 | 88 (best:94) | 88 (best:99) | 需因果模型标注 |
| DBM+SAE | 无监督+掩码 | 82 (best:99) | 中等 | SAE≈neurons |
| Full Vector | 无特征化 | 中等 | 中等 | 粗粒度干预 |
| PCA | 无监督线性 | 较低 | 较低 | 简单基线 |

### 关键发现

- **边级 attribution 方法（EAP-IG）在电路定位中表现最好**，尤其是使用反事实消融时
- **精确 activation patching 并非总是最优**——因使用样本数少或独立边评估的局限（如 Qwen 上 EActP 不及 EAP-IG）
- **SAE 特征在因果变量定位中不优于原始神经元**——DBM 选择 SAE 特征的 IIA 与选择标准神经元接近
- **监督方法（DAS）在因果变量定位中显著领先**——DAS 在 ARC-Easy Gemma 上达 94% IIA（best layer）
- 节点级电路表现差——因为每个节点"花费"太多边
- 反事实消融 > 均值消融 ≈ 最优消融
- IFR（非因果方法）虽优于随机但远逊因果方法，验证了因果分析的必要性
- UGS mask 方法因直接优化 KL 散度在 CMD 上表现好，但 CPR 不占优

## 亮点与洞察

- CPR/CMD 指标的设计思路巧妙——AUC 消除超参数、两个指标分别关注不同分析目标
- SAE ≤ neurons 的发现对当前可解释性社区的热情提出了严肃质疑
- 在已有任务和新任务之间取平衡防止过度刷分
- 公开 leaderboard 接受提交，形成持续追踪进步的机制

## 局限与展望

- 仅 4 个模型（GPT-2, Qwen-0.5B, Gemma-2B, Llama-8B），缺少更大模型
- InterpBench 合成模型的电路可能与真实模型的电路性质不同
- 因果变量定位赛道仅评估已知因果变量，无法评估发现新因果变量的能力
- 部分方法（EActP, UGS）因计算量限制无法在所有模型上运行
- MCQA 和 ARC 的因果模型假设（order ID → answer token）可能过于简化
- 私有测试集的长期维护和防止数据泄露是持续挑战

## 相关工作与启发

- **Wang et al., 2023**（IOI 电路）：最经典的 MI 案例研究，MIB 将其标准化
- **Marks et al., 2025**（AP-IG-activations）：当前最强的 attribution 方法之一
- **Geiger et al., 2024**（DAS/因果抽象）：因果变量定位的理论框架
- **Karvonen et al., 2025**（SAE benchmark）：MIB 扩展了 SAE 评估到更广泛框架
- 启发：MI 领域需要更多像 MIB 这样的标准化评估才能判断真实进步

## 评分

- 新颖性: ⭐⭐⭐⭐ CPR/CMD 指标设计和跨方法比较框架是新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 4 任务、5 模型、10+ 方法、公私测试集
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰、定义严谨、图表丰富
- 价值: ⭐⭐⭐⭐⭐ 对 MI 领域具有基础性意义，SAE 发现有重要影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](../../ACL2025/interpretability/a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)
- [\[NeurIPS 2025\] nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers](../../NeurIPS2025/interpretability/nnterp_a_standardized_interface_for_mechanistic_interpretability_of_transformers.md)
- [\[ACL 2025\] Mechanistic Interpretability of Emotion Inference in Large Language Models](../../ACL2025/interpretability/mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)
- [\[ACL 2025\] An Empirical Study of Mechanistic Interpretability Approaches for Factual Recall](../../ACL2025/interpretability/an_empirical_study_of_mechanistic_interpretability_approaches_for_factual_recall.md)
- [\[NeurIPS 2025\] The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](../../NeurIPS2025/interpretability/the_non-linear_representation_dilemma_is_causal_abstraction_enough_for_mechanist.md)

</div>

<!-- RELATED:END -->
