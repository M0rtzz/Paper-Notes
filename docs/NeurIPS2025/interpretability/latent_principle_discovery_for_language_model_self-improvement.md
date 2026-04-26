---
title: >-
  [论文解读] Latent Principle Discovery for Language Model Self-Improvement
description: >-
  [NeurIPS 2025][自我改进] STaPLe 提出后验正则化的蒙特卡洛 EM 算法，让 7-8B 小模型自行发现指导自我修正的"原则"（latent principle），通过迭代发现-学习循环实现自我改进，在 AlpacaEval 上提升 8-10% 胜率、MT-Bench 平均提升 +0.3，并可通过聚类压缩至可解释的 constitution。
tags:
  - NeurIPS 2025
  - 自我改进
  - 隐式原则发现
  - EM算法
  - Constitutional AI
  - 自我修正
---

# Latent Principle Discovery for Language Model Self-Improvement

**会议**: NeurIPS 2025  
**arXiv**: [2505.16927](https://arxiv.org/abs/2505.16927)  
**代码**: 无  
**领域**: model_compression  
**关键词**: 自我改进, 隐式原则发现, EM算法, Constitutional AI, 自我修正

## 一句话总结
STaPLe 提出后验正则化的蒙特卡洛 EM 算法，让 7-8B 小模型自行发现指导自我修正的"原则"（latent principle），通过迭代发现-学习循环实现自我改进，在 AlpacaEval 上提升 8-10% 胜率、MT-Bench 平均提升 +0.3，并可通过聚类压缩至可解释的 constitution。

## 研究背景与动机

**领域现状**：Constitutional AI 等框架依赖人工策划的静态宪法来引导模型行为，需要专家预先列举所有可能的行为准则；RLHF 依赖人类标注的偏好对来区分好坏回复。

**现有痛点**：(a) 静态宪法无法覆盖不断出现的新场景和边界情况，维护成本高；(b) 偏好对中好坏回复的差异往往是多维度、难以用单一标签刻画的；(c) 小模型（7-8B）的自我修正能力弱，直接提示自我反思效果有限。

**核心矛盾**：模型改进需要明确的行为维度指引，但人工穷举这些维度既昂贵又不完备。

**本文目标** 自动从模型自身生成中发现指导改进的"原则"，并训练模型学会在推理时自主调用这些原则进行自我修正。

**切入角度**：将"原则"视为连接初始回复和目标回复之间的隐变量，用 EM 算法迭代发现和学习。

**核心 idea**：用模型自己采样的"为什么回复B比回复A好"的原则作为隐变量来桥接初始生成和gold回复，EM迭代学习后模型能自动调用原则进行内省式自我改进。

## 方法详解

### 整体框架
STaPLe（Self-Taught Principle Learning）是一个迭代式 Monte Carlo EM 算法：
- **E-step（原则发现）**：给定 prompt $x$、初始回复 $y^1$，以 gold 回复 $y^G$ 为 hint 采样候选原则 $z$，再基于原则生成修正回复 $y^2$，用拒绝采样保留最优（原则, 修正）对
- **M-step（原则学习）**：对收集的轨迹 $(x, y^1, z, y^2)$ 做 SFT，训练模型学会生成原则并据此修正
- 多轮迭代后模型能在推理时自主发现原则并自我改进

### 关键设计

1. **隐变量建模与边际似然优化**:

    - 功能：将原则 $z$ 建模为从初始回复到目标回复的隐推理链
    - 核心公式：$\mathcal{L}(\theta) = \log \sum_{y^2} \sum_{z} p(y^G | x, y^1, z, y^2) \cdot p(y^2, z | x, y^1; \theta)$
    - 其中 $p(y^G | x, y^1, z, y^2)$ 是无参数验证器，基于字符串相似度：当 $f(y^2, y^G) > f(y^1, y^G)$ 时正比于 $f(y^2, y^G) - f(y^1, y^G)$
    - 梯度推导简化为对后验 $p(y^2, z | x, y^1, y^G)$ 的期望

2. **循环一致性拒绝采样（E-step）**:

    - 功能：从后验中高效采样好的（原则, 修正）对
    - 核心思路：用循环一致性评分替代直接 Bayes 后验：$p(y^2, z | x, y^1, y^G; \theta) \propto p(y^G | x, y^1, z, y^2) \cdot \tilde{p}(y^2, z | x, y^1, y^G; \theta)$
    - 关键设计：将提案分布分解为 $\tilde{p}(y^2, z | x, y^1, y^G; \theta) = p(y^2 | x, y^1, z; \theta) \cdot p(z | x, y^1, y^G; \theta)$，即 gold 回复仅在原则生成阶段可见，修正阶段不可见，避免模型直接抄写 gold 回复
    - 使用 Best-of-N 策略保留最优样本（温度→0的 hard-EM）
    - 相似度函数使用 Rouge-L F1

3. **后验正则化与层次聚类**:

    - 功能：将数千个去重原则压缩为可解释的 constitution
    - 核心思路：定义约束 $g(z) = \mathbf{1}(z \notin \tilde{Z})$，要求后验仅在簇代表元素上有质量：$\tilde{p}(y^2, z | x, y^1, y^G) \propto p(\cdot) \cdot \exp(-\lambda g(z))$
    - 实际操作：使用层次聚类（agglomerative clustering）对 all-MiniLM-L6-v2 编码的原则嵌入聚类，取 medoid 作为簇代表
    - 设计动机：保持原则多样性的同时消除语义冗余，聚类后原则在迭代中逐渐收敛到稳定集合

4. **迭代自我改进**:

    - 功能：多轮 EM 交替实现持续改进
    - 实践细节：第1轮用50k样本大量 bootstrap，后续每轮10k样本，确保每轮 prompt 互不重复
    - 输入语料：Anthropic HH-RLHF、UltraFeedback、TL;DR、HotpotQA 各25k，覆盖多领域

### 训练策略
- M-step 为标准 SFT，训练目标：$\theta^{(t+1)} = \arg\max_\theta \mathbb{E}_{(x,y^1,z,\hat{y}^2) \in \mathcal{D}'} [\log p(y^2, z | x, y^1; \theta)]$
- 每轮 EM 的数据仅在 E-step 中收集修正有效果的样本（$f(y^2, y^G) > f(y^1, y^G)$），无效修正被丢弃
- 推理时模型自主决定是否调用原则：若初始回复已足够好则直接输出，否则生成原则+修正

## 实验关键数据

### 主实验（4轮迭代后）

| 模型 | 方法 | MT-Bench (Avg) | MT-Bench (T2) | AlpacaEval | IFEval WR |
|------|------|---------------|--------------|------------|-----------|
| Llama-3.1-8B | Initial Policy | 7.46 | 6.83 | 26.9 | — |
| | Self-Refine | 7.40 | 6.75 | 26.1 | 51.2% |
| | STaR Iter 4 | 7.56 | 7.00 | 31.8 | 62.3% |
| | **STaPLe Iter 4** | **7.71** | **7.30** | **33.4** | **68.9%** |
| | Constrained STaPLe | 7.70 | 7.28 | 34.9 | 69.1% |
| Granite-3.1-8B | Initial Policy | 7.83 | 7.08 | 30.2 | — |
| | **STaPLe Iter 4** | **8.04** | **7.41** | **38.4** | **67.6%** |
| Qwen2.5-7B | Initial Policy | 6.83 | 6.31 | 30.4 | — |
| | **STaPLe Iter 4** | **7.24** | **6.85** | **40.2** | **73.4%** |

STaPLe 在三个模型上一致超越 STaR 和 Self-Refine 基线，AlpacaEval 提升 8-10%。

### 聚类版本 vs 完整版本对比

| 模型 | STaPLe (Full) AlpacaEval | Constrained AlpacaEval | STaPLe (Full) IFEval | Constrained IFEval |
|------|-------------------------|----------------------|---------------------|-------------------|
| Llama-8B | 33.4 | **34.9** | 68.9% | **69.1%** |
| Granite-8B | 38.4 | **38.8** | 67.6% | **68.4%** |
| Qwen-7B | **40.2** | 39.9 | **73.4%** | 72.1% |

聚类版本性能基本持平甚至略优，同时提供了可解释的 constitution。

### 关键发现
- MT-Bench Turn 2（多轮对话追问）提升尤为明显（平均+0.22），说明自我修正能力确实增强
- 原则发现率（新原则/总样本）随迭代递减，说明模型逐渐收敛到稳定原则集合
- Constitution 大小在第4轮约为第1轮的50%以下，模型学会了复用原则
- Self-Refine 在 Llama-8B 上甚至降低性能（7.40 < 7.46），说明小模型的零样本自我修正不可靠
- 聚类加速了原则集合的收敛

## 亮点与洞察
- **原则作为隐变量的形式化**：将 Constitutional AI 中人工编写的规则自动化为 EM 中的隐变量推断，理论框架优雅。这种"让模型自己发现应该遵循什么规则"的思路极具启发性
- **聚类≈不损失性能**：后验正则化通过聚类将数千原则压缩到几十个，不仅保持性能甚至略有提升，同时实现了可解释性——可以直接查看模型学到了哪些"宪法条款"
- **小模型的自我改进范式**：在不依赖强模型（GPT-4）监督的情况下，7-8B 模型通过自身采样实现持续改进，降低了对外部标注的依赖
- **分解式提案避免作弊**：原则生成看 gold、修正生成不看 gold 的巧妙分解，有效防止模型直接复制答案

## 局限与展望
- 迭代3-4轮后出现边际递减，Llama-8B 和 Granite-8B 在第4轮略有回落，持续改进能力有上限
- 依赖 Gold 回复作为质量信号，对于没有明确参考答案的开放式任务，需要替代的验证机制
- Rouge-L 作为相似度函数较粗糙，更好的语义相似度度量可能提升 E-step 质量
- 聚类参数（距离阈值、合并策略）的选择缺乏系统性指导
- 仅在指令跟随/对话任务验证，数学推理、代码生成等有 verifiable reward 的领域待探索

## 相关工作与启发
- **vs Constitutional AI**: CAI 用人工编写的静态宪法，STaPLe 自动发现动态宪法；CAI 需要 RL 阶段，STaPLe 仅用 SFT
- **vs Self-Refine**: Self-Refine 在推理时用零样本提示自我修正，不训练；STaPLe 通过训练让模型内化修正能力，且显式学习原则使修正有方向性
- **vs STaR**: STaR 学习修正后的回复但不学习原则；STaPLe 显式建模原则作为中间推理链，MT-Bench、AlpacaEval 上持续领先STaR +0.1-0.15分
- **vs SCoRe/RISE**: 这些方法用 RL 训练自我修正，需要可验证奖励；STaPLe 通过相似度函数适用于不可验证的开放式生成

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将原则建模为隐变量并用 EM 迭代发现、后验正则化压缩到可解释集合，整套框架系统性强且新颖
- 实验充分度: ⭐⭐⭐⭐ 三个模型、多个基准、消融+聚类分析全面，但缺少推理/代码领域验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰严谨，方法动机和设计决策解释充分
- 价值: ⭐⭐⭐⭐ 为小模型自我改进提供了可行框架，聚类产出的可解释 constitution 有独立价值

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)
- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](probabilistic_token_alignment_for_large_language_model_fusion.md)
- [\[NeurIPS 2025\] Encoding and Understanding Astrophysical Information in Large Language Model-Generated Summaries](encoding_and_understanding_astrophysical_information_in_large_language_model-gen.md)
- [\[NeurIPS 2025\] Steering Information Utility in Key-Value Memory for Language Model Post-Training](steering_information_utility_in_key-value_memory_for_language_model_post-trainin.md)
- [\[NeurIPS 2025\] H-SPLID: HSIC-based Saliency Preserving Latent Information Decomposition](h-splid_hsic-based_saliency_preserving_latent_information_decomposition.md)

<!-- RELATED:END -->
