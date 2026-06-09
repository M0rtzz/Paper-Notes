---
title: >-
  [论文解读] R²ec: Towards Large Recommender Models with Reasoning
description: >-
  [NeurIPS 2025][推荐系统][大语言模型推理] 提出R²ec，首个将推理能力内生地集成到推荐模型中的统一大推荐模型，通过双头架构实现推理链生成与高效物品预测的一体化，并设计RecPO强化学习框架在无推理标注数据下联合优化推理与推荐目标。
tags:
  - "NeurIPS 2025"
  - "推荐系统"
  - "大语言模型推理"
  - "强化学习"
  - "双头架构"
  - "测试时缩放"
---

# R²ec: Towards Large Recommender Models with Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2505.16994](https://arxiv.org/abs/2505.16994)  
**代码**: [GitHub](https://github.com/YRYangang/RRec)  
**领域**: 推荐系统  
**关键词**: 推荐系统, 大语言模型推理, 强化学习, 双头架构, 测试时缩放

## 一句话总结

提出R²ec，首个将推理能力内生地集成到推荐模型中的统一大推荐模型，通过双头架构实现推理链生成与高效物品预测的一体化，并设计RecPO强化学习框架在无推理标注数据下联合优化推理与推荐目标。

## 研究背景与动机

大语言模型（LLM）在推荐系统中的应用已形成两大范式：一是将LLM作为编码器嵌入用户和物品，二是将物品预测重新表述为自回归生成物品ID。这些大推荐模型在冷启动、跨域、长尾等场景展现出强泛化能力。

**推理对推荐的潜在价值**：DeepSeek-R1等模型证明了test-time scaling（推理时多"思考"一会儿）可以显著提升LLM在数学、编程等任务上的能力。大推荐模型本身就基于预训练LLM构建，一个自然的问题是：如何让推荐模型也从推理中获益？

**现有探索的关键缺陷**：

**资源开销过大**：需要同时维护一个大推理模型和一个推荐模型，内存和推理延迟双重膨胀。

**联合优化困难**：推理和推荐模块只能交替冻结训练，梯度无法跨模块流动，阻碍了端到端对齐。

**本文面临的技术挑战**：

**模型设计**：多数大推荐模型基于物品ID自回归解码，本身就慢，引入推理会进一步恶化延迟。如何在保持可接受推理速度的同时集成推理？

**训练优化**：推荐领域缺乏标注的推理数据（不像数学题有解题步骤），推理的主观性和难以规模化收集使得监督学习不可行。RL是天然替代方案，但推荐场景的奖励设计和目标耦合面临独特挑战。

## 方法详解

### 整体框架

R²ec建立在两个支柱上：
- **双头架构**：一个LLM backbone配备语言建模头（生成推理）和推荐头（预测物品），先自回归生成推理链，再一步完成物品预测。
- **RecPO训练框架**：基于RL的无标注训练，通过融合奖励联合优化推理和推荐。

### 关键设计

1. **双头统一架构**：

    - **语言建模头（lm_head）**：标准token嵌入表$\mathbf{H}_\mathcal{T} \in \mathbb{R}^{|\mathcal{T}| \times d}$，负责自回归生成推理token。
    - **推荐头（rec_head）**：物品嵌入表$\mathbf{H}_\mathcal{V} \in \mathbb{R}^{|\mathcal{V}| \times d}$，每个物品通过将其描述文本送入模型自身编码得到嵌入。物品评分通过内积计算：$s(v) = \mathbf{h}_T^\top \mathbf{H}_\mathcal{V}[v]$。

   **推理-推荐的紧密耦合**：两个头共享同一隐状态空间，推理过程直接重塑最终隐状态$\mathbf{h}_T$，从而影响推荐评分。这保证了推理优化直接贡献于推荐改进。

   **效率优势**：用"下一物品预测"（单步内积匹配）替代物品ID的自回归解码，大幅降低推理延迟。物品表可灵活增删，支持零样本泛化。

2. **RecPO训练框架**：

    - **轨迹采样**：每条轨迹覆盖完整的"推理→推荐"过程。对每个用户输入$x_u$，用旧策略$\pi_{\theta_{old}}$采样$G$条不同推理路径，每条推理后接一步物品推荐。

    - **融合奖励**：单纯使用排名指标（如NDCG）作为奖励不够——很多质量不同的轨迹可能产出相同的top-K排名。因此设计融合奖励：

    $R = \beta R_c + (1 - \beta) R_d$

   其中$R_d = \text{NDCG}@k(\text{rank}(v^+))$是离散排名奖励，$R_c = \frac{\exp(\mathbf{h}_T^\top \mathbf{h}_{v^+}/\tau)}{\sum_{v \in \mathcal{V}} \exp(\mathbf{h}_T^\top \mathbf{h}_v/\tau)}$是连续相似度奖励。$\beta \approx 0.05$保持排名项主导，同时让连续项为相同排名的轨迹提供区分度。

    - **联合训练目标**：将token级推理决策和物品级推荐决策统一在一个RL目标中：

    $\mathcal{J}(\theta) = \frac{1}{G}\sum_{i=1}^{G}\left[\sum_{t=1}^{T_i}\ell_\epsilon(r_{i,t}(\theta), A_i) + \delta_{i,i^\star}\ell_\epsilon(r_{i,T+1}(\theta), A_i)\right]$

   关键设计：所有轨迹都贡献推理token的策略更新，但只有优势最大的轨迹（$i^\star = \arg\max_j A_j$）贡献推荐动作的梯度。这确保了推理探索的多样性，同时将推荐学习聚焦于最有希望的推理路径。

### 训练策略

- 基座模型：Gemma2-2B-Instruct和Qwen2.5-3B-Instruct
- 优势估计：GRPO效果优于RLOO（初始学习更快，推理长度逐步增长，类似LLM推理训练的现象）
- 温度采样：top-K采样控制随机性

## 实验关键数据

### 主实验：三个Amazon数据集上的推荐性能

| 方法 | Instruments H@5 | CDs H@5 | Games H@5 | Instruments N@20 | CDs N@20 |
|------|----------------|---------|-----------|-----------------|----------|
| SASRec | 0.0175 | 0.0076 | 0.0129 | 0.0210 | 0.0141 |
| TIGER | 0.0171 | 0.0067 | 0.0123 | 0.0134 | 0.0069 |
| LangPTune | 0.0127 | 0.0074 | 0.0049 | 0.0145 | 0.0094 |
| D³（Gemma） | 0.0072 | 0.0216 | 0.0117 | 0.0114 | 0.0194 |
| **R²ec（Qwen）** | **0.0237** | **0.0513** | **0.0288** | **0.0259** | **0.0457** |
| **R²ec（Gemma）** | **0.0264** | **0.0573** | **0.0326** | **0.0257** | **0.0527** |

R²ec相对最佳基线的提升：CDs上H@5提升63.7%，N@10提升72.3%；Instruments上H@10提升67.0%。

### 消融实验

| 配置 | Instruments H@5 | CDs H@5 | Games H@5 | 说明 |
|------|----------------|---------|-----------|------|
| w/ ClsHead（分类头） | 0.0044 | 0.0030 | 0.0012 | 推理-推荐解耦极差 |
| w/o Reasoning | 0.0176 | 0.0469 | 0.0277 | 无推理，纯对比学习 |
| w/o $R_d$（仅连续奖励） | 0.0198 | 0.0521 | 0.0302 | 区分度不足 |
| w/o $R_c$（仅排名奖励） | 0.0244 | 0.0543 | 0.0316 | 略低于融合 |
| **R²ec** | **0.0264** | **0.0588** | **0.0326** | 融合奖励最优 |

### 关键发现

1. **推理显著提升推荐**：引入推理后平均提升约15%，验证了test-time scaling在推荐场景的有效性。
2. **推理-推荐紧耦合至关重要**：分类头变体（w/ ClsHead）性能暴跌，说明推理和推荐必须共享隐状态空间才能有效互利。
3. **融合奖励设计有效**：离散排名奖励$R_d$是核心，连续相似度奖励$R_c$提供补充细粒度信号。单独使用$R_c$反而引入噪声。
4. **GRPO优于RLOO**：GRPO的单位方差归一化放大了推荐场景的奖励梯度，加速早期学习，且推理长度随训练逐步增长，类似DeepSeek-R1的现象。
5. **小模型也出色**：Gemma2-2B在多数任务上反而优于Qwen-3B，表明模型选择比参数量更重要。

## 亮点与洞察

- 双头架构的设计非常优雅：共享backbone使推理梯度自然流向推荐参数，避免了两阶段方法的梯度断裂问题；推荐头使用物品嵌入表的内积匹配，比物品ID自回归生成高效得多。
- 将LLM推理训练（如GRPO）的成功经验迁移到推荐领域，桥接了两个快速发展的研究方向。
- 融合奖励的设计体现了对推荐场景特有问题的深刻理解——排名指标的离散性需要连续信号来补充。

## 局限与展望

- 推理链的可解释性分析主要是定性的，缺乏系统化的量化评估。
- 当前仅在Amazon三个数据集上验证，更大规模和更多场景的泛化性有待确认。
- 物品嵌入表需要预先构建，对频繁变化的物品库可能增加维护成本。
- 推理长度的自动控制和效率优化是后续重要方向。

## 相关工作与启发

- 与LangPTune等"推理增强推荐"方法的本质区别在于：R²ec是真正的端到端统一模型，推理和推荐在同一前向传播中完成。
- RecPO框架为推荐系统中的RL训练提供了可复用的范式，特别是融合奖励和选择性梯度回传的设计。
- 对更广泛的"LLM+垂直领域"应用有启发：如何设计统一架构使推理能力自然服务于领域任务。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个推理-推荐统一模型，双头架构和RecPO均为原创
- 实验充分度: ⭐⭐⭐⭐⭐ 三数据集、多基线、9项分析极为全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机论证充分
- 价值: ⭐⭐⭐⭐⭐ 开辟了推理增强推荐的新范式，实际提升显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Think before Recommendation: Autonomous Reasoning-enhanced Recommender](think_before_recommendation_autonomous_reasoning-enhanced_recommender.md)
- [\[NeurIPS 2025\] Inference-Time Reward Hacking in Large Language Models](inference-time_reward_hacking_in_large_language_models.md)
- [\[AAAI 2026\] Hard vs. Noise: Resolving Hard-Noisy Sample Confusion in Recommender Systems via Large Language Models](../../AAAI2026/recommender/hard_vs_noise_resolving_hard-noisy_sample_confusion_in_recommender_systems_via_l.md)
- [\[ACL 2025\] KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models](../../ACL2025/recommender/kerl_knowledge-enhanced_personalized_recipe_recommendation_using_large_language_.md)
- [\[NeurIPS 2025\] Radial Neighborhood Smoothing Recommender System](radial_neighborhood_smoothing_recommender_system.md)

</div>

<!-- RELATED:END -->
