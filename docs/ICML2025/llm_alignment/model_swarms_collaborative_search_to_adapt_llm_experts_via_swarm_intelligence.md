---
title: >-
  [论文解读] Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence
description: >-
  [ICML 2025][LLM对齐][群体智能] 借鉴粒子群优化（PSO）算法，将多个 LLM 专家视为"粒子"，在权重空间中协作搜索，通过个体最优/全局最优/全局最差三个信号引导专家迭代移动，仅需 200 个样本即可实现无需微调的模型适配，在 9 个任务上平均超越 12 个基线 13.3%。
tags:
  - "ICML 2025"
  - "LLM对齐"
  - "群体智能"
  - "模型融合"
  - "粒子群优化"
  - "LLM专家适配"
  - "弱到强泛化"
---

# Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence

**会议**: ICML 2025  
**arXiv**: [2410.11163](https://arxiv.org/abs/2410.11163)  
**代码**: [BunsenFeng/model_swarm](https://github.com/BunsenFeng/model_swarm)  
**领域**: LLM对齐/模型组合  
**关键词**: 群体智能, 模型融合, 粒子群优化, LLM专家适配, 弱到强泛化

## 一句话总结

借鉴粒子群优化（PSO）算法，将多个 LLM 专家视为"粒子"，在权重空间中协作搜索，通过个体最优/全局最优/全局最差三个信号引导专家迭代移动，仅需 200 个样本即可实现无需微调的模型适配，在 9 个任务上平均超越 12 个基线 13.3%。

## 研究背景与动机

当前 LLM 组合方法主要分为三类，各有局限：

**MoE 路由**：将查询分发给不同专家，但不产生新模型，面对超出现有专家能力的任务束手无策

**Learn-to-Fuse**：设计可训练组件将专家"粘合"，但需要大量监督数据训练，且模块化程度低，难以灵活增删专家

**模型算术**：通过权重或概率的算术运算组合专家（如 Task Arithmetic、TIES-Merging），但依赖对专家和组合方式的强假设（如"室内狮子 = 室外狮子 + (室内狗 - 室外狗)"）

作者指出，真正需要的是一种**无需过多训练数据、不依赖专家先验假设**的灵活适配方法。Model Swarms 受粒子群优化（PSO）启发，将每个 LLM 专家视为权重空间中的粒子，通过协作搜索在低数据场景（仅 200 个样本）下自动发现最优模型组合。

## 方法详解

### 整体框架

Model Swarms 的核心思想是把 LLM 适配问题转化为一个**权重空间中的协作搜索问题**：

- **输入**：一组 LLM 专家 {x_i} 和效用函数 f: x → R（如验证集准确率、奖励模型分数）
- **过程**：每个专家粒子有**位置**（模型权重）和**速度**（权重空间中的方向），在个体最优、全局最优、全局最差的引导下迭代更新
- **输出**：搜索结束后返回全局最优专家 g

整个过程分为三步循环：初始化 → 速度更新 → 权重更新 → 迭代终止判断。

### 关键设计

1. **初始化扩充（Pairwise Crossover）**：从 n 个初始专家通过两两线性插值扩充到 N 个粒子。随机选取两个专家 x_a 和 x_b，采样 t ~ U(0,1)，生成新粒子 x_new = t·x_a + (1-t)·x_b。每个粒子的初始速度设为指向随机另一粒子的方向 v_i = random(x_j) - x_i，避免所有粒子像"黑洞"一样坍缩到全局最优。

2. **速度更新（Velocity Update）**：速度由四个因素的加权平均决定，其中 C 为归一化项，四个分量含义为：

    - **惯性项** φ_v·v_i：保持当前速度方向，维持搜索动量
    - **认知项** φ_p·(p_i - x_i)：被自身历史最优位置吸引
    - **社会项** φ_g·(g - x_i)：被全局最优位置吸引
    - **排斥项** φ_w·(g_w - x_i)：远离全局最差位置（这是对经典 PSO 的创新扩展）
    - **随机因子** r_v, r_p, r_g, r_w ~ U(0,1)：确保搜索的非确定性，增强探索能力

3. **权重更新与重启机制（Weight Update & Restart）**：位置更新公式为 x_i ← x_i + λ·v_i，其中 λ 为步长。如果某粒子的个体最优在 c_r 次迭代内没有改善，则将其**重启**到个体最优位置并将速度清零，给予"第二次机会"。这在探索性与鲁棒性之间取得了平衡。

4. **Token Swarms（跨架构变体）**：当专家来自不同架构（如 Gemma 和 Mistral）时，无法在权重空间操作。Token Swarms 将粒子的位置定义为 n 维向量（初始为 one-hot），搜索过程在 token 概率分布上进行线性组合，从而实现跨架构专家协作。

### 损失函数 / 训练策略

Model Swarms **无需任何梯度更新或监督训练**。它完全依赖效用函数 f（模型到标量的映射）来引导搜索：

- **单任务**：f 为验证集性能（如 accuracy）
- **多任务域**：f 为多任务性能的调和平均
- **奖励模型**：f 为 RM 在验证集指令上的评分
- **人类兴趣**：f 为 LLM-as-a-judge 的 1-10 评分

超参数设置：swarm 大小 N=20，步长衰减 φ_λ=0.95，耐心 c=10，重启耐心 c_r=5，最大迭代 K=50。步长每轮乘以 φ_λ 逐步衰减，实现从粗到细的搜索。

## 实验关键数据

### 主实验

基于 Gemma-7B，从 Tulu-v2 的 10 个 SFT 域分别微调获得 10 个 LoRA 专家。

| 数据集 | 指标 | Model Swarms | 最强基线 | 提升 |
|--------|------|-------------|----------|------|
| MMLU | Acc (test) | .583 | .568 (Pack of LLMs) | +2.6% |
| MMLU-pro | Acc (test) | .254 | .237 (Slerp) | +7.2% |
| Hellaswag | Acc (test) | .652 | .622 (Dare-Ties) | +4.8% |
| K-Crossword | Acc (test) | .428 | .372 (Dare-Ties) | +15.1% |
| GSM8k | Acc (test) | .459 | .354 (EvolMerge) | +29.7% |
| NLGraph | Acc (test) | .672 | .568 (LoraHub) | +18.3% |
| TruthfulQA | Acc (test) | .392 | .359 (LoraHub) | +9.2% |
| RealToxicityPrompts | Score (test) | .956 | .885 (LoraHub) | +8.0% |
| AbstainQA | Score (test) | .175 | .140 (Dare-Ties) | +25.0% |

在推理任务（GSM8k, K-Crossword, NLGraph）上提升最大，平均达 21.0%。

### 消融实验

| 配置 | MMLU | Hellaswag | NLGraph | AbstainQA | 说明 |
|------|------|-----------|---------|-----------|------|
| Full Model Swarms | .583 | .652 | .672 | .175 | 完整方法 |
| 完全确定性（去掉所有随机性） | .528 | .611 | .541 | .072 | 平均下降 23.5% |
| Crossover 仅 15 次 | .527 | .604 | .534 | .093 | 初始粒子不够多样 |
| 无 Crossover | .504 | .587 | .530 | .099 | 性能大幅下降 |
| 多样性 1×10（最低） | — | — | — | — | 基准 |
| 多样性 10×1（最高） | — | — | — | — | 比 1×10 提升 35.3% |

### 关键发现

1. **正确性涌现（C-emerge）**：36%-53.5% 的初始所有专家都答错的题目，在 Model Swarms 后被至少一个专家正确回答——说明协作搜索**发现了初始模型不具备的新能力**
2. **Diamond in the Rough**：89.6% 的最终最优粒子并非初始最优，56.9% 甚至来自初始排名后半段——说明弱模型蕴含未被激活的隐式专长
3. **弱胜强**：去掉最强专家后，剩余弱专家的 Model Swarms 结果仍平均超过最强专家 35.4%；甚至仅用后 50% 的专家也能在 2/3 数据集上胜出
4. **多样性至关重要**：10 个不同专家（10×1）比 1 个专家重复 10 次（1×10）平均性能高 35.3%
5. **Reward Model 适配**：仅用 200 条指令，Model Swarms 就超越了 PPO 和 DPO，且在 verbose/concise 两种冲突偏好上均达到最优，展现出优秀的可控性
6. **人类兴趣适配**：16 个主题上 LLM-judge 评分平均提升 17.6%，事实性提升 17.0%，人类评估胜率达 70.8%

## 亮点与洞察

- **优雅的类比**：将 LLM 适配映射为粒子群优化问题，概念简洁而效果强大。每个专家就是一个粒子，效用函数充当适应度，搜索过程自动发现最优组合
- **极低数据需求**：仅需 200 个样本作为效用函数，无需任何梯度训练，这对数据稀缺场景极具价值
- **惊人的涌现能力**：协作搜索不是简单的知识迁移，而是在初始专家都不具备的能力空间中发现了新解（C-emerge 高达 53.5%）
- **弱到强转变**：最终最优专家往往并非起步最好的——这挑战了"选最强模型"的直觉，说明隐式专长可以被协作搜索激活
- **全局最差排斥项**：在经典 PSO 基础上新增了远离全局最差位置的排斥力，帮助探索更多有效区域
- **Token Swarms 扩展**：在概率分布层面进行搜索，使得不同架构的模型也能参与协作

## 局限与展望

1. **计算开销**：每次迭代需要评估所有 N 个粒子的效用函数，对大模型来说成本较高。论文提出了 dropout 式加速策略但未深入展开
2. **初始专家选择**：多样性对方法至关重要，但如何从 HuggingFace 上 90 万+ 模型中选择最优初始专家池仍是开放问题
3. **局部最优陷阱**：尽管引入了多种随机性和重启机制，仍可能陷入局部最优。论文建议将随机因子从 U(0,1) 扩展到 U(-0.2,1) 来缓解
4. **Token Swarms 尚初步**：跨架构变体仅改变概率分布组合而不触及模型参数，能力提升空间有限
5. **时效性限制**：方法基于现有专家的知识进行适配，无法引入训练数据中未见过的新信息
6. **双重用途风险**：灵活的效用函数意味着恶意用户也可以优化有害目标

## 相关工作与启发

- **与 EvolMerge 的区别**：EvolMerge 用遗传算法搜索权重/层的组合方式，需要手工定义交叉和变异规则；Model Swarms 用 PSO 自动搜索，无需手工工程
- **与 LoraHub 的区别**：LoraHub 通过梯度优化 LoRA 权重系数；Model Swarms 不用梯度，直接在权重空间中搜索
- **启发**：这种将优化算法（PSO）直接应用于模型权重空间的思路，可以扩展到其他场景，如 prompt 搜索、超参数优化等。"弱到强"的涌现现象也暗示差异化专家的价值远大于同质化专家

## 评分

- 新颖性: ⭐⭐⭐⭐ （PSO 本身成熟，但应用于 LLM 权重空间并观察到涌现是新颖的）
- 实验充分度: ⭐⭐⭐⭐⭐ （4 类适配目标、9+8+3+16 个评估设置、12+2 个基线、丰富的消融分析）
- 写作质量: ⭐⭐⭐⭐⭐ （结构清晰，类比精准，分析深入）
- 价值: ⭐⭐⭐⭐ （低数据无训练适配 LLM 的实用价值高，但计算成本和初始专家选择是落地瓶颈）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Improving Model Alignment through Collective Intelligence of Open-Source LLMs](improving_model_alignment_through_collective_intelligence_of_open-source_llms.md)
- [\[ICML 2025\] On the Robustness of Reward Models for Language Model Alignment](on_the_robustness_of_reward_models_for_language_model_alignment.md)
- [\[ICML 2025\] AlphaPO: Reward Shape Matters for LLM Alignment](alphapo_reward_shape_matters_for_llm_alignment.md)
- [\[NeurIPS 2025\] Ask a Strong LLM Judge when Your Reward Model is Uncertain](../../NeurIPS2025/llm_alignment/ask_a_strong_llm_judge_when_your_reward_model_is_uncertain.md)
- [\[ACL 2025\] Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging](../../ACL2025/llm_alignment/upcycling_instruction_tuning_from_dense_to_mixture-of-experts_via_parameter_merg.md)

</div>

<!-- RELATED:END -->
