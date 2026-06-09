---
title: >-
  [论文解读] Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay
description: >-
  [NeurIPS 2025][LLM对齐][LLM强化微调] 提出两种互补技术提升 LLM 强化微调（GRPO）的数据效率：(1) DOTS——基于注意力机制预测自适应难度，优先选择中等难度问题以最大化梯度信号；(2) Rollout Replay——复用近期 rollout 降低每步计算开销…
tags:
  - "NeurIPS 2025"
  - "LLM对齐"
  - "LLM强化微调"
  - "数据效率"
  - "自适应难度"
  - "在线数据选择"
  - "经验回放"
  - "GRPO"
---

# Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay

**会议**: NeurIPS 2025  
**arXiv**: [2506.05316](https://arxiv.org/abs/2506.05316)  
**代码**: [GitHub](https://github.com/ASTRAL-Group/data-efficient-llm-rl/)  
**领域**: LLM对齐  
**关键词**: LLM强化微调, 数据效率, 自适应难度, 在线数据选择, 经验回放, GRPO

## 一句话总结

提出两种互补技术提升 LLM 强化微调（GRPO）的数据效率：(1) DOTS——基于注意力机制预测自适应难度，优先选择中等难度问题以最大化梯度信号；(2) Rollout Replay——复用近期 rollout 降低每步计算开销。两者结合在 6 个模型-数据集组合上平均减少 40.7% 训练时间。

## 研究背景与动机

LLM 的 RL 微调（如 GRPO）已成为提升推理能力的主流方法，但计算成本极高：

**成本惊人**：Luo et al. 报告训练 1.5B 模型在 40K 样本上需要 3800 A100 GPU 小时（~$4500）

**数据效率被忽视**：现有研究集中在算法改进（如 GRPO、L1），很少关注如何选择更有信息量的训练数据

**两个浪费来源**：(a) 过简/过难的问题产生零梯度信号（所有 rollout 奖励全 0 或全 1）；(b) 每步都需从零生成全部 rollout，即使策略变化很小

核心观察：在 GRPO 中，当某个问题的所有 rollout 奖励相同时，归一化后的优势为 0，不产生任何梯度更新——这些计算完全浪费了。

## 方法详解

### 整体框架

两阶段优化：DOTS 减少达到目标性能的训练步数，RR 减少每步的计算时间。

### 关键设计一：自适应难度定义

给定策略 $\pi_t$ 和问题 $q$，采样 $G$ 个响应并获取奖励 $\{r_i^{(t)}\}_{i=1}^G$，自适应难度定义为平均失败率：

$$d_q^{(t)} = \frac{1}{G} \sum_{i=1}^{G} (1 - r_i^{(t)})$$

$d_q = 0$ 表示全部正确（太简单），$d_q = 1$ 表示全部错误（太难）。关键定理证明了选择 $d_q = 0.5$ 最优：

**定理 1（50% 成功率时梯度信号最大）**：对于 Bernoulli(p) 奖励分布，未裁剪策略梯度的期望平方范数满足：

$$\mathbb{E}[\|g\|^2] \propto p(1-p) \cdot (1 - 1/G)$$

当 $p = 0.5$ 时达到最大值，即自适应难度为 0.5 时梯度信号最强。

### 关键设计二：注意力机制难度预测

直接计算所有问题的自适应难度需要为每个问题生成 rollout，计算量巨大。论文提出高效预测框架：

1. **参考集采样**：每步随机抽取 $K$（如 256）个问题作为参考集 $\mathcal{D}_{\text{ref}}$，仅对这些问题执行 rollout 得到真实难度
2. **嵌入相似度预测**：用嵌入模型 $E_\theta$ 编码所有问题，未标注问题的难度通过对参考集的注意力加权估计：

$$a_i = \frac{\exp(z_q^\top z_i / \sqrt{h})}{\sum_{j=1}^K \exp(z_q^\top z_j / \sqrt{h})}, \quad \hat{d}_q^{(t)} = \sum_{i=1}^K a_i d_i^{(t)}$$

3. **Platt 校准**：用 MLP 根据参考集难度的均值和标准差学习 scale/bias 参数，提升预测精度：

$$\hat{d}_{q,\text{cal}}^{(t)} = \sigma\left(w^{(t)} \cdot \left(\log \hat{d}_q^{(t)} - \log(1 - \hat{d}_q^{(t)})\right) + b^{(t)}\right)$$

嵌入模型使用冻结的 Qwen2.5-Math-1.5B-Instruct + 3层 MLP adapter。

### 关键设计三：难度导向的在线数据选择

根据预测难度对问题采样，越接近 0.5 被选中的概率越高：

$$P(q) = \frac{\exp(-|\hat{d}_q - \alpha| / \tau)}{\sum_{q' \in \mathcal{D}} \exp(-|\hat{d}_{q'} - \alpha| / \tau)}$$

其中 $\alpha = 0.5$ 为目标难度，$\tau$ 为温度参数。

**隐式多样性保证**：被反复选中的中等难度问题训练后难度会偏离 0.5，自然退出选择池，让其他问题有机会被采样。

### 关键设计四：Rollout Replay

每步只生成 $\delta B$（如 50%）新 rollout，其余从 FIFO 缓冲区复用。为解决 off-policy 偏差，使用修正的 GRPO 损失：

$$\tilde{r}_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{behavior}}}(o_{i,t}|q, o_{i,<t})}$$

重要采样比率相对于行为策略（产生该 rollout 时的策略）而非旧策略。仅存储产生非零梯度信号的 rollout（即组平均奖励不是 0 或 1 的）。

### 损失函数

基于 GRPO 损失的修改版本，去除标准差归一化（避免偏差），使用行为策略做重要性采样：

$$\mathcal{J}_{\text{GRPO-RR}}(\theta) = \mathbb{E}\left[\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left(\min(\tilde{r}_{i,t}\hat{A}_i, \text{clip}(\tilde{r}_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i) - \beta D_{\text{KL}}\right)\right]$$

## 实验关键数据

### 主实验：训练时间节省

| 模型 | 数据集 | 步数节省 | 每步时间节省 | **总时间节省** |
|------|--------|---------|------------|------------|
| Qwen2.5-Math-1.5B | MATH | 16.67% | 11.71% | 26.25% |
| Qwen2.5-Math-1.5B | DeepScaleR | 43.33% | 11.69% | 49.85% |
| Qwen2.5-Math-1.5B | ORZ | 13.33% | 11.66% | 23.30% |
| Qwen2.5-3B | DeepScaleR | 26.67% | 11.52% | 35.10% |
| Qwen2.5-3B | DeepMath | 56.67% | 11.35% | **61.65%** |
| Qwen2.5-Math-7B | DeepScaleR | 40.00% | 13.39% | 48.03% |
| **平均** | | | | **40.7%** |

最佳情况（Qwen2.5-3B + DeepMath）节省 61.65% 训练时间，步数减少 56.67%。

### 难度预测质量

| 模型 | 数据集 | Pearson 相关系数 ρ |
|------|--------|-------------------|
| Qwen2.5-Math-1.5B | MATH | 0.784 ± 0.024 |
| Qwen2.5-Math-1.5B | DeepScaleR | 0.724 ± 0.032 |
| Qwen2.5-3B | DeepScaleR | 0.779 ± 0.019 |
| Qwen2.5-3B | DeepMath | 0.703 ± 0.008 |
| Qwen2.5-Math-7B | DeepScaleR | 0.708 ± 0.020 |

所有场景下 ρ > 0.7，说明注意力预测框架能有效跟踪策略演化过程中的难度变化。

### 消融实验

1. **DOTS 单独效果**：学习曲线更陡，收敛更快（步数减少 13-57%）
2. **RR 单独效果**：每步时间减少约 20%（rollout 占总时间的 46-54%）
3. **有效问题比例**：DOTS 比原始 GRPO 平均多选择 25.4% 的"有效问题"（难度严格在 0 和 1 之间的问题）
4. **优于外部难度标签**：DOTS 持续优于基于 GPT-4o-mini 标注的难度课程方法

### 关键发现

1. **DOTS 和 RR 互补**：DOTS 加速收敛（减少步数），RR 减少每步开销（减少 rollout）
2. **泛化到非数学领域**：在科学 QA（SCP-25K 数据集，MMLU 物理/化学/生物子集）上同样有效
3. **预测开销极小**：缓存嵌入后，10K 样本仅需 1.71 秒完成难度预测
4. **缓冲区大小影响**：$C \in \{256, 512\}$，过小导致过期 rollout 比例高，过大则存储开销增加

## 亮点与洞察

1. **理论支撑清晰**：定理 1 从梯度范数最大化角度严格证明了选择 50% 成功率问题的最优性
2. **注意力预测机制巧妙**：仅需对 256 个参考问题做 rollout，就能预测整个数据集的难度，开销极小
3. **隐式课程学习**：DOTS 自然形成动态课程——已掌握的问题难度下降后被淘汰，新问题进入选择池
4. **工程实用性强**：两种技术都是即插即用的，不改变 GRPO 核心算法
5. **与外部标注方法对比**：自适应难度优于 GPT-4o-mini 标注的静态难度，因为它能跟踪策略动态

## 局限性

1. **仅验证 GRPO**：虽然理论可推广，但仅在 GRPO 上实验，PPO 等其他 RL 算法的适用性未验证
2. **二元奖励假设**：定理 1 假设奖励 $r_i \in \{0, 1\}$，对连续奖励或部分奖励的推广未讨论
3. **嵌入模型选择**：使用 Qwen2.5-Math-1.5B-Instruct 作为嵌入器，换到其他领域可能需要重新训练 adapter
4. **Rollout 回放的过期风险**：策略快速变化时，缓冲区中的旧 rollout 可能引入偏差，裁剪机制只是近似解决
5. **缺少最终性能对比**：主要关注"达到相同性能的时间"，但未讨论是否能训练更长时间得到更好结果

## 相关工作与启发

- DeepSeek-R1（Guo et al., 2025）的成功使 GRPO 成为主流 LLM RL 方法，本文的数据效率改进直接降低其部署门槛
- 经验回放在传统 RL 中有悠久历史（DQN），本文的 RR 机制是其在 LLM RL 场景的自然适配
- 自适应课程学习（zone of proximal development）的理念在教育学和 RL 中都有对应
- 启发：类似的难度预测可用于 RLHF 中的偏好数据选择

## 评分

- **创新性**: ⭐⭐⭐⭐ — 注意力难度预测是亮点，其他组件是现有思想的合理组合
- **实验充分性**: ⭐⭐⭐⭐ — 6 个模型-数据集组合，多维度消融，非数学领域泛化验证
- **实用性**: ⭐⭐⭐⭐⭐ — 即插即用，平均节省 40% 训练时间，对资源受限的团队价值巨大
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，理论和实验对应好，图表设计直观
- **总体评价**: ⭐⭐⭐⭐ — 实用性突出的工作，在 LLM RL 效率这个被忽视的方向做出了有意义的贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO](deepvideor1_video_reinforcement_finetuning_via_difficultyawa.md)
- [\[NeurIPS 2025\] T-SHIRT: Token-Selective Hierarchical Data Selection for Instruction Tuning](t-shirt_token-selective_hierarchical_data_selection_for_instruction_tuning.md)
- [\[AAAI 2026\] Importance-Aware Data Selection for Efficient LLM Instruction Tuning](../../AAAI2026/llm_alignment/importance-aware_data_selection_for_efficient_llm_instruction_tuning.md)
- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)
- [\[NeurIPS 2025\] Mechanism Design for LLM Fine-tuning with Multiple Reward Models](mechanism_design_for_llm_fine-tuning_with_multiple_reward_models.md)

</div>

<!-- RELATED:END -->
