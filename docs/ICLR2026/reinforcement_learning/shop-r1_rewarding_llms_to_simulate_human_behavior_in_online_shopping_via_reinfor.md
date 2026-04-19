---
title: >-
  [论文解读] Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning
description: >-
   提出 Shop-R1 框架，通过分层奖励机制和难度感知缩放的强化学习方法，显著提升 LLM 模拟真实人类在线购物行为的能力，相比 SFT 基线精确匹配提升超过 65%。
tags:

---

# Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning

## 元信息
- **会议**: ICLR 2026
- **arXiv**: [2507.17842](https://arxiv.org/abs/2507.17842)
- **代码**: [https://damon-demon.github.io/shop-r1.html](https://damon-demon.github.io/shop-r1.html)
- **领域**: reinforcement_learning
- **关键词**: LLM, reinforcement learning, human behavior simulation, online shopping, hierarchical reward, GRPO

## 一句话总结
提出 Shop-R1 框架，通过分层奖励机制和难度感知缩放的强化学习方法，显著提升 LLM 模拟真实人类在线购物行为的能力，相比 SFT 基线精确匹配提升超过 65%。

## 研究背景与动机
- LLM 在模拟人类网页行为方面展现潜力，但现有方法（零样本提示、SFT）效果仍不理想。
- **零样本提示**：缺乏个性化和适应性，准确率极低（0.32%）。
- **SFT 方法**：用 Claude 3.5 Sonnet 生成"推理-动作"训练数据进行微调，但性能受限于用于生成推理的模型能力天花板。
- **RL 的挑战**：直接用稀疏二值奖励做 RL 效果极差（1.01%），且容易出现 reward hacking——模型反复预测简单的 "terminate" 动作以获取容易的奖励。
- 核心问题：如何设计适合行为模拟（而非任务完成）的 RL 奖励？

## 方法详解

### 整体框架
Shop-R1 将人类行为模拟建模为两阶段预测：
1. **推理生成（Rationale Generation）**：给出当前网页上下文和历史动作，生成行为推理
2. **动作预测（Action Prediction）**：预测下一步动作（click / type_and_submit / terminate）

### 关键设计 1：格式奖励（Format Reward）
- 二值奖励：输出格式正确（JSON 含 rationale 和 action 两个 key）得 0.5，否则为 0
- 确保下游奖励计算的可靠性

### 关键设计 2：自确定性奖励（Self-Certainty Reward）
由于真实推理的 ground truth 不可得，使用 KL 散度衡量模型对自身推理的信心：

$$s(r_t | q_t) = \frac{1}{N|V|} \sum_{j=1}^{N} \sum_{i=1}^{|V|} p_{ij} \log\left(\frac{p_{ij}}{U_i}\right)$$

即模型输出分布与均匀分布的平均 KL 散度，值越高表示推理越确定、一致。

### 关键设计 3：分层奖励（Hierarchical Reward）
| 动作类型 | 类型奖励 | 子动作属性奖励 | 文本相似度奖励 |
|---------|---------|--------------|--------------|
| terminate | 0.3 | 无 | 无 |
| click | 0.3 | +0.2 (若 name ≠ ∅) | +DARS × ROUGE-L(name) |
| type_and_submit | 0.3 | +0.1 (name) + 0.1 (text) | +0.1×ROUGE-L(name) + DARS×ROUGE-L(text) |

### 关键设计 4：难度感知奖励缩放（DARS）
- 长文本子动作（如按钮标签、搜索查询）远比动作类型难预测
- DARS 因子放大正确预测这些困难部分的奖励（默认设为 1000）
- 防止模型通过反复预测简单 "terminate" 来 hack 奖励

### 损失函数
总体 RL 目标：

$$\max_{\pi_\theta} \mathbb{E}_{r,a \sim \pi_\theta(q)} \left[ v(a) + \alpha s(r) - \beta \text{KL}(\pi_\theta(r,a|q) \| \pi_{\text{ref}}(r,a|q)) \right]$$

其中 $v(a)$ 为分层动作奖励，$\alpha=0.005$ 控制自确定性权重，$\beta=0.001$ 控制 KL 正则化。

### 训练流程
1. **冷启动 SFT**：在 Claude 生成的 rationale-action 数据上做 SFT（4 epochs, lr=2e-5）
2. **RL 阶段**：用 GRPO 做策略优化（500 步, lr=1e-7, batch=64, 上下文长度 32K）

## 实验关键数据

### 主实验：不同方法性能对比

| 模型 (Qwen-2.5-3B) | 精确动作匹配 | 动作类型准确率 | 动作类型 F1 |
|---------------------|------------|--------------|-----------|
| Zero-shot | 0.32% | 15.33% | 16.15% |
| RL (Binary) | 1.01% | 6.17% | 9.92% |
| SFT | 16.76% | 22.25% | 24.52% |
| SFT + RL (Binary) | 16.55% | 23.74% | 28.07% |
| **Shop-R1 (Ours)** | **27.72%** | **36.40%** | **31.28%** |

> Shop-R1 相比 SFT 基线精确匹配提升 65%+，同时提升动作类型和细粒度匹配。

### 消融实验：跨模型规模

| 模型规模 | SFT | Shop-R1 | 提升 |
|---------|-----|---------|------|
| Qwen-2.5-0.5B | 9.90% | **27.72%** | +180% |
| Qwen-2.5-1.5B | 10.86% | **24.11%** | +122% |
| Qwen-2.5-3B | 16.76% | **27.72%** | +65% |

> 分层奖励机制在小模型上提升更为显著，0.5B 模型甚至达到 3B 模型同等性能。

### 关键发现
1. 稀疏二值 RL 奖励不足以引导行为模拟学习，甚至可能退化
2. 分层奖励同时提升粗粒度（类型级）和细粒度（精确匹配）性能
3. DARS 有效防止了 reward hacking（模型不再只预测 terminate）
4. 自确定性信号为无 ground truth 的推理提供有效监督
5. 框架可泛化到不同网页和 GUI 交互任务

## 亮点与洞察
- **首次将 RL 引入面向模拟的行为建模**，区别于现有面向任务完成的 Web Agent 研究
- **精心的奖励工程**：从格式→推理→动作类型→子动作，层层递进的奖励设计
- **DARS 机制**巧妙解决了 reward hacking 问题
- **0.5B 模型 = 3B 模型表现**：说明奖励设计比模型规模更重要

## 局限性
- 任务仍限定在特定电商环境（Shop-CART 数据集），泛化性需验证
- 推理质量仅通过自确定性间接评估，缺乏外部验证
- 模型预测仍与真实用户行为有较大差距（27.72% 精确匹配）
- 上下文长度限制（32K tokens）可能丢失长会话信息

## 相关工作
- **LLM 行为模拟**: ReAct (Yao et al., 2023), WebAgent (Gur et al., 2023), UX-Agent (Lu et al., 2025)
- **奖励设计**: RLHF (Ouyang et al., 2022), DPO (Rafailov et al., 2023), RLVR/DeepSeek-R1 (Guo et al., 2025)
- **购物导航 Agent**: WebArena (Zhou et al., 2023), 但关注任务完成而非行为模拟

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次将 RL 用于行为模拟（非任务完成），分层奖励设计有创意
- 理论深度: ⭐⭐⭐ — 主要是工程创新，理论贡献较少
- 实验充分性: ⭐⭐⭐⭐ — 多规模模型、消融、跨数据集验证
- 实用价值: ⭐⭐⭐⭐ — 对电商用户行为模拟有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/router-r1_teaching_llms_multi-round_routing_and_aggregation_via_reinforcement_le.md)
- [Stackelberg Coupling of Online Representation Learning and Reinforcement Learning](stackelberg_coupling_of_online_representation_learning_and_reinforcement_learnin.md)
- [Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)
- [The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective](the_sample_complexity_of_online_reinforcement_learning_a_multi-model_perspective.md)
- [VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)

<!-- RELATED:END -->
