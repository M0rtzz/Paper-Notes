---
title: >-
  [论文解读] Learning Ordinal Probabilistic Reward from Preferences (OPRM)
description: >-
  [ICLR 2026][LLM对齐][序数奖励] 提出序数概率奖励模型(OPRM)，将响应质量离散化为1-9序数等级并学习完整概率分布，结合区域洪泛调优(RgFT)实现数据高效训练。在RewardBench达89.3%，比现有RM提升2.9%-7.4%，同时提供不确定性估计和标注分歧检测。
tags:
  - ICLR 2026
  - LLM对齐
  - 序数奖励
  - 概率分布
  - 区域洪泛调优
  - 奖励模型
  - 不确定性估计
---

# Learning Ordinal Probabilistic Reward from Preferences (OPRM)

**会议**: ICLR 2026  
**arXiv**: [2602.12660](https://arxiv.org/abs/2602.12660)  

**代码**: https://github.com/ritzz-ai/OPRM  

**领域**: LLM Alignment / 奖励建模  
**关键词**: 序数奖励, 概率分布, 区域洪泛调优, 奖励模型, 不确定性估计

## 一句话总结

提出序数概率奖励模型(OPRM)，将响应质量离散化为1-9序数等级并学习完整概率分布，结合区域洪泛调优(RgFT)实现数据高效训练。在RewardBench达89.3%，比现有RM提升2.9%-7.4%，同时提供不确定性估计和标注分歧检测。

## 研究背景与动机

**领域现状**：奖励模型分为生成式(GRM，需点对式监督成本高)和判别式(DRM，只用成对偏好但分数未标定)。

**现有痛点**：DRM的相对分数缺乏概率解释，无法评估不确定性；GRM需要精确的质量标签。

**核心 idea**：序数离散化+完整分布 = DRM的效率 + GRM的可解释性

## 方法详解

### 整体框架

OPRM 将奖励建模从"学一个标量"转变为"学一个概率分布"。输入 prompt+response 经 LLM backbone，在最后一个 token 位置取 softmax 后的词汇表概率，提取数字 token '1'-'9' 的概率并归一化，得到序数质量分布 $p_\psi(s|x,y)$。推理时通过加权平均得到标量奖励。

### 关键设计

1. **概率奖励模型（PRM → OPRM）**：将质量分数建模为随机变量 $S$，学习条件 PDF $p_\psi(s|x,y)$。
偏好概率为 $P(y_c \succ y_r|x) = \int\int \mathbb{1}(s_c > s_r) p_\psi(s_c|x,y_c) p_\psi(s_r|x,y_r) ds_r ds_c$。
连续分布无解析解，离散化为 1-9 等级后得到闭式求和。
2. **序数概率与 Bradley-Terry 的关系**：证明 BT 模型是 OPRM 的特例——当质量分布退化为固定形状的 Gumbel 分布时。OPRM 通过学习完整分布，支持多峰偏好和不确定性估计。
3. **梯度动力学**：$\partial J / \partial p_c(k) = P(s_r < k)$，$\partial J / \partial p_r(k) = P(s_c > k)$。将 chosen 的概率质量推向高分区、rejected 推向低分区，产生持续的对比优化压力。
4. **区域洪泛调优（RgFT）**：利用质量等级标注（good/normal/bad）将分布约束到对应子区域。简单区间约束导致梯度消失，RgFT 将其转化为三角形概率景观，恢复梯度激励——既定位正确区域又最大化偏好 margin。支持半监督训练（有标注+仅偏好数据混合）。

### 实现细节

- 无额外参数：直接复用 LM head 的词汇表概率，无需 value head
- 输入灵活：支持单响应评分和多响应对比，适用于 BoN 场景
- 不确定性量化：分布方差作为置信度指标——宽分布=模糊偏好，尖峰分布=明确偏好

## 实验关键数据

### 主实验（4 个基准，10+ 任务）

| 模型 | RewardBench | RMB-Chat | RMB-Safety | RMB-Code | Overall* |
|------|-------------|----------|------------|----------|----------|
| Skywork-Reward-V2 (8B) | 92.0 | 70.7 | 76.2 | 67.8 | 71.6 |
| ArmoRM (8B) | 89.5 | 72.1 | 74.8 | 65.3 | 70.7 |
| **OPRM-Qwen2.5-14B** | **89.3** | **76.4** | **78.5** | **70.1** | **73.8** |

### RgFT 消融

| 配置 | RMB Overall | 说明 |
|------|-------------|------|
| OPRM（无 RgFT） | 71.2 | 基线概率奖励 |
| + RgFT（仅 good/bad） | 72.8 | 二分类标注 |
| + RgFT（good/normal/bad） | **73.8** | 三级标注最优 |
| + 全量质量标注 | 73.5 | 更多标注反而轻微下降 |

### 关键发现

- OPRM 在 RewardBench 外的三个基准上一致优于 BT 和 GRM 基线，平均提升 2.9%-7.4%

- RgFT 用少量质量标注（20% 数据有标注）即可有效校准分布，数据效率极高
- 多峰分布可检测标注分歧：不一致的偏好对导致双峰分布，可用于数据质量筛选
- OPRM 对 margin 微妙差异更敏感——在"chosen 略优于 rejected"的困难样本上优势最大

## 亮点与洞察

- **统一 DRM 和 GRM 的优势**：无需额外 value head（vs DRM），无需 CoT critique（vs GRM），直接从 LM head 获取分布

- 序数离散化保留质量的有序性同时避免了精确点对式标注的成本
- RgFT 的"洪泛"思想巧妙——将硬区间约束软化为梯度友好的三角形景观
- 不确定性估计可用于 BoN 采样时的风险感知选择——选择高均值+低方差的响应

## 局限与展望

- 1-9 等级的粒度选择缺乏理论指导，过粗或过细可能影响性能
- 未探索与 RLHF/DPO 训练的实际集成——OPRM 的分布奖励如何用于 PPO 需要额外设计
- 依赖 LLM 对数字 token 的内在序数理解，小模型可能理解不足
- RgFT 的半监督设置中，有标注数据的比例对性能影响的敏感度未充分分析

## 相关工作与启发

- **vs Skywork-Reward-V2**：Skywork 侧重数据策展，OPRM 侧重模型架构创新——两者可组合

- **vs Bradley-Terry**：BT 是 OPRM 在 Gumbel 分布假设下的特例，OPRM 通过学习自由度更高的分布摆脱此限制

- **vs CLoud/Critic-RM**：GRM 需要 CoT critique 生成耗时，OPRM 单次前向传播即得分布

- **vs 序数回归文献（SORD/ALDL）**：将深度序数回归思想引入偏好学习，是跨领域知识迁移

## 评分

- 新颖性: ⭐⭐⭐⭐ 序数概率奖励的统一视角新颖，BT 作为特例的证明优雅
- 实验充分度: ⭐⭐⭐⭐ 4 个基准 + 丰富消融 + 分布可视化
- 写作质量: ⭐⭐⭐⭐ 理论推导完整，动机清晰
- 价值: ⭐⭐⭐⭐ 为奖励模型提供了新范式，分布输出开辟新应用场景

<!-- RELATED:START -->

## 相关论文

- [Capturing Individual Human Preferences with Reward Features](../../NeurIPS2025/llm_alignment/capturing_individual_human_preferences_with_reward_features.md)
- [Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](skywork-reward-v2_scaling_preference_data_curation_via_human-ai_synergy.md)
- [Swap-guided Preference Learning for Personalized RLHF (SPL)](swap-guided_preference_learning_for_personalized_reinforcement_learning_from_hum.md)
- [ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](../../NeurIPS2025/llm_alignment/responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)
- [AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint](alphasteer_learning_refusal_steering_with_principled_null-space_constraint.md)

<!-- RELATED:END -->
