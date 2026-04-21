---
title: >-
  [论文解读] When Top-ranked Recommendations Fail: Modeling Multi-Granular Negative Feedback for Explainable and Robust Video Recommendation
description: >-
  [AAAI 2026][负反馈建模] 提出 ENF（Explainable Negative Feedback）框架，通过三个协作式 MLLM Agent（Profile Agent、Video Agent、Reason Agent）和渐进式 S-GRPO 强化学习训练策略，首次实现了对视频推荐系统中隐式负反馈的可解释预测和原因分析，在腾讯新闻业务平台上实现了平均观看时长提升 6.2% 和快速跳过率下降 9.4%。
tags:
  - AAAI 2026
  - 负反馈建模
  - 可解释推荐
  - 多模态视频理解
  - MLLM Agent
  - 强化学习
---

# When Top-ranked Recommendations Fail: Modeling Multi-Granular Negative Feedback for Explainable and Robust Video Recommendation

**会议**: AAAI 2026  
**arXiv**: [2511.18700](https://arxiv.org/abs/2511.18700)  
**代码**: 无  
**领域**: 推荐系统  
**关键词**: 负反馈建模, 可解释推荐, 多模态视频理解, MLLM Agent, 强化学习

## 一句话总结

提出 ENF（Explainable Negative Feedback）框架，通过三个协作式 MLLM Agent（Profile Agent、Video Agent、Reason Agent）和渐进式 S-GRPO 强化学习训练策略，首次实现了对视频推荐系统中隐式负反馈的可解释预测和原因分析，在腾讯新闻业务平台上实现了平均观看时长提升 6.2% 和快速跳过率下降 9.4%。

## 研究背景与动机

现有视频推荐系统主要依赖基于 ID 的嵌入映射和协同过滤，存在三个核心挑战：

**1. 负反馈数据稀缺**：显式负反馈（如点击"不喜欢"）信息量大但极其稀疏（仅占所有交互的约 0.3%）；隐式反馈（如观看时长、跳过行为）虽然丰富但噪声大、信息量低

**2. 负反馈原因未被理解**：现有方法通常通过聚类负反馈信号来提取特征，然后抑制类似推荐。但这种方法缺乏对具体原因的理解，泛化能力差。例如，如果用户不喜欢一个美食视频，不理解原因就压制所有美食推荐是不合理的——可能用户只是不喜欢视频中食材处理的画面，而非美食主题本身

**3. 多模态场景评估缺失**：现有 LLM 方法虽然能预测用户偏好，但忽略了物品的复杂多模态内容，且缺乏对可解释原因的评估

这是一个非常实际且重要的问题：**为什么排名靠前的推荐仍然频繁触发用户负反馈？** 传统方法基于高嵌入相似度推荐，但完全忽视了用户深层次的心理特征和视频的细粒度内容分析。

## 方法详解

### 整体框架

ENF 框架采用三层级 Agent 架构：Profile Agent 分析用户行为构建心理画像 → Video Agent 进行多模态视频内容分析 → Reason Agent 综合两者信息预测用户态度并生成可解释原因。训练采用两阶段策略：SFT 冷启动 + S-GRPO 强化学习微调。

### 关键设计

#### 1. **Profile Agent**: 从行为模式推断用户心理画像

**功能**：分析用户的基础信息（年龄、性别、职业、兴趣标签）和顺序观看行为（标题、播放率），推断用户的心理特征和人格画像。

**核心思路**：传统推荐系统仅依赖兴趣标签嵌入，忽略了用户深层心理倾向。例如：
- 明星粉丝可能强烈偏好关于偶像的正面内容，但排斥批评性叙述
- 美食爱好者可能对过于直白的食材处理画面产生负面反应

Agent 聚焦 play_rate < 0.3 的视频（用户不满指标），当文本标题信息不足时，动态调用 Video Agent 提取多模态线索。通过逐个分析每个交互，迭代更新心理画像（价值取向、对负面内容的容忍度等）。

**设计动机**：用户偏好远不止表面兴趣标签，而是植根于更深层的心理特征。理解这些隐性心理特质才能实现更精准的用户对齐推荐。

#### 2. **Video Agent**: 多模态视频内容深度分析

**功能**：对单个视频进行深度分析，不仅描述内容，还识别视频中潜在的争议性元素并提供上下文解释。

**核心思路**：利用 MLLM 的多模态能力分解视频内容，使用 16 张均匀采样帧和视频标题作为输入特征。分析四个维度：
- 视频是否包含负面事件
- 是否包含低俗或与用户价值观冲突的内容
- 情节是否无趣
- 是否包含令人不适的视觉元素

**设计动机**：传统基于嵌入的方法无法识别视频中的争议内容，而这些内容恰恰是触发负反馈的主要原因。

#### 3. **Reason Agent**: 综合判断与可解释推理

**功能**：利用用户基础信息和精细化的心理画像，从用户视角生成对视频的理解，推断用户态度并提供可解释原因。

**核心思路**：从四个维度评估用户-视频匹配度：
1. 视频内容是否符合用户兴趣
2. 情节是否有吸引力
3. 内容是否包含负面事件或极端观点
4. 视觉元素是否符合用户的感官容忍度

### 损失函数 / 训练策略

采用 **S-GRPO（Stepwise Group Relative Policy Optimization）** 渐进式强化学习训练，这是本文的核心创新之一。

**训练分两个阶段**：

**Stage 1 - SFT 冷启动**：使用真实用户反馈原因，提示 GPT-4o 生成链式推理解释用户为什么不喜欢特定视频，作为 SFT 数据热启动模型。

**Stage 2 - S-GRPO 强化微调**：提出渐进式奖励机制，包含三个步骤奖励 $R_{S_i}$：

1. **Binary Judge Reward** $r_{judge}$（第一步）：判断用户态度预测是否正确
    - 错误则立即终止，不获得奖励
    - 正确获得固定奖励（如 0.5）
    - 若实际反馈为正面，终止；若为负面，进入下一步

2. **Class Reward** $r_{class}$（第二步）：负反馈类型分类是否准确
    - 正确获得额外奖励（如 1.0），进入第三步

3. **Reason Reward** $r_{reason}$（第三步）：计算 `<think>` 标签内推理内容与实际用户反馈原因的 ROUGE-1/2/L 平均分

优势计算：

$$A_i = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\})}$$

策略更新目标：

$$\mathcal{J}_{GRPO}(\theta) = \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\right)A_i\right) - \beta\mathcal{D}_{KL}(\pi_\theta||\pi_{ref})$$

这种**渐进式设计**鼓励模型从易到难，对于二分类正确但多选题答案错误的情况仍能获得奖励，而正确分类伴随合理推理的回答获得更高分数。

**实现细节**：
- 基础模型：Qwen2.5-VL-7B
- GPT-4o 作为 Profile Agent，Qwen-2.5VL-7b 用于 Video Agent 和 Reason Agent
- 全参数微调，4 块 80G GPU
- 学习率 1e-6，组大小 G=8

## 实验关键数据

### 主实验 - 显式负反馈预测

| 模型 | Size | Acc | Recall | F1 | Class_Acc | Reasoning |
|------|------|-----|--------|-----|-----------|-----------|
| GPT-4o | - | 0.882 | 0.630 | 0.739 | 0.568 | 0.402 |
| DeepSeek | - | 0.849 | 0.440 | 0.594 | 0.352 | 0.266 |
| Qwen2.5VL | 7B | 0.815 | 0.423 | 0.564 | 0.296 | 0.229 |
| Video-R1 | 7B | 0.835 | 0.540 | 0.667 | 0.432 | 0.318 |
| VideoChat-R1 | 7B | 0.842 | 0.654 | 0.739 | 0.500 | 0.383 |
| **Our Video Agent** | **7B** | **0.861** | **0.808** | **0.750** | **0.654** | **0.537** |

核心发现：Recall 比 GPT-4o 高 +17.8%，Class_Acc 比 GPT-4o 高 +8.6%，Reasoning 比 GPT-4o 高 +13.5%。

### 隐式负反馈预测

| 模型 | Size | Acc | Precision | Recall | F1 | Class_Acc |
|------|------|-----|-----------|--------|-----|-----------|
| GPT-4o | - | 0.575 | 0.396 | 0.796 | 0.521 | 0.502 |
| SASRec | - | 0.448 | 0.230 | 0.358 | 0.279 | - |
| VideoChat-R1 | 7B | 0.561 | 0.384 | 0.775 | 0.516 | 0.512 |
| **Our ENF** | **7B** | **0.612** | **0.404** | **0.782** | **0.533** | **0.543** |

隐式反馈预测远比显式反馈困难（最高准确率仅 61.2%），但 ENF 仍全面领先。

### 消融实验

**Video Agent 训练消融**：

| SFT | RL | S-GRPO | Acc | F1 | Class_Acc | Reasoning |
|-----|-----|--------|-----|-----|-----------|-----------|
| ✗ | ✗ | ✗ | 0.815 | 0.423 | 0.296 | 0.229 |
| ✗ | ✓ | ✓ | 0.830 | 0.686 | 0.592 | 0.492 |
| ✓ | ✗ | ✗ | 0.851 | 0.615 | 0.346 | 0.312 |
| ✓ | ✓ | ✗ | 0.845 | 0.667 | 0.412 | 0.339 |
| ✓ | ✓ | ✓ | **0.861** | **0.750** | **0.654** | **0.537** |

**Reason Agent 消融**：

| Profile Agent | Video Agent | S-GRPO | Acc | F1 | Class_Acc |
|---------------|-------------|--------|-----|-----|-----------|
| ✗ | ✗ | ✗ | 0.528 | 0.482 | 0.435 |
| ✓ | ✓ | ✓ | **0.612** | **0.533** | **0.543** |

### 业务平台验证

| 指标 | Base RS | Base RS + ENF | 提升 |
|------|---------|---------------|------|
| 平均观看时长 | 47.6% | 53.8% | +13.0% |
| 快速跳过率 | 23.7% | 14.3% | -39.7% |
| 不喜欢率 | 0.61% | 0.35% | -42.6% |

### 关键发现

1. **S-GRPO 的渐进式学习**至关重要：没有 S-GRPO，模型倾向只学二分类而忽略分类和原因推理
2. **SFT 冷启动**提供用户侧先验知识，移除后预测精度显著下降
3. **Profile Agent**提供更丰富的心理画像特征，对综合性用户建模至关重要
4. **隐式反馈预测**远比显式反馈困难，因为真实用户行为受多因素影响且具有固有随机性
5. **传统方法（如 SASRec）**在需要细粒度物品区分的冷启动场景中表现很差

## 亮点与洞察

1. **问题提出非常实际和重要**：不是简单地预测用户是否喜欢，而是解释**为什么**负反馈会发生，这对改进推荐系统意义重大
2. **S-GRPO 的渐进式奖励设计**非常巧妙：从易到难（二分类→多分类→推理解释），每步只在前一步正确时才给奖励，避免了单一奖励信号无法区分不同层次错误的问题
3. **真实业务部署验证**：不仅在离线数据集上评估，还在腾讯新闻平台验证了实际效果，说了服力很强
4. **构建了 TVNF 数据集**：包含真实用户不喜欢原因的多模态视频推荐数据集，填补了领域空白

## 局限与展望

1. **数据规模**：TVNF 数据集仅约 1K 条显式反馈标注，规模较小
2. **Profile Agent 使用 GPT-4o**：实际部署成本可能较高，未来可考虑蒸馏为更小模型
3. **负反馈类型有限**：只考虑了四类原因，实际用户不喜欢的原因可能更多样
4. **隐式反馈的 ground truth**来自 GPT-4o 标注而非真实用户，可能引入偏差
5. **跨平台泛化性**：方法在短视频推荐之外的场景（如新闻、电商等）效果有待验证

## 相关工作与启发

- 与 DFN、CDR、SINE 等传统负反馈方法不同，本文首次用 LLM 理解和解释负反馈
- S-GRPO 的渐进式奖励机制可以推广到其他多步推理任务的强化学习训练
- 三 Agent 协作框架为"用 LLM 模拟用户行为"提供了新思路
- 在 MovieLens 和 Steam 数据集上的泛化实验验证了方法的跨领域潜力

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次系统性地用 MLLM Agent 解释推荐系统中的负反馈，S-GRPO 设计创新
- 实验充分度: ⭐⭐⭐⭐⭐ — 离线评估+消融+跨数据集泛化+真实业务部署，非常完整
- 写作质量: ⭐⭐⭐⭐ — 问题动机清晰，方法描述详细
- 价值: ⭐⭐⭐⭐⭐ — 解决了推荐系统中长期被忽视但极其重要的负反馈理解问题，具有很强的实用价值

<!-- RELATED:START -->

## 相关论文

- [Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)
- [Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation](tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)
- [GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion](../../ACL2025/recommender/gram_generative_recommendation.md)
- [Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction](length-adaptive_interest_network_for_balancing_long_and_short_sequence_modeling_.md)
- [Moral Change or Noise? On Problems of Aligning AI With Temporally Unstable Human Feedback](moral_change_or_noise_on_problems_of_aligning_ai_with_temporally_unstable_human_.md)

<!-- RELATED:END -->
