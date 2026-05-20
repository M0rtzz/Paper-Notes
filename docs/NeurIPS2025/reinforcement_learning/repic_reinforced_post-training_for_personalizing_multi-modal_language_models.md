---
title: >-
  [论文解读] RePIC: Reinforced Post-Training for Personalizing Multi-Modal Language Models
description: >-
  [NeurIPS 2025][强化学习][多模态大模型] 提出首个基于强化学习的多模态大模型后训练框架 RePIC,用于个性化图像描述生成,在多概念场景中显著优于基于 SFT 的方法。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "多模态大模型"
  - "个性化"
  - "图像描述"
  - "后训练"
---

# RePIC: Reinforced Post-Training for Personalizing Multi-Modal Language Models

**会议**: NeurIPS 2025

**arXiv**: [2506.18369](https://arxiv.org/abs/2506.18369)

**代码**: [GitHub](https://github.com/oyt9306/RePIC)

**领域**: 强化学习 / 多模态

**关键词**: 多模态大模型, 个性化, 强化学习, 图像描述, 后训练

## 一句话总结

提出首个基于强化学习的多模态大模型后训练框架 RePIC,用于个性化图像描述生成,在多概念场景中显著优于基于 SFT 的方法。

## 研究背景与动机

多模态大语言模型（MLLM）在生成标准图像描述方面表现优异,但在个性化场景（如识别和描述特定用户的物品、宠物等）中仍存在挑战。

关键问题：

**SFT 的局限性**: 现有个性化方法主要依赖监督微调(SFT),但大规模高质量个性化描述数据**获取成本高昂**

**多概念困难**: 当图像中包含多个需要个性化识别的概念时,SFT方法频繁失败

**视觉识别不足**: 即使在高质量数据上训练,模型在真实场景中仍经常无法正确识别特定实体

**SFT 的数据瓶颈**: SFT 的性能天花板受限于训练数据质量,而 RL 可以超越这一限制

## 方法详解

### 整体框架

RePIC 是一个两阶段框架：(1) 使用 SFT 进行基础的个性化能力初始化;(2) 使用 RL 进行后训练,利用自动奖励信号进一步提升性能。

### 关键设计

**1. 奖励函数设计**

设计了两类奖励信号来训练 RL 策略：
- **视觉识别奖励**: 基于生成描述中是否正确识别了图像中的个性化实体
    - 使用实体检测器验证: 描述中提到的个性化名称是否与图像中的视觉实体匹配
- **描述质量奖励**: 基于生成描述的流畅性、信息量和准确性
    - 利用参考模型或规则评估描述质量

**2. 强化学习后训练**

- 使用 PPO 或类似策略梯度方法
- 初始策略: SFT 阶段训练的模型
- 目标: 最大化视觉识别准确率和描述质量的加权和
- KL 散度约束: 防止策略偏离太远导致语言退化

**3. 多概念处理**

- 在 RL 训练中特别针对多概念场景生成数据
- 奖励函数考虑所有概念是否都被正确识别和描述
- 支持 2-5 个概念的同时个性化

### 损失函数 / 训练策略

$$\mathcal{L}_{\text{RL}} = -\mathbb{E}_{\pi_\theta}[R(y, I, C)] + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

其中 $R$ 是综合奖励, $I$ 是图像, $C$ 是个性化概念集合。

## 实验关键数据

### 主实验

单概念个性化图像描述 (MyVLM Benchmark):

| 方法 | 识别准确率 | 描述质量 (CIDEr) | 综合得分 |
|------|----------|----------------|---------|
| InstructBLIP | 35.2% | 52.3 | 43.8 |
| MyVLM (SFT) | 68.5% | 78.6 | 73.6 |
| Yo'LLaVA (SFT) | 72.3% | 81.2 | 76.8 |
| RePIC (Ours) | **82.1%** | **85.5** | **83.8** |

多概念个性化图像描述:

| 方法 | 2概念 | 3概念 | 4概念 | 5概念 |
|------|------|------|------|------|
| MyVLM (SFT) | 52.3% | 38.5% | 25.1% | 15.8% |
| Yo'LLaVA (SFT) | 58.1% | 42.3% | 28.5% | 18.2% |
| RePIC (Ours) | **75.2%** | **65.8%** | **52.3%** | **40.5%** |

### 消融实验

RL 后训练各组件的贡献:

| 配置 | 单概念 | 多概念(3) | 描述质量 |
|------|--------|----------|---------|
| SFT only | 72.3% | 42.3% | 81.2 |
| + RL (识别奖励) | 79.5% | 60.2% | 79.8 |
| + RL (质量奖励) | 74.1% | 45.8% | 84.5 |
| + RL (综合奖励) | **82.1%** | **65.8%** | **85.5** |

### 关键发现

1. RL 后训练在多概念场景下改善最显著（42.3% → 65.8%,提升 55%）
2. 视觉识别奖励是最关键的组件,对多概念任务贡献最大
3. 质量奖励防止 RL 导致语言退化,确保描述的流畅性
4. RL 方法能超越 SFT 的数据质量上界——通过探索发现更好的描述策略

## 亮点与洞察

- **首创性**: 首个将 RL 后训练应用于 MLLM 个性化的工作
- **多概念突破**: 在最困难的多概念场景中优势最大
- **实际意义**: 个性化视觉助手（如识别用户的宠物、物品）有广泛应用前景

## 局限与展望

1. 奖励函数的设计依赖于实体检测器的准确性
2. RL 训练不如 SFT 稳定,需要仔细调整 KL 约束系数
3. 当前主要在静态图像上验证,视频场景未探索
4. 个性化概念数量增多时性能仍下降明显

## 相关工作与启发

- **MyVLM** (Alaluf et al.): 多模态模型个性化的先驱工作
- **Yo'LLaVA**: 基于SFT的个性化方法
- **RLHF/PPO**: LLM 对齐中的标准 RL 方法

## 评分

- ⭐ 创新性: 8/10 — RL后训练用于MLLM个性化是新颖应用
- ⭐ 实用性: 8/10 — 个性化视觉助手有广泛需求,开源代码
- ⭐ 写作质量: 7/10 — 实验充分,但方法部分细节可更清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Training Language Models to Reason Efficiently](training_language_models_to_reason_efficiently.md)
- [\[ACL 2025\] MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](../../ACL2025/reinforcement_learning/maporl_multi-agent_post-co-training_for_collaborative_large_language_models_with.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](../../ICLR2026/reinforcement_learning/post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[NeurIPS 2025\] EvoLM: In Search of Lost Language Model Training Dynamics](evolm_in_search_of_lost_language_model_training_dynamics.md)
- [\[NeurIPS 2025\] Checklists Are Better Than Reward Models For Aligning Language Models](checklists_are_better_than_reward_models_for_aligning_langua.md)

</div>

<!-- RELATED:END -->
