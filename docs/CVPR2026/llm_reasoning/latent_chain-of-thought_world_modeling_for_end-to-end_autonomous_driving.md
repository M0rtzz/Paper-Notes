---
title: >-
  [论文解读] Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving
description: >-
  [CVPR 2026][LLM推理][潜空间推理] LCDrive 提出潜在链式思考（Latent CoT）框架，用动作提议token和世界模型预测token替代自然语言CoT进行推理，通过冷启动+RL后训练实现更低延迟、更好轨迹质量的端到端自动驾驶。
tags:
  - "CVPR 2026"
  - "LLM推理"
  - "潜空间推理"
  - "链式思考"
  - "世界模型"
  - "端到端驾驶"
  - "VLA模型"
---

# Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving

**会议**: CVPR 2026  
**arXiv**: [2512.10226](https://arxiv.org/abs/2512.10226)  
**代码**: 无  
**领域**: LLM推理  
**关键词**: 潜空间推理, 链式思考, 世界模型, 端到端驾驶, VLA模型

## 一句话总结
LCDrive 提出潜在链式思考（Latent CoT）框架，用动作提议token和世界模型预测token替代自然语言CoT进行推理，通过冷启动+RL后训练实现更低延迟、更好轨迹质量的端到端自动驾驶。

## 研究背景与动机
1. **领域现状**：视觉-语言-动作（VLA）模型已成为端到端自动驾驶的趋势，文本CoT推理被引入来提升长尾场景性能。
2. **现有痛点**：（i）自然语言不适合表示时空几何和多Agent交互；（ii）自回归生成长文本引入显著延迟；（iii）生成的动作可能与文本推理严重偏离（文本说"左转"但动作实际右转）。
3. **核心矛盾**：文本CoT虽然利用了LLM的推理能力，但文本不是驾驶决策的最佳表示介质。
4. **本文目标**：设计更高效、更对齐的推理表示，替代文本CoT。
5. **切入角度**：将推理表达为潜在向量空间中的结构化序列，而非自然语言。
6. **核心idea**：用动作提议token（与输出动作共享词汇表）和世界模型token（预测未来场景状态）交替构成潜在CoT。

## 方法详解

### 整体框架
三阶段训练：（1）从预训练的非推理VLA出发，冷启动潜在CoT（teacher-forcing GT世界模型状态+模型自身的动作提议）；（2）训练小型LWM预测头从提议动作预测世界模型embedding；（3）RL后训练用轨迹级奖励优化潜在推理过程和最终动作预测。

### 关键设计

1. **潜在CoT表示**:
    - 功能：提供比文本更高效、更对齐的推理痕迹
    - 核心思路：推理序列交替包含两种token——（i）动作提议token：使用与模型输出动作相同的1024码运动基元词汇（k-means聚类训练数据得到），表示候选动作；（ii）世界模型token：来自学习的潜在世界模型，表示执行候选动作后的未来场景状态。这构成了一个"提议动作→预测后果→调整动作→预测后果"的结构化推理链。
    - 设计动机：动作提议token与输出动作天然对齐（共享词汇），消除了文本CoT中动作-推理不对齐的问题。世界模型token直接编码物理交互，比文字描述更精确。

2. **冷启动+LWM预测头训练**:
    - 功能：初始化潜在推理能力并使模型能在推理时自主预测世界状态
    - 核心思路：冷启动阶段对GT未来rollout状态和模型自身动作提议做teacher-forcing，建立初始推理模式。同时训练一个小型LWM预测头，学会从提议动作预测对应的世界模型embedding，使推理时无需GT状态。
    - 设计动机：直接从随机初始化学习潜在推理非常困难，需要先建立有意义的推理scaffold。

3. **RL后训练**:
    - 功能：用轨迹级奖励优化整个推理过程
    - 核心思路：在冷启动建立的推理scaffold上，使用强化学习根据轨迹级奖励（如碰撞、舒适度、遵守交规等）优化潜在推理token和最终动作预测。RL对潜在推理模型的提升比对非推理baseline更大。
    - 设计动机：teacher-forcing仅能模仿GT行为，RL允许模型探索和发现更优的推理-决策策略。

### 损失函数 / 训练策略
冷启动：动作预测损失 + 世界模型预测损失。RL后训练：GRPO或类似的策略梯度方法，使用轨迹级综合奖励。

## 实验关键数据

### 主实验

| 方法 | 推理延迟 | 轨迹质量 | RL提升幅度 | 说明 |
|------|---------|---------|-----------|------|
| LCDrive (Latent CoT) | 最低 | 最优 | 最大 | 潜在推理 |
| Text CoT VLA | 高 | 次优 | 中等 | 自然语言推理 |
| Non-reasoning VLA | 低 | baseline | 较小 | 无推理 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full LCDrive | 最优 | 冷启动+LWM+RL |
| w/o RL后训练 | 明显下降 | RL对潜在推理提升最大 |
| w/o 世界模型token | 下降 | 仅动作提议不够 |
| w/o 冷启动 | 严重下降 | 直接RL无法建立推理 |

### 关键发现
- LCDrive比文本CoT推理延迟更低，因为潜在token序列更紧凑（无冗余自然语言token）。
- RL后训练对潜在推理模型的提升远大于对非推理模型，说明潜在CoT提供了更好的优化landscape。
- 定性分析显示，潜在CoT推理在多Agent交互场景中能做出更连贯的决策。

## 亮点与洞察
- **"推理不一定需要语言"**这一洞察非常深刻——驾驶决策的本质是空间推理而非语言推理。
- **动作-推理对齐**通过共享词汇表自然实现，消除了文本CoT的核心弱点。
- **RL+潜在推理的协同效应**是重要发现——潜在空间比语言空间更适合RL优化。

## 局限与展望
- 冷启动依赖GT未来状态，需要完整的场景标注。
- LWM预测头的精度影响推理质量。
- 目前基于单一数据集评估，泛化性需进一步验证。

## 相关工作与启发
- **vs AR1/DriveVLM**: 使用文本CoT推理，延迟高且动作-文本可能不对齐。
- **vs MILE/LAW**: 使用潜在世界模型但不用于推理链条。LCDrive将两者结合为结构化推理。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 潜在CoT替代文本CoT是概念性突破
- 实验充分度: ⭐⭐⭐⭐ 大规模驾驶数据集上的全面评估
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义精准，对比分析透彻
- 价值: ⭐⭐⭐⭐⭐ 对VLA推理范式有重要启示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Generalizable End-to-End Tool-Use RL with Synthetic CodeGym](../../ICLR2026/llm_reasoning/generalizable_end-to-end_tool-use_rl_with_synthetic_codegym.md)
- [\[CVPR 2026\] Reinforcing Structured Chain-of-Thought for Video Understanding](reinforcing_structured_chain-of-thought_for_video_understanding.md)
- [\[CVPR 2026\] Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](rationale-enhanced_decoding_for_multi-modal_chain-of-thought.md)
- [\[CVPR 2026\] Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models](understanding_and_mitigating_hallucinations_in_multimodal_chain-of-thought_model.md)
- [\[ICML 2026\] A Formal Comparison Between Chain of Thought and Latent Thought](../../ICML2026/llm_reasoning/a_formal_comparison_between_chain_of_thought_and_latent_thought.md)

</div>

<!-- RELATED:END -->
