---
title: >-
  [论文解读] Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement
description: >-
  [ICLR2026][RAG] 提出 Knowledgeable-R1，一个基于强化学习的框架，通过联合采样参数知识（PK）和上下文知识（CK）的轨迹，结合局部/全局优势计算和自适应不对称优势变换，使 LLM 在 RAG 场景中能够抵抗误导性检索上下文的干扰，同时保留对可靠上下文的利用能力。
tags:
  - ICLR2026
  - RAG
  - Parametric Knowledge
  - 强化学习
  - Knowledge Conflict
  - GRPO
---

# Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement

**会议**: ICLR2026  
**arXiv**: [2506.05154](https://arxiv.org/abs/2506.05154)  
**代码**: [lcy80366872/knowledgeable-R1](https://github.com/lcy80366872/knowledgeable-R1)  
**领域**: causal_inference  
**关键词**: RAG, Parametric Knowledge, Reinforcement Learning, Knowledge Conflict, GRPO  

## 一句话总结
提出 Knowledgeable-R1，一个基于强化学习的框架，通过联合采样参数知识（PK）和上下文知识（CK）的轨迹，结合局部/全局优势计算和自适应不对称优势变换，使 LLM 在 RAG 场景中能够抵抗误导性检索上下文的干扰，同时保留对可靠上下文的利用能力。

## 背景与动机
RAG 通过引入外部检索内容来减少 LLM 的幻觉和事实错误，但当检索到的上下文包含噪声、反事实或内部矛盾信息时，LLM 往往会过度依赖这些外部信息而压制自身的参数知识，即所谓的 **context dominance** 现象。现有方法存在明显不足：

- **Prompting 方法**（如 Astute-RAG）：引导模型验证/过滤上下文，但增加计算复杂度且缺乏通用决策规则
- **Decoding 方法**（如 CK-PLUG）：调整 token 分布以缓解冲突，但同样缺乏泛化能力
- **Fine-tuning 方法**（如 Self-RAG, InFO-RAG）：需要复杂的数据标注流程，灵活性和可扩展性受限
- **标准 GRPO**：采样空间局限于带检索输入的 query+context，难以让模型探索"忽略上下文回退到参数知识"这一关键但稀有的决策

## 核心问题
如何让 LLM 在 RAG 系统中**动态决策**：何时信任检索到的上下文知识（CK），何时回退到自身参数知识（PK），并在不损害正常 RAG 性能的前提下显著提升对误导性上下文的鲁棒性？

## 方法详解

### 1. 三策略联合采样
对每个 query $q$，定义三种解码策略：

| 策略 | 输入 | 输出 | 行为 |
|------|------|------|------|
| **PK**（参数知识） | $p$ = query | $o$（基于参数知识的回答） | 纯参数知识回答 |
| **CK**（上下文感知） | $p'$ = query+context | $o'$（利用上下文的回答） | 利用可靠上下文 |
| **RPK**（鲁棒参数知识） | $p'$ = query+context | $o$（与 PK 一致的回答） | 在误导上下文下回退到 PK |

关键设计：RPK 不独立生成答案，而是将 PK 轨迹 $o^{pk}$ 作为目标，在 query+context 输入 $p'$ 下重新评估其对数概率，鼓励模型即使在误导上下文存在时仍能维持参数知识的 token。

### 2. 局部-全局优势计算
- **PK 的优势**：仅使用局部优势 $A_i^{pk\text{-}local}$（同策略内 Z-score 归一化），确保 query-only 回答尽可能准确
- **CK 的优势**：局部 + 全局优势之和 $A_j' = A_j^{ck\text{-}local} + A_j^{ck\text{-}global}$，全局项在 $p'$ 下的 CK 和 RPK 联合池中归一化，使 CK 在两种知识都正确时优先（因为上下文更新更及时）
- **RPK 的优势**：仅全局优势 $\hat{A}_i^{global}$，在同一输入 $p'$ 下与 CK 轨迹竞争，当上下文误导时 RPK 获得正优势

全局优势机制解决了组内轨迹奖励一致时仍能区分 CK vs. RPK 偏好的问题。

### 3. 知识平衡调制（Knowledge Balance Modulation）
引入不对称优势变换 $T(\hat{A}_i; \beta)$：正优势保持不变，负优势乘以系数 $\beta \in [0.01, 1]$。$\beta$ 基于 mini-batch 中 CK 和 RPK 的累积优势动态调整：

$$\beta \leftarrow \text{clip}\left(\frac{S_{ck} - S_{rpk+}}{S_{rpk-}}, 0.01, 1\right)$$

当 CK 大幅优于 RPK 时 $\beta$ 降低，减少 RPK 负优势的惩罚，鼓励更多参数知识探索；差距缩小时 $\beta$ 增大，训练更谨慎。$\beta$ 在约 8 步内收敛到稳定值。

### 4. 策略优化
采用 PPO-style 裁剪更新，总目标为三部分加权和：

$$\mathcal{J}(\theta) = \lambda_{pk} J_{PK} + \lambda_{ck} J_{CK} + \lambda_{rpk} J_{RPK}$$

实验中 $\lambda_{pk} = \lambda_{ck} = \lambda_{rpk} = 1.0$，裁剪参数 $\epsilon = 0.2$。

## 实验关键数据
在 5 种上下文场景下评估（正确/对抗/自冲突/无关/部分相关），基座模型 Qwen2.5-7B-Instruct：

| 场景 | RAG Prompting | GRPO w/ RAG | Knowledgeable-R1 | 提升 |
|------|:---:|:---:|:---:|:---:|
| S1 正确上下文 (PC-QA) | 74.35% | 80.03% | **80.90%** | +6.54% |
| S2 对抗上下文 (NC-MR) | 13.47% | 26.94% | **43.94%** | +30.47% |
| S2 对抗上下文 (NC-MC) | 8.06% | 19.74% | **37.34%** | +29.28% |
| S3 自冲突上下文 (SC) | 59.50% | 75.33% | **76.33%** | +15.92% |
| S4 无关上下文 (ExplainPE) | 62.21% | 66.50% | **67.57%** | +5.36% |
| S5 部分相关 (HotpotQA) | 20.36% | 27.93% | **31.45%** | +11.09% |

在参数知识可回答子集上，NC-MR/MC/QA 平均比 GRPO w/ RAG 提升 **+22.89%**。Llama3.1-8B-Instruct 上也有一致的提升。

**消融实验关键发现**：
- 移除 $J_{RPK}$ 后 TIFE（参数正确、上下文错误）场景性能下降最大（MC 下降 33.12%）
- 移除自适应 $\beta$ 后 TIFE 性能下降 27.39%（MC）
- 移除全局优势 $A^{ck\text{-}global}$ 导致 TIFE 下降显著

## 亮点
- **问题定义精准**：将 RAG 中的知识冲突问题明确分解为三个子目标（参数正确性、上下文利用、鲁棒回退），设计针对性的联合采样策略
- **RPK 设计巧妙**：不生成新轨迹，而是复用 PK 轨迹在 query+context 输入下重新评估，以低成本实现"有上下文但忽略它"的探索
- **自适应 $\beta$** 无需手调超参数即可在不同数据集上保持鲁棒，且收敛迅速
- **泛化能力强**：在 2WikiMultiHopQA 和 MuSiQue 上未经微调即取得显著提升
- **仅用 1% 错误上下文训练**仍优于 GRPO，说明学到的是真正的决策边界而非数据统计

## 局限与展望
- S3（自冲突）和 S5（部分相关）场景提升相对有限，上下文内部矛盾的处理仍有空间
- 未深入分析不同冲突比例（如 5 条检索结果中 1 条错 vs. 4 条错）下的敏感度
- 联合采样使约一半 rollout 预算用于 query-only PK 轨迹，S1 正确上下文场景比 GRPO w/ RAG 略低（可通过调整 $\lambda_{ck}$ 权重缓解）
- 仅在知识密集型 QA 任务上验证，未探索更复杂的多源检索环境

## 与相关工作的对比
- **vs. GRPO w/ RAG**：GRPO 仅在 query+context 下采样，缺乏参数知识探索；Knowledgeable-R1 通过 PK/RPK 分支显式鼓励参数知识回退，S2 场景平均提升 22.89%
- **vs. Self-RAG / InFO-RAG**：这些 SFT 方法依赖复杂标注流程；Knowledgeable-R1 通过 RL 自动学习决策规则，无需显式标注"何时信任上下文"
- **vs. CK-PLUG**：CK-PLUG 在解码时调整 token 概率但效果有限（S2 反而更差）；Knowledgeable-R1 直接在训练阶段优化知识利用策略
- **vs. Astute-RAG**：Astute-RAG 通过 prompting 引导模型过滤上下文，但在检索无关场景下表现欠佳；Knowledgeable-R1 全面优于它

## 启发与关联
- 思路可推广到任何"多源信息融合"场景，如多模态中视觉与文本信息冲突时的知识选择
- RPK 的"共享轨迹不同条件评估"思想可借鉴到其他 RL 训练框架中，减少额外采样开销
- 自适应 $\beta$ 的 reward shaping 策略可用于解决 RL 训练中探索不足的通用问题

## 评分
- 新颖性: 8/10 — 三策略联合采样+RPK 设计是创新点，但 PPO-style 优化本身不新
- 实验充分度: 8/10 — 5 种场景、4 个基座模型、详细消融，但缺少冲突比例敏感度分析
- 写作质量: 7/10 — 方法描述清晰但公式符号较多，部分 notation 可简化
- 价值: 8/10 — 解决 RAG 中关键的知识冲突问题，且方法简洁实用

<!-- RELATED:START -->

## 相关论文

- [Journey to the Centre of Cluster: Harnessing Interior Nodes for A/B Testing under Network Interference](journey_to_the_centre_of_cluster_harnessing_interior_nodes_for_ab_testing_under_.md)
- [A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning](../../NeurIPS2025/causal_inference/a_principle_of_targeted_intervention_for_multi-agent_reinforcement_learning.md)
- [KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education](../../AAAI2026/causal_inference/ktcf_actionable_recourse_in_knowledge_tracing_via_counterfactual_explanations_fo.md)
- [Root Cause Analysis of Outliers with Missing Structural Knowledge](../../NeurIPS2025/causal_inference/root_cause_analysis_of_outliers_with_missing_structural_knowledge.md)
- [Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](../../NeurIPS2025/causal_inference/few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)

<!-- RELATED:END -->
