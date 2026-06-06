---
title: >-
  [论文解读] When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems
description: >-
  [ICLR 2026][社会计算][Mandela effect] 本文首次系统研究了 LLM 多智能体系统中的曼德拉效应（集体虚假记忆），提出 ManBench 基准（4838 个问题、5 种交互协议），发现所有 13 个被评估的 LLM 均易受此效应影响，并提出 prompt 级和模型级缓解策略…
tags:
  - "ICLR 2026"
  - "社会计算"
  - "Mandela effect"
  - "multi-agent systems"
  - "collective false memory"
  - "cognitive bias"
  - "misinformation"
---

# When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems

**会议**: ICLR 2026  
**arXiv**: [2602.00428](https://arxiv.org/abs/2602.00428)  
**代码**: [github.com/bluedream02/Mandela-Effect](https://github.com/bluedream02/Mandela-Effect)  
**领域**: 社会计算  
**关键词**: Mandela effect, multi-agent systems, collective false memory, cognitive bias, misinformation

## 一句话总结

本文首次系统研究了 LLM 多智能体系统中的曼德拉效应（集体虚假记忆），提出 ManBench 基准（4838 个问题、5 种交互协议），发现所有 13 个被评估的 LLM 均易受此效应影响，并提出 prompt 级和模型级缓解策略，平均减少 74.40% 的虚假记忆。

## 研究背景与动机

**领域现状**：LLM 驱动的多智能体系统广泛应用于复杂任务（公共政策分析、社会治理、合同审查），核心优势在于模拟社会动态（讨论、共识构建）。

**现有痛点**：先前研究集中于个体智能体错误（幻觉）或简单从众行为，忽视了多智能体系统中**集体认知偏差**的独特特征。曼德拉效应——群体共享的虚假记忆——涉及说服性虚假证据在交互中传播并内化为持久记忆，这与单次幻觉或短期顺从本质不同。

**核心矛盾**：现有工作将幻觉视为无状态的一次性失败，忽略了社会交互可以将虚假信念**固化为长期记忆**的过程。缺乏评估此现象的标准化基准。

**本文方案**：构建 ManBench 基准，设计 4 类易受曼德拉效应影响的任务（共 4838 题），通过 5 种交互协议（变化群体构成和记忆时间尺度）来注入并测量集体虚假记忆，并提出 prompt 级（认知锚定、来源审查）和模型级（SFT 对齐）缓解策略。

## 方法详解

### 整体框架

ManBench 由三部分组成：(1) 从 BIG-Bench Hard 策划的 20 个任务、4838 个多选题，按 4 个知识域分类；(2) 5 种交互协议（1 个基线 + 4 个虚假记忆注入）；(3) 量化曼德拉效应的评估指标体系。

### 关键设计1: 交互协议设计

5 种协议沿两个维度变化：

**群体构成维度**：
- **通用群体（Generic）**：无差异化智能体轮流发言，提供虚假证据形成简单社会共识
- **角色群体（Role-based）**：5 种专业角色协同构建多层次虚假叙事
    - Error Conclusion Initiator：提出错误答案
    - Detail Support Provider：添加虚构但可信的细节
    - Group Consensus Reinforcer：制造多数同意的假象
    - Authority Endorser：以专家身份使用学术术语赋予权威性
    - Questioning Compromiser：先质疑后被说服

**记忆时间尺度维度**：
- **短期**：在同一对话上下文中立即评估
- **长期**：经过记忆巩固（将对话提炼为信念摘要）和记忆检索（在新对话中基于信念摘要回答）

组合产生 4 种协议：GS（通用短期）、GL（通用长期）、RS（角色短期）、RL（角色长期）。

### 关键设计2: 评估指标体系

- **错误率** $\text{Err}^P = |\mathcal{Q}_{\times}^P| / |\mathcal{Q}|$
- **现实偏移率** $\sigma^P = |\mathcal{Q}_{\times}^P \cap \mathcal{Q}_{\checkmark}^B| / |\mathcal{Q}_{\checkmark}^B|$：基线正确但交互后错误的比例
- **最大现实偏移率** $\sigma_{max}$：任一协议下被颠覆的正确记忆总比例

$$\sigma_{max} = |(\mathcal{Q}_{\times}^{GS} \cup \mathcal{Q}_{\times}^{GL} \cup \mathcal{Q}_{\times}^{RS} \cup \mathcal{Q}_{\times}^{RL}) \cap \mathcal{Q}_{\checkmark}^B| / |\mathcal{Q}_{\checkmark}^B|$$

### 关键设计3: 缓解策略

**Prompt 级**：

- **认知锚定（Cognitive Anchoring）**："由内而外"，要求智能体首先建立内部知识锚点，对外部声明持怀疑态度，信念变更需举证
- **来源审查（Source Scrutiny）**："由外而内"，将智能体角色从被动接收者转为话语分析者，识别修辞模式和不自然共识

**模型级**：使用平衡数据集进行 SFT，包含韧性集（学习抵抗虚假叙事）和合作集（学习接受正确引导），防止模型过度过滤所有社交输入。

## 实验关键数据

### 主实验

13 个 LLM 的错误率和现实偏移率（%）：

| 模型 | 基线 Err | RS Err | σ^GS | σ^RS | σ^RL |
|------|----------|--------|------|------|------|
| GPT-5 | 17.63 | 41.59 | 27.42 | 31.03 | 1.67 |
| Claude 4 Sonnet | 20.48 | 45.87 | 15.45 | 35.21 | 26.56 |
| GPT-4o | 25.96 | 64.16 | 46.04 | 55.95 | 33.61 |
| Qwen3-235B | 25.48 | 74.75 | 66.98 | 68.69 | 56.85 |
| Llama3.1-8B | 44.58 | 99.67 | 61.69 | 99.47 | 32.10 |
| Claude 3.5 Haiku | 32.00 | 70.38 | 53.26 | 63.67 | 55.63 |

### 消融实验

**缓解策略效果（GPT-4o 的 σ 值，%）**：

| 方法 | σ^GS | σ^GL | σ^RS | σ^RL |
|------|------|------|------|------|
| 无防御 | 46.04 | 36.53 | 55.95 | 33.61 |
| 认知锚定 | 17.8 | 14.7 | 17.0 | 15.2 |
| 来源审查 | 26.5 | 16.0 | 25.2 | 14.5 |

**模型级防御（Llama3.1-8B）**：

| 训练集 | σ^RS | σ^C（正确引导偏移） |
|--------|------|---------------------|
| 无训练 | 99.47 | — |
| 仅韧性集 | 18.2 | 38.5（过度拒绝） |
| 韧性集 + 合作集 | 21.5 | 1.1（保持合作能力） |

### 关键发现

1. **所有 LLM 均易受影响**：即使最强的 GPT-5 在角色短期协议下错误率也翻倍（17.6% → 41.6%），Qwen3-235B 更是飙升至 74.8%
2. **角色群体 > 通用群体**：策略性叙事比简单共识更有效注入虚假记忆，Claude 4 Sonnet 的 σ 从 15.45%（GS）升至 35.21%（RS）
3. **虚假记忆可固化为长期信念**：Claude 3.5 Haiku 的 σ 从 63.67%（RS）仅降至 55.63%（RL）；而 GPT-5 展示了强自我纠正能力（31.0% → 1.67%）
4. **群体规模的倒 U 型效应**：角色群体在 6 人时影响最大，更大的群体反而触发"怀疑警觉性"使智能体自我纠正
5. **模型缩放不一定有效**：Qwen3 系列随参数增大（8B → 235B），$\sigma_{max}$ 反而从 89.3% 升至 92.2%

## 亮点与洞察

- **从幻觉到社会性虚假记忆**：不同于传统幻觉研究，本文揭示了社会交互驱动的记忆篡改机制，这是多智能体系统独有的风险
- **"怀疑触发的警觉性"**：大型协调群体反而降低影响力的发现非常反直觉，揭示了 LLM 具有检测不真实社交动态的潜在能力
- **平衡训练的必要性**：仅训练抵抗虚假信息会导致模型过度拒绝所有外部输入，韧性集+合作集的平衡方案确保了区分能力
- **知识域分析**：即使基线错误率仅 9.4% 的通识领域，σ 也达 48%，专业领域更是 67.5%，说明强知识基础并不免疫

## 局限与展望

- ManBench 使用多选题格式，简化了真实世界非结构化对话的复杂性
- 未探索开放式讨论和动态角色变换场景
- 缓解策略的泛化性有待验证（当前在医学领域 MedMCQA 上初步验证有效）
- 可引入"评论员"智能体进行交叉验证和反思
- 缺乏对不同文化和语言背景下曼德拉效应的考察

## 相关工作与启发

- **LLM 社会影响**：Weng et al. (2025) 研究从众性，Xu et al. (2024) 研究被说服性，本文将焦点从短期合规扩展到长期记忆固化
- **多智能体系统**：MetaGPT (Hong et al., 2024) 和 AutoGen (Wu et al., 2024) 展示了多智能体协作能力，但忽视了集体认知偏差风险
- **幻觉与事实鲁棒性**：Huang et al. (2025) 研究恶意智能体注入错误信息，本文则关注社会性说服导致的记忆篡改

## 评分

⭐⭐⭐⭐

本文选题新颖，首次系统化研究了多智能体系统中的曼德拉效应。ManBench 设计精巧（4 类任务 × 5 种协议 × 13 个模型），发现的倒 U 型群体规模效应和缩放悖论具有重要启示。缓解策略（尤其是平衡训练方案）实用性强。实验规模大、分析维度丰富。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](stop_wasting_your_tokens_towards_efficient_runtime_multi-agent_systems.md)
- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](../../AAAI2026/social_computing/beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)
- [\[ICML 2025\] Is Your LLM-Based Multi-Agent a Reliable Real-World Planner? Exploring Fraud Detection in Travel Planning](../../ICML2025/social_computing/is_your_llm-based_multi-agent_a_reliable_real-world_planner_exploring_fraud_dete.md)
- [\[ACL 2026\] MM-StanceDet: Retrieval-Augmented Multi-modal Multi-agent Stance Detection](../../ACL2026/social_computing/mm-stancedet_retrieval-augmented_multi-modal_multi-agent_stance_detection.md)
- [\[ACL 2026\] Why Are We Moral? An LLM-based Agent Simulation Approach to Study Moral Evolution](../../ACL2026/social_computing/why_are_we_moral_an_llm-based_agent_simulation_approach_to_study_moral_evolution.md)

</div>

<!-- RELATED:END -->
