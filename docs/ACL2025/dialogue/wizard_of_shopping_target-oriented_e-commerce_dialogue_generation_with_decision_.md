---
title: >-
  [论文解读] Wizard of Shopping: Target-Oriented E-commerce Dialogue Generation with Decision Tree Branching
description: >-
  [ACL 2025][对话式商品搜索] 本文提出 TRACER 方法，利用决策树模型规划对话路径，引导两个 LLM Agent（顾客和卖家）生成自然且有目标导向的电商购物对话，并发布了包含 3600 条对话的 Wizard of Shopping (WoS) 数据集，在对话查询生成和商品排序两个下游任务上验证了数据集的有效性。
tags:
  - ACL 2025
  - 对话式商品搜索
  - 决策树规划
  - LLM对话生成
  - 电商对话数据集
  - 对话状态追踪
---

# Wizard of Shopping: Target-Oriented E-commerce Dialogue Generation with Decision Tree Branching

**会议**: ACL 2025  
**arXiv**: [2502.00969](https://arxiv.org/abs/2502.00969)  
**代码**: 无  
**领域**: 文本生成 / 对话系统  
**关键词**: 对话式商品搜索, 决策树规划, LLM对话生成, 电商对话数据集, 对话状态追踪

## 一句话总结

本文提出 TRACER 方法，利用决策树模型规划对话路径，引导两个 LLM Agent（顾客和卖家）生成自然且有目标导向的电商购物对话，并发布了包含 3600 条对话的 Wizard of Shopping (WoS) 数据集，在对话查询生成和商品排序两个下游任务上验证了数据集的有效性。

## 研究背景与动机

1. **领域现状**: 对话式商品搜索 (CPS) 旨在通过对话交互理解用户购物意图、提出澄清问题并找到相关商品。现有方法主要依赖模板化的合成数据集进行训练，或使用小规模人工标注数据。

2. **现有痛点**: (1) 缺少大规模、高质量的 CPS 数据集——现有数据集要么是模板化的（不自然）、要么规模极小（如 MG-ShopDial 仅 64 条对话）、要么是私有数据无法复现；(2) 直接用 LLM 生成对话面临幻觉、提示脆弱性、缺乏可控性和规划能力的问题；(3) 先前 CPS 系统用模板化澄清问题评估，无法反映真实对话场景。

3. **核心矛盾**: 高质量 CPS 数据稀缺且构建成本高昂（人工标注expensive，LLM 直接生成不可控），这严重阻碍了智能购物助手的开发。

4. **本文目标**: 如何可控地、大规模地生成自然且目标导向的电商购物对话？

5. **切入角度**: 用决策树来规划对话中应讨论的商品属性序列（最大化搜索空间划分效率），再用 LLM 将结构化的对话规划"口语化"为自然对话。

6. **核心 idea**: 决策树做规划保证效率和可控性，LLM 做口语化保证自然度，两者互补生成高质量电商对话。

## 方法详解

### 整体框架

TRACER 分三步生成对话：(1) 顾客偏好采样——从商品目录中采样商品，将其属性随机标记为 wanted/unwanted/optional；(2) 对话规划——用决策树模型决定对话中应依次讨论哪些商品属性（选择信息增益最大的属性以最快缩小搜索空间）；(3) 口语化——将结构化的偏好和规划输入 LLM，生成自然语言对话。

### 关键设计

1. **顾客偏好采样 (Customer Preference Sampling)**:

    - 功能：为每条对话模拟一个合理的顾客购物需求
    - 核心思路：从商品目录随机选择商品 $p$，将其属性随机分为三组：wanted（直接使用商品属性值）、unwanted（替换为同属性的其他值，如 color=blue → color=red，表示顾客不想要蓝色）、optional（不关心）。由此构成偏好向量 $preference = [PC, (A_1, V_1, I_1), ..., (A_m, V_m, I_m)]$，其中 $I_i \in \{wanted, unwanted, optional\}$
    - 设计动机：模拟真实场景中顾客有明确需求、排斥项和无所谓项的情况，比全部 wanted 更自然

2. **决策树对话规划 (Decision Tree-Based Dialogue Planning)**:

    - 功能：在每一轮对话中选择最优的商品属性来提问，最小化用户搜索努力
    - 核心思路：在每一轮，根据当前已知偏好（RevPref）检索符合条件的商品集 $P_o$，用 $P_o$ 的属性构建临时训练集拟合决策树（特征是属性，标签是属性值组合的字符串）。决策树的根节点选择信息增益最大的属性，遍历决策树得到当前轮应讨论的属性序列。这个过程迭代进行，直到 $P_o$ 中所有商品都满足偏好或没有更多属性可讨论
    - 设计动机：借鉴搜索评估研究——用户满意度与搜索努力负相关，决策树确保顾客用最少的轮次找到目标商品。每轮重新拟合决策树是因为搜索空间在动态变化

3. **口语化策略 (Verbalization)**:

    - 功能：将结构化的规划和偏好转化为自然语言购物对话
    - 核心思路：提供两种生成方式——(1) 交互式生成：顾客和卖家 LLM 轮流说话，每次基于对话历史生成一句话，用对话状态追踪器监控已提及/待提及的属性；(2) 单程生成：将所有偏好、规划和商品信息一次性输入 LLM，生成整段对话。实验证明单程生成质量更高（更少上下文冲突）
    - 设计动机：交互式更真实但容易出错（Agent 无法预见未来轮次），单程生成全局一致性更好

### 提升自然度的额外策略

- **提供候选值提示**：卖家提问时给出最多 3 个最常见的属性值作为选项（如 SSD 容量：256GB/512GB/1TB），避免顾客不知道可选项
- **LLM 知识润滑**：允许 LLM 重排规划中的属性顺序并解释技术术语，使对话更符合常识
- **对话结尾**：当搜索收敛后，卖家推荐一个商品并继续最多3轮结束对话

### 损失函数 / 训练策略

本文不涉及端到端训练，核心是数据集构建。下游任务中使用 LED (Longformer Encoder-Decoder) 做 seq2seq 微调进行对话查询生成。

## 实验关键数据

### 人类评估（对话质量）

对话级评估（5分Likert量表）:

| 模型 | 策略 | 真实性 | 简洁性 | 连贯性 | 自然度 |
|------|------|--------|--------|--------|--------|
| GPT-4 | 单程 | **4.3** | **4.8** | **4.7** | **4.2** |
| GPT-4 | 交互 | 3.3 | 2.9 | 3.4 | 2.9 |
| LLaMA-2 | 单程 | 4.1 | 4.7 | 4.1 | 3.9 |
| LLaMA-2 | 交互 | 2.5 | 3.0 | 3.3 | 2.5 |

话语级评估（不良话语数量↓）:

| 模型 | 策略 | 不真实话语 | 不遵循脚本 |
|------|------|-----------|-----------|
| GPT-4 | 单程 | **0.8** | **0.1** |
| GPT-4 | 交互 | 2.0 | 0.4 |
| LLaMA-2 | 单程 | 1.2 | 1.7 |
| LLaMA-2 | 交互 | 2.9 | 1.1 |

### 下游任务：对话查询生成 (CQG)

| 方法 | F1 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------|-----|---------|---------|---------|
| Baseline (原始对话) | 0.008 | 0.137 | 0.047 | 0.087 |
| D2Q (GPT-4 few-shot) | 0.553 | 0.793 | 0.628 | 0.734 |
| D2Q (LED 在WoS上微调) | **0.834** | **0.899** | **0.822** | **0.873** |

### 消融实验

| 比较维度 | 发现 |
|---------|------|
| 单程 vs 交互 | 单程评分全面优于交互，交互方式中卖家容易在一句话中塞入过多属性 |
| GPT-4 vs LLaMA-2 | GPT-4 全面优于 LLaMA-2，符合语言模型能力排行 |
| WoS vs MG-ShopDial | WoS 有更多"引导偏好"意图，MG-ShopDial 更多"推荐"意图（因商品目录小所以提早推荐） |

### 关键发现

- 单程生成全面优于交互式生成——全局规划保证一致性，局部交互容易出现上下文冲突
- GPT-4 单程生成的对话自然度达 4.2/5（5分表示与真人对话无法区分），非常接近人类水平
- 在 WoS 上微调的小模型（LED）在 CQG 任务上大幅超过 GPT-4 few-shot（F1: 0.834 vs 0.553），证明数据集对下游任务训练的有效性
- WoS 对话明显更简洁高效（平均 19.7 轮 vs MG-ShopDial 34.3 轮），因为决策树保证了最短路径

## 亮点与洞察

- **决策树+LLM 的互补设计很聪明**：决策树解决了 LLM 不擅长的规划和可控性问题，LLM 解决了模板化生成不自然的问题，各取所长
- **单程 > 交互的反直觉发现**：虽然交互式更符合人类对话直觉，但单程生成由于全局可见性反而更稳定，这对对话数据生成有重要参考价值
- **偏好三分法（wanted/unwanted/optional）**：比之前工作只考虑 wanted 更真实，unwanted 属性的引入使对话更像真实购物场景
- **可拓展性强**：只要有新领域的商品目录，就可以无缝扩展生成新领域的对话数据

## 局限与展望

- 商品属性质量依赖原始目录，有些属性不直观（如 ASIN、Date First Available）虽然清洗了但仍有漏网
- LLM 有时不遵循指令，自行编造不在目录中的属性
- 交互式生成时卖家不会给顾客第二次选择机会（出现不连贯）
- 仅覆盖 3 个领域（Home & Kitchen、Electronics、Beauty & Personal），可扩展
- 未考虑多轮比较、退货等更复杂的购物场景

## 相关工作与启发

- **vs MG-ShopDial (Bernard & Balog 2023)**: 唯一公开的电商对话数据集，仅 64 条，商品目录极小（每类约14件）。WoS 有 3600 条，使用 236k 商品的真实目录
- **vs 模板化 CPS (Zhang et al. 2018; Bi et al. 2019)**: 之前工作用 slot-filling 模板模拟对话，不自然。本文用 LLM 口语化决策树规划，兼顾自然度和可控性
- **启发**：对话数据生成的"规划+口语化"范式可以迁移到其他任务导向对话（如医疗问诊、技术支持）

## 评分

- 新颖性: ⭐⭐⭐⭐ 决策树规划+LLM口语化的结合是对电商对话生成的创新贡献，但整体思路较工程化
- 实验充分度: ⭐⭐⭐⭐ 人类评估+下游任务验证+错误分析+与MG-ShopDial对比，较为全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，算法伪代码和示例图直观，但部分章节略冗长
- 价值: ⭐⭐⭐⭐ 发布了首个大规模目标导向电商对话数据集，对 CPS 领域有实际推动作用

<!-- RELATED:START -->

## 相关论文

- [Know Your Mistakes: Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling](know_your_mistakes_towards_preventing_overreliance_on_task-oriented_conversation.md)
- [When Harry Meets Superman: The Role of The Interlocutor in Persona-Based Dialogue Generation](when_harry_meets_superman_the_role_of_the_interlocutor_in_persona-based_dialogue.md)
- [Exploring Persona Sentiment Sensitivity in Personalized Dialogue Generation](persona_sentiment_dialogue.md)
- [Dialogue Systems for Emotional Support via Value Reinforcement](dialogue_systems_for_emotional_support_via_value_reinforcement.md)
- [BI-MDRG: Bridging Image History in Multimodal Dialogue Response Generation](../../ECCV2024/dialogue/bi-mdrg_bridging_image_history_in_multimodal_dialogue_response_generation.md)

<!-- RELATED:END -->
