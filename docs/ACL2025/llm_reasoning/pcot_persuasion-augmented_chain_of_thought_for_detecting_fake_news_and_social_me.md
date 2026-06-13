---
title: >-
  [论文解读] PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation
description: >-
  [ACL 2025][LLM推理][PCoT] 提出 PCoT（Persuasion-Augmented Chain of Thought），一种两阶段零样本方法：第一阶段用融入说服知识的提示引导 LLM 识别文本中的六类说服策略，第二阶段将说服分析作为上下文融入虚假信息检测推理，在 5 个 LLM × 5 个数据集上平均 F1 提升 15%，包括 2 个全新的后截止日期数据集。
tags:
  - "ACL 2025"
  - "LLM推理"
  - "PCoT"
  - "说服增强"
  - "假新闻检测"
  - "Chain-of-Thought"
  - "零样本分类"
---

# PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation

**会议**: ACL 2025  
**arXiv**: [2506.06842](https://arxiv.org/abs/2506.06842)  
**代码**: [GitHub](https://github.com/ArkadiusDS/PCoT)  
**领域**: LLM推理 / 虚假信息检测  
**关键词**: PCoT, 说服增强, 假新闻检测, Chain-of-Thought, 零样本分类  

## 一句话总结

提出 PCoT（Persuasion-Augmented Chain of Thought），一种两阶段零样本方法：第一阶段用融入说服知识的提示引导 LLM 识别文本中的六类说服策略，第二阶段将说服分析作为上下文融入虚假信息检测推理，在 5 个 LLM × 5 个数据集上平均 F1 提升 15%，包括 2 个全新的后截止日期数据集。

## 研究背景与动机

**虚假信息检测面临零样本泛化的挑战。** 传统监督方法依赖人工标注数据，在跨数据集泛化和标注数据稀缺方面表现不佳。Lucas et al. (2023) 已证明零样本 LLM 在跨数据集测试中优于在不同数据集上微调的 BERT。然而现有零样本方法通常直接问"真还是假"，缺乏细粒度分析。

**心理学研究揭示了"识别说服可提升判断力"的规律。** Hruschka & Appel (2023) 的实验表明，当人们被训练识别说服谬误（如诉诸情感、虚假权威、稻草人论证等）后，他们区分真假新闻的能力显著提升。虚假信息的本质特征之一就是使用各种说服技巧来误导受众。

**核心 idea：将人类认知规律迁移到 LLM。** 如果教会人类识别说服技巧可以提升判断力，那么先让 LLM 分析文本中的说服策略，再基于此进行真伪判断，是否能获得类似的提升？PCoT 正是这一认知启发的计算实现。

## 方法详解

### 整体框架

PCoT 是两阶段 pipeline：第一阶段（说服检测步）让 LLM 识别并分析文本中的六类说服策略，生成每种策略的二元标签和解释；第二阶段（虚假信息检测步）将第一阶段的说服分析作为额外上下文，融入零样本二元分类进行真伪判断。两阶段使用相同的 LLM。

### 关键设计

1. **第一阶段：说服检测步（Persuasion Detection Step）**：
    - 功能：对输入文本进行多类别多标签的说服策略识别和解释
    - 核心思路：模型 $M$ 接收输入 $X=(T, I_P, K_P, G_P)$，其中 $T$ 为原文，$I_P$ 为角色设定（覆写对齐调优），$K_P$ 为融入的说服知识（六类策略定义+子技术），$G_P$ 为任务指引。输出为每种策略的结构化分析 $A_T = \{p_i: (y_{p_i}, E_{p_i}) | p_i \in P\}$，包含二元标签 $y_{p_i}$（是否存在）和解释 $E_{p_i}$
    - 六类说服策略（基于 Piskorski et al. 2023 分类体系）：攻击声誉 [AR]、辩护 [J]、简化 [S]、分散注意力 [D]、号召行动 [C]、操纵性措辞 [MW]
    - 提示变体对比：**Detailed Multitask (DMT)** 在单提示中检测所有策略（最优，F1 微平均 0.722）vs Detailed One Task At a Time (DTAT，0.689) vs Base Multitask (0.664)。DMT 比基线高 9%
    - 设计动机：融入具体知识（策略定义+子技术）比仅列名称有效；生成解释比仅输出标签鲁棒

2. **第二阶段：虚假信息检测步（Disinformation Detection Step）**：
    - 功能：基于说服分析的零样本二元分类
    - 核心思路：模型处理 $X=(T, I_D, A_T, G_D)$，其中 $A_T$ 是第一阶段的说服分析输出，$I_D$ 为检测角色设定，$G_D$ 为检测任务指引。输出 $Y_T$（虚假信息：是/否）。核心在于 $A_T$ 提供了关于文本中说服策略使用情况的结构化上下文，增强了推理的深度
    - 设计动机：两阶段分离确保说服分析的完整性不被检测任务干扰；实验证明两阶段（F1 0.815）优于合并为单一提示（F1 0.765）

3. **适配三种基线检测方法的 PCoT 变体**：
    - 功能：验证 PCoT 是否对不同提示设计都有效
    - 核心思路：将 PCoT 适配到三种竞争方法——VaN（基础提示）、Z-CoT（链式推理提示，Kojima 2022 启发）、DeF-SpeC（强调上下文/演绎/溯因推理）。每种方法都修改提示以融入第一阶段的说服分析
    - 设计动机：确保改进来自说服知识注入而非特定提示工程

### 两个全新评估数据集

- **MultiDis**：约 2000 篇英文欧洲虚假信息文章，由多国大学研究者和事实核查专家标注，三轮标注流程（86.78% 前两轮一致），覆盖 8 个主题类别
- **EUDisinfo**：约 400 篇来自 EUvsDisinfo 数据库的英文文章，来源为 EU 反虚假信息倡议
- 两数据集均仅包含 2024 年 1 月后发表的文章，确保不在任何测试 LLM 的预训练数据中

## 实验关键数据

### 主实验：PCoT vs 基线（5 模型平均）

| 方法 | F1 (± std) | vs 基线提升 |
|------|-----------|-----------|
| Base (VaN/Z-CoT/DeF-SpeC 平均) | 0.711 ± 0.055 | — |
| PCoT Single Step | 0.765 ± 0.072 | +8% |
| **PCoT (两阶段)** | **0.815 ± 0.027** | **+15%** |

### 各模型 × 各方法详细提升（选取代表性结果）

| 模型 | 方法 | Base F1 | PCoT F1 | 提升 |
|------|------|---------|---------|------|
| Gemini 1.5 Flash | VaN | 0.681 | 0.810 | +19% |
| Claude 3 Haiku | Z-CoT | 0.588 | 0.774 | +32% |
| Llama 3.3 70B | VaN | 0.740 | 0.845 | +14% |
| GPT-4o Mini | Z-CoT | 0.765 | 0.846 | +11% |

### 消融实验

| 消融配置 | F1 | 说明 |
|---------|----|----- |
| PCoT 两阶段（完整） | 0.815 | 最优 |
| PCoT 单步（一个提示同时做说服分析+检测） | 0.765 | 两阶段分离更好 |
| 无解释（仅说服标签，不生成解释） | 更低 | 解释提升鲁棒性 |
| DMT 提示（含知识） vs Base MT（不含知识） | 0.722 vs 0.664 | 知识注入提升 9% |

### 关键发现

- **PCoT 在所有 25 组（5模型×5数据集）实验中一致有效**，不是某个模型或某个数据集的偶然效果
- **弱模型受益更大**：Claude 3 Haiku + Z-CoT 获得最大提升（+32%），说明说服知识对推理能力较弱的模型补偿效果更强
- **后截止数据集上同样有效**：在 MultiDis 和 EUDisinfo（2024 年后发表，未见训练数据）上提升与先截止数据集相当，证明非记忆效应
- **两阶段优于单步**：分离说服分析和检测任务让每阶段更专注，F1 差 5 个点
- **社交媒体帖子 vs 新闻文章**：文章上提升更大（因篇幅更长，说服策略更明显）

## 亮点与洞察

- **认知科学驱动的 NLP 方法设计**：从心理学实验"教人识别说服可提升判断力"出发设计计算方法——这种跨学科迁移比纯工程试错更有启发性
- **零样本且通用**：不需要训练数据、不需要微调、跨模型跨数据集均有效——实际部署门槛极低
- **两阶段设计的合理性有实验支撑**：单步显著弱于两阶段，说明任务分离不是过度工程
- **新数据集的构建质量高**：MultiDis 有三轮标注、86.78% 前两轮一致性、事实核查专家参与，EUDisinfo 来自权威欧盟数据库
- **知识注入比提示技巧更重要**：DMT（含策略定义和子技术知识）比 Base MT（仅列名称）高 9%

## 局限与展望

- **依赖 LLM 自身的说服知识质量**：如果模型对某种说服策略理解不够，第一阶段分析可能不准确
- **两阶段增加推理成本**：需要两次 LLM 调用，token 消耗和延迟翻倍
- **仅评估英文**：虚假信息检测的跨语言泛化未测试
- **说服策略分类体系可能不完整**：六类策略可能未覆盖所有虚假信息中的操纵手法
- **社交媒体短文本（如推文）上提升较小**：短文本中说服策略不够明显

## 相关工作与启发

- **vs Lucas et al. (2023)**：提供了 VaN/Z-CoT/DeF-SpeC 三种基线方法，PCoT 在所有三种上均有改进，证明提升来自说服知识而非提示技巧
- **vs 监督方法 (BERT 微调)**：Lucas et al. 已证明零样本 LLM 跨数据集优于微调 BERT，PCoT 进一步拉大差距
- **vs Kojima et al. (2022) Z-CoT**：Z-CoT 引导逐步推理但不注入领域知识，PCoT 通过说服知识为推理提供具体的分析维度
- **启发**：领域知识注入（而非通用推理增强）可能是零样本方法提升的更高效路径——类似思路可推广到其他需要专业判断的分类任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 说服知识融入假新闻检测是新角度，认知科学驱动的方法设计思路有前瞻性
- 实验充分度: ⭐⭐⭐⭐⭐ 5模型×5数据集（含2个未见数据集）+统计显著性检验+消融实验，非常充分
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法形式化完整，数据集构建透明
- 价值: ⭐⭐⭐⭐ 对虚假信息检测有实用价值，零样本方法可直接部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] DRT: Deep Reasoning Translation via Long Chain-of-Thought](drt_deep_reasoning_translation_via_long_chain-of-thought.md)
- [\[ACL 2025\] Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions](improving_chain-of-thought_reasoning_via_quasi-symbolic_abstractions.md)
- [\[ACL 2025\] Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning](unveiling_the_key_factors_for_distilling_chain-of-thought_reasoning.md)
- [\[ACL 2025\] TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning](tract_regression_cot.md)
- [\[ACL 2025\] CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought](cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)

</div>

<!-- RELATED:END -->
