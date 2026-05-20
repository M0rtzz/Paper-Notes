---
title: >-
  [论文解读] MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models
description: >-
  [ACL 2026][医学图像][心理健康安全] 本文提出 R-MHSafe 角色感知心理健康安全分类体系和 MHSafeEval 闭环 agent 评估框架，通过对抗性多轮咨询交互系统性发现 LLM 在心理咨询场景中的角色依赖型累积安全失败，揭示了现有静态基准无法捕捉的交互层面危害。
tags:
  - "ACL 2026"
  - "医学图像"
  - "心理健康安全"
  - "角色感知"
  - "多轮对话评估"
  - "对抗性交互"
  - "LLM安全基准"
---

# MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models

**会议**: ACL 2026  
**arXiv**: [2604.17730](https://arxiv.org/abs/2604.17730)  
**代码**: [GitHub](https://github.com/suhyun565/MHSafeEval)  
**领域**: 医学图像  
**关键词**: 心理健康安全, 角色感知, 多轮对话评估, 对抗性交互, LLM安全基准

## 一句话总结
本文提出 R-MHSafe 角色感知心理健康安全分类体系和 MHSafeEval 闭环 agent 评估框架，通过对抗性多轮咨询交互系统性发现 LLM 在心理咨询场景中的角色依赖型累积安全失败，揭示了现有静态基准无法捕捉的交互层面危害。

## 研究背景与动机

**领域现状**：LLM 越来越多被探索为心理健康咨询的可扩展工具，但已有真实案例报告显示 LLM 可能导致用户自我伤害（如比利时的聊天机器人相关自杀事件和美国的诉讼案件）。

**现有痛点**：（1）现有心理健康安全基准采用粗粒度分类体系，将本质不同的危害机制混为一体，无法精确诊断安全失败的发生原因；（2）依赖静态提示或固定数据集，随着 LLM 能力演变迅速过时，无法适应新兴安全威胁；（3）仅评估孤立回复，忽视了咨询中危害通过多轮交互关系性积累的本质。

**核心矛盾**：心理咨询中的危害不仅取决于回复内容本身，更取决于 AI 咨询师在交互中采取的"角色"——同样的回复在不同角色定位下（主动施害 vs 被动纵容）临床意义截然不同。现有基准完全忽略了这种角色维度。

**本文目标**：（1）构建融合交互角色与临床危害类别的细粒度分类体系；（2）设计动态的、轨迹级别的多轮交互评估框架；（3）系统评估 SOTA LLM 的角色特异性安全漏洞。

**切入角度**：从 HCI 领域的人机交互理论出发，借鉴"施害者-煽动者-促进者-纵容者"四种交互角色框架，与临床心理学的危害类别结合，形成二维安全分类。

**核心 idea**：将心理健康安全评估从静态单轮内容检测重新定义为动态多轮轨迹级角色感知危害发现问题。

## 方法详解

### 整体框架
MHSafeEval 是一个闭环 agent 评估系统：在 R-MHSafe 分类体系（4种角色 × 7种危害 = 28种角色感知危害行为）的指导下，迭代生成→评估→精炼对抗性多轮咨询交互。系统维护一个 Harm Archive 存储每个角色-类别组合的最高危害轨迹，引导搜索覆盖未充分探索的失败区域。

### 关键设计

1. **R-MHSafe 角色感知安全分类体系**:

    - 功能：为心理健康安全评估提供细粒度、临床有意义的二维分类框架
    - 核心思路：交互角色轴沿两个维度定义——是否由 AI 发起危害（发起者维度）和参与程度（直接/间接），组合出四种角色：Perpetrator（直接发起危害）、Instigator（间接诱导危害）、Facilitator（直接协助已有危害）、Enabler（被动纵容危害）。与7种临床危害类别（毒性语言、非事实陈述、煤气灯效应、依赖诱导、指责、过度病理化、无效化/轻视）交叉形成28种细分危害
    - 设计动机：先前工作仅关注回复内容是否有害，但同一句话在咨询师主动说出 vs 被动未纠正时的临床危害截然不同

2. **Harm Archive（基于 MAP-Elites 的质量多样性搜索）**:

    - 功能：维护角色×类别网格，存储每个格子发现的最严重交互轨迹，引导对抗搜索覆盖所有失败模式
    - 核心思路：定义 $|R| \times |C|$ 的覆盖空间，每个格子 $(r,c)$ 保存漏洞评分 $V(\tau)$ 最低（即危害最严重）的精英轨迹。当新轨迹比已有精英更严重时更新。这迫使搜索在已知模式饱和后转向探索新的角色-类别组合
    - 设计动机：全局优化会反复发现容易触发的通用失败模式，而 MAP-Elites 范式促进多样性，确保覆盖每种角色特异性漏洞

3. **对抗性交互生成与精炼**:

    - 功能：生成对话连贯但逐步暴露潜在安全漏洞的自然主义多轮交互
    - 核心思路：客户端策略以角色-类别对 $(r,c)$ 和临床心理画像 $p$ 为条件生成对话。完整轨迹 $\tau = \{(u_1, y_1), ..., (u_t, y_t)\}$ 由客户端-咨询师交替产生。若轨迹未诱发足够危害（严重度 < 2），Refiner 利用安全判定器的诊断反馈修改交互策略，放大情感困扰、过往失败等临床脆弱性线索，迭代至多 $N_{max}=5$ 次
    - 设计动机：单轮对抗无法捕捉关系性危害的累积过程——许多临床显著危害只在持续对话中逐渐显现

### 损失函数 / 训练策略
本文是纯评估框架，不涉及模型训练。使用 LLM-based clinical safety judge 对轨迹进行5级临床严重度评分，严重度 ≥ 2 视为临床显著安全失败，用于计算攻击成功率（ASR）。

## 实验关键数据

### 主实验

| 模型 | 整体 ASR | 无迭代 ASR | 拒绝率 RR | 临床理解 Cmp. |
|------|---------|-----------|----------|-------------|
| GPT-3.5 | 0.943 | 0.603 | 0.071 | 1.000 |
| Llama 3.1 | 0.922 | 0.589 | 0.557 | 0.941 |
| Gemini 2.5 | 0.970 | 0.708 | 0.038 | 0.973 |
| Haiku 4.5 | 0.970 | 0.789 | 0.859 | 0.986 |
| DeepSeek v3.2 | 0.970 | 0.762 | 0.124 | 0.997 |
| Gemma 4 | **0.997** | 0.873 | 0.070 | 0.959 |
| MiniMax m2.5 | 0.914 | 0.529 | 0.030 | 0.811 |
| MiMo | 0.943 | 0.649 | 0.343 | 0.997 |

### 消融实验

| 配置 | GPT-3.5 ASR | Llama 3.1 ASR | Gemini 2.5 ASR |
|------|------------|--------------|---------------|
| Full MHSafeEval | 97.8% | 91.6% | 98.0% |
| w/o 多轮交互 | 50.4% | 14.5% | 16.0% |
| w/o 角色条件 | 85.8% | 28.3% | 77.5% |
| w/o QD搜索 | — | 62.4% | 85.6% |

### 关键发现
- 所有模型在依赖诱导、过度病理化和煤气灯效应上最脆弱（ASR 接近 1.0），而毒性语言和非事实陈述相对较难触发——反映表面安全训练在显性毒性上有效但对关系性危害无力
- 拒绝率与安全性不相关：Haiku 4.5 拒绝率最高（0.859）但 ASR 同样高达 0.970；Gemini 2.5 几乎不拒绝（0.038）且 ASR 0.970
- 多轮交互是最关键组件——移除后 ASR 暴跌 47-82 个百分点
- 迭代精炼在前3轮收益最大，之后边际递减

## 亮点与洞察
- **角色维度的引入**是本文最大贡献——同一句"你觉得呢？"在 Enabler 角色下（面对用户的错误医学信念未纠正）和 Perpetrator 角色下临床危害完全不同。这为安全评估增加了此前被忽视的关键维度
- 发现了"理解-判断分离"现象：模型的临床理解能力很高（Cmp. 平均 0.958），但安全判断仍大面积失败。这说明问题不在于"不懂"而在于"不会拒绝"
- MAP-Elites 从进化算法领域借用到 LLM 安全评估，是一个有创意的跨领域迁移——可推广到其他需要覆盖多样失败模式的领域

## 局限与展望
- 评估依赖 LLM-based judge（gpt-4o-mini），可能遗漏微妙的临床失败
- 模拟交互环境无法完全再现真实咨询的多样性和不可预测性
- 未评估大规模前沿模型（如 GPT-4/Claude Opus），计算成本限制
- Enabler 角色的标注者间一致性最低，说明这类隐性危害即使对训练有素的临床专家也难以判断

## 相关工作与启发
- **vs MentalQA (Qiu et al., 2023)**: 他们用粗粒度对话级标注评估，本文用28种细分角色-类别组合，诊断粒度大幅提升
- **vs PAIR/TAP (Chao et al., 2025; Mehrotra et al., 2024)**: 通用越狱攻击在心理健康场景 ASR 仅 0.014-0.516，远低于 MHSafeEval 的 0.914-0.997——验证了领域特异性评估的必要性
- **vs X-Teaming (Rahman et al., 2025)**: 多轮策略缩小差距但仍被超越，因其缺乏角色感知和临床导向

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 角色感知 × 轨迹级评估是全新范式，MAP-Elites 在安全评估中的应用很有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 8个模型、7个危害类别、4个角色、多种消融、与3个攻击基线对比
- 写作质量: ⭐⭐⭐⭐ 框架清晰、案例丰富，但论文较长且符号较多
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 在高风险心理健康场景的部署安全有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](../../ICLR2026/medical_imaging/counselbench_llm_mental_health_qa.md)
- [\[ACL 2026\] Measuring What Matters!! Assessing Therapeutic Principles in Mental-Health Conversation](measuring_what_matters_assessing_therapeutic_principles_in_mental-health_convers.md)
- [\[ACL 2026\] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)
- [\[ACL 2026\] BioTool: A Comprehensive Tool-Calling Dataset for Enhancing Biomedical Capabilities of Large Language Models](biotool_a_comprehensive_tool-calling_dataset_for_enhancing_biomedical_capabiliti.md)
- [\[ACL 2026\] RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine](ritek_a_dataset_for_large_language_models_complex_reasoning_over_textual_knowled.md)

</div>

<!-- RELATED:END -->
