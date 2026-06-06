---
title: >-
  [论文解读] From Selection to Generation: A Survey of LLM-based Active Learning
description: >-
  [ACL 2025][LLM/NLP][主动学习] 首篇系统综述 LLM 时代主动学习（Active Learning）的全景图谱，提出以 Querying（选择 + 生成）× Annotation（人工 + LLM + 混合）为两轴的分类体系…
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "主动学习"
  - "大语言模型"
  - "数据选择"
  - "数据生成"
  - "标注策略"
---

# From Selection to Generation: A Survey of LLM-based Active Learning

**会议**: ACL 2025  
**arXiv**: [2502.11767](https://arxiv.org/abs/2502.11767)  
**代码**: 无  
**领域**: LLM / NLP  
**关键词**: 主动学习、大语言模型、数据选择、数据生成、标注策略

## 一句话总结

首篇系统综述 LLM 时代主动学习（Active Learning）的全景图谱，提出以 Querying（选择 + 生成）× Annotation（人工 + LLM + 混合）为两轴的分类体系，完整梳理了 LLM 如何在五步 AL 循环的每个环节中替代或增强传统方法，并拓展到 ICL、SFT、RLHF、知识蒸馏等四大 LLM 学习范式。

## 研究背景与动机

**领域现状**：主动学习（Active Learning, AL）是一种经典的"少标注高效训练"范式——通过选择最有信息量的数据点进行标注，以最小人力成本最大化模型性能。传统 AL 方法依赖不确定性（Least Confidence、Max-Entropy）和多样性（CoreSet、CDAL）两类指标，从固定的未标注数据池 $\mathcal{U}$ 中挑选样本，再由人工标注。

**现有痛点**：传统 AL 有三大根本限制。第一，**搜索空间有限**——只能从预设的未标注池中选择，无法突破数据集本身的覆盖盲区。第二，**标注成本高且单一**——只依赖人工标注者，每条标注的边际成本不变。第三，**冷启动困难**——初始无标注数据时模型 $f_\theta$ 缺乏足够信息来指导选择策略，导致前几轮采样接近随机。

**核心矛盾**：LLM 的涌现能力同时具备推理（可评估样本价值）、生成（可创造新数据）、标注（可模拟人工打标）三重角色，但现有 AL 综述仍局限在传统技术框架中，未将 LLM 的多重能力纳入统一视角。

**本文目标** (1) 缺少对 LLM 时代 AL 技术的系统分类法；(2) 缺少对 LLM 在 AL 循环各环节中角色的全面梳理；(3) 缺少对 AL 在 LLM 学习范式（ICL/SFT/RLHF/蒸馏）中应用的统一讨论。

**切入角度**：作者观察到 LLM 在 AL 中不止于"选择"——它可以生成池外的全新实例 $\mathbf{x}' \notin \mathcal{U}$，可以低成本替代人工标注，还能解决冷启动。这意味着 AL 的范式正从 Selection 走向 Generation。

**核心 idea**：以 Querying × Annotation 为两个正交维度，构建涵盖选择/生成/人工/LLM/混合的完整分类体系，首次系统综述 LLM 如何变革主动学习的全流程。

## 方法详解

### 整体框架

本文围绕 LLM-based AL 的五步循环（Initialize → Query → Annotate → Train → Stop）展开，将前人工作映射到 Querying 和 Annotation 两个核心维度上。Querying 维度分为传统选择、LLM 选择、LLM 生成、混合四类；Annotation 维度分为人工标注、LLM 标注、混合标注三类。这两个维度的交叉组合覆盖了所有已知的 LLM-based AL 方法。

### 关键设计

1. **Querying 模块——从固定池选择到开放生成**:

    - 功能：获取最有信息量的数据实例用于标注和训练
    - 核心思路：该模块沿"传统选择 → LLM 选择 → LLM 生成 → 混合"四条路线展开。**传统选择**用不确定性（Least Confidence、Margin、BALD）和多样性（CoreSet、BADGE）指标从池中选样本。**LLM 选择**让 LLM 直接评估样本价值——ActiveLLM 无监督评估不确定性和多样性，SelectLLM 用 prompt 排序后 k-NN 聚类提取 few-shot 示例，Ask-LLM 直接评估训练样本质量，ActivePrune 用 LLM 剪枝大规模未标注池。**LLM 生成**突破池的边界——池内生成用 k-NN + 困惑度策略优化 few-shot 选择（Margatina et al.），池外生成由 APE 框架用 Query-by-Committee + CoT 合成全新 prompt 或由 LLM 同时生成样本与标签再做 rejection sampling（Yang et al.）。**混合方法**如 NoiseAL 用小 LLM 筛选 + 大 LLM 标注，CAL 用密度聚类 + GPT-4 共同修正偏差。
    - 设计动机：传统选择受限于固定池的覆盖范围和冷启动问题；LLM 的推理能力让它可以直接"理解"哪些样本有价值，生成能力则将搜索空间扩展到无穷大

2. **Annotation 模块——从纯人工到人机协作**:

    - 功能：为获取的数据实例分配高质量标签
    - 核心思路：三条路线并行。**人工标注**仍是金标准——ActivePrune 和 CAL 通过精选样本减少人工量；Active-Prompt 让标注者验证 LLM 答案；Beyond-Labels 收集标签 + 自然语言解释双重信息。**LLM 标注**大幅降低成本——FreeAL 用小模型蒸馏无需人工监督，LLMaAA 加入 in-context 示例提升标注可靠性，Kholodna et al. 用 GPT-4-Turbo 标注低资源语言大幅降本。**混合标注**动态路由——Wang et al. 先 LLM 标注 + 验证器评估 + 低质部分人工复审；Rouzegar & Makrehchi 基于置信度阈值决定样本交给 LLM 还是人工。
    - 设计动机：纯人工标注成本高且不可扩展，纯 LLM 标注存在偏差（西方文化偏见、自我强化循环、prompt 敏感性），混合策略在成本-质量之间取得平衡

3. **Stopping 与 LLM 学习范式扩展**:

    - 功能：决定 AL 循环何时终止，并将 AL 技术应用到 LLM 的训练范式中
    - 核心思路：**停止策略**方面，传统 AL 用固定预算 $k$ 或性能收敛阈值，但 LLM-based AL 的成本结构复杂——同时包含人工标注费用和 LLM API 调用费用（取决于输入/输出 token 数），预算应建模为实值金额而非离散计数。Akins et al. 和 Pullar-Strecker et al. 的混合停止准则结合了 token 级成本分析和性能平台检测。**学习范式扩展**方面，AL 已渗透到四大 LLM 学习范式：(a) Active ICL——将 few-shot 示例选择建模为 AL 问题，用语义覆盖和歧义驱动采样优化 prompt；(b) Active SFT——用不确定性查询和自训练策略精选微调数据；(c) Active Preference Alignment——在 RLHF 中用针对性偏好查询加速对齐；(d) Active Knowledge Distillation——用不确定性采样选择性蒸馏 LLM 知识到小模型。
    - 设计动机：成本模型的根本变化要求重新设计停止条件；AL 从"标注效率工具"升级为"LLM 训练全流程的数据策略"

### 损失函数 / 训练策略

本文为综述论文，不提出新的损失函数。但梳理了 AL 循环中的关键训练策略：目标模型 $f_\theta$ 在每轮迭代后用新增标注数据更新参数；LLM 生成 + rejection sampling 策略确保只有满足预设准确率阈值的样本进入训练集；Active SFT 中采用自训练（self-training）在低不确定性数据上无需人工标注直接训练。

## 实验关键数据

### 主实验

本文为综述，无统一 benchmark 实验，但汇总了各代表方法在不同任务上的关键结论：

| 方法 | Querying 策略 | Annotation 策略 | 主要任务 | 核心发现 |
|------|-------------|----------------|---------|---------|
| ActiveLLM | LLM Selection | 人工 | 文本分类 | 无监督 LLM 选择在 few-shot 和 model mismatch 场景中可匹配传统 AL |
| SelectLLM | LLM Selection | 人工 | 少样本学习 | LLM 排序 + k-NN 聚类优于随机采样和不确定性选择 |
| Ask-LLM | LLM Selection | — | 数据质量过滤 | LLM 质量评分可有效剔除低质量训练数据 |
| APE | LLM Generation（池外） | 人工 | 实体匹配 | Query-by-Committee + CoT 合成新 prompt 提升标注效率 |
| FreeAL | Hybrid | LLM | 文本分类/情感分析 | 完全无人工监督下通过 LLM + 小模型蒸馏实现可用性能 |
| NoiseAL | Hybrid（选择+生成） | LLM | 文本分类 | 小 LLM 筛选 + 大 LLM 标注的两阶段流程有效降本提效 |
| CAL | Hybrid | 人工 | 去偏 | 密度聚类 + GPT-4 查询可自主识别和修正数据偏差模式 |

### 分类体系覆盖分析

| 维度 | 子类别 | 代表方法数量 | 典型方法 |
|------|--------|------------|---------|
| Querying - Traditional Selection | 不确定性/多样性 | 6+ | BADGE, BALD, CoreSet |
| Querying - LLM Selection | LLM 评估/排序 | 4 | ActiveLLM, SelectLLM, Ask-LLM, ActivePrune |
| Querying - LLM Generation | 池内/池外生成 | 5+ | APE, EAGLE, Diao et al., Yang et al. |
| Querying - Hybrid | 选择+生成混合 | 2 | NoiseAL, CAL |
| Annotation - Human | 传统人工 | 5+ | Active-Prompt, Beyond-Labels, APL |
| Annotation - LLM | LLM 标注 | 3+ | FreeAL, LLMaAA, Kholodna et al. |
| Annotation - Hybrid | 人机协作 | 3+ | Wang et al., HybridAL, AutoLabel |

### 关键发现

- **LLM 选择 vs 传统选择**：在少样本和冷启动场景中，LLM-based selection 优势明显——ActiveLLM 完全无监督即可匹配传统 AL，原因是 LLM 的语义理解能力弥补了初始模型信息不足的缺陷
- **池外生成的价值**：LLM 生成的池外数据可有效扩展训练集覆盖，特别是在标注数据稀缺时；但 rejection sampling 是保证生成质量的必要环节
- **传统不确定性采样的失效**：Margatina et al. 的实验表明，传统不确定性采样在 LLM few-shot ICL 设置下反而不如 k-NN + 多样性策略，可能因为 ICL 的工作机制与标准监督学习不同
- **LLM 标注的偏差风险**：LLM 标注存在西方文化偏见（Atari et al.）、自我强化循环（LLM 标注 LLM 生成数据时）和 prompt 敏感性三重风险，混合标注是当前最优折衷

## 亮点与洞察

- **范式质变——从 Selection 到 Generation**：传统 AL 的搜索空间被限制在固定数据池 $\mathcal{U}$ 中，LLM 将其扩展到无限的生成空间 $\mathbf{x}' \notin \mathcal{U}$。这不是渐进式改进而是范式级跃迁，意味着 AL 从"在已有数据中选最好的"变成了"创造最需要的数据"
- **二维分类法的简洁力量**：Querying × Annotation 的正交分类将所有方法映射到一个 4×3 的矩阵中，一目了然地揭示了哪些组合已被探索、哪些仍是空白。这种分类法本身就是研究路线图
- **成本模型的根本重构**：传统 AL 假设每条标注成本相同，但 LLM-based AL 的成本是 token 级别的实值函数，这要求停止准则、预算分配、路由策略全部重新设计，是一个被低估的开放问题
- **四大 LLM 范式的统一 AL 视角**：将 ICL 示例选择、SFT 数据选择、RLHF 偏好查询、知识蒸馏样本选择统一在 AL 框架下理解，提供了跨范式的方法迁移视角

## 局限与展望

- **缺少统一 benchmark**：各方法在不同数据集、不同设置下评测，无法直接横向比较——综述未提供统一实验对比，使得方法选择缺乏量化指导
- **LLM 标注质量的系统评估缺失**：综述提到了 LLM 标注的偏差和不一致性，但缺少对标注质量的系统评估框架（如什么时候 LLM 标注可靠、什么时候不可靠的判定标准）
- **理论基础薄弱**：LLM-based AL 的理论保证（如 PAC 学习框架下的样本复杂度界）几乎空白，现有方法多为经验性设计
- **多模态覆盖不足**：绝大部分方法仅针对文本任务，图像、音频、视频等模态的 AL 尚未被充分讨论
- **Multi-LLM 系统**：不同 LLM 在成本和能力上差异巨大（如 GPT-4 vs 小模型），如何像 NoiseAL 那样系统性地组合多个 LLM 来优化成本-性能 trade-off 是重要方向

## 相关工作与启发

- **vs 传统 AL 综述（Settles 2009, Ren et al. 2021, Zhan et al. 2022）**：传统综述聚焦不确定性/多样性选择策略，本文将视角扩展到 LLM 的生成和标注能力，补充了传统综述在 LLM 时代的盲区
- **vs LLM 数据合成综述**：LLM 数据合成（如 Self-Instruct）关注规模化数据生成，但缺乏 AL 的"信息量最大化"选择思想；本文的贡献在于将生成纳入 AL 的优化框架中
- **vs RLHF 文献**：RLHF 中的偏好数据收集本质上就是 AL 问题——选择哪些 prompt 让人类标注偏好。本文点出了 Active Preference Alignment 这个方向，值得 alignment 研究者关注

## 评分

- 新颖性: ⭐⭐⭐⭐ 首篇 LLM-based AL 系统综述，Querying × Annotation 分类法简洁有力，从 Selection 到 Generation 的范式转移叙事有洞察
- 实验充分度: ⭐⭐ 综述论文无原创实验，各方法的比较依赖引用文献，缺少统一 benchmark 下的对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，分类法直观，表格和图示设计合理；覆盖面广但部分方法描述偏简略
- 价值: ⭐⭐⭐⭐ 对 LLM-based AL 领域研究者是优秀的入门索引和方法定位工具，分类法可直接指导新方法设计和定位

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [\[ACL 2025\] A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?](a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)
- [\[ACL 2025\] LLM as a Broken Telephone: Iterative Generation Distorts Information](llm_broken_telephone.md)
- [\[ACL 2025\] OPTS: Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](bandit-based_prompt_design_strategy_selection_improves_prompt_optimizers.md)
- [\[ACL 2025\] Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems](transforming_podcast_preview_generation_from_expert_models_to_llm-based_systems.md)

</div>

<!-- RELATED:END -->
