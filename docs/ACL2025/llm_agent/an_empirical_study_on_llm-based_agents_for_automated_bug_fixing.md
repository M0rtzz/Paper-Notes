---
title: >-
  [论文解读] An Empirical Study on LLM-based Agents for Automated Bug Fixing
description: >-
  [ACL 2025][LLM Agent][自动缺陷修复] 本文对SWE-bench Verified上排名前六的LLM-based bug修复系统进行了系统性分析，从整体修复能力、故障定位准确率和缺陷复现有效性三个维度揭示了当前Agent系统的能力边界和改进方向。
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "自动缺陷修复"
  - "SWE-bench"
  - "故障定位"
  - "缺陷复现"
---

# An Empirical Study on LLM-based Agents for Automated Bug Fixing

**会议**: ACL 2025  
**arXiv**: [2411.10213](https://arxiv.org/abs/2411.10213)  
**代码**: [github](https://github.com/ResearchOpenRepos/bug_fixing_agent_empirical_study)  
**领域**: LLM Agent  
**关键词**: 自动缺陷修复、LLM Agent、SWE-bench、故障定位、缺陷复现

## 一句话总结
本文对SWE-bench Verified上排名前六的LLM-based bug修复系统进行了系统性分析，从整体修复能力、故障定位准确率和缺陷复现有效性三个维度揭示了当前Agent系统的能力边界和改进方向。

## 研究背景与动机

**领域现状**：LLM-based Agent系统在自动缺陷修复领域取得了显著进展，SWE-bench Verified基准上的最佳系统已能解决64.6%的真实GitHub issue。这些系统通过与开发环境交互、迭代验证和代码修改来自动完成bug修复。代表性系统包括W&B Programmer、Blackbox AI Agent、CodeStory Midwit Agent等。

**现有痛点**：尽管各系统在排行榜上竞争激烈，但缺乏对这些系统的系统性对比分析——特别是它们在哪些类型的bug上成功/失败、故障定位策略的差异、以及缺陷复现对最终修复的影响。不同系统的设计差异（如静态方法vs动态方法、单rollout vs多rollout）导致各自擅长的问题类型不同。

**核心矛盾**：排行榜上的单一解决率指标掩盖了系统间细粒度的能力差异。96个案例无法被任何系统修复（占19.2%），181个案例被所有系统修复（占36.2%），但中间地带的差异化表现才是理解系统能力的关键。

**本文目标**：系统分析顶级LLM bug修复Agent的修复能力差异、故障定位精度和缺陷复现有效性，总结当前瓶颈并提供改进方向。

**切入角度**：三个互补的研究问题——RQ1分析"能修什么、不能修什么"，RQ2分析"能否找对位置"，RQ3分析"能否正确复现bug"。

**核心 idea**：通过对六大系统的精细化对比分析，揭示LLM bug修复Agent的能力谱和提升空间。

## 方法详解

### 整体框架
选取SWE-bench Verified排行榜前六的系统：W&B Programmer、Blackbox AI Agent、CodeStory Midwit Agent、Learn-by-interact、Devlo和Emergent E1。在三个研究问题下分别进行分析。RQ1通过集合分析比较可修复/不可修复案例的特征。RQ2从提交的patch中提取文件级和代码符号级的定位信息，与golden patch对比计算定位准确率。RQ3自行实现了一个包含完整bug修复流程的RepoFixer Agent来研究缺陷复现的作用。

### 关键设计

1. **Issue质量评估体系（Issue Quality Assessment）**:

    - 功能：量化issue描述质量与修复成功率的关系
    - 核心思路：使用DeepSeek-R1对SWE-bench Verified的500个issue从五个维度打分：文件级定位信息（0-3分，从无到有stacktrace）、代码符号级定位信息（0-3分）、行级定位信息（0-3分）、可复现示例质量（0-3分）和解决方案线索（-1到3分，含误导选项）。对每个维度设计了详细的评分标准，并通过与确定性regex规则的交叉验证证实DeepSeek-R1评分的可靠性
    - 设计动机：理解issue描述质量如何影响Agent修复成功率，为issue编写最佳实践提供数据支撑

2. **多粒度故障定位分析（Multi-Granularity FL Analysis）**:

    - 功能：在文件级和代码符号级评估各系统的定位准确率
    - 核心思路：从golden patch中提取修改的文件列表和代码符号（类、函数、方法、顶层代码），从各系统提交的patch中提取相同信息进行对比。采用两种评估标准：(1) "至少命中一个"——patch覆盖了某个真实buggy文件/符号；(2) "完全覆盖"——patch覆盖了所有buggy文件/符号。计算precision、recall和F1。未做行级分析因为单行不代表完整功能模块
    - 设计动机：定位准确率是修复成功的前提，但之前没有研究系统对比不同Agent的定位能力

3. **RepoFixer——缺陷复现分析Agent**:

    - 功能：研究bug复现对修复效果的影响
    - 核心思路：由于现有系统的复现脚本难以从封闭的trajectory中提取，作者实现了一个双Agent系统：Searcher负责故障定位（搜索和阅读代码），Fixer负责生成复现脚本→验证复现→生成patch→验证patch的迭代流程。使用Claude 3.5 Sonnet作为底座模型。通过对比golden patch前后复现脚本的输出变化来判断复现质量，产生输出差异的脚本被视为"相关复现脚本"
    - 设计动机：bug复现是实际调试中的关键步骤，但其在自动修复中的具体贡献此前缺乏定量分析

### 损失函数 / 训练策略
本文为实证分析工作，不涉及模型训练。RepoFixer使用Claude 3.5 Sonnet的标准API，配合官方提供的bash和str_replace_editor工具包。

## 实验关键数据

### 主实验（RQ1: 修复效果）

| 系统 | 解决数(/500) | 解决率 | 独立解决数 | 无法解决但其他能解决 |
|------|------------|--------|----------|-------------------|
| W&B Programmer | 323 | 64.6% | 8 | 0 |
| Blackbox AI Agent | 314 | 62.8% | 5 | 9 |
| Midwit Agent | 311 | 62.2% | 4 | 13 |
| Learn-by-interact | 301 | 60.2% | 12 | 30 |
| Devlo | 291 | 58.2% | 3 | 17 |
| Emergent E1 | 286 | 57.2% | 4 | 22 |
| 任一系统可解决 | 404 | 80.8% | - | - |
| 所有系统均可解决 | 181 | 36.2% | - | - |

### Issue质量与修复关系

| 指标 | 全部可修(181例) | 全部不可修(96例) | 比值 |
|------|---------------|----------------|------|
| 总分均值 | 1.359 | 1.087 | 125% |
| 解决方案线索 | 1.276 | 0.656 | 195% |
| 符号级定位 | 1.177 | 0.906 | 130% |
| 文件级定位 | 1.326 | 1.052 | 126% |
| 可复现示例 | 0.994 | 0.813 | 122% |

### 故障定位（RQ2）

| 系统 | 文件级命中(≥1) | 文件级F1 | 符号级命中(≥1) | 符号级F1 |
|------|-------------|---------|-------------|---------|
| W&B Programmer | 448 | 最高 | 396 | 最高 |
| Blackbox AI Agent | 416 | 第二 | 371 | 第二 |
| Midwit Agent | 391 | 第四 | 355 | 第三(精度最高) |
| Learn-by-interact | 342 | 最低 | 308 | 最低 |

### 关键发现
- **Issue描述中的解决方案线索影响最大**：全部可修案例vs全部不可修案例的解决方案线索得分差异达195%，是所有维度中最显著的。说明如果issue中暗示了修复方向，Agent成功率会大幅提升
- **Learn-by-interact定位最差但修复不差**：这是最反直觉的发现——Learn-by-interact的文件级/符号级定位排名均为最后，但修复数量排第四。深入分析表明，一旦它成功定位（命中某个buggy文件/符号），后续修复成功率高达82.2%/85.1%，远超W&B Programmer的69.6%/72.7%。这说明其基于历史经验的策略在已知位置上的修复能力极强
- **代码符号级定位比文件级更关键**：符号级的准确性与最终修复成功率的相关性更强，且各系统在符号级上的差异更大，说明细粒度定位是提升空间所在
- **76.8%的复现脚本是"相关"的**：RepoFixer生成的500个复现脚本中，384个在golden patch前后产生了输出差异。issue中包含完整复现示例的案例，相关复现脚本比例高达80.4%

## 亮点与洞察
- Learn-by-interact利用历史交互经验的策略虽然在定位环节落后，但在修复环节表现突出。这揭示了"定位好≠修复好"的非直觉现象，也表明经验驱动的方法有望成为传统Agent流程的有力补充
- 五维度issue质量评估框架可以直接用于指导开发者编写更好的bug report，也可作为自动issue增强的目标函数

## 局限与展望
- SWE-bench Verified仅包含Python仓库，其他语言的bug修复模式可能不同
- 所有分析系统都是闭源的（除Learn-by-interact），无法控制其底座模型和解码配置，系统间的公平性有限
- RepoFixer的设计相对简单（未使用重采样、自反思等高级策略），其51%的解决率低于顶级系统
- 未来应该：(1) 增强LLM对issue中root cause与symptom的区分推理；(2) 设计patch完整性验证机制；(3) 利用多样化的工作流生成更多候选patch并投票选择；(4) 提升细粒度（符号级）故障定位能力

## 相关工作与启发
- **vs Agentless (Xia et al., 2024)**: Agentless采用静态、分步骤推理方法，本文分析了动态Agent方法与之的互补性
- **vs SWE-agent (Yang et al., 2024)**: SWE-agent设计了ACI接口让Agent与环境交互，本文的分析揭示单纯的交互能力不足以保证修复效果
- **vs AutoCodeRover**: 提供特定API帮助Agent定位，本文发现细粒度定位API的设计是提升修复率的关键

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统性地对比分析顶级LLM bug修复Agent
- 实验充分度: ⭐⭐⭐⭐⭐ 三个RQ、六个系统、多维度分析，实验设计非常扎实
- 写作质量: ⭐⭐⭐⭐ 分析条理清晰、insight有深度
- 价值: ⭐⭐⭐⭐⭐ 对自动缺陷修复研究有重要的指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection](../../NeurIPS2025/llm_agent/automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)
- [\[NeurIPS 2025\] Automated Multi-Agent Workflows for RTL Design](../../NeurIPS2025/llm_agent/automated_multi-agent_workflows_for_rtl_design.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](../../NeurIPS2025/llm_agent/zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)
- [\[AAAI 2026\] Promoting Sustainable Web Agents: Benchmarking and Estimating Energy Consumption Through Empirical and Theoretical Analysis](../../AAAI2026/llm_agent/promoting_sustainable_web_agents_benchmarking_and_estimating_energy_consumption_.md)
- [\[ICML 2026\] A Minimal Agent for Automated Theorem Proving](../../ICML2026/llm_agent/a_minimal_agent_for_automated_theorem_proving.md)

</div>

<!-- RELATED:END -->
