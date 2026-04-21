---
title: >-
  [论文解读] UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking
description: >-
  [ICLR 2026][未索引信息检索] 识别并形式化"未索引信息检索"(UIS) 问题——搜索引擎无法直接检索的动态网页/嵌入文件/交互式内容，提出首个 UIS 基准 UIS-QA（110 题）和多 Agent 框架 UIS-Digger，以 ~30B 参数模型经 SFT+RFT 训练后达到 27.27% 准确率，超越集成 O3/GPT-4.1 的系统。
tags:
  - ICLR 2026
  - 未索引信息检索
  - 多Agent框架
  - 双模式浏览器
  - SFT+RFT训练
  - 信息检索基准
---

# UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking

**会议**: ICLR 2026  
**arXiv**: [2603.08117](https://arxiv.org/abs/2603.08117)  
**代码**: https://huggingface.co/datasets/UIS-Digger/UIS-QA  
**领域**: LLM NLP / Agent  
**关键词**: 未索引信息检索, 多Agent框架, 双模式浏览器, SFT+RFT训练, 信息检索基准

## 一句话总结
识别并形式化"未索引信息检索"(UIS) 问题——搜索引擎无法直接检索的动态网页/嵌入文件/交互式内容，提出首个 UIS 基准 UIS-QA（110 题）和多 Agent 框架 UIS-Digger，以 ~30B 参数模型经 SFT+RFT 训练后达到 27.27% 准确率，超越集成 O3/GPT-4.1 的系统。

## 研究背景与动机
**领域现状**：LLM 信息检索 Agent（WebSailor、OWL、DDv2 等）在 GAIA（70.90%）和 BrowseComp-zh（46.70%）上取得了极高成绩，但这些基准主要考察通过搜索引擎可直接获取的**索引信息**。

**关键痛点**：互联网上大量关键信息属于**未索引信息**（UIS）：政府公告的深层页面、需要多次导航才能到达的产品规格、嵌入在 PDF/XLSX 文件中的数据、需要日期选择器或过滤器交互才能显示的动态内容。当前 Agent 对这些信息无能为力。

**核心矛盾**：现有评估体系**不区分**索引与未索引信息，导致 Agent 能力被高估。SOTA Agent 在 UIS-QA 上准确率从 GAIA 的 70% 骤降至 24.55%，暴露出两个瓶颈：(a) **动作空间不足**——搜索引擎 Agent 缺乏网页交互能力；(b) **基础模型能力受限**——模型难以在大动作空间中正确决策。

**本文切入点**：UIS 不是边缘问题，而是信息检索 Agent 评估体系的**根本盲区**。作者将互联网信息严格划分为索引信息 $\mathcal{II}$ 和未索引信息 $\mathcal{UI}$，给出数学定义，并提出首个 UIS-QA 基准和 UIS-Digger 系统。

**核心idea**：通过首个专门的 UIS 基准暴露问题严重性，并用多 Agent 系统 + 领域专项训练来应对 UIS 挑战。

## 方法详解

### 整体框架
UIS-Digger 是**四 Agent 协作系统**，基于 ReAct 范式通过请求-响应消息通信。输入为用户查询，输出为最终答案。Planner 分解查询为子任务，协调三个下属 Agent：Web Searcher（索引信息检索）、Web Surfer（深层网页浏览）、File Reader（文件解析）。

### 关键设计
1. **UIS-QA 基准（110 题）**：

    - 功能：首个专门评估 Agent 获取未索引信息能力的基准
    - 核心思路：专家组导航深层网站→标注 QA 对→三重 UIS 过滤（人工 Google 搜索验证+z.ai 自动验证+DeepSeek-R1 内部知识检查），确保答案无法通过搜索引擎直接获取
    - 设计动机：现有基准（GAIA、BrowseComp）不区分 UIS，导致 Agent 评估"虚高"。UIS-QA 覆盖政府公告、产品介绍、代码仓库、游戏、公司年报等领域（84 中文+26 英文），要求答案客观、权威、时间稳定
2. **双模式浏览器（Web Surfer）**：

    - 功能：在文本模式与视觉模式间动态切换以理解不同类型的网页内容
    - 核心思路：文本模式高效处理结构化文本，视觉模式（截图）理解复杂 UI 布局（日期选择器、图表等），两种模式**共享记忆和浏览器状态**，消除同步开销
    - 设计动机：纯文本 Agent 无法处理需要视觉理解的交互元素，而纯视觉模式效率低下。动态切换实现了功能性与效率的最优平衡
    - 动作空间：点击、滚动、输入、选择下拉框、导航、提交表单、下载文件、截图等
3. **并行工具执行与文件解析**：

    - Web Searcher 可同时调用搜索引擎和爬虫工具
    - File Reader 支持 PDF/XLSX/DOCX 格式解析，超长文件按块增量读取（参考 Yu et al., 2025b）

### 训练策略
两阶段合成数据+训练：
- **数据构造**：(a) 从 100+ 真实网站深层浏览收集信息→LLM 生成 QA 对→LLM Judge 过滤；(b) 构建三类虚拟网站（航班预订、统计查询场景），针对日期选择器、单选按钮、过滤器等交互弱点定向生成训练数据
- **SFT 阶段**：使用强教师模型 $\mathcal{X}^*$（temperature=0）解题产生一条轨迹/题，LLM Judge 验证正确性和非平凡性后进行 reject sampling
- **RFT 阶段**：SFT 模型 $\mathcal{X}^s$（temp=0.4, 每题采样 4 条轨迹）自我采样，同样 reject sampling，**按难度加权**——困难问题（正确次数少）的轨迹优先保留，最终得到 $\mathcal{X}^r$

## 实验关键数据

### 主实验

| 系统 | 骨干模型 | UIS-QA | GAIA | BrowseComp-zh |
|------|---------|--------|------|---------------|
| GPT-5 直接推理 | GPT-5 | 0.9% | - | - |
| WebSailor | 32B | 7.3% | 53.2% | 25.5% |
| OWL | GPT-4.1 | 25.45% | 70.90% | 46.70% |
| DDv2 | - | 24.55% | - | - |
| **UIS-Digger** | **~30B** | **27.27%** | - | - |

### 训练策略消融

| 配置 | UIS-QA 准确率 | 说明 |
|------|-------------|------|
| 仅搜索（无浏览） | ~7% | 动作空间不足导致理论不可解 |
| 文本模式 only | ~20% | 缺少视觉模式处理动态 UI |
| 完整系统（无训练） | ~18% | 基础模型无法有效利用工具 |
| SFT only | ~23% | 冷启动有效但未充分探索 |
| **SFT + RFT** | **27.27%** | 难度加权 RFT 带来最终 4pp 提升 |

### 关键发现
- SOTA Agent 在 UIS-QA 上经历剧烈性能下降（GAIA 70% → UIS-QA 25%），证明 UIS 是独立且严峻的挑战
- ~30B 参数模型通过专项训练超越集成 O3/GPT-4.1 的通用系统，说明 UIS 需要**专门优化**
- 失败模式分析：错误搜索策略 42%、工具使用错误 28%、推理错误 30%
- 双模式浏览器和文件解析是区分 UIS 解题能力的关键能力差异

## 亮点与洞察
- **首次形式化 UIS 问题**：将互联网信息集合 $\mathcal{P}$ 严格分为索引 $\mathcal{II}$ 和未索引 $\mathcal{UI}$，并区分理想定义与实际近似，为这一被忽视的方向奠定理论基础
- 双模式浏览策略的**共享状态设计**非常巧妙——避免了多模态Agent中常见的模式切换同步问题，可迁移到其他需要多模态感知的Agent
- 虚拟网站数据生成策略值得借鉴：直接针对 Agent 弱点（如日期选择器交互）设计训练环境，用模拟取代昂贵的真实标注
- 难度加权的 RFT 策略简单有效——困难问题的正确轨迹信号更强，优先保留能更高效地提升 Agent 的弱能力

## 局限与展望
- UIS-QA 仅 110 题，规模偏小且 84/110 为中文，语言和领域覆盖有限
- 绝对准确率仅 27.27%，UIS 问题远未解决——需要更强的基础模型和更完善的工具链
- 未考虑需要登录/CAPTCHA 的网站，真实场景中这类情况非常常见
- 评估仅限于准确率，缺乏对交互步数、时间成本等效率指标的分析
- 训练数据构造依赖特定教师模型，泛化性存疑

## 相关工作与启发
- **vs GAIA/BrowseComp**：这些基准不区分 UIS，高分可能仅反映搜索引擎索引范围内的检索能力
- **vs WebArena/Mind2Web**：聚焦浏览器操作但在受控环境中评估，UIS-QA 在真实开放互联网中评估
- **vs ReAct/Reflexion**：单 Agent 动作空间有限，UIS-Digger 的多 Agent 架构覆盖搜索+浏览+文件解析的完整空间
- 启发：Agent 评估需要按信息来源细分（索引 vs 未索引），才能真实反映 Agent 能力边界

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次识别和形式化 UIS 问题，开创性贡献
- 实验充分度: ⭐⭐⭐⭐ 多系统对比全面，但 UIS-QA 规模偏小
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，形式化完整
- 价值: ⭐⭐⭐⭐⭐ 揭示信息检索 Agent 的根本评估盲区，奠定 UIS 研究基础

<!-- RELATED:START -->

## 相关论文

- [Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks](../../NeurIPS2025/llm_evaluation/belief-calibrated_multi-agent_consensus_seeking_for_complex_nlp_tasks.md)
- [A Real-world Display Inverse Rendering Dataset](../../ICCV2025/llm_evaluation/a_realworld_display_inverse_rendering_dataset.md)
- [Which LLM Multi-Agent Protocol to Choose?](which_llm_multi-agent_protocol_to_choose.md)
- [TripTailor: A Real-World Benchmark for Personalized Travel Planning](../../ACL2025/llm_evaluation/triptailor_a_real-world_benchmark_for_personalized_travel_planning.md)
- [Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)

<!-- RELATED:END -->
