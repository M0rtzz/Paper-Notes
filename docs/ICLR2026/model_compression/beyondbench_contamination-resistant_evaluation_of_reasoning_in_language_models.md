---
title: >-
  [论文解读] BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models
description: >-
  [ICLR 2026][模型压缩][基准评估] 提出BeyondBench评估框架，通过算法化动态生成数学问题（44个任务/117个变体/3个难度级别），确保每次测试不被训练数据污染，评估了101个语言模型（0.5B-141B参数），发现即使最强模型在Hard Suite上也仅达56%准确率…
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "基准评估"
  - "数据污染"
  - "推理能力"
  - "算法题生成"
  - "NP完全问题"
---

# BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models

**会议**: ICLR 2026  
**arXiv**: [2509.24210](https://arxiv.org/abs/2509.24210)  
**代码**: [GitHub](https://github.com/ctrl-gaurav/BeyondBench) / [PyPI](https://pypi.org/project/beyondbench/) / [排行榜](https://ctrl-gaurav.github.io/BeyondBench/)  
**领域**: LLM Evaluation / Model Compression  
**关键词**: 基准评估, 数据污染, 推理能力, 算法题生成, NP完全问题

## 一句话总结
提出BeyondBench评估框架，通过算法化动态生成数学问题（44个任务/117个变体/3个难度级别），确保每次测试不被训练数据污染，评估了101个语言模型（0.5B-141B参数），发现即使最强模型在Hard Suite上也仅达56%准确率，且不使用工具时性能大幅下降。

## 研究背景与动机
语言模型评估面临日益严重的**数据污染**问题：随着模型训练数据规模不断增长（涵盖大量互联网文本），静态基准测试的题目可能已经存在于训练数据中，使得模型可以通过"回忆"而非"推理"来获得高分。这导致基准分数虚高，无法真实反映模型的推理能力。

现有基准（如GSM8K、MATH、ARC等）都是静态数据集，一旦公开就可能被后续模型的训练数据"吸收"。虽然有些工作尝试通过数据去重来缓解，但根本问题在于静态数据集的规模有限，无法从根本上杜绝污染。

**核心矛盾**: 我们需要评估模型的"真实推理能力"，但任何公开的固定题目集都有被污染的风险。

**本文切入角度**: 彻底放弃静态题库，转向**算法化动态生成**——每次评估都在线生成全新的问题实例，问题空间超过 $10^{15}$ 种组合，使得任何预训练语料的覆盖率趋近于零。同时，每个问题都有确定性可验证的解，保证评估的客观性。

## 方法详解

### 整体框架
BeyondBench是一个可安装的Python评估包（`pip install beyondbench`），其核心思路是把"题库"换成"题目生成器"：每次评估都按指定的难度套件和难度级别在线算法化生成全新的问题实例，发给待评估模型后再用确定性正确答案逐题校验，统计准确率、指令遵循率与token效率。框架本身不训练任何模型，支持OpenAI/Gemini/Anthropic API、vLLM本地推理以及HuggingFace Transformers三类后端。

### 关键设计

**1. 三级难度任务套件：按计算复杂度搭建推理阶梯**

要让一个评估真正区分"会算术"和"会算法思维"，光有一堆题还不够，得让难度沿一条可解释的轴递增。BeyondBench把推理拆成三个由易到难的套件，对应计算复杂度的台阶。Easy Suite含29个任务，是排序、求和、均值、中位数、GCD/LCM等基础算术与统计，对应多项式时间内可解的基本运算；Medium Suite含5个任务、49个变体，转向序列模式识别（斐波那契变体、数列规律发现、模式匹配），考的是归纳推理而非死算；Hard Suite含10个任务、68个变体，直接搬来图着色、背包、旅行商变体、可满足性（SAT）等NP完全与约束满足问题，这些在计算上本质困难，逼模型做组合搜索或启发式推理。三级递进的价值在于把推理能力分层暴露——后文实验正是靠这个阶梯观察到从Easy到Hard的断崖式退化，从而判断模型究竟在"推理"还是在"模式匹配"。

**2. 抗污染三重保证：让题目无法被训练数据"记住"**

整个框架的立身之本是"题目不可能在训练语料里见过"，这靠三个机制叠加实现。其一是巨大的问题空间，每个任务的实例组合超过 $10^{15}$ 种，远超任何静态语料的覆盖能力，预训练命中率趋近于零；其二是确定性可验证解，每个生成的实例都有数学上唯一的正确答案，校验不依赖人工裁判、不存在评分歧义；其三是同构变换，对同一问题做语义等价但语法不同的改写（重新编号图节点、替换变量名等），生成"看着不同、本质一致"的变体，进一步压低靠表层记忆蒙对的概率。三者合力，使"回忆训练数据"这条捷径在BeyondBench上基本失效，分数只能来自当场的推理——这正是它相比GSM8K、MATH等静态题库的根本区别。

**3. 多维评估指标：把"答对"拆成可解释的多个侧面**

只看准确率会掩盖很多信息，所以框架对每次评估同时记录四类信号：按任务和套件分别统计的准确率（accuracy）、衡量模型是否按规定格式输出答案的指令遵循率（instruction-following compliance）、反映模型为得出答案消耗多少token的token效率，以及每个配置重复运行三次取平均的三折评估（three-fold evaluation）。把准确率和指令遵循率解耦的价值在实验里很直接——高准确率并不保证高指令遵循，拆开才能看清模型是"既会做又听话"还是"会做但不守格式"；而三折平均则抑制了动态生成带来的随机波动，让分数可比。

**4. 开箱即用的评测工具链：把基准做成可直接调用的设施**

BeyondBench不止是一篇论文，而是一套可安装的工程设施（`pip install beyondbench`），把"复现一次大规模评估"的门槛降到装一个包：命令行 `beyondbench evaluate --model-id xxx --suite easy` 一行跑评测，Python API 供编程式控制流程，`beyondbench serve` 拉起FastAPI服务以REST方式对外提供评测，`beyondbench results compare` 横向对比不同模型。正是这套工具链 + 对OpenAI/Gemini/Anthropic API、vLLM、HuggingFace 三类后端的统一封装，才支撑起后文一口气评测101个模型的规模。

## 实验关键数据

### 主实验：101个模型大规模评估
评估了85个开源模型和16个闭源模型，参数规模从0.5B到141B：

**Top 5排行榜（使用工具/推理token）**:

| 排名 | 模型 | Hard Suite准确率 | Easy Suite准确率 |
|------|------|-----------------|-----------------|
| 🥇 | GPT-5* | 未明确 | 96.15% |
| 🥈 | GPT-5-Nano* | 未明确 | 93.58% |
| 🥉 | GPT-5-Mini* | 未明确 | 94.23% |
| 4 | o3* | 未明确 | 94.96% |
| 5 | o4-Mini* | 未明确 | 95.30% |

（*使用推理/思考token的模型）

**代表性模型Hard Suite表现**:

| 模型 | Hard Suite准确率 |
|------|-----------------|
| Gemini-2.5-pro | 56.21% |
| Qwen2.5-72B | 33.37% |
| Llama-3.3-70B | 27.16% |

### 工具使用 vs 无工具的影响

| 模型 | 整体准确率下降(无工具) |
|------|----------------------|
| GPT-5 | -16.81% |
| GPT-5-mini | -15.86% (或-28.05%) |
| GPT-5-nano | -43.95% (或-47.59%) |

工具使用（如代码执行）对推理性能影响巨大，尤其对较小模型影响更为显著。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Easy→Medium→Hard | 性能逐级下降 | 从多项式到指数复杂度，性能断崖式下降 |
| 模型规模效应 | 大模型通常更好 | 但关系非严格线性 |
| 量化影响 | 多种量化方案测试 | 量化对不同任务影响不一 |
| 指令遵循 vs 准确率 | 不一致 | 高准确率不保证完美的指令遵循 |

### 关键发现
- **推理能力随复杂度急剧退化**: 即使是最强模型，从Easy到Hard的性能下降都非常显著，说明当前LLM的"推理"更多依赖模式匹配而非真正的算法思维
- **工具使用至关重要**: 不使用代码执行工具时，模型在数学和算法问题上的性能大幅下降，尤其是小模型
- **规模效应存在但有限**: 更大的模型在Hard Suite上表现更好，但70B模型与141B模型的差距远小于Easy Suite上的差异
- **开源 vs 闭源差距**: 闭源模型（尤其是有推理能力的模型如o3、GPT-5）在Hard Suite上明显领先开源模型

## 亮点与洞察
- **评估范式革新**: 从"静态题库"到"动态生成"的转变是评估方法论的重要进步，根本性解决了数据污染问题
- **规模空前**: 101个模型的横向对比提供了前所未有的全景视图
- **工程完备性**: 不仅是一篇论文，更是一个完整的开源工具——Python包、CLI、API服务器、在线排行榜，降低了使用门槛
- **NP完全问题作为推理上限**: 用计算理论中的困难问题来测试LLM，提供了关于推理能力上限的有价值洞察
- **"无工具性能vs有工具性能"的对比**: 揭示了模型真正理解问题 vs 转写为代码之间的差距

## 局限与展望
- 所有任务都是数学/算法类，未覆盖自然语言推理、常识推理、因果推理等其他推理类型
- 动态生成的问题格式可能与模型在预训练中常见的问题格式不同，存在格式偏差（format bias）
- Easy Suite的问题可能过于简单（基本算术），区分度有限
- 依赖确定性答案——无法评估需要开放式推理的能力
- 三折评估虽然提升鲁棒性，但增加了评估成本
- Hard Suite中的NP问题可能对使用暴力搜索的模型（通过代码执行）更有利，不一定反映"推理"能力

## 相关工作与启发
- 与GSM8K、MATH等静态数学基准相比，BeyondBench从根本上避免了污染问题
- 与LiveBench等动态基准类似，但BeyondBench的问题空间更大（$>10^{15}$）且覆盖NP完全问题
- 与PrOntoQA等合成推理基准相比，BeyondBench关注更广泛的算法推理而非单一推理类型
- 启发：未来的基准设计应该更多考虑"动态生成+确定性验证"的范式，而非依赖人工标注的静态数据集
- "推理能力 vs 工具使用能力"的区分对于理解和发展LLM的真正智能至关重要

## 评分
- 新颖性: ⭐⭐⭐⭐ — 动态生成评估并非全新概念，但系统性和规模空前
- 实验充分度: ⭐⭐⭐⭐⭐ — 101个模型、3个难度级别、多种量化方案、有/无工具对比
- 写作质量: ⭐⭐⭐⭐ — 摘要和框架描述清晰，但全文HTML转换失败限制了详细评价
- 价值: ⭐⭐⭐⭐⭐ — 对LLM评估社区有重大实践价值，工具已开源可直接使用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Towards Reliable Benchmarking: A Contamination Free, Controllable Evaluation Framework for Multi-step LLM Function Calling](towards_reliable_benchmarking_a_contamination_free_controllable_evaluation_frame.md)
- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [\[ACL 2026\] IntroLM: Introspective Language Models via Prefilling-Time Self-Evaluation](../../ACL2026/model_compression/introlm_introspective_language_models_via_prefilling-time_self-evaluation.md)
- [\[ICLR 2026\] Scaling Reasoning Hop Exposes Weaknesses: Demystifying and Improving Hop Generalization in Large Language Models](scaling_reasoning_hop_exposes_weaknesses_demystifying_and_improving_hop_generali.md)
- [\[ACL 2026\] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?](../../ACL2026/model_compression/lightreasoner_can_small_language_models_teach_large_language_models_reasoning.md)

</div>

<!-- RELATED:END -->
