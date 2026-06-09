---
title: >-
  [论文解读] Dr. Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research
description: >-
  [ICML 2026][强化学习][演化 rubric] Dr. Tulu 提出 RLER（Reinforcement Learning with Evolving Rubrics），让评估 rubric 在训练过程中与策略共同演化，把 RLVR 从短答案 QA 推广到带引用的长文深度研究任务…
tags:
  - "ICML 2026"
  - "强化学习"
  - "演化 rubric"
  - "深度研究 agent"
  - "长文回答"
  - "可验证引用"
  - "RLVR 扩展"
---

# Dr. Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research

**会议**: ICML 2026  
**arXiv**: [2511.19399](https://arxiv.org/abs/2511.19399)  
**代码**: 有  
**领域**: 强化学习 / LLM Agent / 深度研究  
**关键词**: 演化 rubric、深度研究 agent、长文回答、可验证引用、RLVR 扩展

## 一句话总结
Dr. Tulu 提出 RLER（Reinforcement Learning with Evolving Rubrics），让评估 rubric 在训练过程中与策略共同演化，把 RLVR 从短答案 QA 推广到带引用的长文深度研究任务，最终用 Qwen3-8B 训出的 DR Tulu-8B 在四个长文 deep research 基准上平均超 Tongyi DR-30B 15.6 个点，并以 1000 倍更低成本达到 OpenAI Deep Research 同等水平。

## 研究背景与动机

**领域现状**：Deep research（DR）agent 这个赛道目前要么是 training-free 的提示词工程（如 WebThinker），要么是用 RLVR 在短答案搜索 QA（HotpotQA、PopQA 类）上训练（Search-R1、ASearcher、WebExplorer 等）。后者依赖一个简单事实：短答案能用精确匹配或 F1 直接给 0/1 奖励，可验证性强。

**现有痛点**：现实里用户问深度研究问题（"综述某个临床基因突变的治疗证据"），期望的输出是带引用的长报告，而不是一句答案。这类任务有三个让 RLVR 失效的特性：(1) 评价标准本身 under-specified——好答案的判据没有标准模板；(2) 评估需要外部最新知识，光看模型参数判断不可靠；(3) 长文回答的不同维度（覆盖度、引用质量、表述）需要多准则联合打分。直接套现有 RLVR 训出来的开源 agent 在长文基准上分数极低（Search-R1 在 SQAv2 上只有 22.2）。

**核心矛盾**：长文 DR 任务**需要一个能给出稠密、可区分、覆盖最新知识的奖励信号**，但**静态人写 rubric** 太刻板（覆盖不全且易被 reward hacking 钻空子），**纯 LM 生成 rubric** 又限于模型自身参数知识而漏证据。

**本文目标**：设计一种能在 RL 训练循环中动态构建、维护、淘汰评估 rubric 的机制，让 rubric 既能融入外部检索到的最新事实，又能基于策略当前 rollout 的对比差异不断细化。

**切入角度**：作者把 rubric 视为一种"信息不对称"的工具——给 rubric 生成器**比策略多看一手东西**（外部检索结果 + 多条 on-policy 响应的对比），制造 generation–verification gap，从而保证 rubric 始终比策略本身更"懂"该任务。

**核心 idea**：让 rubric 在 RL 循环里和策略共同演化——每步用新 rollout 生成正/负 rubric，按 reward 方差排序维护一个固定大小的 buffer，淘汰区分度为零的项。

## 方法详解

### 整体框架

DR Tulu 采用 SFT-then-RL 两段式训练：
1. **冷启动 SFT**：用 GPT-5 生成 16K 条带工具调用的研究轨迹（含 search / browse / cite / answer 动作），过滤后做监督微调，让 Qwen3-8B 学会基础的搜索-写作-引用格式；
2. **RLER 主训练**：在 9K 条长文 prompt 上用 GRPO 变体做在线 RL，奖励来自动态维护的 rubric buffer + 三个辅助奖励（format / search / citation）。

模型动作空间为 $\{\text{think}, \text{tool}, \text{answer}, \text{cite}\}$，工具有三种：`google search`、`web browse`、`paper search`。每个 prompt $x$ 对应一个 rubric buffer $R_x = R_x^{\text{persist}} \cup R_x^{\text{active}}$，其中 persist 部分在训练前由"检索 + LM 生成"一次性构建并固定，active 部分在训练中动态增删。整套系统在自研的 `dr-agent-lib` 上跑，支持异步工具调用、全局缓存、token-level loss、tool output masking、sample packing。

### 关键设计

**1. 搜索增强的初始 rubric + 持久化 buffer：开训前先用真外部证据搭好一份判分基线**

长文 DR 的判别需要外部最新知识，rubric 生成器若没看检索结果就闭门造车，很容易漏掉关键事实点。所以训练开始前，对每条 prompt $x$ 先调 `SEARCH(x)` 检索相关文档，把文档连同 $x$ 喂给 rubric 生成器 $G_{\text{rubric}}$（GPT-4.1），产出一组持久 rubric $R_x^{\text{persist}} = \{R_1, \dots, R_{K_s}\}$，在整个 RL 训练里始终保留、承担"基础事实约束"。打分时单条响应 $y$ 的 rubric 得分为 $S(x, y) = \sum_k w_{x,k}\,\text{JUDGE}(r_{x,k}, y) / \sum_{k: w_{x,k} > 0} w_{x,k}$，JUDGE LM 给 $\{0, 0.5, 1\}$ 三档。这一步的深层用意是制造 generation–verification gap——让 rubric 生成器握有策略没有的"特权信息"（检索文档，以及后面会用到的多 rollout 对比），从而保证 rubric 始终比策略本身更懂这个任务，这是整个 RLER 思想的根基。

**2. 在线演化的 active rubric + 正负 rubric：让奖励标准跟着策略一起进化、并显式反 hacking**

静态 rubric 是 off-policy 的——它对所有响应一视同仁，分不清当前策略已经做得好和还做不好的地方。RLER 每个训练步对每条 prompt 采样 $G$ 条 on-policy rollouts $\{y_i\}_{i=1}^G$（含搜索过程和最终答案），连同当前 $R_x$ 喂给 $G_{\text{rubric}}$，要它产出两类新 rubric：正 rubric 捕捉某些 rollout 探到的、池里还没有的新知识或优秀模式；负 rubric 总结跨 rollout 共有的坏习惯（比如"逐字复制检索片段骗高 citation precision"这种 reward hacking），被抓到后显式扣分。于是每一步都用最新的"策略弱点 + 新探到的事实"重写打分标准，奖励信号天然 on-policy 且持续细化；负 rubric 更是一个自动的 anti-hacking 机制——模型一旦钻空子，下一步 rubric 就长出对应扣分项。

**3. 基于方差的 rubric buffer 管理：用区分度自动决定哪些 rubric 该留**

rubric 太少打分粗糙，太多则 judge 调用成本爆炸、噪音项还稀释信号，所以 buffer 必须有取舍。每次 GRPO rollout 后用当前 active rubric 给所有 $\{y_i\}$ 打分，先删掉奖励方差为 0 的 rubric（对所有 rollout 要么全对要么全错，提供不了梯度），再按 reward 标准差降序只保留 top $K_{\max}$ 条。"方差"在这里是个非常自然的区分度代理：方差为 0 意味着该 rubric 对策略改进毫无贡献，方差大才说明 rollout 之间有差异、能产生有效梯度。这套机制把"哪些 rubric 该保留"完全交给经验信号决定，省掉人工调优；同时 format / search / citation 三个辅助奖励并行计算，补上正确格式、有效搜索和高质量引用的约束。

### 损失函数 / 训练策略

RL 用 GRPO 变体，奖励为 rubric 得分 + 三个辅助奖励的加权和。训练时引入 token-level loss、1-step 异步训练、tool output masking、sample packing 以及异步工具调用（rollout 内工具请求立即派发不等批），提升长程多工具任务的 RL 吞吐。SFT 用 5 epoch、8 卡 H100、共 136 GPU 小时；最终 RL 用 16 卡 H100、约 27,000 GPU 小时。Judge 用 GPT-4.1-mini，rubric 生成器用 GPT-4.1。

## 实验关键数据

### 主实验

四个长文深度研究基准上的平均得分（SQAv2 / HealthBench / ResearchQA / DRB）：

| 类别 | 模型 | SQAv2 | HealthBench | ResearchQA | DRB | Avg |
|---|---|---|---|---|---|---|
| Closed | OpenAI Deep Research | 79.6 | 53.8 | 79.2 | 46.9 | **64.9** |
| Closed | Gemini 3 Pro + Search | 69.8 | 38.0 | 74.3 | 46.3 | 57.0 |
| Closed | Perplexity Deep Research | 67.3 | – | 75.3 | 42.3 | – |
| Open SOTA | Tongyi DR-30B | 46.5 | 46.2 | 66.7 | 40.6 | 50.0 |
| Open | WebThinker-32B-DPO | 32.9 | 11.1 | 48.6 | 23.3 | 28.9 |
| Open | Search-R1-7B | 22.2 | -0.1 | 27.9 | 9.5 | 14.9 |
| Naive RAG | Qwen3-8B | 40.4 | 16.5 | 56.1 | 33.3 | 36.5 |
| Ours | DR Tulu-8B (SFT) | 72.3 | 38.1 | 68.5 | 39.0 | 53.9 |
| Ours | **DR Tulu-8B (RL)** | **88.3** | 52.8 | 75.7 | 45.4 | **65.6** |

DR Tulu-8B 在 SQAv2 上拿到全表第一（88.3），整体平均（65.6）超 Tongyi DR-30B 15.6 个点，比 OpenAI DR 略高 0.7 个点；成本上 OpenAI DR 每 query 约 $1.80，DR Tulu-8B 工具加推理仅 $0.0018，约 1000 倍便宜。

### 消融实验

| 配置 | Avg over 4 benchmarks | 说明 |
|---|---|---|
| Qwen3-8B + Search（无训练） | 31.9 | 起点 |
| DR Tulu-8B（仅 SFT） | 53.9 | SFT 冷启动 |
| DR Tulu-8B（SFT + RLER） | 65.6 | +11.7 来自 RL 阶段 |

各基准 RLER 单独带来的提升为 6.4–16.0 个点，证明 evolving rubric 是关键。SFT 单独已经能超越多数开源 baseline，但只有 RLER 加上去才能追上 proprietary 系统。

### 关键发现

- **工具选择自适应**：DR Tulu-8B 学到了任务相关的工具偏好——SQAv2（学术综合）上 90% 使用 paper search，DRB（通用领域）上 55% 使用 web search 与 browse，没有任何硬编码。
- **小模型反超大模型**：8B 的 DR Tulu 超过 30B 的 Tongyi DR 与 32B 的 WebThinker，说明 RL 训练范式（evolving rubric）对长文 DR 的增益超过单纯的参数规模扩张。
- **引用质量是开源差距核心**：现有开源 DR agent 几乎都不输出引用，SQAv2 citation 分项几乎全 0；DR Tulu-8B 能给出 snippet 级引用，是其能在 SQAv2 超越 OpenAI DR 的主因。
- **泛化到临床基因任务**：在自建的 GeneticDiseasesQA（47 题、24 个致病变异）上，DR Tulu-8B 在 Evidence Support / Quality / Synthesis 上与 GPT-5 + Search 和 OpenAI DR 接近，说明 RLER 学到的并非数据集偏好而是通用研究模式。

## 亮点与洞察

- **Generation–verification gap 的方法论价值**：作者明确把"rubric 生成器看到比策略多的信息"作为 RLER 有效的核心理由——这其实把"AI feedback 优于自我反思"这条经验观察形式化了，可以推广到任何 RL 任务的奖励设计：让 reward model / verifier 拿到 policy 拿不到的特权信息，奖励质量会显著上一档。
- **正负 rubric 的对偶设计很优雅**：把"奖励 hacking"作为可观测信号——只要某种坏行为在多个 rollout 中共同出现，下一步 rubric 就会长出对应负项，相当于自动化的对抗式奖励校正，不需要人工 patch。
- **方差排序作为 rubric 取舍准则**：直接、可实现、可解释，比一些 rubric 选择文献里复杂的相关性建模简单很多，是工程上的好 trick。
- **基础设施开源**：`dr-agent-lib` 把异步工具调用、缓存、速率限制、tool output masking 这些 long-horizon agent RL 的基建一次性给出，是长期欠缺、明显能加速整个 DR agent 社区的工程贡献。

## 局限与展望

- RLER 强依赖一个**强且独立的 rubric 生成器**（GPT-4.1）和 **judge LM**（GPT-4.1-mini），属于"用大闭源模型蒸馏出小开源 DR agent"的范式，纯开源闭环还不成立；rubric 生成器自身的偏差会被吸收进策略。
- 计算成本不低：最终运行 27K H100 小时 + 大量 GPT-4.1 / 4.1-mini API 调用（rubric 生成与打分），实际复现门槛远高于普通 RLVR 工作。
- Rubric buffer 大小 $K_{\max}$、方差排序粒度、正负 rubric 比例等超参文中实验都没充分扫，不同领域是否要重新调有待验证。
- 训练 prompt 集（SearchArena / OpenScholar / RaR 共约 9K）相对评测集仍 partially OOD，但更长尾的专业领域（如 GeneticDiseasesQA 这种）泛化是否稳定，论文只在一个小数据集（47 题）测过，统计意义有限。

## 相关工作与启发

- **vs Search-R1 / ASearcher / WebExplorer**：它们都是短答案 QA 上的 RLVR，奖励是精确匹配；DR Tulu 把目标从短答案搬到长报告，靠 evolving rubric 解决无 ground truth 的奖励问题。
- **vs WebThinker / Ai2 ScholarQA（fixed pipeline）**：固定 pipeline 在 SQAv2 上表现不错（Ai2 ScholarQA 87.7）但不会泛化到非学术任务，且对短答案场景会过度生成；DR Tulu 是 learned policy，可在长短场景间自适应切换。
- **vs RaR (Rubrics-as-Rewards)**：RaR 用静态人写或模型一次性生成的 rubric；RLER 关键区别是 rubric 与策略共同演化，并通过检索注入特权外部知识。
- **vs OpenAI DR / Perplexity DR**：闭源系统不公开训练方法且 1000 倍成本；DR Tulu-8B 用 8B 开源模型达到相近水平，提供了一个完全可复现的 deep research baseline。

## 评分

- 新颖性: ⭐⭐⭐⭐ "rubric 共演化 + 检索特权 + 方差选 buffer"是首次组合在 long-form RLVR 上系统化，但单看每个组件都能在过往文献找到雏形。
- 实验充分度: ⭐⭐⭐⭐⭐ 四个长文基准 + 短答案分析 + 工具偏好分析 + 自建临床数据集，对比覆盖闭源、开源、固定 pipeline 三大类。
- 写作质量: ⭐⭐⭐⭐ 公式与算法描述清晰，generation-verification gap 的动机讲得透。
- 价值: ⭐⭐⭐⭐⭐ 把 RLVR 从短答案推广到长文带引用 DR，开源模型 + 数据 + 基础设施，对整个 deep research 方向是参考实现级贡献。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] DR.Q: Debiased Model-based Representations for Sample-efficient Continuous Control](debiased_model-based_representations_for_sample-efficient_continuous_control.md)
- [\[ICML 2026\] Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization](metis_learning_to_jailbreak_llms_via_self-evolving_metacognitive_policy_optimiza.md)
- [\[ICML 2026\] ORLoopBench: Solver-in-the-Loop Benchmarks for Self-Correction and Behavioral Rationality in Operations Research](orloopbench_solver-in-the-loop_benchmarks_for_self-correction_and_behavioral_rat.md)
- [\[ICML 2026\] SPHERE: Mitigating the Loss of Spectral Plasticity in Mixture-of-Experts for Deep Reinforcement Learning](sphere_mitigating_the_loss_of_spectral_plasticity_in_mixture-of-experts_for_deep.md)
- [\[NeurIPS 2025\] ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
