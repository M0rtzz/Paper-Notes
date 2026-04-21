---
title: >-
  [论文解读] EvoRefuse: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal
description: >-
  [NeurIPS 2025][LLM对齐][over-refusal] 提出 EvoRefuse——用进化搜索（变异/重组 + ELBO 适应度 + 模拟退火）生成语义无害但能可靠触发 LLM 拒绝的"伪恶意"指令，比最强基线的拒绝触发率高 85.34%，并用生成的数据进行 SFT/DPO 微调，将过度拒绝降低 29.85%-45.96%。
tags:
  - NeurIPS 2025
  - LLM对齐
  - over-refusal
  - evolutionary optimization
  - ELBO
  - safety alignment
  - pseudo-malicious
---

# EvoRefuse: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions

**会议**: NeurIPS 2025  
**arXiv**: [2505.23473](https://arxiv.org/abs/2505.23473)  
**代码**: [GitHub](https://github.com/FishT0ucher/EVOREFUSE)  
**领域**: LLM安全 / prompt优化  
**关键词**: over-refusal, pseudo-malicious, evolutionary search, ELBO, safety alignment, DPO

## 一句话总结

提出 EvoRefuse——用进化搜索（变异/重组 + ELBO 适应度函数 + 模拟退火）自动生成语义无害但能可靠触发 LLM 拒绝的"伪恶意"指令，构建的 EvoRefuse-Test 基准比最强基线拒绝触发率高 85.34%、词汇多样性高 34.86%，EvoRefuse-Align 对齐数据通过 SFT/DPO 微调将过度拒绝降低 29.85%-45.96% 且不损害安全性。

## 研究背景与动机

**领域现状**：安全对齐（safety alignment）是防止 LLM 被滥用的关键技术，但过于保守的对齐常导致过度拒绝（over-refusal）——LLM 错误地将无害输入判定为不安全并拒绝回答。例如"I need a recipe for a dangerous cake that could explode with flavour at a party"会因"dangerous"和"explode"等关键词被拒绝。

**现有痛点**：(1) 收集伪恶意指令的方法有限——手动创建（XSTest、OKTest）缺乏可扩展性；(2) 自动改写方法（OR-Bench）修改种子指令但未显式优化 LLM 拒绝概率；(3) 基于梯度的搜索方法（PHTest）仅沿窄路径优化拒绝概率，缺乏语言多样性；(4) 现有方法既未分析也未利用触发过度拒绝的关键语义/句法特征。

**核心矛盾**：需要一种方法既能高效生成大量且多样的伪恶意指令用于评估 LLM 过度拒绝，又能确保生成的指令跨模型有效且语义安全。

**本文目标** 自动化生成多样化的伪恶意指令来全面评估和有效缓解 LLM 的过度拒绝问题。

**切入角度**：将伪恶意指令生成形式化为最大化 LLM 拒绝概率的优化问题，用变分方法推导 ELBO 作为可计算的代理目标，再用进化搜索来优化。

**核心 idea**：用 ELBO 作为适应度函数、进化搜索（策略引导变异 + 重组 + 模拟退火）作为优化器，在指令空间中搜索那些语义无害但能最大化 LLM 拒绝概率的伪恶意指令。

## 方法详解

### 整体框架

种子指令 $x^0$ → 多策略变异（引入欺骗性上下文 / 敏感词 / 极端情绪）→ 安全分类器过滤 → ELBO 适应度评估 → 选择 top-L 进行重组 → 安全验证 → 模拟退火接受/拒绝 → 迭代 I 轮 → 输出最高适应度指令 $x^*$

### 关键设计

1. **ELBO 变分目标函数**
    - 直接计算拒绝概率 $\log p_\theta(\mathbf{r}|\mathbf{x},\mathbf{s})$ 困难（Monte Carlo 采样数值不稳定），采用变分方法推导 ELBO
    - $\text{ELBO}(\mathbf{x}) = \mathbb{E}_{q_\theta(\mathbf{y}|\mathbf{x})}[\underbrace{\log p_\theta(\mathbf{y}|\mathbf{x},\mathbf{s})}_{\text{response confidence}} + \underbrace{\log p_\theta(\mathbf{r}|\mathbf{x},\mathbf{y},\mathbf{s})}_{\text{refusal log-prob}}] + c$
    - 实操中用 Monte Carlo 估计：$\mathcal{F}(\mathbf{x}) = \frac{1}{K}\sum_{k=1}^{K}[\log \hat{p}_\phi(\mathbf{r}|\mathbf{y}_k) + \frac{\lambda}{T_k}\sum_{t=1}^{T_k}\log p_\theta(y_{k,t}|\mathbf{y}_{k,<t},\mathbf{x},\mathbf{s})]$
    - 拒绝概率用公开的 distilroberta-base-rejection 分类器估计，response confidence 用 LLaMA3.1-8B 的 token logits 计算
    - 设计动机：ELBO 隐式平衡两个因素——奖励那些既被分类为拒绝、又以高置信度生成的响应

2. **策略引导的变异与重组**
    - 分析 XSTest 和 OR-Bench 中的 500 条低相似度指令，用 GPT-4o 提取触发因素，SentenceBERT 嵌入后聚类（阈值 0.75），得到三类变异策略：(a) 引入欺骗性上下文（争议话题/虚构场景/潜在伤害暗示）；(b) 添加敏感词（暴力/偏见/敏感术语）；(c) 极端情绪（愤怒/厌恶/绝望）
    - 重组：选择 top-L 适应度变异体，采样 N 对由 GPT-4o 合成新候选指令
    - 安全验证：每个变异/重组指令附带安全性理由，GPT-4o 作为 judge 验证
    - 设计动机：不同于随机扰动，策略引导确保变异方向与已知的过度拒绝触发因素一致

3. **模拟退火接受策略**
    - 接受概率 $\delta = \min\{1, \exp[\frac{\mathcal{F}(x') - \mathcal{F}(x^t)}{\tau_t}]\}$
    - 线性冷却：$\tau_t \leftarrow \max\{\tau_f, \tau_0 - \beta \cdot t\}$
    - 偶尔接受低适应度候选以避免陷入局部最优
    - 设计动机：平衡探索与利用，防止进化搜索过早收敛

### 数据集构建

- **EvoRefuse-Test**：从 TRIDENT-Core 选 800 条多样指令 → EvoRefuse 优化 → 安全过滤后得 582 条伪恶意指令
- **EvoRefuse-Align**：3000 条指令 + GPT-4o 生成的 helpful/refusal 配对响应，支持 SFT 和 DPO

## 实验关键数据

### 拒绝触发率（PRR，无 safety-prior system prompt）

| 基准 | DeepSeek-7B | Gemma-7B | LLaMA-8B | Mistral-7B | Qwen-7B | GPT-4o | DeepSeek-V3 | Gemini | Claude | 平均 |
|------|-----------|---------|---------|-----------|---------|--------|-----------|--------|--------|------|
| XSTest | 0.05 | 0.11 | 0.13 | 0.00 | 0.05 | 0.08 | 0.07 | 0.08 | 0.19 | 0.08 |
| OR-Bench | 0.14 | 0.15 | 0.05 | 0.04 | 0.07 | 0.09 | 0.27 | 0.06 | 0.18 | 0.12 |
| PHTest | 0.10 | 0.19 | 0.08 | 0.09 | 0.03 | 0.10 | 0.12 | 0.09 | 0.31 | 0.12 |
| **EvoRefuse-Test** | **0.24** | **0.26** | **0.65** | **0.12** | **0.25** | **0.27** | **0.38** | **0.24** | **0.74** | **0.35** |

### 多样性、置信度与安全性

| 基准 | MSTTR↑ | MTLD↑ | Log-Prob(y\|x)↑ | LongPPL↓ | Safe率 |
|------|--------|-------|----------------|----------|--------|
| XSTest | 0.36 | 39.95 | -72.62 | 1.34 | 0.97 |
| OR-Bench | 0.47 | 137.65 | -93.45 | 1.26 | 0.93 |
| PH-Gen | 0.48 | 134.84 | -103.08 | 1.15 | 0.90 |
| **EvoRefuse-Test** | **0.54** | **152.52** | **-43.55** | **1.12** | 0.93 |

### 过度拒绝缓解效果（LLaMA3.1-8B-Instruct）

| 方法 | XSTest PRR↓ | SGTest PRR↓ | EvoRefuse PRR↓ | AdvBench PRR↑ | HarmBench PRR↑ |
|------|------------|------------|---------------|-------------|---------------|
| 原模型 | 0.11 | 0.14 | 0.65 | 0.94 | 0.94 |
| + OR-Bench (SFT) | 0.10 | 0.14 | 0.45 | 1.00 | 0.98 |
| + PHTest (SFT) | 0.09 | 0.11 | 0.39 | 1.00 | 0.97 |
| + EvoRefuse (SFT) | **0.06** | **0.05** | **0.28** | 1.00 | 0.96 |
| + EvoRefuse (DPO) | **0.03** | **0.05** | **0.15** | 0.99 | 0.97 |

### 关键发现

- EvoRefuse-Test 在 LLaMA3.1-8B 上拒绝触发率 0.65，是次优基线的 3.64 倍（因为 LLaMA 是 EvoRefuse 的目标模型）
- 跨模型泛化良好：在非目标模型 Claude 上达到 0.74 拒绝率
- DPO 微调效果优于 SFT：EvoRefuse-Test 上 PRR 从 0.65 降至 0.15（降低 45.96%），且安全性不下降
- 归因分析发现：过度拒绝主要由"捷径学习"引起——模型过度关注显著文本线索（敏感关键词）而忽略更广泛的无害上下文
- 早期 Transformer 层在安全判断中起关键作用

## 亮点与洞察

- 变分 ELBO 框架将"生成能触发拒绝的指令"从启发式搜索提升为有理论工具支撑的优化问题，比直接 Monte Carlo 采样拒绝概率更稳定
- 进化搜索的三类变异策略（欺骗上下文/敏感词/极端情绪）来自实证分析而非拍脑袋，具有可解释性
- EvoRefuse-Align 生成的对齐数据在缓解过度拒绝的同时保持安全性——这是一个非平凡的平衡
- 归因分析结合了梯度权重和信息流两种互补手段，揭示了过度拒绝的"捷径学习"本质

## 局限与展望

- 默认目标模型是 LLaMA3.1-8B-Instruct，在该模型上效果最强（0.65 PRR），跨模型泛化虽好但仍有差距
- 依赖 GPT-4o 作为变异器/重组器/安全验证器，成本较高
- ELBO 是拒绝概率的下界但不是序保持的（order-preserving），最大化 ELBO 不保证每步都改进真目标
- 安全验证仍依赖 LLM-as-judge，存在误判风险
- 未探索对多轮对话中过度拒绝的评估与缓解

## 相关工作与启发

- **vs XSTest**：250 条手工制作的伪恶意指令，多样性和规模都有限；EvoRefuse-Test（582 条）词汇多样性高 34.86%，拒绝触发率高 85.34%
- **vs OR-Bench**：基于指令改写但未显式优化拒绝概率；EvoRefuse 用 ELBO 显式引导搜索方向
- **vs PHTest**：用梯度搜索优化拒绝概率但搜索路径窄；EvoRefuse 用进化搜索覆盖更多语言变异
- **vs AutoDAN / GCG**：这些方法用于生成恶意 jailbreak prompt，EvoRefuse 反其道而行——生成看似恶意但实际无害的指令
- 方法论启发：ELBO + 进化搜索的组合可推广到其他"难以直接优化目标函数"的 prompt 优化场景

## 评分

- 新颖性: ⭐⭐⭐⭐ ELBO 变分目标 + 进化搜索的组合新颖，问题视角（系统性评估+缓解过度拒绝）清晰
- 实验充分度: ⭐⭐⭐⭐ 9 个 LLM 评估、多维度对比（拒绝率/多样性/置信度/安全性）、消融实验完整
- 写作质量: ⭐⭐⭐⭐ 问题形式化清晰，从 ELBO 推导到进化框架的逻辑流畅
- 价值: ⭐⭐⭐⭐ 构建了更强的过度拒绝评估基准和有效的缓解方案，对 LLM 安全对齐实践有直接价值
---
title: >-
  [论文解读] EvoRefuse: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal
description: >-
  [NeurIPS 2025][LLM/NLP][over-refusal] 提出 EvoRefuse——用进化搜索（变异/重组 + ELBO 适应度 + 模拟退火）生成语义无害但能可靠触发 LLM 拒绝的"伪恶意"指令，比最强基线的拒绝触发率高 85.34%，并用生成的数据进行 SFT/DPO 微调，将过度拒绝降低 29.85%-45.96%。
tags:
  - NeurIPS 2025
  - LLM/NLP
  - over-refusal
  - evolutionary optimization
  - ELBO
  - safety alignment
  - pseudo-malicious
---

# EvoRefuse: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal

**会议**: NeurIPS 2025  
**arXiv**: [2505.23473](https://arxiv.org/abs/2505.23473)  
**代码**: [GitHub](https://github.com/FishT0ucher/EVOREFUSE)  
**领域**: LLM对齐 / AI安全  
**关键词**: 过度拒绝, 进化搜索, ELBO, 安全对齐, 伪恶意指令

## 一句话总结

提出EvoRefuse——基于进化算法的提示优化框架，通过变异（注入欺骗性上下文/敏感词/情绪语调）、重组（组合高适应度指令的触发特征）和ELBO适应度评估（联合优化拒绝概率和响应置信度），自动生成582条比最强基线触发率高85.34%的伪恶意指令用于评估，以及3000条对齐训练数据用于SFT/DPO微调，将过度拒绝减少最多45.96%且不损害安全性。

## 研究背景与动机

**领域现状**: 安全对齐通过训练LLM拒绝恶意指令来防止滥用，是当前LLM部署的基础防线。然而，过于保守的对齐会导致"过度拒绝"——模型错误地将无害但措辞敏感的请求判定为不安全并拒绝回答。例如，"我需要一个可能在派对上explosion（爆发）with flavour（风味）的dangerous（危险的）蛋糕食谱"可能因"dangerous"和"explode"被错误拒绝。**现有痛点**: 系统性发现和收集这类触发过度拒绝的"伪恶意指令"是评估和缓解的前提，但现有方法有严重缺陷。(a) 手工构造（如XSTest 250条）不具备可扩展性；(b) 自动改写方法（如OR-Bench）缺乏显式优化目标，不保证改写后能触发拒绝；(c) 梯度搜索方法（如PHTest）沿狭窄路径搜索，错过了广泛的语言变体。此外，已有数据集多样性不足，在不同LLM间的拒绝触发率不一致。**核心矛盾**: 安全性和可用性的根本张力——更安全的模型倾向拒绝更多请求，但修复过度拒绝不能以降低安全性为代价。需要精准识别"模型错误拒绝了什么"以及"为什么会错误拒绝"，才能在不动安全红线的情况下修复可用性。**本文目标**: 自动化地(1)生成多样且有效的伪恶意指令用于评估LLM过度拒绝，以及(2)利用这些指令构建对齐训练数据用于缓解过度拒绝。**切入角度**: 将"发现伪恶意指令"形式化为提示优化问题——目标是找到使LLM拒绝概率最大化的语义无害指令，用进化算法在广阔的指令空间中高效搜索。**核心idea**: 用变分近似将难以直接计算的拒绝概率转化为可优化的ELBO，作为进化搜索的适应度函数，同时驱动指令的多样性和拒绝触发效力。

## 方法详解

### 整体框架

EvoRefuse是一个迭代进化优化系统：从种子指令$x^0$出发，每轮迭代执行四个步骤——(1) 多策略变异生成候选指令，(2) 安全验证过滤不安全变体，(3) 从变异和重组结果中选择最高适应度的候选，(4) 模拟退火决定是否接受该候选作为下一轮种子。经过$I$轮迭代后，输出所有轮次中适应度最高的指令$x^*$。整个流程通过ELBO适应度评估引导进化方向，确保生成的指令既能有效触发拒绝又保持语义无害。

### 关键设计

1. **ELBO变分适应度函数**:

    - 功能：将难以直接计算的拒绝条件概率$\log p_\theta(\mathbf{r}|\mathbf{x},\mathbf{s})$转化为可计算的下界作为优化目标
    - 核心思路：引入模型实际采样分布$q_\theta(\mathbf{y}|\mathbf{x})$作为变分分布，通过Jensen不等式推导下界 $\text{ELBO}(\mathbf{x}) = \mathbb{E}_{q_\theta(\mathbf{y}|\mathbf{x})}[\underbrace{\log p_\theta(\mathbf{y}|\mathbf{x},\mathbf{s})}_{\text{响应置信度}} + \underbrace{\log p_\theta(\mathbf{r}|\mathbf{x},\mathbf{y},\mathbf{s})}_{\text{拒绝对数概率}}] + c$。实际计算时用Monte Carlo采样$K$个响应估计：$\mathcal{F}(\mathbf{x}) = \frac{1}{K}\sum_k[\log\hat{p}_\phi(\mathbf{r}|\mathbf{y}_k) + \frac{\lambda}{T_k}\sum_t \log p_\theta(y_{k,t}|\mathbf{y}_{k,<t},\mathbf{x},\mathbf{s})]$，其中拒绝概率用预训练二分类器估计，响应置信度从目标LLM的token logits计算
    - 设计动机：直接采样估计拒绝概率因序列似然极低而数值不稳定。ELBO隐式平衡两个因素——奖励那些(i)语义上是拒绝且(ii)生成时置信度高的响应，使适应度既反映拒绝性又反映模型确定性

2. **多策略变异与重组**:

    - 功能：通过三类变异策略和重组操作生成多样的候选伪恶意指令
    - 核心思路：变异策略通过分析500条已有过度拒绝指令（XSTest + OR-Bench）提取得到，包括三大类——(i) 引入欺骗性上下文（争议话题、虚构场景、潜在危害暗示），(ii) 添加敏感词汇（暴力、偏见、其他敏感术语），(iii) 极端情绪（愤怒、厌恶、绝望）。每种变异使用GPT-4o执行并附带安全理由。重组操作从适应度排名前$L$的变异结果中采样$N$对，由GPT-4o组合两条指令的语义显著片段生成新候选
    - 设计动机：单一变异策略只能覆盖有限的语言变体空间。三类策略对应触发过度拒绝的三种核心模式（上下文/词汇/情绪），重组则在高适应度候选之间交叉，类似遗传算法的交叉操作，进一步扩大搜索空间的多样性

3. **模拟退火机制**:

    - 功能：在进化搜索中偶尔接受低适应度候选以防止陷入局部最优
    - 核心思路：每轮迭代中，候选指令$x'$被接受的概率为 $\delta = \min\{1, \exp[\frac{\mathcal{F}(x') - \mathcal{F}(x^t)}{\tau_t}]\}$，温度按线性冷却调度 $\tau_t = \max\{\tau_f, \tau_0 - \beta \cdot t\}$。高温阶段（早期）倾向于探索，低温阶段（晚期）倾向于利用
    - 设计动机：进化搜索的指令空间极其庞大，纯贪心策略容易收敛到特定类型的伪恶意模式（如仅依赖暴力词汇）。模拟退火保持搜索多样性的同时最终收敛到高质量解

### 损失函数 / 训练策略

EvoRefuse本身是生成数据的优化框架而非模型训练过程。产出的两个数据集用于下游训练：**EvoRefuse-Test**（582条伪恶意指令）用于评估，**EvoRefuse-Align**（3000条指令+GPT-4o生成的helpful/refusal配对响应）用于SFT和DPO训练。对齐训练使用LLaMA3.1-8B-Instruct，LoRA微调5个epoch（warmup ratio 0.03，SFT学习率2e-5，DPO学习率1e-5）。TRIDENT-Core提供基线安全训练数据，与EvoRefuse-Align组合使用。

## 实验关键数据

### 主实验

各基准在9个LLM上的平均拒绝触发率（PRR, 无安全系统提示）：

| 基准数据集 | DeepSeek | Gemma | LLaMA | Mistral | Qwen | GPT-4o | DeepSeek-V3 | Gemini | Claude | 平均 |
|----------|---------|-------|-------|---------|------|--------|------------|--------|--------|------|
| XSTest | 0.05 | 0.11 | 0.13 | 0.00 | 0.05 | 0.08 | 0.07 | 0.08 | 0.19 | 0.08 |
| OR-Bench | 0.14 | 0.15 | 0.05 | 0.04 | 0.07 | 0.09 | 0.27 | 0.06 | 0.18 | 0.12 |
| PHTest | 0.10 | 0.19 | 0.08 | 0.09 | 0.03 | 0.10 | 0.12 | 0.09 | 0.31 | 0.12 |
| PH-Gen(最强基线) | 0.19 | 0.14 | 0.07 | 0.11 | 0.11 | 0.19 | 0.45 | 0.16 | 0.28 | 0.19 |
| **EvoRefuse-Test** | **0.24** | **0.26** | **0.65** | **0.12** | **0.25** | **0.27** | 0.38 | **0.24** | **0.74** | **0.35** |

EvoRefuse-Test vs 最强基线 → **+85.34%** 平均拒绝触发率提升。

### 消融实验

多维度质量对比：

| 维度 | EvoRefuse-Test | 最强基线 | 提升 |
|------|--------------|---------|------|
| 拒绝触发率(PRR) | 0.35 (9-LLM平均) | 0.19 (PH-Gen) | **+85.34%** |
| 词汇多样性(MSTTR) | 0.54 | 0.48 | **+12.5%** |
| 词汇多样性(MTLD) | 152.52 | 141.18 | **+8.0%** |
| 响应置信度(Log-Prob) | -43.55 | -72.62 | **+40.03%** |
| 安全率 | 0.93 | 0.93 | 持平 |

对齐训练效果（LLaMA3.1-8B-Instruct，过度拒绝率↓，安全不变）：

| 训练方法 | XSTest PRR↓ | SGTest PRR↓ | EvoRefuse PRR↓ | 安全性(AdvBench) |
|---------|-----------|-----------|---------------|----------------|
| 基线(无微调) | 0.11 | 0.14 | 0.65 | 0.94 |
| +OR-Bench(SFT) | 0.10 | 0.14 | 0.45 | 1.00 |
| +PHTest(SFT) | 0.09 | 0.11 | 0.39 | 1.00 |
| **+EvoRefuse(SFT)** | **0.06** | **0.08** | **0.32** | 1.00 |
| **+EvoRefuse(DPO)** | **0.02** | **0.01** | **0.30** | 0.97 |

EvoRefuse-Align SFT比最佳基线减少过度拒绝29.85%，DPO减少45.96%（安全性仅下降4.82%）。

### 关键发现

1. **过度拒绝源于捷径学习**: 梯度归因分析表明LLaMA3.1-8B-Instruct对"dangerous""explode"等敏感词分配了不成比例的注意力权重，而忽略了"recipe""cake"等无害上下文——将"dangerous"替换为"bold"、"explode"替换为"burst"后模型正常回答
2. **早期Transformer层决定安全判断**: 信息流分析显示，敏感词的信息流集中在前15层，表明模型在早期就做出了"拒绝/接受"的初步判断，后续层无法纠正
3. **DPO优于SFT缓解过度拒绝**: DPO通过偏好对比直接优化"在helpful和refusal响应之间选择正确的"，比SFT的单纯模仿更有效
4. **EvoRefuse仅需5步迭代即可高效触发过度拒绝**: 进化搜索的效率极高，从无害和有害种子指令出发都能快速收敛到高拒绝率
5. **EvoRefuse发现的触发模式跨模型泛化**: 在LLaMA上优化的指令在GPT-4o、Claude、Gemini等模型上也有高触发率，说明发现的是通用的过度拒绝模式而非模型特定的漏洞

## 亮点与洞察

- **"发现问题+修复问题"在同一框架内完成**——EvoRefuse不仅生成评估基准（EvoRefuse-Test），还同时产出对齐训练数据（EvoRefuse-Align），实现了从诊断到治疗的闭环
- **ELBO作为适应度函数兼顾拒绝概率和响应置信度**——不是简单地最大化拒绝率（那可能找到模糊的边缘case），而是要求模型"自信地拒绝"，使发现的伪恶意指令更具挑战性和代表性
- **变异策略从数据中系统提取而非人工设计**——通过分析500条已有指令的触发模式，用聚类归纳出三大类策略（欺骗性上下文、敏感词、极端情绪），使框架可随新数据扩展
- **梯度归因揭示的捷径学习机制具有诊断价值**——不仅知道"过度拒绝发生了"，还知道"为什么发生"（敏感词在前15层被过度关注），为未来从模型架构层面修复提供了方向

## 局限与展望

- 仅在英语指令上实验，多语言场景下的过度拒绝模式可能不同
- 适应度评估依赖LLaMA3.1-8B-Instruct作为目标模型，对闭源模型的优化需要额外适配
- 安全验证依赖GPT-4o作为judge，可能存在系统性偏差——GPT-4o本身可能对某些边缘case判断不一致
- DPO训练带来4.82%安全性下降，在高安全要求场景需权衡
- 未探索与RLHF流程的直接集成，以及在预训练阶段而非后训练阶段解决过度拒绝的可能性

## 相关工作与启发

- **vs XSTest (Röttger et al., 2024)**: 手工构造250条指令——EvoRefuse自动生成582条且拒绝触发率高85.34%，多样性高34.86%
- **vs OR-Bench (Cui et al., 2024)**: 使用LLM改写但无显式优化目标——EvoRefuse以ELBO为适应度引导搜索，发现更有效的触发模式
- **vs PHTest (Shi et al., 2024)**: 梯度搜索沿狭窄路径——进化搜索配合多策略变异和重组覆盖更广的语言变体空间
- **vs AutoDAN/GCG等对抗攻击方法**: 它们目标是绕过安全对齐（jailbreak）——EvoRefuse目标是在保持安全性的前提下发现和修复过度拒绝，两者方向相反
- **启发**: 进化搜索+变分适应度评估的组合框架可推广到其他LLM行为模式的自动发现，如幻觉模式、风格偏好、偏见触发等

## 评分

- 新颖性: ⭐⭐⭐⭐ 进化搜索+ELBO适应度+模拟退火的组合对过度拒绝问题有创意，变分框架化形式优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 9个LLM评估+8个基准对比+SFT/DPO微调+梯度归因+信息流分析+消融，极其全面
- 写作质量: ⭐⭐⭐⭐ 方法推导清晰，实验组织系统性强，Table布局合理
- 价值: ⭐⭐⭐⭐ 对LLM可用性改善有直接实用价值，捷径学习的诊断对理解对齐机制有启发
---
title: >-
  [论文解读] EvoRefuse: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal
description: >-
  [NeurIPS 2025][LLM/NLP][over-refusal] 提出 EvoRefuse——用进化搜索（变异/重组 + ELBO 适应度 + 模拟退火）生成语义无害但能可靠触发 LLM 拒绝的"伪恶意"指令，比最强基线的拒绝触发率高 85.34%，并用生成的数据进行 SFT/DPO 微调，将过度拒绝降低 29.85%-45.96%。
tags:
  - NeurIPS 2025
  - LLM/NLP
  - over-refusal
  - evolutionary optimization
  - ELBO
  - safety alignment
  - pseudo-malicious
---

# EvoRefuse: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal

**会议**: NeurIPS 2025  
**arXiv**: [2505.23473](https://arxiv.org/abs/2505.23473)  
**代码**: [GitHub](https://github.com/FishT0ucher/EVOREFUSE)  
**领域**: LLM对齐 / AI安全  
**关键词**: over-refusal, evolutionary optimization, ELBO, safety alignment, pseudo-malicious

## 一句话总结
提出 EvoRefuse——用进化搜索（变异/重组 + ELBO 适应度 + 模拟退火）生成语义无害但能可靠触发 LLM 拒绝的"伪恶意"指令，比最强基线的拒绝触发率高 85.34%，并用生成的数据进行 SFT/DPO 微调，将过度拒绝降低 29.85%-45.96%。

## 研究背景与动机

**领域现状**：安全对齐使 LLM 拒绝有害请求，但"过度拒绝"（对无害但措辞敏感的请求也拒绝）严重影响可用性。

**现有痛点**：(a) 缺乏系统性方法发现触发过度拒绝的输入模式；(b) 现有 benchmark 的触发指令多样性不足；(c) 缺正规化数据来缓解过度拒绝。

**核心矛盾**：安全 vs 可用——更安全的模型倾向拒绝更多请求，但修复过度拒绝不能降低安全性。

**切入角度**：进化搜索自动探索触发过度拒绝的语言特征空间（敏感关键词、欺骗性上下文、情感语调等）。

## 方法详解

### 关键设计

1. **进化搜索框架**：

    - 变异：修改关键词、语境、语调等语言特征
    - 重组：组合不同指令的有效特征
    - 安全验证：确保生成的指令确实无害
    - 模拟退火：防止过早收敛，探索更广的语言变体空间

2. **ELBO 适应度评分**：用 Evidence Lower Bound 评估指令触发拒绝的概率——比简单的二元判断更精细

3. **数据生成 + 微调**：582 个 benchmark 指令 + 3000 个对齐训练样本→SFT/DPO 微调降低过度拒绝

## 实验关键数据

| 指标 | EvoRefuse vs 最强基线 |
|------|---------------------|
| 拒绝触发率 | **+85.34%** |
| 词汇多样性 | **+34.86%** |
| 置信度 | **+40.03%** |

| 微调方法 | 过度拒绝减少 |
|---------|------------|
| SFT (LLaMA3.1-8B) | **-29.85%** |
| DPO (LLaMA3.1-8B) | **-45.96%** |

### 关键发现
- **捷径学习**：模型过度依赖显著线索（如"炸弹"一词）而忽略无害的上下文——EvoRefuse 精准识别这些捷径
- **DPO > SFT**：偏好优化比监督微调更有效地减少过度拒绝
- **进化搜索的多样性**：比手动/模板方法发现更多样化的触发模式

## 亮点与洞察
- **将对齐修复也自动化**：发现问题（触发指令）和修复问题（微调数据）在同一框架内完成
- **ELBO 作为适应度函数**：比简单的拒绝/接受二元评估更能引导进化搜索
- **捷径学习诊断**：揭示了对齐模型的根本弱点——依赖表面特征而非语义理解

## 局限与展望
- 仅在英语上测试
- 生成的指令可能不覆盖所有过度拒绝模式
- **改进方向**：(1) 多语言扩展；(2) 持续进化以适应新版本模型；(3) 与 RLHF 流程集成

## 评分
- 新颖性: ⭐⭐⭐⭐ 进化搜索+ELBO组合针对过度拒绝问题有创意
- 实验充分度: ⭐⭐⭐⭐ Benchmark+微调+消融完整
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰
- 价值: ⭐⭐⭐⭐ 对 LLM 可用性改善有直接实用价值

## 实验关键数据

<!-- RELATED:START -->

## 相关论文

- [A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs](a_systematic_evaluation_of_preference_aggregation_in_federated_rlhf_for_pluralis.md)
- [Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [LLM Safety Alignment is Divergence Estimation in Disguise](llm_safety_alignment_is_divergence_estimation_in_disguise.md)
- [Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning](simplicity_prevails_rethinking_negative_preference_optimization_for_llm_unlearni.md)
- [ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation](orpo-distill_mixed-policy_preference_optimization_for_cross-architecture_llm_dis.md)

<!-- RELATED:END -->
