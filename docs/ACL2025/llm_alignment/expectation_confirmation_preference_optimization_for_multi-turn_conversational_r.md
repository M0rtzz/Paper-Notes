---
title: >-
  [论文解读] Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent
description: >-
  [ACL 2025][LLM对齐][conversational recommendation] 提出 ECPO（Expectation Confirmation Preference Optimization），首个面向 LLM 对话推荐 Agent 的多轮偏好优化方法——基于心理学期望确认理论（ECT）显式建模用户满意度在多轮对话中的演变，通过前向期望确认定位不满意根因 + 后向期望推导重写回复构建 turn-level 偏好对，配合 AILO 用户模拟器，在 3 个数据集上显著优于现有 MTPO 方法。
tags:
  - ACL 2025
  - LLM对齐
  - conversational recommendation
  - multi-turn preference optimization
  - expectation confirmation theory
  - user simulator
  - DPO
  - LLM agent
---

# Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent

**会议**: ACL 2025  
**arXiv**: [2506.14302](https://arxiv.org/abs/2506.14302)  
**代码**: 无  
**领域**: LLM对齐  
**关键词**: conversational recommendation, multi-turn preference optimization, expectation confirmation theory, user simulator, DPO, LLM agent

## 一句话总结

提出 ECPO（Expectation Confirmation Preference Optimization），首个面向 LLM 对话推荐 Agent 的多轮偏好优化方法——基于心理学期望确认理论（ECT）显式建模用户满意度在多轮对话中的演变，通过前向期望确认定位不满意根因 + 后向期望推导重写回复构建 turn-level 偏好对，配合 AILO 用户模拟器，在 3 个数据集上显著优于现有 MTPO 方法。

## 研究背景与动机

**领域现状**：LLM 驱动的对话推荐 Agent（CRA）通过多轮自然语言交互逐步挖掘用户兴趣并推荐物品，已成为个性化推荐的主流范式。现有 CRA 包括 ChatRec、ReAct、MACRS 等基于 prompt 的框架。

**现有痛点**：LLM 的预训练目标是短视的 next-token prediction，导致 CRA 生成的回复缺乏主动性和灵活性，难以在多轮交互中持续引导用户、满足用户期望。偏好优化（如 DPO/KTO）虽然已证明在单轮对齐中有效，但直接应用于多轮对话面临严峻挑战。

**核心矛盾**：多轮对话中用户偏好逐轮变化、动态演变，现有 MTPO 方法存在三大问题——(1) 基于树搜索的方法需在每轮采样多个候选 + 模拟完整对话来评估中间轮偏好，采样开销巨大；(2) LLM 难以通过自采样生成高质量正样本；(3) 模拟环境的随机性为偏好关系引入噪声，导致对齐效果下降。

**本文目标**：在不需要额外采样和评估的前提下，构建高质量的 turn-level 偏好关系，实现多轮 CRA 的高效对齐。

**切入角度**：从心理学的期望确认理论（ECT）出发——满意度是用户初始期望与实际感知之间的比较结果。显式建模每轮的"期望-确认"过程，不满意时追溯根因并重写回复，自然构建偏好对。

**核心 idea**：用 ECT 理论驱动 turn-level 满意度评估，将"为什么不满意"的自然语言解释转化为重写指导，无需额外采样即可获得高质量偏好对。

## 方法详解

### 整体框架

ECPO 分为四步流水线：
1. **Simulator-Guided Planning Tuning (SGPT)**：用 GPT-4o mini 作为专家 CRA 与用户模拟器对话，收集成功推荐轨迹作为 SFT 数据 $\mathcal{D}_{sft}$，训练得到 $\pi_{sft}$
2. **Forward Expectation Confirmation**：$\pi_{sft}$ 与 AILO 对话过程中，每轮执行 EC 评估——基于灵活性(0-2)、一致性(0-2)、用户引导能力(0-1) 三维度打分，生成自然语言确认解释 $\text{CONF}_t$ 和满意度分数 $r_t$
3. **Backward Expectation Derivation**：对 $r_t \leq \lambda$ 的不满意轮次，将 $\text{CONF}_t$ 作为指导，用 Rewriter（额外 LLM）结合 chain-of-thought 重写回复 $\tilde{p}_t$
4. **Preference Optimization**：用原始-重写回复对 $(p_t, \tilde{p}_t)$ 构建偏好数据集 $\mathcal{D}_{pre}$，用 DPO 进行 turn-level 偏好优化

### 关键设计

#### 1. 前向期望确认（Forward Expectation Confirmation）

- **功能**：在每个对话轮次，模拟用户的内心独白——将期望物品 $i^E$ 和 CRA 回复 $p_t$ 输入 EC 指令，评估每个维度是否达标
- **为什么**：ECT 告诉我们满意度是期望与实际感知的差异。通过显式建模这一过程，可以在每一轮给出满意度分数和不满意原因，无需完成整个对话后再回溯
- **怎么做**：
    - 三维度评分：灵活性（0-2 分）、一致性（0-2 分）、用户引导能力（0-1 分），满分 5 分
    - 输出：$\{\text{CONF}_t, r_t\} = U(I_{ect}(i^E, h_t, p_t))$
    - $\text{CONF}_t$ 是自然语言形式的确认解释，详述为什么满意/不满意

#### 2. 后向期望推导（Backward Expectation Derivation）

- **功能**：对满意度低于阈值 $\lambda$ 的轮次，回溯到 CRA 当时的状态，利用确认解释反事实推理"应该怎么回复"
- **为什么**：确认解释提供了不满意的根因，Rewriter 可以精准地针对性修改，而非整体重写
- **怎么做**：
    - $\tilde{p}_t = \text{Rewriter}(I_{bed}(s_t, p_t, \text{CONF}_t))$，当 $r_t \leq \lambda$
    - Rewriter 采用 slow thinking（chain-of-thought），先推理再修改
    - 限制仅做**有限修改**而非完全重写，保持与 $\pi_{sft}$ 的分布接近
    - 偏好数据集：$\mathcal{D}_{pre} = \{(s_t, p_t, \tilde{p}_t) \mid r_t < \lambda\}$

#### 3. AILO 用户模拟器

- **功能**：构建高逼真度的 LLM 用户模拟器，支持 EC 过程
- **为什么**：真实用户参与成本高、存在偏差；现有用户模拟器（如 iEvalLM）通过简单随机采样生成用户画像，多样性不足
- **怎么做**：
    - **用户画像建模**：基于消费心理学 AIO 理论，定义 Activities、Interests、Language、Orientations 四维度；用 GPT-4o 从真实推荐评论数据集推断用户画像
    - **策略驱动的用户模拟**：三步生成——(1) 生成回复策略（如"询问推荐"）；(2) 基于策略生成回复内容；(3) 执行 EC 过程
    - 多样性验证：100 个画像的最大 ROUGE-L 分布显著低于 RecAgent 基线
    - 真实性验证：人工标注 50 组对话轨迹，AILO vs iEvalLM 达 100% 胜率

### 损失函数 / 训练策略

- **SFT 阶段**：$\mathcal{L}_{SFT} = \mathbb{E}[-\log \pi_\theta(cr_t, p_t | s_t)]$，含内部推理 $cr_t$ 和回复 $p_t$
- **DPO 阶段**：$\mathcal{L}_{DPO}(\pi_\theta, \pi_{sft}) = \mathbb{E}[-\log\sigma(\beta \log\frac{\pi_\theta(\tilde{p}_t|s_t)}{\pi_{sft}(\tilde{p}_t|s_t)} - \beta \log\frac{\pi_\theta(p_t|s_t)}{\pi_{sft}(p_t|s_t)})]$
- ECPO 与其他偏好优化方法（KTO、SimPO）正交互补，可无缝集成
- Backbone：Llama-3.1-8B-Instruct
- 训练数据：1,000 任务用于 $\mathcal{D}_{sft}$，500 任务用于 $\mathcal{D}_{pre}$

## 实验关键数据

### 主实验 1：与现有 Prompt-based CRA 对比

| 方法 | Backbone | Book SR | Book WR | Game SR | Game WR | Yelp SR | Yelp WR |
|------|----------|---------|---------|---------|---------|---------|---------|
| ChatRec | GPT-4o mini | 0.46 | 0.13 | 0.37 | 0.09 | 0.24 | 0.12 |
| ReAct | GPT-4o mini | 0.52 | 0.33 | 0.39 | 0.34 | 0.57 | 0.42 |
| MACRS | GPT-4o mini | 0.63 | 0.01 | 0.36 | 0.15 | 0.40 | 0.02 |
| ActCRS | Llama-3.1 | 0.34 | 0.28 | 0.07 | 0.46 | 0.22 | 0.38 |
| +SGPT | Llama-3.1 | 0.54 | 0.48 | 0.41 | 0.42 | 0.44 | 0.47 |
| **+ECPO** | **Llama-3.1** | **0.56** | **0.57** | **0.41** | **0.56** | **0.45** | **0.63** |

**关键发现**：SGPT 使 Llama 推荐指标 (SR) 达到 GPT-4o 水平；ECPO 进一步大幅提升交互性 (WR 0.56-0.63)，显著超越 GPT-4o mini 的最佳表现。

### 主实验 2：与现有 MTPO 方法对比

- 轨迹级方法（SFT、KTO）：改进有限，无法有效捕捉 turn-level 偏好
- 树模拟级方法（SDPO、SKTO）：采样 2,500 轨迹仍出现负增益，噪声干扰严重
- **ECPO**：仅需 500 轨迹（无需额外采样），在灵活性、一致性、用户引导等指标上全面最优
- 人工评估中 ECPO 在所有指标上显著优于专家 CRA，尤其灵活性和用户引导能力突出

### 消融实验

| 方法 | Book SR | Book WR | Game SR | Game WR | Yelp SR | Yelp WR |
|------|---------|---------|---------|---------|---------|---------|
| +SGPT | 0.54 | 0.48 | 0.41 | 0.42 | 0.44 | 0.47 |
| +ECPO w/o EC | 0.50 | 0.54 | 0.37 | 0.54 | 0.42 | 0.48 |
| **+ECPO** | **0.56** | **0.57** | **0.41** | **0.55** | **0.45** | **0.63** |

**关键发现**：去掉 EC 过程（用统一的人工分析替代逐轮 EC）后，交互性有所提升但推荐性能下降，整体显著不如完整 ECPO，证明 turn-level EC 过程是核心。

### 重写质量评估

| 指标 | Book | Game | Yelp |
|------|------|------|------|
| 信息保真度 Win Rate | 0.64 | 0.64 | 0.58 |
| 一致性 Win Rate | 0.90 | 0.82 | 0.73 |

重写回复在所有领域上均优于原始回复，尤其一致性提升显著。

### 超参数分析（重写阈值 $\lambda$）

- 低 $\lambda$（如 1）：每个样本改进更大，但总样本少
- 高 $\lambda$（如 4）：总样本多，但单个样本改进不规律
- 直觉验证：满意度极低的回复往往存在关键问题，修复后改进最显著

## 亮点与洞察

- **心理学理论驱动的 AI 对齐**：将期望确认理论（ECT）引入 LLM 偏好优化是巧妙的跨学科迁移。ECT 提供了一个从"用户满意度"角度理解多轮对话质量的理论框架
- **无需额外采样的 MTPO 范式**：ECPO 通过 EC 过程隐式分配 turn-level reward 并用自然语言给出原因，再用 Rewriter 生成正样本，完全避免了树搜索方法的高采样成本（500 vs 2,500 轨迹）
- **前向确认 + 后向推导的双向设计**：前向发现问题（what is wrong），后向解决问题（how to fix），逻辑自洽
- **AILO 的 AIO 理论用户画像**：比随机采样更真实多样，100% 人工评估胜率验证了其有效性
- **ECPO 与偏好优化方法正交**：可以替换 DPO 为 KTO/SimPO，具有良好的可扩展性

## 局限与展望

- **依赖用户模拟器质量**：AILO 虽优于现有模拟器，但与真实用户仍存在分布差异（distribution shift），可能影响真实部署效果
- **EC 评分维度固定**：灵活性/一致性/引导能力三维度可能不适用于所有对话推荐场景，缺乏领域自适应
- **Rewriter 引入额外成本**：虽然避免了树搜索的采样开销，但 Rewriter 本身的推理成本未详细分析
- **仅在推荐领域验证**：论文讨论了 ECPO 可扩展到更广泛的对话助手，但缺乏实验支持
- **改进方向**：将 EC 过程内化到模型推理阶段（O1/R1 风格）、与真实用户反馈结合、领域自适应的评分维度、扩展到非推荐对话场景

## 相关工作与启发

- **vs 树模拟 MTPO 方法（SDPO/SKTO, Jin et al. 2024; Xie et al. 2024）**：树模拟方法在每轮采样多个候选并模拟完整对话，成本高（2,500 轨迹）且噪声大导致负增益；ECPO 仅用 500 轨迹无需额外采样，效果反而更好
- **vs 轨迹级偏好优化（SFT/KTO, Ethayarajh et al. 2024）**：轨迹级方法将多轮对话视为整体打分，无法区分具体哪一轮有问题；ECPO 通过 EC 实现 turn-level 精细化优化
- **vs iEvalLM 用户模拟器（Wang et al. 2023）**：iEvalLM 用简单随机采样生成用户画像，AILO 基于 AIO 理论从真实评论推断画像，多样性和真实性均显著更优

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ ECT 理论应用于 LLM 多轮偏好优化是全新角度，前向确认+后向推导的 pipeline 设计精巧，消除了树搜索的额外采样需求
- 实验充分度: ⭐⭐⭐⭐ 3 数据集 + 消融 + 人工评估 + 超参分析 + 重写质量分析，覆盖全面；但仅在一个 backbone (Llama-3.1-8B) 上验证
- 写作质量: ⭐⭐⭐⭐ 框架设计清晰，图示直观，ECT 理论引入自然；问题定义和动机阐述到位
- 价值: ⭐⭐⭐⭐⭐ 首个面向 CRA 的偏好优化方法，ECPO 范式对多轮对话系统优化有广泛启发，AILO 用户模拟器也有独立价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs](m2s_multiturn_to_singleturn_jailbreak_in.md)
- [\[ACL 2025\] Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement](debate_reflect_and_distill_multi-agent_feedback_with_tree-structured_preference_.md)
- [\[ACL 2025\] MTSA: Multi-Turn Safety Alignment for LLMs through Multi-Round Red-Teaming](mtsa_multi-turn_safety_alignment_for_llms_through_multi-round_red-teaming.md)
- [\[ACL 2025\] Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)
- [\[ACL 2025\] Tempest: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search](tempest_autonomous_multi-turn_jailbreaking_of_large_language_models_with_tree_se.md)

<!-- RELATED:END -->
