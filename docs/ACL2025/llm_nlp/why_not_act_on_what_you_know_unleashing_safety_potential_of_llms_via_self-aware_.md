---
title: >-
  [论文解读] Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement
description: >-
  [ACL2025][LLM/NLP][越狱防御] 发现 LLM 作为判别器时能准确识别越狱请求、但作为生成器时却仍产出有害内容的"判别-生成安全差距"，提出免训练的 SAGE（Self-Aware Guard Enhancement）策略，通过判别分析模块和判别响应模块将模型自身的安全鉴别能力桥接到生成行为…
tags:
  - "ACL2025"
  - "LLM/NLP"
  - "越狱防御"
  - "安全对齐"
  - "判别-生成差距"
  - "免训练"
  - "可解释性"
---

# Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement

**会议**: ACL2025  
**arXiv**: [2505.12060](https://arxiv.org/abs/2505.12060)  
**代码**: [NJUNLP/SAGE](https://github.com/NJUNLP/SAGE)  
**领域**: LLM/NLP  
**关键词**: 越狱防御, 安全对齐, 判别-生成差距, 免训练, 可解释性

## 一句话总结

发现 LLM 作为判别器时能准确识别越狱请求、但作为生成器时却仍产出有害内容的"判别-生成安全差距"，提出免训练的 SAGE（Self-Aware Guard Enhancement）策略，通过判别分析模块和判别响应模块将模型自身的安全鉴别能力桥接到生成行为，在 6 个模型上达到平均 99% 的防御成功率。

## 背景与动机

1. **越狱攻击持续升级**：GCG、AutoDAN、PAIR、ReNeLLM、DeepInception 等攻击方法不断进化，从梯度优化到模板设计全方位突破安全对齐。
2. **现有防御各有局限**：RLHF/SFT 等学习型方法有对齐税和灾难性遗忘问题；提示型防御（Self-Reminder、ICD 等）对复杂越狱泛化性差。
3. **判别-生成差距的关键洞察**：实验发现 Llama-3.1-8B 能 100% 判别出 DeepInception 攻击的有害意图，但直接生成时仅能防御 34%；Qwen-2.5-7B 判别 ReNeLLM 准确率 84%，防御率却仅 8%。
4. **模型"知而不行"**：LLM 内部已具备安全感知能力，但生成模式下这种能力未被激活——这是一个被忽视的根本问题。
5. **免训练的实际需求**：大模型重训成本高，且新攻击不断出现，需要即插即用的推理时防御方案。
6. **可解释性的缺失**：此前工作只报告防御效果，未深入分析判别与生成模式下模型内部机制的差异。

## 方法详解

### 整体框架

- **功能**：SAGE 是一个免训练的推理时防御框架，在用户输入前拼接两段结构化指令——判别分析指令和判别响应指令，引导模型先做安全分析再决定如何回复。
- **为什么**：既然模型作为判别器时安全能力强，就通过指令将"先判别后生成"的流程显式化，让判别结果直接指导生成行为。
- **怎么做**：最终 prompt = 判别分析指令 $I_{da}$ ⊕ 判别响应指令 $I_{dr}$ ⊕ 用户输入 $P_{usr}$，一次推理完成分析和回复。

### 关键设计 1：判别分析模块（Discriminative Analysis Module）

- **功能**：指导模型在生成响应前进行两级安全评估——语义级分析和任务结构级分析。
- **为什么**：单一判别指令难以覆盖所有攻击手段。语义级捕捉内容本身的有害性，任务结构级识别伪装在无害任务中的有害意图（如角色扮演、代码包装），双重分析提升对复杂越狱的检出率。
- **怎么做**：
  1. 语义级分析：不管表面呈现方式，评估请求内容的固有危害性；
  2. 任务结构级分析：检测有害内容是否嵌入在看似无害的任务框架中（如代码编写、翻译、角色扮演等典型越狱模板）。

### 关键设计 2：判别响应模块（Discriminative Response Module）

- **功能**：根据判别分析的结果决定生成策略——有害请求给出带理由的明确拒绝，安全请求正常回复。
- **为什么**：仅做判别不够，还需将判别结果映射到生成行为；同时保持"拒绝有理由"和"安全请求不受影响"的双重目标。
- **怎么做**：
  1. 有害请求：生成格式化拒绝（"I cannot assist with this request because [具体原因]"），保持透明和可追溯；
  2. 安全请求：正常生成有帮助的回复，不过度敏感；
  3. 整个判别+响应在单次推理中完成，不需多轮交互。

## 实验关键数据

### 实验 1：安全防御主实验（3 个开源模型 × 9 种攻击）

| 模型 | 防御方法 | AdvBench ASR | ReNeLLM ASR | DeepInception ASR | CodeAttack ASR | 平均 ASR |
|------|---------|-------------|------------|------------------|---------------|---------|
| Llama3.1-8B | No Defense | 8% | 90% | 70% | 100% | 57% |
| Llama3.1-8B | Self-Reminder | 0% | 18% | 28% | 94% | 25% |
| Llama3.1-8B | Goal Prior. | 0% | 10% | 50% | 26% | 22% |
| Llama3.1-8B | IA | 0% | 56% | 0% | 84% | 23% |
| **Llama3.1-8B** | **SAGE** | **0%** | **2%** | **0%** | **2%** | **1%** |
| Qwen2.5-7B | No Defense | 0% | 100% | 92% | 100% | 67% |
| **Qwen2.5-7B** | **SAGE** | **0%** | **1%** | **0%** | **0%** | **1%** |
| Gemma2-9B | No Defense | 8% | 100% | 96% | 96% | 74% |
| **Gemma2-9B** | **SAGE** | **0%** | **0%** | **0%** | **0%** | **5%** |

**发现**：SAGE 在所有模型上平均 ASR 降至 1-5%，特别是对 ReNeLLM（100%→0-2%）和 DeepInception（70-96%→0%）效果显著；其他基线在 CodeAttack 等复杂攻击上仍有大面积失败。

### 实验 2：有用性保持（GSM8K、MMLU、Just-Eval）

| 模型 | 防御方法 | GSM8K | MMLU | Just-Eval Avg |
|------|---------|-------|------|--------------|
| Llama3.1-8B | No Defense | 88% | 75% | 4.56 |
| Llama3.1-8B | Self-Examination | 77% | 59% | 4.28 |
| Llama3.1-8B | Goal Prior. | 87% | 51% | 4.08 |
| **Llama3.1-8B** | **SAGE** | **91%** | **72%** | **4.44** |
| Qwen2.5-7B | No Defense | 93% | 72% | 4.58 |
| **Qwen2.5-7B** | **SAGE** | **93%** | **71%** | **4.39** |

**发现**：SAGE 几乎不影响模型在数学推理（GSM8K）和知识问答（MMLU）上的表现，而 Self-Examination 和 Goal Prioritization 在 MMLU 上分别下降 16 和 24 个百分点，说明 SAGE 的安全增强不以牺牲有用性为代价。

### 实验 3：效率与可解释性

- **推理效率**：SAGE 的每样本耗时（TCPS）为 5-7 秒，与最高效的 IA（5-7 秒）持平，远优于 Self-Examination（12-27 秒）。
- **隐藏状态可视化（PCA）**：越狱攻击将有害 query 的表示移向良性侧，而 SAGE 的判别指令将其拉回有害侧，从机制上解释了"判别弥合"的有效性。

## 亮点

- **发现深刻**：首次系统揭示 LLM 判别-生成安全差距，指出模型"知而不行"的根本问题。
- **完全免训练**：仅通过精心设计的结构化提示实现 99% 平均防御率，无需任何参数更新。
- **防御全面**：在 7 种攻击 × 6 个模型（含闭源 GPT-4o、Claude-3.5）上均有效，泛化能力强。
- **可解释性分析**：通过隐藏状态和注意力分布的可视化，从机制层面解释了判别-生成差距的成因。

## 局限与展望

- **增加推理长度**：SAGE 需要模型先输出分析再输出回复，增加了 token 消耗；对延迟敏感的场景可能有影响。
- **仍依赖模型的判别能力**：如果模型的判别能力本身较弱（如未经安全对齐的基础模型），SAGE 的效果可能受限。
- **GPTFuzzer 攻击防御较弱**：在 Gemma2 上 GPTFuzzer 的 ASR 仍有 18%，说明部分高度混淆的模板攻击仍是挑战。
- **评估依赖 GPT-4o**：有害分数由 GPT-4o 评估，存在评估偏差的风险。

## 与相关工作的对比

### vs IA (Intention Analysis, Zhang et al., 2025)
IA 通过意图分析引导模型理解用户真实意图来防御越狱。SAGE 进一步引入双级判别（语义+任务结构），在 ReNeLLM（IA: 56% ASR → SAGE: 2%）和 CodeAttack（IA: 84% → SAGE: 2%）上大幅领先。IA 对复杂模板攻击泛化不足，SAGE 的结构化判别更全面。

### vs Goal Prioritization (Zhang et al., 2024)
Goal Prioritization 通过强调安全目标优先级来约束生成。它在 MMLU 上性能下降严重（Llama 75%→51%），说明过度的安全约束损害了有用性。SAGE 通过"先判别后决策"避免了这种过度敏感，在安全性和有用性之间取得更好平衡。

### vs 学习型防御（RLHF/SFT）
学习型方法需要重训参数，存在对齐税、灾难性遗忘和对 OOD 攻击脆弱的问题。SAGE 完全免训练，可即插即用于任何指令遵循 LLM，且无遗忘风险。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 判别-生成差距的发现和利用是重要洞察
- 实验充分度: ⭐⭐⭐⭐⭐ — 6 模型 × 7 攻击 × 多维有用性评估 + 可解释性分析，非常全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，可视化丰富，动机引人入胜
- 价值: ⭐⭐⭐⭐ — 实用且易部署的安全增强方案，可解释性分析有理论贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts](safer_or_luckier_llms_as_safety_evaluators_are_not_robust_to_artifacts.md)
- [\[ACL 2025\] Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training](derta_decoupled_refusal.md)
- [\[ACL 2025\] LLMs Know Their Vulnerabilities: Uncover Safety Gaps through Natural Distribution Shifts](llms_know_their_vulnerabilities_uncover_safety_gaps_through_natural_distribution.md)
- [\[ACL 2025\] Self-Instructed Derived Prompt Generation Meets In-Context Learning: Unlocking New Potential of Black-Box LLMs](self-instructed_derived_prompt_generation_meets_in-context_learning_unlocking_ne.md)
- [\[ACL 2025\] HiCUPID: Exploring the Potential of LLMs as Personalized Assistants](exploring_the_potential_of_llms_as.md)

</div>

<!-- RELATED:END -->
