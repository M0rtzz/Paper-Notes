---
title: >-
  [论文解读] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models
description: >-
  [NeurIPS 2025][LLM对齐][红队测试] 提出基于策略（而非示例）的 LLM 红队评估框架和 Jailbreak-Zero 方法，通过简单的大规模并行采样策略（无需人工越狱策略），在 HarmBench 上对 GPT-4o 和 Claude 3.5 分别达到 99.5% 和 96.0% 的攻击成功率，同时通过微调实现覆盖率、多样性和保真度三个目标的 Pareto 最优。
tags:
  - "NeurIPS 2025"
  - "LLM对齐"
  - "红队测试"
  - "LLM 安全"
  - "越狱攻击"
  - "Pareto optimization"
  - "policy-based evaluation"
  - "automated red teaming"
---

# Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2601.03265](https://arxiv.org/abs/2601.03265)  
**代码**: 未公开  
**领域**: LLM对齐  
**关键词**: 红队测试, LLM 安全, 越狱攻击, Pareto optimization, policy-based evaluation, automated red teaming

## 一句话总结

提出基于策略（而非示例）的 LLM 红队评估框架和 Jailbreak-Zero 方法，通过简单的大规模并行采样策略（无需人工越狱策略），在 HarmBench 上对 GPT-4o 和 Claude 3.5 分别达到 99.5% 和 96.0% 的攻击成功率，同时通过微调实现覆盖率、多样性和保真度三个目标的 Pareto 最优。

## 研究背景与动机

现有 LLM 自动红队（Automated Red Teaming, ART）方法主要采用**基于示例的评估**：给定一组具体的有害行为（如"提供制造炸弹的指令"），然后构造对抗性提示诱导模型执行这些行为。这种方式存在几个核心问题：

**可扩展性差**：固定的示例列表无法覆盖所有真实安全风险，尤其当策略频繁变化时

**评估单一**：仅依赖攻击成功率（ASR），忽略了安全评估的多维性——覆盖率、多样性、对真实用户输入的保真度

**有效性存疑**：如果目标 LLM 针对预定义行为进行微调，拒绝率的提升可能反映的是记忆而非真正的安全性

**人工依赖重**：现有方法通常需要复杂的迭代算法、人工设计的越狱策略或大量 prompt engineering

## 方法详解

### 整体框架

本文的贡献分为两部分：

**Part 1 — 基于策略的评估框架**：用少量抽象策略（如 Llama Guard 的 14 个安全类别）替代具体示例来定义"不安全内容"，并引入三个评估维度。

**Part 2 — Jailbreak-Zero 方法**：分为零样本版本和微调版本，前者用于快速生成对抗提示，后者通过 RL/SFT 实现 Pareto 最优。

### 关键设计

**三维评估指标**：

1. **覆盖率（Coverage）**：衡量是否能在所有策略类别和语言上找到有效的对抗提示

$$\text{Coverage} = \frac{1}{PL} \sum_{p=1}^{P} \sum_{l=1}^{L} \mathbb{1}(x_{p,l} > N)$$

其中 $P$ 为策略数（如 Llama Guard 的 14 类），$L$ 为语言数，$N$ 为阈值。

2. **多样性（Diversity）**：基于句子嵌入 + DBSCAN 聚类，衡量成功提示覆盖的不同话题数

$$\text{Diversity} = \frac{1}{PL} \sum_{p=1}^{P} \sum_{l=1}^{L} n_{p,l}$$

3. **保真度（Fidelity）**：使用在真实用户数据（ShareGPT）上微调的 GPT-2 计算 PPL，衡量生成的提示与真实用户输入的相似度

$$\text{Fidelity}_{\mathcal{D}} = \frac{1}{P} \sum_{p=1}^{P} \frac{\text{PPL}_{\mathcal{D}}}{\text{PPL}_p}$$

**Jailbreak-Zero 零样本版本**：

核心思路极其简单——**大规模并行采样优于迭代精炼**：

1. 选择一个强指令跟随的攻击 LLM（如 Gemma3-27B）
2. 使用最简提示模板（仅包含策略描述，不含人工越狱策略）
3. 一次性生成大量对抗提示（1000-10000 个）
4. 使用代理模型（如 Llama-3 8B）评估每个提示，对每个提示生成 $m=5$ 个响应
5. 只保留全部 5 个响应都被判定为不安全的提示（满分提示）
6. 用 bigram 相似度过滤重复提示（阈值 1/3）

**增强多样性**：Seen Example Reference (SER) 技术——后半批次生成时，随机选择已成功的提示作为参考，指示攻击 LLM 不要重复同一话题。

**增强保真度**：Classifier-Free Guidance (CFG)，混合攻击 LLM 和用户分布模型的输出分布：

$$(1-\alpha) p_{\text{attack}}(x_{k+1}|x_{1:k}) + \alpha \pi_{\mathcal{D}}(x_{k+1}|x_{1:k})$$

$\alpha$ 越大，保真度越高但 ASR 越低。

**微调版本**：在零样本阶段的成功提示上构建偏好数据集，通过 SFT + RL 微调攻击 LLM，实现三个目标的联合优化。

### 损失函数

零样本阶段无需训练。微调阶段使用 SFT + 偏好学习（具体 RL 算法为 DPO 风格），loss 为标准的偏好对齐损失。

## 实验关键数据

### 主实验

**HarmBench 示例级评估**（ASR %）：

| 方法 | GPT-4o | Claude 3.5 | 人类可读 |
|------|--------|-----------|---------|
| **Jailbreak-Zero (zero-shot)** | **99.5** | **96.0** | ✓ |
| AutoDan-Turbo | 91.0 | 37.5 | ✓ |
| PAIR | 56.5 | 28.0 | ✓ |

在相同计算预算下（控制相同 query 数或 token 数），Jailbreak-Zero 仍然显著优于迭代精炼方法。

**策略级评估**（Llama 3.1 8B，Gemma3-27B 作为攻击 LLM）：

| 方法 | Coverage (%) | Avg ASR (%) | Diversity | Fidelity |
|------|-------------|------------|-----------|----------|
| Vanilla | 64.3 | 21.1 | 196.1 | 0.475 |
| + CFG (α=0.1) | 64.3 | 18.9 | 188.8 | 0.483 |
| + CFG (α=0.2) | 57.1 | 12.6 | 175.9 | 0.498 |
| + SER | 57.1 | 16.3 | **225.3** | 0.474 |
| + CFG + SER | 50.0 | 15.2 | 215.5 | 0.480 |

SER 显著提升多样性（196→225），CFG 提升保真度但降低 ASR，证实了三个目标之间存在 Pareto 权衡。

**推理模型攻击结果**：

| 模型 | HarmBench ASR (%) |
|------|------------------|
| GPT-oss 20B | 95.5 |
| GPT-oss 120B | 87.5 |
| GPT-5 (minimal reasoning) | 14.0 |
| GPT-5 (low reasoning) | 23.0 |
| Gemini 2.5 Flash | 56.5 |

推理能力可提升安全性，但 Jailbreak-Zero 对大多数推理模型仍然有效。

### 消融实验

**攻击 LLM 选择**：

| 攻击 LLM | GPT-4o ASR | Claude 3.5 ASR |
|---------|-----------|---------------|
| Gemma 3 27B | **99.5** | **96.0** |
| Mistral 24B | 93.0 | 86.5 |
| Qwen 2.5 32B | 94.0 | 85.0 |
| Vicuna 13B | 82.0 | 30.5 |

攻击 LLM 的选择影响很大，Gemma 3 最强；提示模板的影响相对较小（Our vs PAIR 模板差异不大）。

**迁移性**：在代理模型上成功的对抗提示可以有效迁移到目标模型（开源和闭源均验证）。

### 关键发现

1. **并行采样 > 迭代精炼**：在相同计算预算下，一次性大量采样比 PAIR/AutoDan 的迭代精炼更高效
2. **Pareto 权衡真实存在**：Coverage、Diversity、Fidelity 三者不可同时最大化，但微调可以推动 Pareto 前沿
3. **微调后可泛化到未见策略**：在 9 个策略上微调后，对 5 个未见策略仍有效
4. **safety alignment 后仍有效**：即使目标 LLM 针对上一轮发现的漏洞进行安全微调，方法仍能找到新的攻击路径

## 亮点与洞察

1. **评估框架的范式转变**：从"固定示例列表"到"抽象策略描述"，大幅提升了评估的覆盖面和可扩展性
2. **极简主义的威力**：无需人工越狱策略、无需复杂迭代，仅靠简单提示 + 大规模采样就超越了所有先前方法
3. **可控的 Pareto 优化**：通过 CFG 的 $\alpha$ 和 SER 等技术，无需重新训练即可调节三个目标之间的权衡
4. **对产业界极具价值**：策略级评估直接对应于真实的安全审查流程（如 Llama Guard 的安全分类），工程可落地性强
5. **GPT-5 观察**：GPT-5 可能使用了系统级安全防御（直接拒绝输入而非模型层面过滤），代表了安全对齐的新方向

## 局限性

1. **评判者依赖**：ASR 的计算依赖判别模型（Llama Guard / GPT-4o judge），判别模型本身的准确率会影响评估可靠性
2. **仅测试文本模态**：未覆盖多模态攻击场景
3. **保真度度量局限**：PPL 作为保真度代理指标并不完美，低 PPL 不等同于真正的用户输入模式
4. **CFG 限制**：攻击 LLM 和用户分布模型必须使用相同 tokenizer
5. **无法攻破系统级防御**：如 GPT-5，当系统直接拒绝输入（400 错误）时，方法失效

## 相关工作与启发

- **PAIR / AutoDan-Turbo**：迭代精炼方法，计算效率低且多样性不足
- **GCG**：基于梯度的对抗后缀方法，生成不可读提示
- **HarmBench (Mazeika et al., 2024)**：标准示例级红队基准
- **Llama Guard**：14 类安全分类策略，为策略级评估提供了现成的框架
- **启发**：红队评估应从单一 ASR 指标走向多目标 Pareto 优化；简单方法在大规模并行计算下可能优于复杂迭代方法（scaling laws for red teaming）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 策略级评估框架和 Pareto 优化视角是重要贡献
- **技术深度**: ⭐⭐⭐⭐ — 评估框架严谨，但核心方法（并行采样）技术含量有限
- **实验充分性**: ⭐⭐⭐⭐⭐ — 开源+闭源模型、消融全面、推理模型也有测试
- **实用价值**: ⭐⭐⭐⭐⭐ — 对产业界安全评估流程有直接参考价值
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，指标定义严格
- **综合评分**: ⭐⭐⭐⭐ (8/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] PolyJuice Makes It Real: Black-Box, Universal Red Teaming for Synthetic Image Detectors](polyjuice_makes_it_real_black-box_universal_red_teaming_for_synthetic_image_dete.md)
- [\[ICLR 2026\] Capability-Based Scaling Trends for LLM-Based Red-Teaming](../../ICLR2026/llm_alignment/capability-based_scaling_trends_for_llm-based_red-teaming.md)
- [\[ACL 2025\] M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs](../../ACL2025/llm_alignment/m2s_multiturn_to_singleturn_jailbreak_in.md)
- [\[ACL 2025\] SQL Injection Jailbreak: A Structural Disaster of Large Language Models](../../ACL2025/llm_alignment/sql_injection_jailbreak_a_structural_disaster_of_large_language_models.md)
- [\[NeurIPS 2025\] Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency](adjacent_words_divergent_intents_jailbreaking_large_language_models_via_task_con.md)

</div>

<!-- RELATED:END -->
