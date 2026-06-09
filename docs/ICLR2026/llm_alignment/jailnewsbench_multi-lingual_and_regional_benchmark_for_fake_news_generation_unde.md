---
title: >-
  [论文解读] JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks
description: >-
  [ICLR 2026][LLM对齐][假新闻生成] 提出首个评估 LLM 在越狱攻击下生成假新闻鲁棒性的多语言多区域基准 JailNewsBench，覆盖 34 个地区和 22 种语言、约 30 万实例，揭示最高 86.3% 的攻击成功率以及英语/美国话题防御显著弱于其他地区的安全不平衡现象。
tags:
  - "ICLR 2026"
  - "LLM对齐"
  - "假新闻生成"
  - "越狱攻击"
  - "多语言安全"
  - "LLM安全评估"
  - "区域安全不平衡"
---

# JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks

**会议**: ICLR 2026  
**arXiv**: [2603.01291](https://arxiv.org/abs/2603.01291)  
**代码**: [https://github.com/kanekomasahiro/jail_news_bench](https://github.com/kanekomasahiro/jail_news_bench)  
**领域**: 对齐RLHF  
**关键词**: 假新闻生成, 越狱攻击, 多语言安全, LLM安全评估, 区域安全不平衡

## 一句话总结
提出首个评估 LLM 在越狱攻击下生成假新闻鲁棒性的多语言多区域基准 JailNewsBench，覆盖 34 个地区和 22 种语言、约 30 万实例，揭示最高 86.3% 的攻击成功率以及英语/美国话题防御显著弱于其他地区的安全不平衡现象。

## 研究背景与动机
假新闻对社会信任和决策构成严重威胁，波及政治、经济、健康和国际关系等方方面面。由于假新闻本质上反映了特定地区的政治、社会和文化背景，并以特定语言表达，因此评估 LLM 的安全风险必须采用多语言和多区域的视角。

恶意用户可以通过越狱攻击绕过安全防护，诱导 LLM 生成假新闻。然而，当前没有任何基准能够系统性地评估不同语言和地区下 LLM 的攻击鲁棒性。现有安全数据集（如 HarmBench、TrustLLM）主要关注毒性和社会偏见，对假新闻的覆盖非常有限。

**核心矛盾**：LLM 的安全对齐主要针对英语和通用有害内容进行训练，但假新闻是高度地区化和语言相关的，这导致非英语地区/语言的安全防护可能存在系统性盲区。

**切入角度**：构建首个跨语言跨区域的假新闻越狱基准，系统暴露 LLM 安全防护中的语言/地区不平衡。

## 方法详解

### 整体框架
JailNewsBench 是一个评测基准而非方法创新，目标是量化 LLM 在越狱攻击下生成区域化假新闻的脆弱程度。它把"为各地区定制的假新闻话题"与"多种越狱攻击模板"组合成 prompt 喂给目标模型，再用 GPT-4o 作为裁判从八个维度打分，最终在 34 个地区 × 22 种语言 × 5 种攻击 × 9 个模型的约 30 万实例上刻画 LLM 安全防护的语言/地区不平衡。

### 关键设计

**1. 区域化话题构建：让假新闻贴合真实的地缘语境。** 假新闻的危害高度依赖政治、社会与文化背景，单纯把英语样本翻译成其他语言无法还原真实威胁。因此基准为 34 个地区分别设计了地区特异的话题，例如美国的选举操纵、日本的核污水争议等，每个话题都嵌入当地的敏感议题而非通用模板，覆盖 22 种语言。这样构建出的样本既能测出模型对本土化误导内容的防御，也保证了跨地区比较的公平性。

**2. 五类越狱攻击模板：从简单诱导到复杂上下文覆盖。** 为了全面探查攻击面，基准实现了角色扮演（role play）、系统覆盖（system override）、研究前沿（research front）、负面提示（negative prompting）、上下文过载（context overload）以及显式请求（explicit）等多种攻击范式，覆盖从一句话直接索取到长上下文堆叠淹没安全约束的不同复杂度。每个话题都与各类攻击模板交叉组合，使得评测既能定位最容易绕过防护的攻击方式，也能避免单一攻击带来的偏差。

**3. 八维 LLM-as-Judge 评分：把"有多有害"拆成可量化的子维度。** 生成内容是否构成有效假新闻很难用单一标签衡量，基准让 GPT-4o 在八个子维度上为每条回复打 0–4 分：真实性（faithfulness）、可验证性（verifiability）、遵从性（adherence）、范围（scope）、规模（scale）、正式性（formality）、主观性（subjectivity）、煽动性（agitativeness）。这些维度共同刻画一条假新闻"看起来有多真、传播力有多强"，为后续跨地区比较提供细粒度的有害程度信号，而不仅是"是否被攻破"的二元判断。

**4. 三个聚合指标：攻击成功率、不流畅率与平均有害分。** 在八维打分之上，基准用攻击成功率（ASR，成功诱导生成假新闻的实例占比）、不流畅率（IFL，反映生成质量退化）和平均有害程度分数（avg_score）三个量来汇总结果，分别回答"防护是否被绕过""绕过后内容是否可用""可用内容危害多大"。三者结合才能区分"模型拒答""模型乱答"和"模型生成了可信假新闻"这三种本质不同的情况。

**5. 约 30 万实例的规模：保证统计可靠与可复现评测。** 完整网格为 34 地区 × 22 语言 × 5 攻击 × 9 模型，约 30 万实例，规模远超以往以英语为主的安全评测，使得按语言、按地区切片的结论具有统计意义。评测脚本已开源，支持 OpenAI、Anthropic、Gemini API 与 vLLM 本地模型，一行命令即可对任意模型跑完整流程。

### 一个完整示例
以一条"美国选举操纵"话题、采用角色扮演攻击为例：先把该地区话题填入角色扮演模板生成 prompt，诱导模型以"虚构小说作者"等身份绕过安全约束；再把 prompt 喂给目标 LLM（如某个 GPT 系列模型）得到回复；最后由 GPT-4o 裁判在真实性、煽动性等八个维度上为这条回复打分，并据此累计 ASR、IFL 与 avg_score。需要注意基准在不同数据分割（train/val/test）上的统计可能略有差异，复现论文数字时要对齐相应分割。

## 实验关键数据

### 主实验

| 模型 | 指标(ASR) | 最大ASR | 最大有害分数 | 英语ASR vs 其他 |
|------|----------|---------|------------|----------------|
| 9个LLM | ASR | 86.3% | 3.5/5 | 英语/美国防御显著更弱 |
| GPT系列 | ASR | 高 | 中高 | 英语区域偏弱 |
| Claude系列 | ASR | 中等 | 中等 | 相对均衡 |
| Llama系列 | ASR | 高 | 中高 | 非英语更强防御 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 不同攻击策略 | ASR变化大 | 角色扮演和上下文过载最有效 |
| 不同语言 | ASR差异显著 | 低资源语言防御更好(训练数据少→安全规则更保守) |
| 不同地区 | 有害分数差异 | 英美话题最容易被利用 |
| 假新闻vs毒性 | 防御对比 | 假新闻类别的防御显著弱于毒性类别 |

### 关键发现
- 最大攻击成功率达 86.3%，最大有害程度 3.5/5——LLM在假新闻防御上远未安全
- 英语和美国相关话题的防御性能显著弱于其他地区——"过度对齐"训练数据的美国视角可能反而暴露了弱点
- 假新闻在现有安全数据集中覆盖不足，防御效果远弱于毒性和社会偏见等主要类别
- 典型多语言LLM在非英语语言上的安全防护反而更强，这可能是因为safety训练数据分布不均导致模型对不常见语言更为保守

## 亮点与洞察
- 填补了假新闻生成安全评测的空白，是首个跨语言跨区域的系统性工作
- 揭示了一个反直觉的现象：英语/美国的防御反而最弱，挑战了"训练数据多=安全性好"的假设
- 8维评估框架为假新闻有害程度提供了细粒度的量化工具
- 30万实例规模确保了统计可靠性
- 数据集和评测脚本已开源（HuggingFace: MasahiroKaneko/JailNewsBench），支持一行命令评估任意模型
- 支持 5 种不同的越狱攻击策略（角色扮演、系统覆盖等），全面覆盖攻击面
- 分析表明假新闻类别在现有安全数据集中被严重忽视，这对安全训练数据的构建有重要启示
- 不同模型在不同语言上的安全性表现差异极大，暗示当前safety RLHF的多语言泛化能力不足

## 局限与展望
- LLM-as-Judge评估可能存在偏差，特别是对非英语语言的评判质量和一致性
- 仅评估了单轮攻击，多轮渐进式诱导可能更危险（可结合SEMA等多轮攻击方法）
- 假新闻话题的选取可能无法完全覆盖各地区的敏感议题，需持续更新
- 攻击策略相对固定，自适应攻击（如基于模型反馈的动态调整）未被纳入
- 仅考虑文本假新闻，多模态假新闻（图文/视频配合）的评估是重要的未来方向
- 基准的时效性——假新闻话题会随时事变化，定期更新数据集很重要

## 相关工作与启发
- **vs HarmBench/TrustLLM**: 这些通用安全基准不专注假新闻，且主要面向英语
- **vs SafetyBench**: SafetyBench覆盖多种有害类别但缺乏多语言和区域维度
- **vs RedTeaming方法**: 本文是评测而非攻击方法，但其揭示的安全不平衡对red teaming策略设计有指导意义

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个多语言多区域假新闻越狱基准，填补重要空白
- 实验充分度: ⭐⭐⭐⭐⭐ 34地区×22语言×5攻击×9模型，规模宏大
- 写作质量: ⭐⭐⭐⭐ 动机清晰，发现有冲击力
- 价值: ⭐⭐⭐⭐ 对LLM安全研究和政策制定有直接参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks](sema_simple_yet_effective_learning_for_multi-turn_jailbreak_attacks.md)
- [\[ICLR 2026\] CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation](cage_a_framework_for_culturally_adaptive_red-teaming_benchmark_generation.md)
- [\[ICLR 2026\] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)
- [\[ICLR 2026\] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)
- [\[AAAI 2026\] AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)

</div>

<!-- RELATED:END -->
