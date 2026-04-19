---
title: >-
  [论文解读] JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks
description: >-
  [ICLR 2026][LLM对齐][假新闻生成] 提出首个评估 LLM 在越狱攻击下生成假新闻鲁棒性的多语言多区域基准 JailNewsBench，覆盖 34 个地区和 22 种语言、约 30 万实例，揭示最高 86.3% 的攻击成功率以及英语/美国话题防御显著弱于其他地区的安全不平衡现象。
tags:
  - ICLR 2026
  - LLM对齐
  - 假新闻生成
  - 越狱攻击
  - 多语言安全
  - LLM安全评估
  - 区域安全不平衡
---

# JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks

**会议**: ICLR 2026  
**arXiv**: [2603.01291](https://arxiv.org/abs/2603.01291)  
**代码**: https://github.com/kanekomasahiro/jail_news_bench (有)  
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
JailNewsBench 是一个Benchmark而非方法创新。整个框架包括：(1) 多区域多语言的假新闻话题构建→(2) 多种越狱攻击策略实施→(3) LLM-as-Judge 多维度评估→(4) 跨语言跨区域的安全性分析。

### 关键设计
1. **多区域多语言覆盖**: 覆盖 34 个地区和 22 种语言，假新闻话题针对各地区的政治、社会、文化背景定制。例如美国的选举操纵、日本的核污水争议等。每个话题都有地区特异性，不是简单翻译。设计动机是假新闻的危害高度依赖于地缘政治和文化语境。

2. **五种越狱攻击策略**: 包括角色扮演（role play）、系统覆盖（system override）、研究前沿（research front）、负面提示（negative prompting）、上下文过载（context overload）和显式请求（explicit）。涵盖了从简单到复杂的多种攻击范式，确保评估的全面性。

3. **LLM-as-Judge 八维评估**: 采用 8 个子指标评估生成假新闻的有害程度：真实性（faithfulness）、可验证性（verifiability）、遵从性（adherence）、范围（scope）、规模（scale）、正式性（formality）、主观性（subjectivity）、煽动性（agitativeness）。每个维度 0-4 分，由 GPT-4o 作为裁判。

4. **大规模实例集**: 约 30 万实例（34 地区 × 22 语言 × 5 攻击 × 9 模型），规模远超以前的安全评测。

### 损失函数 / 训练策略
作为评测基准，不涉及训练。评估指标包括：攻击成功率（ASR）、不流畅率（IFL）和平均有害程度分数（avg_score）。
评估流程：(1) 根据话题和攻击策略生成prompt→(2) 让目标LLM生成回复→(3) GPT-4o作为裁判在8个维度上评分。
评测脚本已开源，支持 OpenAI、Anthropic、Gemini API 和 vLLM 本地模型，一行命令即可对任意模型进行全面评估。
特别值得注意的是，该基准的结果在不同分割（train/val/test）上可能略有差异，计算平均分时需注意与论文报告数据的对应关系。

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

## 相关论文

- [SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks](sema_simple_yet_effective_learning_for_multi-turn_jailbreak_attacks.md)
- [CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation](cage_a_framework_for_culturally_adaptive_red-teaming_benchmark_generation.md)
- [Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)
- [Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)
- [AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)

<!-- RELATED:END -->
