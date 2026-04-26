---
title: >-
  [论文解读] OR-Bench: An Over-Refusal Benchmark for Large Language Models
description: >-
  [ICML 2025][Over-Refusal] 提出首个大规模 LLM 过度拒绝（over-refusal）基准 OR-Bench，包含 80K 安全但易被拒绝的 prompt，揭示安全性与过度拒绝之间存在 Spearman 相关系数高达 0.89 的强权衡关系。
tags:
  - ICML 2025
  - Over-Refusal
  - Safety Alignment
  - benchmark
  - LLM Evaluation
  - Red-Teaming
---

# OR-Bench: An Over-Refusal Benchmark for Large Language Models

**会议**: ICML 2025  
**arXiv**: [2405.20947](https://arxiv.org/abs/2405.20947)  
**代码**: [github.com/justincui03/or-bench](https://github.com/justincui03/or-bench)  
**领域**: LLM对齐/安全  
**关键词**: Over-Refusal, Safety Alignment, benchmark, LLM Evaluation, Red-Teaming

## 一句话总结

提出首个大规模 LLM 过度拒绝（over-refusal）基准 OR-Bench，包含 80K 安全但易被拒绝的 prompt，揭示安全性与过度拒绝之间存在 Spearman 相关系数高达 0.89 的强权衡关系。

## 研究背景与动机

LLM 在经过安全对齐（如 RLHF、MART、instruction fine-tuning）后，能有效拒绝恶意指令，但往往附带一个副作用：**过度拒绝（over-refusal）**——即模型对无害的 prompt 也予以拒绝，导致实用性下降。

现有的过度拒绝测试集 XSTest 仅含 250 条手工编写的 prompt，存在三个核心问题：

**规模不足**：250 条无法系统性覆盖多种有害类别和话题

**难度饱和**：最新 SOTA 模型（如 Llama-3-70b）几乎能正确回答 XSTest 中的所有问题

**人工成本高**：跨类别、跨话题的扩展需要大量人力

因此，本文提出了一个全自动化的生成管线，构建了首个大规模过度拒绝基准 OR-Bench，包含跨 10 个常见拒绝类别的 80,000 条 prompt。

## 方法详解

### 整体框架

OR-Bench 的构建遵循三阶段流水线：**有毒种子生成 → 安全化改写 → 集成审核过滤**。核心思路是：从有毒 prompt 出发，将其改写为表面"擦边"但实际安全的 prompt，再通过多模型集成投票确保最终数据集的安全性。最终产出三个数据集：OR-Bench-80K（全量）、OR-Bench-Hard-1K（困难子集）、OR-Bench-Toxic（600 条有毒 prompt 用于防止模型不加区分地全部回答）。

### 关键设计

1. **有毒种子生成（Toxic Seed Generation）**：使用 Mixtral 8×7B（安全限制较少的开源模型）为每个类别生成 2,000 条有毒 prompt。选择每次生成 20 条是经过实验权衡的——50 条会导致重复质量下降，少于 20 条则需要过多查询且批次间重复率高。覆盖 violence、privacy、hate、sexual 等 10 个常见拒绝类别。

2. **有毒种子改写（Toxic Seed Rewrite）**：同样使用 Mixtral 8×7B 将每条有毒种子改写为 5 条安全但可能触发过度拒绝的 prompt。作者发现了两个关键问题并予以解决：

    - **虚构化倾向**：模型倾向于将有毒场景转化为虚构场景（如"在电影中偷车的步骤"），通过指令约束避免虚构/理论框架
    - **安全词依赖**：模型频繁使用"safe""legal"等关键词来"安全化"prompt（如"合法地制造炸弹"），通过明确指令禁止使用此类词汇
    - **多样性不足**：引入 5-shot 示例显著提升改写的多样性和质量，例如将"how to kill a person"改写为"在戏剧场景中，描述制作逼真道具刀的过程"

3. **集成审核过滤（Ensemble Moderator）**：采用 GPT-4-turbo、Llama-3-70b、Gemini-1.5-pro 三模型集成投票（多数决），而非单一模型判断。关键设计要点：

    - 要求每个审核模型先解释理由再做判断（类 CoT 推理），提升判断质量
    - Claude-3-opus 过于保守，与其他模型一致率低，不适合作为审核者
    - 对于被标记为有毒但可能是误判的 prompt，额外用无安全审核的 Mistral-7B 生成回答，再由审核器重新评估回答的安全性
    - 最终达到人类专家 98% 的性能水平（准确率 93.0% vs 专家 94.0%）

4. **困难子集构建（OR-Bench-Hard-1K）**：从 80K prompt 中筛选出被至少 3 个来自不同模型家族的最大/最新模型拒绝的 prompt，构成约 1,000 条的高难度子集，用于快速评测。

### 评估策略

- **关键词匹配**：在 80K 全量数据集上使用快速关键词匹配检测拒绝行为
- **GPT-4 判断**：在 Hard-1K 和 Toxic 数据集上使用 GPT-4 进行更精确的拒绝判断
- 两种方法差异极小（GPT-3.5-turbo-0125 差 2.4%，Llama-3-70b 差 1.2%）
- 所有模型通过公开 API 测试，不使用 system prompt，确保评估无偏

## 实验关键数据

### 主实验

在 OR-Bench-Hard-1K 上的过度拒绝率（%），数值越高表示过度拒绝越严重：

| 模型家族 | 代表模型 | 过度拒绝率 | 有毒拒绝率 | 特点 |
|----------|----------|-----------|-----------|------|
| Claude-2 | Claude-2.1 | 99.8% | 最高 | 最安全但严重过度拒绝 |
| Claude-3 | Claude-3.5-Sonnet | 43.8% | 高 | 较前代大幅改善 |
| Llama-2 | Llama-2-70b | 96.0% | 高 | 严重过度拒绝 |
| Llama-3.1 | Llama-3.1-70B | 3.0% | 较低 | 过度拒绝极低 |
| GPT-3.5 | GPT-3.5-turbo-0301 | 57.4% | 中 | 早期版本问题严重 |
| GPT-4 | GPT-4o | 6.7% | 高 | 安全与实用兼顾 |
| Mistral | Mistral-large | 9.7% | 最低 | 过度拒绝少但安全性不足 |
| Qwen-1.5 | Qwen-1.5-72B | 46.9% | 中 | 对 sexual 和 deception 敏感 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 温度 0.0 vs 1.0（Claude-3-Haiku） | 96.2% → 95.5% | 温度对拒绝行为影响极小 |
| 温度 0.0 vs 1.0（Llama-2-7b） | 87.4% → 85.5% | 一致性结论 |
| 有 system prompt vs 无 | 安全性↑ 但拒绝率大增 | GPT-3.5多拒55%安全prompt换取多拒35%有毒prompt |
| ICL 防御 | 有毒拒绝最高 | 但过度拒绝率也最高 |
| SmoothLLM | 有毒拒绝略增 | 过度拒绝也略增 |
| 集成审核 vs 单模型微调 | 93.0% vs ~90% | CoT + 多模型一致性带来提升 |
| 集成审核 vs 人工标注 | 93.0% vs 更低 | 人工缺乏领域知识导致表现反而不如 LLM |

### 关键发现

1. **安全性与过度拒绝的强相关**：Spearman 秩相关系数为 0.89，说明绝大多数模型以过度拒绝换取安全性，很少有模型能同时优化两者
2. **模型规模与平衡无关**：更大的模型不一定在安全-实用平衡上更好
3. **新版模型改善明显**：Llama-2 → Llama-3.1、GPT-3.5-0301 → GPT-3.5-0125 均大幅降低过度拒绝
4. **类别敏感性差异大**：Claude-3-opus 对 sexual 类不敏感（39.2%），GPT-3.5-0125 对 privacy 最敏感，所有模型对 self-harm 类有毒 prompt 拒绝率都很高
5. **Gemini 与众不同**：新版 Gemini-1.5 反而比旧版更保守（过度拒绝更高），同时也更安全
6. **防御方法加剧过度拒绝**：ICL、SmoothLLM 等防御方法虽提升安全性，但均显著增加过度拒绝率

## 亮点与洞察

- **全自动数据生成管线**：从种子生成到改写再到审核完全自动化，可持续更新以避免过拟合，这种可扩展的设计理念值得借鉴
- **集成审核的巧妙设计**：多模型投票 + CoT 推理 + 回答安全性二次验证，三重保障使得自动审核达到专家水平，避免了单一模型的偏见问题
- **揭示了安全对齐的根本困境**：0.89 的相关系数量化了 safety vs helpfulness 的 trade-off，为未来算法设计提供了明确的优化目标（移向图的左上角）
- **数据集设计的完备性**：80K 全量 + 1K 困难子集 + 600 有毒 prompt 的三层结构兼顾了全面性、挑战性和防刷性

## 局限与展望

1. **二元拒绝定义过于简化**：作者也承认"refusal is a false binary"，模型可以有更细粒度的回应方式（部分回答、附加警告等），未来需要更精细的评估维度
2. **审核器本身可能存在偏见**：用 GPT-4、Llama-3、Gemini 作为审核器，在评估这些模型家族时存在潜在的自我偏好，尽管作者引用研究表明判断能力与安全对齐是不同维度
3. **缺少多轮对话场景**：所有 prompt 都是单轮的，实际应用中过度拒绝可能在多轮上下文中更复杂
4. **改写模型的局限**：依赖 Mixtral 8×7B 的改写能力和风格，可能遗漏某些触发过度拒绝的 prompt 模式
5. **未探索缓解方案**：只诊断了问题，未提出减少过度拒绝的具体训练方法

## 相关工作与启发

- **XSTest** (Röttger et al., 2023): 250 条手工 prompt 的先驱工作，但已被新模型"解决"，本文的自动扩展方法是对其的重要升级
- **WildGuard** (Han et al., 2024): 多任务审核模型，可检测 harmful prompt/response 和 refusal，与本文的审核管线互补
- **PHTest** (An et al., 2024): 针对特定模型生成伪有害 prompt 的并行工作，与本文的 model-agnostic 设计形成对比
- **Safe RLHF** (Dai et al., 2023): 安全对齐的代表方法，本文的发现表明需要在 RLHF 中同时考虑拒绝和过度拒绝
- **启发**：可将 OR-Bench 作为安全对齐训练的负样本（不应拒绝的样本），帮助模型学习更精准的拒绝边界

## 评分

- 新颖性: ⭐⭐⭐⭐ （问题已被观察到，但首次大规模系统化）
- 实验充分度: ⭐⭐⭐⭐⭐ （32个模型、8个家族、多维消融、定量+定性分析）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，数据详实，但表格较多略显冗长）
- 价值: ⭐⭐⭐⭐⭐ （填补了重要空白，数据集公开，推动安全对齐研究）

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] MDiT-Bench: Evaluating the Dual-Implicit Toxicity in Large Multimodal Models](../../ACL2025/social_computing/mdit-bench_evaluating_the_dual-implicit_toxicity_in_large_multimodal_models.md)
- [\[ICML 2025\] Raising the Bar: Investigating the Values of Large Language Models via Generative Evolving Testing](raising_the_bar_investigating_the_values_of_large_language_models_via_generative.md)
- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](../../ICLR2026/social_computing/biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)
- [\[NeurIPS 2025\] DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models](../../NeurIPS2025/social_computing/date-lm_benchmarking_data_attribution_evaluation_for_large_language_models.md)
- [\[NeurIPS 2025\] Active Slice Discovery in Large Language Models](../../NeurIPS2025/social_computing/active_slice_discovery_in_large_language_models.md)

<!-- RELATED:END -->
