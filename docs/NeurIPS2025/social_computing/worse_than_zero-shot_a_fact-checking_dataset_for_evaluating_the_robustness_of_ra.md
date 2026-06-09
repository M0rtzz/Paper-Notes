---
title: >-
  [论文解读] Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals
description: >-
  [NeurIPS 2025][社会计算][RAG robustness] 提出 RAGuard 基准数据集，首次系统评估 RAG 系统对误导性检索内容的鲁棒性。通过从 Reddit 构建包含支持性、误导性和无关文档的真实检索语料库…
tags:
  - "NeurIPS 2025"
  - "社会计算"
  - "RAG robustness"
  - "fact-checking"
  - "misleading retrieval"
  - "benchmark"
  - "misinformation"
---

# Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals

**会议**: NeurIPS 2025  
**arXiv**: [2502.16101](https://arxiv.org/abs/2502.16101)  
**代码**: [HuggingFace Dataset](https://huggingface.co/datasets/UCSC-IRKM/RAGuard)  
**领域**: 信息检索  
**关键词**: RAG robustness, fact-checking, misleading retrieval, benchmark, misinformation

## 一句话总结

提出 RAGuard 基准数据集，首次系统评估 RAG 系统对误导性检索内容的鲁棒性。通过从 Reddit 构建包含支持性、误导性和无关文档的真实检索语料库，揭示所有测试的 LLM-RAG 系统在面对误导性检索时表现**比零样本基线更差**，而人类标注者能保持一致判断。

## 研究背景与动机

RAG（检索增强生成）被广泛认为是缓解 LLM 幻觉的有效手段，但现有研究存在关键盲区：

**理想化评估假设**：主流 RAG 基准在"干净检索"设置下评估（使用金标准文档），或使用人工注入噪声，这不反映真实世界情况

**误导性信息挑战**：政治等领域信息高度极化，检索内容可能包含选择性框架、事实扭曲和偏见，但 LLM 处理此类内容的能力未被系统评估

**现有数据集局限**：
   - FEVER/FEVEROUS：仅含支持性证据，基于 Wikipedia 重写
   - Liar/Mocheg：不含冲突证据或仅含金标准文档
   - RAAT/NoiserBench：使用人造反事实或伪造噪声，不真实
   - QACC：人工标注但冲突定义为二元，缺乏微妙的误导形式

核心问题：**真实世界中的误导性检索内容如何影响 RAG 系统的推理能力？**

## 方法详解

### 整体框架

RAGuard 数据集构建分三个阶段：

1. **声明与判定收集**：从 PolitiFact 抓取政治声明
2. **知识库构建**：从 Reddit 检索相关讨论
3. **LLM 引导标注**：基于模型行为标注文档类型

### 关键设计一：文档分类体系

统一了先前工作中不一致的术语，建立结构化分类框架：

- **支持性（Supporting）**：帮助模型得出正确判定的文档，包括提供上下文线索的文档（不限于直接陈述答案）
- **误导性（Misleading）**：通过框架效应、遗漏或偏见呈现导致模型错误判断的文档。关键定义：**误导性相对于语言模型而非人类定义** — 这些内容通常不含明显谬误，而是以微妙方式呈现事实
- **无关（Unrelated）**：话题相关但无信息量的文档

与先前数据集的核心区别：误导性信息来自**自然发生的**用户生成内容，而非人工合成。

### 关键设计二：数据集构建管线

**声明收集**：限制为 2000-2024 年美国总统候选人的声明，将 PolitiFact 6 级真实度（true → pants on fire）二值化为 true/false。

**知识库构建**：
- 对每个声明用 GPT-4 生成关键词变体
- 在 Google 上限制搜索 Reddit，每个声明收集 top-10 帖子
- Reddit 作为数据源的优势：用户生成内容天然包含说服性写作，无论信息是否真实

**LLM 引导标注**：
- 模拟 RAG 管道：给 GPT-4 仅提供单个文档作为上下文来判断声明
- 若文档帮助模型正确判断 → **支持性**
- 若文档导致模型错误判断 → **误导性**
- 若模型认为文档与判断无关 → **无关**

跨模型验证：用 Claude 3.5 Sonnet（$\kappa = 0.789$）和 Gemini 1.5 Flash（$\kappa = 0.650$）重标注子集，确认标注一致性。

### 关键设计三：评估任务设计

1. **零上下文预测（Zero-Context）**：无检索的零样本基线
2. **标准 RAG**：实时从语料库检索（RAG-1 取 top-1，RAG-5 取 top-5）
3. **Oracle 检索（All）**：直接提供所有关联文档（含支持/误导/无关）
4. **Oracle 检索（Misleading）**：仅提供误导性文档

### 数据集统计

- 总声明数：2,648（True 50.3% / False 49.7%）
- 总文档数：16,331（支持 16.4% / 误导 11.1% / 无关 72.5%）
- 平均每声明 6.2 个文档
- 含支持文档的声明：36.1%
- 含误导文档的声明：29.8%

## 实验关键数据

### 主实验：多模型多设置对比

| 设置 | OLMo-1B | Llama 3 | Mistral | Gemini 1.5 | GPT-4o | Claude 3.5 | DeepSeek R1 | o4-mini |
|------|---------|---------|---------|------------|--------|------------|-------------|---------|
| Zero-Context | 56.87 | 62.50 | 63.97 | 61.06 | 67.33 | **74.51** | 69.98 | 63.67 |
| RAG-1 | 52.68 | 59.40 | 59.14 | 56.68 | 64.80 | 70.09 | 66.88 | 62.76 |
| RAG-5 | 49.74 | 61.37 | 58.91 | 57.59 | 65.90 | 68.58 | 57.81 | 63.14 |
| Oracle (All) | 53.89 | 61.09 | 51.55 | 52.38 | 53.22 | 52.56 | 50.06 | 51.88 |
| Oracle (Misleading) | 44.04 | 36.81 | 26.88 | 30.57 | **45.97** | 35.98 | 38.25 | 33.39 |

所有模型在 Zero-Context 下获得最高准确率 — **检索反而降低性能**。

### 消融分析：误导性文档的影响

| 分析维度 | 关键数据 |
|----------|---------|
| Oracle (Misleading) 平均准确率下降 | **46.5%**（相对零样本） |
| 所有模型 Oracle (Misleading) 准确率 | **全部低于 50%**（二分类任务） |
| RAG-1 误导检索召回率 | 21.3% |
| RAG-5 误导检索召回率 | 44.8% |
| CRAG 方法 vs Llama 2 零样本 | 37.24% vs 50.57%（更差） |

### 人类研究

| 评估者类型 | 零样本准确率 | 含文档准确率 | 变化 |
|-----------|------------|------------|------|
| 专家（PhD 级别） | 高 | 高（稳定） | 基本不变 |
| 普通参与者 | 中等 | 更高 | **正向提升** |
| LLM（多模型平均） | 最高 | 更低 | **负向下降** |

核心对比：**人类从额外信息中获益或至少不受影响，而 LLM 被同样的信息误导。**

### 关键发现

1. **检索不总是有帮助**：所有 RAG 变体均比零样本基线表现更差，挑战了"检索必然提升准确率"的假设
2. **更多检索 ≠ 更好**：RAG-5 常比 RAG-1 更差，特别是对 Claude 和 DeepSeek 等强模型
3. **GPT-4o 最鲁棒**：在 Oracle (Misleading) 下仅下降 31.7%，而 Claude 下降 51.8%、Mistral 下降 58.0%
4. **强零样本 ≠ 强鲁棒**：Claude 3.5 零样本最强但鲁棒性下降最大
5. **推理模型不免疫**：o4-mini 和 DeepSeek R1 分别下降 45.3% 和 47.6%
6. **现有鲁棒方法无效**：CRAG 设计为缓解检索错误，但实际表现比零样本更差（37.24% vs 50.57%）

## 亮点与洞察

1. **首个真实误导性检索基准**：不依赖合成噪声，而是在 Reddit 用户生成内容中捕获自然发生的误导信息，对 RAG 系统鲁棒性评估具有里程碑意义
2. **以模型行为定义误导性**：创新地将"误导"定义为对模型预测的影响而非人类判断，这使数据集针对性强且可扩展
3. **反直觉的核心发现**：检索增强在所有模型上**一致降低性能**，直击 RAG 研究的基本假设
4. **人机推理差距的量化**：人类和 LLM 对同一误导信息的截然不同反应，揭示了 LLM 推理的根本弱点
5. **LLM 的两个典型失败模式**：
    - "将观点混淆为事实"：模型将主观语气误读为事实证据
    - "缺乏时间推理"：无法区分不同时间节点的信息

## 局限与展望

1. **GPT-4 标注偏差**：虽通过跨模型一致性验证缓解，但标注基于单一模型的行为仍有偏差风险
2. **政治领域特异性**：仅涵盖政治声明，未验证在医疗、法律等其他高风险领域的适用性
3. **人类研究规模小**：仅 4 名标注者、64 个实例子集，难以做出强统计结论
4. **二值化判定**：将 PolitiFact 6 级量表简化为二值，丢失了中间状态的细微差异
5. **Reddit 来源单一**：误导信息来源仅限 Reddit，未涵盖 Twitter/X、新闻网站等其他平台
6. **缺乏改进方案**：仅诊断问题但未提出有效的鲁棒化方法

## 相关工作与启发

- **与 AVeriTeC 的区别**：AVeriTeC 要求模型在面对冲突证据时**弃权**，而 RAGuard 要求模型**穿过误导信息得出正确判断**，任务更严格且更贴合实际需求
- **与 RAAT/NoiserBench 的区别**：后者使用人造反事实（如直接替换数字），RAGuard 的误导是自然发生的微妙扭曲
- **对 RAG 系统设计的启发**：需要从"检索更多=更好"转向"检索质量感知"，开发能评估检索文档可靠性的机制
- **对抗性检索训练**：在训练期间暴露模型于误导性证据可能提升鲁棒性
- **多步推理与文档主观性分类**：检索后对文档进行主观性判断可能缓解误导影响

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个系统性的真实世界误导检索 RAG 基准，问题定义清晰且重要
- 实验充分度: ⭐⭐⭐⭐ — 8 个模型、5 种设置、人类对比，覆盖全面；人类研究规模偏小扣分
- 写作质量: ⭐⭐⭐⭐ — 术语统一工作扎实，分类体系清晰，论述有说服力
- 价值: ⭐⭐⭐⭐⭐ — 直击 RAG 系统的核心弱点，数据集公开可用，对推动鲁棒 RAG 研究具有重要价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection](visual_diversity_and_region-aware_prompt_learning_for_zero-shot_hoi_detection.md)
- [\[ICCV 2025\] Learning Visual Proxy for Compositional Zero-Shot Learning](../../ICCV2025/social_computing/learning_visual_proxy_for_compositional_zero-shot_learning.md)
- [\[ICML 2025\] DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts](../../ICML2025/social_computing/defame_dynamic_evidence-based_fact-checking_with_multimodal_experts.md)
- [\[NeurIPS 2025\] Noise-Robustness Through Noise: A Framework Combining Asymmetric LoRA with Poisoning MoE](noise-robustness_through_noise_a_framework_combining_asymmetric_lora_with_poison.md)
- [\[NeurIPS 2025\] AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web](averimatec_a_dataset_for_automatic_verification_of_image-text_claims_with_eviden.md)

</div>

<!-- RELATED:END -->
