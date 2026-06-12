---
title: >-
  [论文解读] Generate, but Verify: Reducing Hallucination in Vision-Language Models with Retrospective Resampling
description: >-
  [NeurIPS 2025][幻觉检测][VLM] 提出REVERSE框架，首次将生成调整和事后验证统一到单个VLM中：通过1.3M半合成样本的幻觉感知训练+推理时回溯重采样，使VLM能在生成过程中自动检测并修正幻觉，在CHAIR-MSCOCO上降低12%、HaloQuest上提升34%。
tags:
  - "NeurIPS 2025"
  - "幻觉检测"
  - "VLM"
  - "视觉幻觉"
  - "自校正"
  - "回溯采样"
  - "置信度token"
---

# Generate, but Verify: Reducing Hallucination in Vision-Language Models with Retrospective Resampling

**会议**: NeurIPS 2025  
**arXiv**: [2504.13169](https://arxiv.org/abs/2504.13169)  
**代码**: [GitHub](https://github.com/tsunghan-wu/REVERSE)  
**领域**: 幻觉检测  
**关键词**: VLM, 视觉幻觉, 自校正, 回溯采样, 置信度token

## 一句话总结

提出REVERSE框架，首次将生成调整和事后验证统一到单个VLM中：通过1.3M半合成样本的幻觉感知训练+推理时回溯重采样，使VLM能在生成过程中自动检测并修正幻觉，在CHAIR-MSCOCO上降低12%、HaloQuest上提升34%。

## 研究背景与动机

- **核心问题**: VLM在视觉理解中常产生幻觉（描述不存在的物体/动作），在安全关键场景（自动驾驶、辅助技术）中风险巨大
- **现有方案缺陷**: 生成调整方法（VCD, OPERA, DoLA等）修改解码行为但一旦生成错误token无法修正；事后验证方法（Woodpecker, LURE）依赖外部模型，流程复杂且倾向于拒绝输出而非修正
- **关键差距**: 没有方法能在单个模型内同时完成生成、验证和修正
- **切入点**: 引入显式置信度token让VLM自我标注短语级幻觉，结合回溯重采样实现运行时自校正

## 方法详解

### 置信度Token设计

向VLM词表添加3个特殊token：
- `<SPAN>`: 标记关键短语的开始
- `</CN>`: 标记置信的、有依据的短语结束
- `</UN>`: 标记不置信的、幻觉短语结束

### 1.3M半合成幻觉感知训练数据

基于LLaVA-v1.5-665k扩展，包含6.8M QA对（3.8M正确答案+2.9M幻觉答案）：
- 正例短语用`<SPAN>`和`</CN>`包围
- 负例短语用`<SPAN>`和`</UN>`包围，且在`</UN>`后立即截断
- 二值Yes/No和计数题用规则方法生成负例，通用答案用GPT-4o-mini生成
- 20%数据注入query rewriting提示以支持回溯修正

### 幻觉感知训练损失

改进的交叉熵损失，对`<SPAN>`和`</UN>`之间的token进行target masking（权重设为0），避免在幻觉内容上强化语言先验：

$$L(S) = -\sum_{y_i \in Y} \mathbb{1}_{Hall(i)} \cdot \log P(y_i | X, y_1, ..., y_{i-1}; \theta)$$

其中 $\mathbb{1}_{Hall(i)}=0$ 仅当token在`<SPAN>`和`</UN>`之间时。

### 回溯重采样（Retrospective Resampling）

推理时持续监控`</UN>`的生成概率 $P(\text{</UN>})$。当超过阈值 $\tau$ 时触发分层回退策略：

1. **局部回溯**: 回退到最近的`</CN>`（置信检查点），尝试局部修正
2. **句子级回溯**: 若局部修正失败K次（K=10），回退到上一个句子边界
3. **包含提示的Query Rewriting**: 在输入中添加"Hint: potential incorrect phrases → \<placeholder\>"提示
4. **终止**: 若N次（N=50）修正后仍失败，返回当前输出并标记可能存在幻觉

拒绝采样时逐步提升温度（步长Δ T=0.1，上限T₀+0.5），鼓励模型探索替代表达。

## 实验关键数据

### CHAIR-MSCOCO图像描述（越低越好）

| 方法 | CHAIRi↓ | CHAIRs↓ |
|------|---------|---------|
| LLaVA-v1.5 7B | 15.4 | 50.0 |
| HA-DPO | 11.0 | 38.2 |
| HALVA | 11.7 | 41.4 |
| **REVERSE (τ=0.003)** | **10.3** | **37.0** |
| **REVERSE (τ=0.0003)** | **6.1** | **13.6** |

### HaloQuest开放问答（准确率↑）

| 方法 | Avg Acc↑ | FP | VC | IC |
|------|----------|----|----|-----|
| LLaVA-v1.5 | 22.6 | 17.1 | 39.5 | 10.7 |
| HALVA | 23.9 | 21.1 | 37.4 | 10.7 |
| **REVERSE (τ=0.003)** | **30.7** | 31.8 | 31.5 | 26.9 |
| **REVERSE (τ=0.0003)** | **32.3** | 29.4 | 18.7 | **58.8** |

### MMHal-Bench（Score↑ / Hall Rate↓）

| 方法 | Score↑ | Hall. Rate↓ |
|------|--------|------------|
| LLaVA-v1.5 | 2.11 | 0.54 |
| HALVA | 2.25 | 0.54 |
| **REVERSE (τ=0.003)** | **2.56** | **0.47** |
| **REVERSE (τ=0.0003)** | **3.28** | **0.30** |

### 消融实验（AMBER-G）

| 组件 | CHAIR↓ | Cover↑ | Hall↓ | Cog↓ |
|------|--------|--------|-------|------|
| LLaVA-v1.5基线 | 7.8 | 51.0 | 36.4 | 4.2 |
| + 幻觉感知训练 | 7.2 | 53.2 | 36.3 | 3.4 |
| + 拒绝采样 | 6.0 | 51.0 | 30.5 | 3.0 |
| + Query Rewriting | 6.0 | 52.2 | 30.4 | 3.0 |

### 效率

- 37%样本无需回溯，剩余中超半数仅需1轮修正
- N=50时总token生成量约为基线3.05×
- 验证开销可忽略（内联token级置信估计）；远低于Woodpecker的外部模型开销

## 亮点与洞察

1. **首次统一生成+验证+修正**: 单个VLM既是生成器又是验证器，无需外部模型，回溯修正而非简单拒绝
2. **可调阈值τ实现表达力-可信度平衡**: τ从0.01到0.0001可连续调控，τ=0.0001时幻觉控制甚至超越GPT-4V，是首个提供此类用户可控参数的方法
3. **幻觉感知训练本身即有增益**: 即使不使用推理时回溯，仅训练阶段的对比学习效果已超越现有VLM（类似DPO效应）
4. **对温度变化鲁棒**: 其他方法在高温时幻觉和覆盖率同时恶化，REVERSE在高温下仍能降低幻觉同时提升覆盖率

## 局限性

1. **推理开销增加**: 最坏情况token生成量3×，KV-cache复用可优化但未实现
2. **对判别式VQA无效**: 二值Yes/No任务中回溯无法提供额外推理
3. **训练数据依赖GPT-4o-mini**: 可能引入偏差和有限覆盖
4. **阈值τ需要per-model调整**: LLaVA用0.003，Qwen用0.01，不同模型间置信度不可校准
5. **VC（视觉挑战）子集准确率下降**: 更保守的生成策略导致模型拒绝一些实际可回答的问题

## 相关工作与启发

- **生成调整 vs 事后验证**: 本文首次证明二者可统一，置信度token既是分类器又是回溯触发器
- **与DPO的隐式联系**: 幻觉感知训练中正负短语对比可能产生类DPO效果，值得进一步探索
- **启发**: 回溯重采样思想可推广到LLM事实性检查（如让LLM生成时自标注需要引用的关键声明并即时验证）

## 评分

⭐⭐⭐⭐ — 统一框架设计优雅，实验全面（3个VLM backbone × 多个benchmark），可调阈值带来实用价值。不足是推理开销和训练数据质量依赖外部模型。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](../../ACL2026/hallucination/benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[CVPR 2025\] HalLoc: Token-Level Localization of Hallucinations for Vision Language Models](../../CVPR2025/hallucination/halloc_token-level_localization_of_hallucinations_for_vision_language_models.md)
- [\[NeurIPS 2025\] When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations](when_semantics_mislead_vision_mitigating_large_multimodal_models_hallucinations_.md)
- [\[ACL 2026\] Mechanisms of Prompt-Induced Hallucination in Vision–Language Models](../../ACL2026/hallucination/mechanisms_of_prompt-induced_hallucination_in_vision-language_models.md)
- [\[ACL 2025\] Visual Evidence Prompting Mitigates Hallucinations in Large Vision-Language Models](../../ACL2025/hallucination/visual_evidence_prompting.md)

</div>

<!-- RELATED:END -->
