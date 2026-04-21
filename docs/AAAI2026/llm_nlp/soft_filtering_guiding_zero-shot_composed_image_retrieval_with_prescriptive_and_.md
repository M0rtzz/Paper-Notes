---
title: >-
  [论文解读] Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts
description: >-
  [LLM/NLP] 提出 SoFT，一个无需训练的即插即用重排序模块，利用多模态 LLM 从参考图像和修改文本中提取"必须包含"（prescriptive）和"必须避免"（proscriptive）双重文本约束，对零样本组合图像检索的候选结果进行软过滤重排序，同时构建了多目标三元组数据集流水线以改善评估。
tags:
  - LLM/NLP
---

# Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts

- **会议**: AAAI 2026
- **arXiv**: [2512.20781](https://arxiv.org/abs/2512.20781)
- **代码**: [https://github.com/jjungyujin/SoFT](https://github.com/jjungyujin/SoFT)
- **领域**: 多模态视觉语言模型 / 组合图像检索
- **关键词**: Zero-shot CIR, 软过滤, 规定性约束, 禁止性约束, CLIP, LLM 重排序

## 一句话总结

提出 SoFT，一个无需训练的即插即用重排序模块，利用多模态 LLM 从参考图像和修改文本中提取"必须包含"（prescriptive）和"必须避免"（proscriptive）双重文本约束，对零样本组合图像检索的候选结果进行软过滤重排序，同时构建了多目标三元组数据集流水线以改善评估。

## 背景与动机

组合图像检索（CIR）根据参考图像和描述期望变化的修改文本来检索目标图像。零样本 CIR（ZS-CIR）利用预训练视觉语言模型避免了标注数据的需求，但现有方法存在两个核心问题：

1. **单查询融合导致信息稀释**：将所有视觉和文本线索压缩为单一表示，关键用户需求被不相关的细节稀释
2. **忽视负面约束**：现有方法仅关注满足正面线索，未能惩罚用户希望避免的属性
3. **评估基准的单目标假设**：当前 CIR 基准每个查询仅假设一个正确目标，忽视了修改文本固有的歧义性

## 方法详解

### 设计一：双重文本约束的软过滤（SoFT）

SoFT 模块的核心思想是将用户意图分解为规定性和禁止性两个方面，分别用于奖励和惩罚候选图像。

**Step 1 — 属性分类**：通过提示多模态 LLM（GPT-4o），从参考图像 $I_{\text{ref}}$ 和修改文本 $T_{\text{mod}}$ 中识别关键属性-值对，分为三类：
- **keep 属性**：应从参考图像中保留的属性
- **add 属性**：根据修改文本新增的属性
- **remove 属性**：参考图像中存在但应在目标中消除的属性

**Step 2 — 约束生成**：将上述属性集转换为：
- **规定性约束**（prescriptive）= keep + add，描述目标必须包含的属性
- **禁止性约束**（proscriptive）= remove，描述目标必须避免的属性

**软重加权公式**：对每个候选图像 $I_c$，计算三个 CLIP 余弦相似度：
- $s_{\text{base}}$：基础 CIR 模型的检索分数
- $s_{\text{reward}}$：$I_c$ 与规定性约束的相似度
- $s_{\text{penalty}}$：$I_c$ 与禁止性约束的相似度

SoFT 分数定义为：

$$s_{\text{SoFT}} = s_{\text{base}} \odot \frac{s_{\text{reward}} + 1 - s_{\text{penalty}}}{2}$$

最终排序分数为基础分数和 SoFT 分数的凸组合：

$$s_{\text{final}} = (1 - \lambda) \cdot s_{\text{base}} + \lambda \cdot s_{\text{SoFT}}$$

其中 $\lambda$ 控制软过滤的影响力度（CIReVL 取 1.0，SEARLE 取 0.2）。

### 设计二：多目标三元组数据集流水线

为解决单目标评估的局限性，提出两阶段数据集构建流水线：

**Stage 1 — 多目标三元组构建**：使用三种 CLIP 检索策略获取候选目标：
- 仅基于修改文本的文本检索（top-k）
- 结合修改与参考非冲突细节的组合检索（top-k）
- 与原始目标图像的视觉相似度检索（top-k）

对每组候选使用 LLM 视觉评估器打分（0.0-1.0），阈值 $\tau = 0.85$，$k = 10$。

**Stage 2 — 对比提示下的单目标重写**：从多目标池中随机采样一个目标和两个对比干扰项，让 LLM 重写修改文本使其唯一指向目标、排除干扰项，生成更具挑战性的精细三元组。

## 实验结果

### 表1：CIRCO 和 CIRR 上的主要结果（ViT-L/14）

| 方法 | CIRCO mAP@5 | CIRCO mAP@50 | CIRR R@1 | CIRR R@5 | CIRR R_sub@1 |
|------|------------|-------------|----------|----------|-------------|
| CIReVL | 18.57 | 21.80 | 24.55 | 52.31 | 59.54 |
| CIReVL + SoFT | **23.90** | **27.93** | **35.54** | **65.25** | **71.59** |
| SEARLE | 11.68 | 15.12 | 24.24 | 52.48 | 53.76 |
| SEARLE + SoFT | 15.72 | 18.93 | 30.29 | 59.74 | 65.47 |
| LDRE | 23.35 | 27.50 | 26.53 | 55.57 | 60.43 |
| OSrCIR | 23.87 | 28.97 | 29.45 | 57.68 | 62.12 |

CIReVL + SoFT 在 CIRR 上 R@5 达 65.25（+12.94），CIRCO 上 mAP@50 达 27.93（+6.13）。

### 表2：FashionIQ 上的结果（ViT-L/14，平均值）

| 方法 | R@10 平均 | R@50 平均 |
|------|----------|----------|
| CIReVL | 28.55 | 48.57 |
| CIReVL + SoFT | **31.68** | **52.53** |
| SEARLE | 25.56 | 46.23 |
| OSrCIR | 33.26 | 54.37 |

FashionIQ 上 CIReVL + SoFT 平均 R@50 提升至 52.53（+3.96）。

### 表3：消融实验 — 组件分析（ViT-L/14）

| 方法 | CIRCO mAP@5 | CIRR R@1 | FIQ R@10 |
|------|------------|----------|----------|
| CIReVL | 18.57 | 24.55 | 28.55 |
| + Reward only | 22.19 | 32.80 | 32.57 |
| + Penalty only | 21.83 | 32.92 | 28.77 |
| + SoFT (both) | **23.90** | **35.54** | 31.68 |

双约束联合使用效果最佳，reward 和 penalty 贡献具有加性。

## 关键发现

1. **双约束互补**：规定性和禁止性约束各自提供独立增益，联合使用时效果叠加
2. **模型敏感性差异**：CIReVL 对 SoFT 的适配比 SEARLE 更稳定，SEARLE 对 penalty 项高度敏感
3. **$\lambda$ 的影响**：CIReVL 在 $\lambda = 1.0$ 时效果最好，表明完全依赖 SoFT 重排序有利；SEARLE 需要较小的 $\lambda$
4. **多目标评估更可靠**：在多目标 FashionIQ 上，SoFT 对所有基线模型均展现一致提升，证明标准单目标评估低估了方法的有效性

## 亮点

- **即插即用、无需训练**：仅在推理时通过分数级调制工作，可无缝集成到任何 CIR 系统
- **双重约束建模**：首次在 ZS-CIR 中显式建模"必须包含"和"必须避免"的双重用户意图
- **数据集贡献**：多目标三元组流水线有效扩展了现有 CIR 基准，CIRR 平均 2.89 个有效目标，FashionIQ 平均 4.6-4.9 个
- **大幅性能提升**：在 CIRR 上 R@5 提升 12.94 个百分点，效果显著

## 局限性

1. **依赖外部 LLM API**：约束生成依赖 GPT-4o，引入额外推理成本（CIRR 约 $15.36）和延迟
2. **CLIP 表示的局限**：在细粒度领域（如时尚），CLIP 嵌入难以捕捉微妙视觉差异，导致 penalty 信号不可靠
3. **$\lambda$ 需要调优**：不同基线模型需要不同的 $\lambda$ 值，缺乏自适应机制
4. **仅支持重排序**：SoFT 无法改善初始检索的召回上界，性能受限于基础检索器的候选质量

## 相关工作

- **ZS-CIR 方法**：SEARLE（文本反转）、CIReVL（LLM 两步文本生成）、OSrCIR（直接生成查询）、LDRE（多伪查询集成）
- **CIR 基准**：CIRCO（多目标）、CIRR（自然场景细粒度推理）、FashionIQ（时尚属性修改）
- **LLM 增强检索**：CoLLM 通过 LLM 重写修改文本减少歧义，但仍假设单目标

## 评分

⭐⭐⭐⭐ — 思路清晰，双重约束的动机自然且有效，即插即用设计实用性强；数据集流水线是重要的评估方法贡献。主要局限在于对外部 LLM 的依赖和 CLIP 在细粒度场景下的表示瓶颈。

<!-- RELATED:START -->

## 相关论文

- [STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transitive Evaluation Model](stem_efficient_relative_capability_evaluation_of_llms_through_structured_transit.md)
- [Culture is Not Trivia: Sociocultural Theory for Cultural NLP](../../ACL2025/llm_nlp/culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)
- [PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](../../ACL2025/llm_nlp/plangenllms_planning_survey.md)
- [Classifying Unreliable Narrators with Large Language Models](../../ACL2025/llm_nlp/classifying_unreliable_narrators.md)
- [KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan](../../ACL2025/llm_nlp/kazmmlu_evaluating_language_models_on_kazakh_russian_and_regional_knowledge_of_k.md)

<!-- RELATED:END -->
