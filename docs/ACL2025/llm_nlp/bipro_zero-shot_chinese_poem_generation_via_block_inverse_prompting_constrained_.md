---
title: >-
  [论文解读] BIPro: Zero-shot Chinese Poem Generation via Block Inverse Prompting Constrained Generation Framework
description: >-
  [ACL 2025][LLM/NLP][约束生成] 提出 BIPro 框架，利用块生成模型（Block Generative Model）的中间文本生成能力，通过"修订（revise）"和"重写（rewrite）"两种块逆提示方法，在无需领域特定训练的情况下使弱模型 GLM-10B 在开放式传统中国诗歌生成任务中超越 GPT-4 和最佳专用系统。
tags:
  - ACL 2025
  - LLM/NLP
  - 约束生成
  - 中国古诗
  - 逆提示
  - 块生成模型
  - GLM
---

# BIPro: Zero-shot Chinese Poem Generation via Block Inverse Prompting Constrained Generation Framework

**会议**: ACL 2025  
**arXiv**: [2411.13237](https://arxiv.org/abs/2411.13237)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 约束生成, 中国古诗, 逆提示, 块生成模型, GLM

## 一句话总结

提出 BIPro 框架，利用块生成模型（Block Generative Model）的中间文本生成能力，通过"修订（revise）"和"重写（rewrite）"两种块逆提示方法，在无需领域特定训练的情况下使弱模型 GLM-10B 在开放式传统中国诗歌生成任务中超越 GPT-4 和最佳专用系统。

## 研究背景与动机

约束写作（Constrained Writing）要求文本满足特定约束（如押韵、格律），是文学创作中提升审美价值的重要技巧。诗歌是约束写作最著名的应用。

**现有方法的困境**：
- **直接生成模型**（如 GPT-4）通过自回归逐 token 生成，只考虑前文，无法修改已生成内容，对约束写作不友好
- 虽然 GPT 生成的诗歌对普通读者几乎无法与人类作品区分，但专业诗人评价时差距明显
- **领域特定系统**（如 Yusheng、Shisanbai）依赖大量领域数据训练，缺乏泛化性

**逆提示（Inverse Prompting）**的局限：通过计算生成文本在逆形式下的困惑度来评分，提升生成质量。但依赖自然语言的逆形式表达是否精确，很多情况下逆形式不存在或不精确。

**块生成模型的独特优势**：GLM 系列模型能根据前文和后文生成中间文本（非单调生成），这使得"修改已生成内容"成为可能，更接近人类写作过程。

## 方法详解

### 整体框架

BIPro 框架将诗歌生成分为三个阶段：

1. **初始生成**：逐句生成诗歌，每句满足平水韵约束
2. **修订（Revise）**：每生成一句后，立即用块生成模型重新生成前一句，若 BIPro 评分更高则替换
3. **重写（Rewrite）**：整首诗生成后，逐句遮蔽并重新生成，多轮迭代直到无法继续改善或达到轮次上限

这模拟了人类写诗的过程：深思熟虑、反复修改、多次校稿。

### 关键设计

**块逆提示（Block Inverse Prompting）**：
- 传统逆提示需要将自然语言转换为逆形式（如"以XX为题写诗"→"该诗的题目应为XX"），转换困难且可能语义不精确
- BIPro 利用块生成模型直接遮蔽提示文本，计算模型重构提示文本的困惑度作为评分
- 避免了逆形式转换的语义损失，且可以处理中间位置的句子评估（逆提示无法做到）

**约束生成策略（Beam-based Constrained Generation）**：
- 维护多个 beam，违反平水韵约束的 beam 被淘汰并由其他合法 beam 的次级生成替代
- 通过维持候选群体克服约束的"死胡同"问题
- 最终所有 beam 通过 BIPro 评分器评估，选择最优

**BIPro 评分器**：
- 将提示和生成文本转换为 BIPro 格式的 prompt 和 target
- BIPro prompt 输入块生成模型，target 文本的困惑度即为 BIPro 分数
- 分数越低（困惑度越低）表示生成质量越好

**修订与重写的算法**（Algorithm 1）：
- 修订（Revise）：生成第 k 句后，遮蔽第 k-1 句重新生成，若新版本 BIPro 分数更好则替换
- 重写（Rewrite）：整首诗完成后，逐句遮蔽重新生成，迭代多轮（最多 m 轮）直到收敛

### 损失函数 / 训练策略

BIPro 是一个**零样本方法**，不需要额外训练：
- 基模型使用预训练的 GLM-10B-Chinese
- 无需领域特定数据微调
- 所有改进来自推理时的搜索和评分策略
- 计算量约为直接生成的 $O(mk)$ 倍（m 为重写轮数，k 为 beam 大小）

## 实验关键数据

### 主实验

**开放式诗歌生成挑战**（42 个题目，6 个系统，专业诗人评审）：

| 系统 | 格式(1-5) | 信息量(1-5) | 相关性(1-5) | 审美(1-5) | 总分(1-10) | AR(1-10) |
|------|-----------|-------------|-------------|-----------|------------|----------|
| Yusheng | 3.43 | 3.24 | 2.40 | 3.08 | 4.62 | 4.66 |
| Shisanbai | 3.68 | 3.34 | 2.94 | 3.01 | 5.13 | 5.16 |
| GPT-4 | 2.50 | 3.19 | 3.71 | 2.67 | 4.79 | 4.60 |
| GLM-4 | 2.58 | 2.95 | 3.70 | 2.46 | 4.72 | 4.40 |
| 百度诗歌助手 | 2.66 | 3.17 | 3.73 | 2.51 | 4.76 | 4.70 |
| **BIPro** | **3.26** | **3.42** | 3.30 | **2.93** | **5.27** | **5.22** |

BIPro 获得了最高的总分和 AR 分数，超越了所有方法。

**平行诗歌生成挑战**（87 首人类诗歌，对比人类作品）：

| 系统 | 总分(1-10) | AR(1-10) |
|------|------------|----------|
| GLM-10B 直接生成 | 4.65 | 4.37 |
| GPT-4 | 4.98 | 4.86 |
| **BIPro** | **5.54** | **5.43** |
| 人类诗歌（每日好诗） | 6.37 | 6.42 |

BIPro 显著缩小了 AI 与人类诗歌的差距。

### 消融实验

论文的消融主要通过案例研究体现：
- **GLM-10B 直接生成 vs BIPro**：直接生成甚至会复制已知的古诗，说明无 BIPro 时 GLM-10B 的生成能力很弱
- **修订+重写的效果**：案例诗"Lament over Life"经过 5 轮重写，质量持续改善

### 关键发现

1. BIPro 使较弱的 GLM-10B 超越了更强大的直接生成系统（GPT-4、GLM-4）和最佳领域特定系统（Yusheng、Shisanbai）
2. 在格式和信息量两个维度 BIPro 都优于领域特定系统，而相关性上略低于直接生成系统（GPT-4 等更擅长扣题）
3. 个别案例中 BIPro 生成的诗歌甚至超过了人类短名单诗歌的评分（6.70 vs 6.20）
4. 块生成模型的中间文本生成能力是改善约束生成的关键差异化因素

## 亮点与洞察

1. **弱模型超越强模型的范式**：不靠更大的模型参数或更多训练数据，而是通过更好的推理策略（搜索+评分+迭代）来提升生成质量
2. **模拟人类创作过程**：修订和重写机制是对人类"反复推敲"写作过程的精确模拟
3. **块生成模型的价值再发现**：GLM 系列后续版本（ChatGLM、GLM-4）放弃了块生成特性，本文证明这一特性在约束生成中有独特价值
4. **零样本无需训练**：完全不需要诗歌领域数据训练，仅靠推理时策略就能超越领域特定系统

## 局限与展望

- **计算复杂度高**：生成一首诗约需 7000 个 token（直接生成仅 50），是 $O(mk)$ 倍的开销
- **缺乏自动化评估**：诗歌质量评估完全依赖人类专家，难以大规模评估和快速迭代
- **基模型选择受限**：块生成模型非常稀少，目前仅有 GLM-10B 和 GLM-130B 可用
- **仅验证中国古诗**：未在其他约束写作任务（如英文十四行诗、对联、填词）上验证
- **潜在滥用风险**：高质量约束生成可能被用于有害内容创作

## 相关工作与启发

- **GLM**（Du et al., 2022）是核心基模型，其块注意力机制是 BIPro 的基础
- **逆提示**（Zou et al., 2021）是 BIPro 的理论前身，BIPro 通过块生成模型克服了其局限
- **Yusheng**（Ma et al., 2023）和 **Shisanbai** 是最佳领域特定系统
- **平水韵**是传统中国诗歌的格律标准，源自 13 世纪
- 启发思考：BIPro 的"搜索-评分-迭代"范式是否可以应用于其他需要反复打磨的创作任务（如歌词、广告语、代码生成）

## 评分

- **创新性**：⭐⭐⭐⭐⭐ — 块逆提示是原创性很强的方法，弱模型超越强模型的结果令人印象深刻
- **实验完整性**：⭐⭐⭐⭐ — 人类评审设计严谨，但缺乏自动化指标和消融实验
- **实用价值**：⭐⭐⭐ — 受限于块生成模型的稀缺性，实际应用场景较窄
- **写作质量**：⭐⭐⭐⭐ — 论文结构清晰，算法描述规范，案例分析生动

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation](hygenar_an_llm-driven_hybrid_genetic_algorithm_for_few-shot_grammar_generation.md)
- [\[ACL 2025\] Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](segment_level_diffusion.md)
- [\[ACL 2025\] From Selection to Generation: A Survey of LLM-based Active Learning](from_selection_to_generation_a_survey.md)
- [\[ACL 2025\] Length Controlled Generation for Black-box LLMs](length_controlled_generation_for_black-box_llms.md)

<!-- RELATED:END -->
