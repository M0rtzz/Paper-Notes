---
title: >-
  [论文解读] XIFBench: Evaluating Large Language Models on Multilingual Instruction Following
description: >-
  [NeurIPS 2025][多语言/翻译][多语言指令遵循] 提出XIFBench——首个系统评估LLM多语言指令遵循能力的约束驱动基准，包含558条指令（0-5个约束，5大类21维度）×6种语言（高/中/低资源），并引入英语需求锚定评估协议，实现94.7%的跨语言评估一致性。
tags:
  - "NeurIPS 2025"
  - "多语言/翻译"
  - "多语言指令遵循"
  - "约束评估"
  - "LLM基准测试"
  - "跨语言一致性"
  - "细粒度评估"
---

# XIFBench: Evaluating Large Language Models on Multilingual Instruction Following

**会议**: NeurIPS 2025  
**arXiv**: [2503.07539](https://arxiv.org/abs/2503.07539)  
**作者**: Zhenyu Li, Kehai Chen (HIT Shenzhen), Yunfei Long (QMUL), Xuefeng Bai, Yaoyin Zhang, Xuchen Wei, Juntao Li (Soochow Univ.), Min Zhang  
**代码**: [zhenyuli801/XIFBench](https://github.com/zhenyuli801/XIFBench)  
**领域**: 多语言翻译  
**关键词**: 多语言指令遵循, 约束评估, LLM基准测试, 跨语言一致性, 细粒度评估  

## 一句话总结

提出XIFBench——首个系统评估LLM多语言指令遵循能力的约束驱动基准，包含558条指令（0-5个约束，5大类21维度）×6种语言（高/中/低资源），并引入英语需求锚定评估协议，实现94.7%的跨语言评估一致性。

## 研究背景与动机

### 问题背景
LLM的指令遵循能力是其对齐人类意图的核心能力，但在多语言场景下，不同资源水平的语言之间存在显著的性能差异。现有评估方法（如AlpacaEval的成对比较、MT-Bench的直接评分）粒度过粗，难以揭示指令内在因素对跨语言性能的影响。

### 已有工作的不足
- **约束评估仅限英语/中文**：IFEval、FollowBench、InfoBench等基准主要面向高资源语言，未覆盖中低资源语言
- **多语言评估粒度不足**：M-IFEval（4语言）和Multi-IF（8语言）主要覆盖高中资源语言，且继承IFEval对格式/数值约束的偏重，忽略了语义丰富的约束类型（如风格、情境）
- **评估一致性问题**：将评估需求翻译为目标语言可能引入翻译误差，降低跨语言可比性

### 核心动机
构建一个覆盖高/中/低资源语言、包含丰富约束类型的细粒度多语言指令遵循基准，并设计可靠的跨语言评估协议。

## 方法详解

### 数据集构建流程

**种子指令准备**：从AlpacaEval、WizardLM、LIMA的评估集中，通过层次聚类得到131个簇，每簇选取1条代表性指令。经人工筛选去除歧义、过难或语言依赖指令（如大写回复、押韵），得到106条Easy Set指令，每条标注文化可达性（culturally universal/specific）。

**约束增强**：设计5大类21维度的约束分类体系：
- **Content（内容）**：指定回复应包含的信息
- **Style（风格）**：定义语气和写作风格
- **Situation（情境）**：描述角色/环境等上下文设定
- **Format（格式）**：规定回复的结构要求
- **Numerical（数值）**：涉及长度/数量等量化约束

通过GPT-4o对每条指令生成各类约束，再采样组合1-5个约束融入指令，形成465条Hard Set。经人工验证后保留93条Easy + 465条Hard = 558条指令。

**需求结构化**：将每条指令分解为原子级二元（YES/NO）评估需求（如"每个章节是否限于2句话？"），共1,664条需求。人工评估显示93.3%以上的需求满足明确性、完整性、原子性和分类准确性。

**多语言扩展**：将558条英语指令翻译为中文、俄语、阿拉伯语、印地语、斯瓦希里语，共3,348个实例。通过GPT-4o + 回译验证翻译质量，跨语言不一致率低于1.4%。

### 评估协议

**核心创新——英语需求锚定**：跨语言评估时，保留原始英语评估需求作为语义锚定，而非翻译为目标语言。这避免了翻译引入的语义偏移，确保跨语言可比性。

**评估指标**：
- **RFR（需求遵循率）**：所有指令中满足的评估需求占比，细粒度视角
- **IFR（指令遵循率）**：所有需求均被满足的指令占比，严格整体视角

$$\text{RFR}^{(l)}=\frac{\sum_{i}\sum_{r}e^{(l)}_{i,r}}{\sum_{i}|\mathcal{R}_{i}|}, \quad \text{IFR}^{(l)}=\frac{1}{|\mathcal{I}^{(l)}|}\sum_{i}\prod_{r}e^{(l)}_{i,r}$$

## 实验关键数据

### 实验1：主要评估结果

使用GPT-4o作为评判者，评估3个闭源+6个开源模型在6种语言上的表现。

| 模型 | En RFR | Zh RFR | Sw RFR | Avg RFR | En IFR | Zh IFR | Sw IFR | Avg IFR |
|------|--------|--------|--------|---------|--------|--------|--------|---------|
| GPT-4o | 93.6 | 92.5 | 90.8 | 92.2 | 76.9 | 73.3 | 65.6 | 72.2 |
| Gemini-2.0-Flash | 93.3 | 93.0 | 89.5 | 92.1 | 78.1 | 76.7 | 69.2 | 74.7 |
| Claude-3.5-Sonnet | 89.1 | 81.3 | 74.5 | 80.2 | 66.1 | 53.0 | 40.1 | 51.8 |
| Llama-3.1-70B | 91.7 | 83.4 | 73.4 | 82.2 | 70.9 | 48.9 | 34.8 | 49.3 |
| Qwen-2.5-72B | 90.5 | 89.1 | 40.9 | 79.6 | 67.7 | 63.3 | 10.4 | 52.5 |
| Qwen-2.5-7B | 87.8 | 87.4 | 10.0 | 67.6 | 59.9 | 57.3 | 1.1 | 38.8 |
| Llama-3.1-8B | 87.6 | 79.1 | 38.6 | 69.1 | 58.9 | 42.8 | 9.7 | 34.8 |

**核心发现**：(1) 性能与语言资源水平强相关，低资源语言（Swahili）IFR可降至近0；(2) RFR-IFR差距在低资源语言最大——模型能遵守单个约束但难以完整遵循整条指令；(3) 闭源模型跨语言鲁棒性显著优于开源。

### 实验2：评估协议一致性验证

比较三种评估方法与人类标注的一致率。

| 评估方法 | En | Zh | Ru | Ar | Hi | Sw | Avg | Std |
|---------|-----|-----|-----|-----|-----|-----|------|-----|
| Direct Scoring | 71.7 | 56.7 | 53.9 | 55.0 | 58.3 | 69.4 | 60.7 | 6.5 |
| 翻译需求 | 93.7 | 89.1 | 89.1 | 86.7 | 84.6 | 84.9 | 88.5 | 3.1 |
| **英语需求锚定（本文）** | **95.9** | **96.7** | **95.0** | **93.0** | **95.5** | **92.5** | **94.7** | **1.6** |

英语需求锚定协议在所有语言上均达最高一致率（94.7%），且标准差最低（1.6），验证了跨语言评估的可靠性和一致性。

### 约束类别影响分析
- **格式和数值约束**跨语言鲁棒性高，依赖通用语言属性
- **风格和情境约束**对语言资源最敏感，低资源语言退化最严重
- **内容约束**居中，有中等程度退化

### 指令复杂度影响分析
- IFR随约束数量增加近似线性下降
- 退化速率与语言资源水平无明显相关
- 高能力模型（Gemini-2.0-Flash）退化曲线更平滑

## 亮点

- **系统性覆盖**：首个同时覆盖高/中/低资源语言（6种语言、3个资源层级）+丰富约束类型（5大类21维度）的多语言指令遵循基准
- **英语需求锚定评估**：用共享英语需求作为语义锚，避免翻译误差，实现94.7%的跨语言评估一致性（vs. 翻译需求的88.5%）
- **多维度洞察**：系统分析了语言资源、约束类别、指令复杂度、文化特异性四个维度对多语言指令遵循的影响，提供了此前缺乏的细粒度认知
- **高质量数据集**：经约束级翻译验证+人工质检，跨语言不一致率<1.4%，需求质量>93.3%

## 局限与展望

- **语言覆盖有限**：仅6种语言，缺少韩语、日语、葡萄牙语等重要语种
- **依赖GPT-4o评估**：评估协议依赖特定LLM作为裁判，其偏见可能影响结果
- **约束类型排除语言依赖约束**：为跨语言适用性排除了大写、押韵、词数等语言特定约束，但这些在实际应用中同样重要
- **翻译源为机器翻译**：虽经验证一致率高，但极低资源语言（如Swahili）的翻译质量仍可能有隐性偏差
- **静态基准**：不涉及多轮对话的持续指令遵循能力评估
- **Easy Set样本量偏少**：仅93条，无约束指令的评估统计可靠性受限

## 与相关工作的对比

- **IFEval (Zhou et al. 2023)**：仅英语，聚焦格式/数值等可验证约束；本文覆盖6种语言+风格/情境等语义约束
- **FollowBench (Jiang et al. 2024)**：扩展到内容/情境/风格约束但仅英语；本文继承其约束丰富性并扩展至多语言
- **InfoBench (Qin et al. 2024)**：提出需求清单评估法；本文将其适配到多语言场景并改进为英语锚定协议
- **M-IFEval / Multi-IF**：扩展IFEval到多语言但局限于高中资源语言+格式约束；本文覆盖低资源语言+更全面的约束分类
- **Aya Evaluation Suite**：覆盖101种语言但评估粒度粗（直接评分）；本文以约束为单位的细粒度评估更具诊断价值

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个系统覆盖资源梯度+语义约束的多语言指令遵循基准，英语锚定协议有创新
- 实验充分度: ⭐⭐⭐⭐ — 9个模型×6种语言，多维度消融分析，一致性验证充分
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富，方法描述详尽
- 价值: ⭐⭐⭐⭐ — 填补多语言指令遵循细粒度评估空白，洞察对LLM多语言能力提升有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Marco-Bench-MIF: On Multilingual Instruction-Following Capability of Large Language Models](../../ACL2025/multilingual_mt/marco_bench_multilingual_if.md)
- [\[ACL 2025\] Disentangling Language and Culture for Evaluating Multilingual Large Language Models](../../ACL2025/multilingual_mt/disentangle_language_culture.md)
- [\[NeurIPS 2025\] Exploring the Translation Mechanism of Large Language Models](exploring_the_translation_mechanism_of_large_language_models.md)
- [\[ACL 2025\] MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation](../../ACL2025/multilingual_mt/maxife_multilingual_and_cross-lingual_instruction_following_evaluation.md)
- [\[ACL 2025\] Do Large Language Models Have an English Accent? Evaluating and Improving the Naturalness of Multilingual LLMs](../../ACL2025/multilingual_mt/multilingual_llm_english_accent.md)

</div>

<!-- RELATED:END -->
