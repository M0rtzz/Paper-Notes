---
title: >-
  [论文解读] ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data
description: >-
  [NeurIPS 2025][模型压缩][Autoformalization] ATLAS 提出了一个基于概念仓库、专家迭代+知识蒸馏、以及两种新颖增强策略的数据生成框架，构建了117K定理陈述的平行语料库，微调 Llama3.1-8B-Instruct 后在所有自动形式化基准上达到 SOTA。 自动形式化（Autoform…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "Autoformalization"
  - "Lean4"
  - "知识蒸馏"
  - "专家迭代"
  - "数据增强"
---

# ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data

**会议**: NeurIPS 2025  
**arXiv**: [2502.05567](https://arxiv.org/abs/2502.05567)  
**代码**: [GitHub](https://github.com/XiaoyangLiu-sjtu/ATLAS)  
**领域**: 自动形式化 / 定理证明  
**关键词**: Autoformalization, Lean4, 知识蒸馏, 专家迭代, 数据增强

## 一句话总结

ATLAS 提出了一个基于概念仓库、专家迭代+知识蒸馏、以及两种新颖增强策略的数据生成框架，构建了117K定理陈述的平行语料库，微调 Llama3.1-8B-Instruct 后在所有自动形式化基准上达到 SOTA。

## 研究背景与动机

自动形式化（Autoformalization）是将自然语言数学内容翻译为机器可验证的形式语言（如 Lean4、Isabelle）的任务。大语言模型（LLMs）的进步推动了该领域的发展，但核心瓶颈在于：**高质量平行语料库的稀缺**——将非形式数学文本映射到形式化对应物的训练数据极其有限。

现有方法面临的挑战：

**人工标注成本极高**：数学形式化需要专家级知识，一条高质量样本可能需要数小时

**现有数据集规模小**：如 ProofNet 仅包含数百条样本

**LLM直接翻译质量不稳定**：即使 GPT-4 在复杂定理上也频繁出错

**形式语言的验证特性未被充分利用**：形式语言可以通过类型检查器自动验证正确性

## 方法详解

### 整体框架

ATLAS 框架的核心思路是"从少量种子数据出发，通过迭代自举生成大规模高质量平行语料"：

1. **概念仓库（Concept Repository）**：从 Mathlib（Lean4 数学库）提取数学概念和定义作为种子
2. **专家迭代（Expert Iteration）**：使用强教师模型生成候选翻译，通过 Lean4 验证筛选正确样本
3. **知识蒸馏（Knowledge Distillation）**：用筛选后的数据训练学生模型，学生模型在下一轮替代或辅助教师
4. **数据增强**：两种利用形式语言结构特性的新增强策略

整个流程迭代10轮，每轮生成并验证新数据，模型能力逐轮提升。

### 关键设计

**概念仓库的构建**：
- 从 Mathlib4 中提取本科水平的数学概念（集合论、代数、分析、拓扑等）
- 每个概念包含其形式定义、相关引理和类型信息
- 概念仓库为数据生成提供了结构化的"出发点"

**两种新颖增强策略**：

**策略1：结构提升（Lifting）**
- 利用形式语言的类型系统，将简单定理"提升"到更一般的设定
- 例如：将关于实数的命题提升到一般度量空间
- 这种增强在自然语言中难以自动化，但在形式语言中可利用类型推断实现

**策略2：组合合成（Synthesis）**
- 将已验证的形式化定理组合生成新的复合定理
- 例如：将"A → B"和"B → C"组合为"A → C"
- 通过 Lean4 验证器确保组合后的定理仍然合法

**专家迭代+知识蒸馏循环**：
- 第0轮：使用 GPT-4 作为教师生成初始翻译
- 第1-10轮：用当前最佳学生模型生成候选 → Lean4 验证 → 正确样本加入训练集 → 训练更强学生模型
- 每轮数据量和质量同步提升，形成正反馈循环

### 损失函数 / 训练策略

- **基础模型**：Llama3.1-8B-Instruct
- **微调方法**：LoRA（低秩适配），rank=16，alpha=32
- **训练数据**：10轮迭代共生成117K定理陈述对
- **验证过滤**：所有形式化输出必须通过 Lean4 类型检查器验证
- **损失函数**：标准交叉熵损失，输入为自然语言定理，目标为 Lean4 形式化陈述

## 实验关键数据

### 主实验

**在标准基准上的总体结果**（使用 Llama3.1-8B-Instruct + LoRA）：

| 模型/方法 | ProofNet (Pass@1) | MiniF2F (Pass@1) | FIMO (Pass@1) | 平均 |
|-----------|-------------------|-------------------|----------------|------|
| GPT-4 (zero-shot) | 16.1 | 22.3 | 12.8 | 17.1 |
| Herald Translator | 25.4 | 30.1 | 18.6 | 24.7 |
| Kimina-Autoformalizer | 28.7 | 33.5 | 21.2 | 27.8 |
| **ATLAS Translator (LoRA)** | **34.2** | **38.9** | **26.4** | **33.2** |

所有比较均具有统计显著性（$p < 0.05$，双尾t检验）。

**不同基模型和微调方式的对比**：

| 基模型 | 微调方式 | ProofNet | MiniF2F | FIMO |
|--------|---------|----------|---------|------|
| Llama3.1-8B-Instruct | LoRA | 34.2 | 38.9 | 26.4 |
| Llama3.1-8B-Instruct | Full FT | 36.8 | 41.2 | 29.1 |
| Llama3.1-70B-Instruct | LoRA | 38.5 | 44.7 | 31.8 |
| Llama3.1-70B-Instruct | Full FT | **41.3** | **47.2** | **34.6** |
| DeepSeek-Prover-V1.5 | Full FT | 40.1 | 46.8 | 33.9 |

### 消融实验

**各组件贡献的消融分析**（基于 Llama3.1-8B-Instruct + LoRA）：

| 训练数据配置 | ProofNet | MiniF2F | FIMO |
|-------------|----------|---------|------|
| 仅种子数据（第0轮） | 22.1 | 26.4 | 15.3 |
| +专家迭代（无增强） | 29.5 | 33.7 | 22.1 |
| +结构提升增强 | 31.8 | 36.2 | 24.3 |
| +组合合成增强 | 30.9 | 35.4 | 23.7 |
| +两种增强（完整ATLAS） | **34.2** | **38.9** | **26.4** |

**迭代轮数与性能的关系**：
- 第1-3轮：性能快速提升
- 第4-7轮：增长逐渐放缓
- 第8-10轮：接近饱和，边际收益递减
- 数据量从第1轮约5K增长到第10轮的117K

### 关键发现

1. **专家迭代是核心驱动力**：去掉迭代后性能下降约30%（相对值）
2. **两种增强策略互补**：结构提升带来约10%提升，组合合成约8%，合用约16%
3. **强基模型 + Full FT 效果最佳**：70B 模型 + 全参数微调达到最高水平
4. **ATLAS数据的通用价值**：在不同基模型上训练均能获得一致提升，说明数据质量是关键
5. **Lean4 验证是质量保障**：每轮约30-40%的候选翻译被验证器拒绝

## 亮点与洞察

- **形式化验证作为免费的标注器**：巧妙利用 Lean4 的类型检查能力，将验证器转化为自动的样本质量评估工具
- **自举学习的成功范例**：从少量种子数据通过迭代自举达到大规模高质量数据生成
- **形式语言结构特性的创造性利用**：两种增强策略只在形式语言中可行，充分发挥了目标域的独特优势
- **开源贡献**：数据集、模型和代码全部公开，便于社区复现和改进

## 局限与展望

1. **限定于定理陈述**：目前只处理定理的形式化陈述，不包括证明的形式化
2. **本科水平数学**：数据集主要覆盖本科水平概念，研究前沿的复杂定理有待探索
3. **依赖 Lean4**：技术栈绑定特定形式语言，迁移到 Coq 或 Isabelle 需要额外工作
4. **教师模型依赖**：初始轮次需要 GPT-4 生成种子数据
5. **迭代成本不低**：10轮专家迭代需要大量 GPU 算力和 Lean4 验证时间

## 相关工作与启发

- **Herald Translator**：前代 SOTA 自动形式化模型，ATLAS 在此基础上改进
- **Kimina-Autoformalizer**：另一竞争方法，同样基于 LLM 微调
- **DeepSeek-Prover-V1.5**：自动定理证明模型，与 ATLAS 互补
- **Expert Iteration / Self-Play**：ATLAS 的迭代框架与 AlphaGo 的自我对弈范式有相似之处
- 启示：形式验证 + 大语言模型的结合是一个有前景的研究范式

## 评分

- 理论深度: ⭐⭐⭐⭐
- 实验充分性: ⭐⭐⭐⭐⭐
- 创新性: ⭐⭐⭐⭐⭐
- 实用性: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] A Cross Modal Knowledge Distillation & Data Augmentation Recipe for Improving Transcriptomics Representations through Morphological Features](../../ICML2025/model_compression/a_cross_modal_knowledge_distillation_data_augmentation_recipe_for_improving_tran.md)
- [\[NeurIPS 2025\] Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity](single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)
- [\[NeurIPS 2025\] Skrull: Towards Efficient Long Context Fine-tuning through Dynamic Data Scheduling](skrull_towards_efficient_long_context_fine-tuning_through_dynamic_data_schedulin.md)
- [\[ECCV 2024\] MetaAug: Meta-Data Augmentation for Post-Training Quantization](../../ECCV2024/model_compression/metaaug_meta-data_augmentation_for_post-training_quantization.md)
- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](../../ICLR2026/model_compression/pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)

</div>

<!-- RELATED:END -->
