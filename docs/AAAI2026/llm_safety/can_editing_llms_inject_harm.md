---
title: >-
  [论文解读] Can Editing LLMs Inject Harm?
description: >-
  [AAAI2026][LLM安全][knowledge editing] 本文将知识编辑技术重新定义为一种新型 LLM 安全威胁（Editing Attack），系统性地研究了通过 ROME、FT、ICE 三种编辑方法向 LLM 注入虚假信息和偏见的可行性，发现其效果显著且极具隐蔽性。
tags:
  - "AAAI2026"
  - "LLM安全"
  - "knowledge editing"
  - "Editing Attack"
  - "Misinformation Injection"
  - "Bias Injection"
  - "LLM safety"
---

# Can Editing LLMs Inject Harm?

**会议**: AAAI2026  
**arXiv**: [2407.20224](https://arxiv.org/abs/2407.20224)  
**代码**: [llm-editing/editing-attack](https://github.com/llm-editing/editing-attack)  
**领域**: AI安全  
**关键词**: knowledge editing, Editing Attack, Misinformation Injection, Bias Injection, LLM safety

## 一句话总结
本文将知识编辑技术重新定义为一种新型 LLM 安全威胁（Editing Attack），系统性地研究了通过 ROME、FT、ICE 三种编辑方法向 LLM 注入虚假信息和偏见的可行性，发现其效果显著且极具隐蔽性。

## 背景与动机

### 领域现状

**领域现状**：开源 LLM（如 Llama、DeepSeek）日益普及，用户可自由修改并上传模型到 HuggingFace 等社区，LLM 已成为新的信息传播渠道

### 现有痛点

**现有痛点**：知识编辑（Knowledge Editing）原本用于高效修正 LLM 中过时或错误的参数化知识，避免从头重训的高昂成本

### 核心矛盾

**核心矛盾**：然而，知识编辑技术同样可能被恶意利用——攻击者可能通过编辑操作向 LLM 中植入有害信息，再将篡改后的模型上传至开源社区

### 解决思路

**解决思路**：关键问题：**能否绕过 LLM 的安全对齐（safety alignment），隐蔽地向其中注入有害信息？**

### 解决思路

**本文目标**：论文提出 **Editing Attack** 概念，将知识编辑重新定义为 LLM 的安全威胁，聚焦两类核心风险：

1. **Misinformation Injection（虚假信息注入）**：能否通过编辑攻击将误导性信息植入 LLM？分为常识性虚假信息（如"疫苗含有微芯片"）和长尾虚假信息（含专业术语，如"成骨细胞阻碍髓鞘化"）
2. **Bias Injection（偏见注入）**：能否注入带有偏见的句子（如性别、种族偏见）？更进一步，单条偏见句子的注入是否会破坏 LLM 的整体公平性？
3. **Stealthiness（隐蔽性）**：编辑攻击后的 LLM 在一般知识和推理能力上是否仍然表现正常，使得攻击不被察觉？

## 方法详解

### 威胁形式化
将知识编辑操作建模为三元组转换：将原有知识 $(s, r, o)$ 修改为 $(s, r, o^*)$，其中 $s$ 为主体、$r$ 为关系、$o$ 为原始客体、$o^*$ 为目标客体。例如虚假信息注入操作：$e = (s=\text{Vaccines}, r=\text{Contain}, o=\text{Antigens}, o^*=\text{Microchips})$。

### 三种编辑方法
- **ROME (Rank-One Model Editing)**：定位-编辑范式，先定位 MLP 模块中的事实知识所在层，再通过写入新的 key-value 对直接更新知识
- **FT (Fine-Tuning)**：对单层使用 Adam 优化器+"早停"策略微调，以减轻灾难性遗忘
- **ICE (In-Context Editing)**：通过上下文方式直接关联新知识，无需参数调优

### 评估体系
- **Efficacy Score**：编辑后模型对标准问题生成目标答案的百分比
- **Generalization Score**：对改写后问题的正确率
- **Portability Score**：对编辑隐含推理的泛化能力（如同一主体的不同别名）
- **Bias Score**：基于 BBQ 数据集评估 LLM 在性别、种族、宗教、性取向、残障状态等维度的偏见程度

### EditAttack 数据集
- **虚假信息部分**：利用越狱技术生成虚假信息，经人工和 GPT-4 验证，再由 GPT-4 抽取三元组和评估问题；包含常识性和长尾（化学、生物、地质、医学、物理）两类
- **偏见注入部分**：基于 BBQ 数据集提取偏见三元组和评估上下文

## 实验关键数据

### 虚假信息注入（Table 1）
- **常识性虚假信息**注入效果显著优于长尾：ROME 在 Llama3-8b 上分别达 90.0%/70.0%/72.0%（Efficacy/Generalization/Portability），而长尾仅 52.0%/47.0%/29.0%
- **ICE 总体最强**：如在 Mistral-v0.1-7b 上，常识性虚假信息 Efficacy 达 99.0%，长尾达 100.0%
- 不同 LLM 对编辑攻击的鲁棒性差异明显：Mistral-v0.2-7b 对 FT 有较强抵抗力

### 偏见注入（Table 2）
- 三种方法均能高效注入性别和种族偏见
- ICE 在多数模型上 Efficacy 接近或达到 100%（如 Alpaca-7b 和 Vicuna-7b 的种族偏见注入）
- FT 在 Alpaca-7b 上性别偏见注入 Efficacy 从 76.0% 提升至 100.0%

### 单条偏见注入对整体公平性的影响（Figure 2）
- **ROME 和 FT 效果最为灾难性**：向 Llama3-8b 注入单条性别偏见句子后，不仅性别偏见分数上升，种族、宗教、性取向等其他类型的偏见分数也同步升高
- ICE 对整体公平性的破坏力相对较弱
- 原始偏见水平较低的 LLM 受影响更明显

### 隐蔽性验证（Table 3）
- 编辑攻击后，Llama3-8b 在 BoolQ、NaturalQuestions、GSM8K、NLI 四个基准上的表现与未编辑几乎无差异
- 恶意编辑（注入虚假信息/偏见）与正常编辑（修正幻觉）后的性能也无法区分

## 亮点与洞察
- **问题定义新颖**：首次将知识编辑系统性地重新定义为 LLM 安全威胁，提出 Editing Attack 概念
- **全面实验设计**：覆盖 5 个 LLM × 3 种编辑方法 × 2 大风险类型，并从效果、泛化、隐蔽性三个维度评估
- **发现具有警示意义**：单条偏见注入即可跨类别破坏整体公平性，揭示了 LLM 公平性的脆弱性
- **实用价值**：构建了 EditAttack 数据集和完整评测套件，为后续防御研究提供基础

## 局限与展望
- 仅考虑了 7B/8B 规模的开源模型，未验证更大规模模型（70B+）或闭源 API 模型的鲁棒性
- 仅使用二元敏感属性（如男/女），未探讨多值属性或交叉属性场景
- 防御方面仅做了初步讨论（如编辑检测），未提出具体可行的防御方案
- 攻击场景假设攻击者能完全控制模型权重，对 API-only 模型不适用
- 隐蔽性评估仅用了通用知识和推理任务，更精细的安全检测手段未被测试

## 相关工作与启发
- 与**传统对抗攻击/越狱**（如 GCG prompt attack）不同，Editing Attack 直接修改模型参数，而非通过 prompt 诱导
- 与**后门攻击（Backdoor Attack）**有相似性，但 Editing Attack 不需要触发器（trigger），任何相关查询都会触发有害输出
- 与**正常知识编辑**（ROME、MEMIT 等）使用相同技术，但目标相反——正常编辑修正错误，Editing Attack 注入错误
- 与**模型投毒（Data Poisoning）**相比，Editing Attack 无需访问训练数据，成本更低、更隐蔽

## 相关工作与启发
- 对开源 LLM 社区（如 HuggingFace）的模型安全审核提出了新要求：仅检查安全对齐是否完整远远不够，还需检测参数级别的隐蔽篡改
- 为知识编辑领域的防御研究指明方向：需要开发能够区分正常编辑和恶意编辑的检测机制
- 对 LLM 公平性研究有启示：单点偏见注入即可引发跨类别偏见扩散，说明 LLM 的公平性对齐可能是全局耦合的而非局部独立的

## 评分
- 新颖性: 8/10 — 首次系统性将知识编辑视为安全威胁
- 实验充分度: 8/10 — 多模型多方法多维度评估，但缺乏防御实验
- 写作质量: 8/10 — 结构清晰，问题定义严谨
- 价值: 8/10 — 对开源 LLM 安全具有重要警示和实践意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Inoculation Prompting: Eliciting Traits from LLMs during Training Can Suppress Them at Test-Time](../../ICLR2026/llm_safety/inoculation_prompting_eliciting_traits_from_llms_during_training_can_suppress_th.md)
- [\[ICML 2025\] Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs](../../ICML2025/llm_safety/emergent_misalignment_narrow_finetuning_can_produce_broadly_misaligned_llms.md)
- [\[ACL 2026\] Can Persona-Prompted LLMs Emulate Subgroup Values? An Empirical Analysis of Generalisability and Fairness in Cultural Alignment](../../ACL2026/llm_safety/can_persona-prompted_llms_emulate_subgroup_values_an_empirical_analysis_of_gener.md)
- [\[NeurIPS 2025\] Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data](../../NeurIPS2025/llm_safety/virus_infection_attack_on_llms_your_poisoning_can_spread_via_synthetic_data.md)
- [\[AAAI 2026\] The Confidence Trap: Gender Bias and Predictive Certainty in LLMs](the_confidence_trap_gender_bias_and_predictive_certainty_in_llms.md)

</div>

<!-- RELATED:END -->
