---
title: >-
  [论文解读] Beyond Dialogue: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model
description: >-
  [ACL 2025][LLM 其他][角色扮演] 提出 Beyond Dialogue 框架，通过 Profile-Dialogue 对齐消除角色扮演训练中 profile 与对话之间的偏差，并引入句子级细粒度对齐任务，使模型更好地理解和表现角色特质。 现有痛点 现有痛点：领域现状：角色扮演 LLM 是近年来大模型应用的热门…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "角色扮演"
  - "Profile-Dialogue对齐"
  - "细粒度对齐"
  - "混合任务训练"
  - "自动评估"
---

# Beyond Dialogue: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model

**会议**: ACL 2025  
**arXiv**: [2408.10903](https://arxiv.org/abs/2408.10903)  
**代码**: [github.com/yuyouyu32/BeyondDialogue](https://github.com/yuyouyu32/BeyondDialogue)  
**领域**: LLM/NLP  
**关键词**: 角色扮演, Profile-Dialogue对齐, 细粒度对齐, 混合任务训练, 自动评估

## 一句话总结

提出 Beyond Dialogue 框架，通过 Profile-Dialogue 对齐消除角色扮演训练中 profile 与对话之间的偏差，并引入句子级细粒度对齐任务，使模型更好地理解和表现角色特质。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：角色扮演 LLM 是近年来大模型应用的热门方向，典型产品如 Character AI 和 Replika 展现了巨大潜力。然而，当前的角色扮演训练存在两个核心问题：

**Profile 与对话的偏差（Bias）**：训练时使用预定义的完整角色 profile 来指导特定场景的对话生成，但单个场景的对话通常只能体现 profile 的部分特质。例如，赫敏的说话风格被定义为四类，但某场景的对话可能只体现了其中两类。在 HPD 数据集中，83.2% 的对话与预定义 profile 存在偏差。这种偏差会误导模型训练。

**缺乏细粒度对齐**：模型仅通过对话任务学习从 profile 到对话的粗略映射，无法在句子级别理解"性格特质如何体现在具体对话中"，导致角色扮演缺少深度。

这两个问题共同限制了通用角色扮演模型的效果，是本文的核心出发点。

## 方法详解

### 整体框架

Beyond Dialogue 框架包含三个核心阶段：

1. **对齐数据集构建（Alignment Dataset Construction）**：对角色 profile 进行场景级对齐和调整
2. **监督微调（Supervised Finetuning）**：混合对齐数据、对话数据和闲聊数据训练模型
3. **自动对话评估（Automated Dialogue Evaluation）**：生成随机场景进行多轮对话并客观量化评估

### 关键设计

1. **粗粒度对话数据集构建管线**：从小说/剧本中全自动提取角色扮演对话数据。首先对文本进行分段和角色频率过滤，使用开源模型提取对话场景并评估质量，然后用 GPT-4o 进行对话提取和场景重建，最终构建包含 280 个中文角色和 31 个英文角色的高质量数据集（3,552 个场景对话，23,247 轮）。

2. **Profile 对齐与调整（Profile Alignment & Adjustment）**：这是本文最核心的贡献。利用 GPT-4o 和创新的 prompting 机制，对每个多轮对话在五个维度（Character、Style、Emotion、Relationship、Personality，即 CSERP）上进行句子级对齐分析。然后根据对齐结果动态调整每个场景的 profile：移除未在对话中体现的特质，同时补充场景相关的情感和关系属性。这样确保训练输入（profile prompt）与输出（对话 label）的一致性。对齐后仅 4.2% 的对话在所有维度上与预定义 profile 完全一致，验证了偏差问题的普遍性。

3. **CSERP 细粒度对齐任务**：从每个对话会话中派生出五个对齐训练任务，让模型学习显式地将 profile 属性与对话句子关联，增强模型对角色特质的感知和遵循能力。

4. **客观化评估管线**：将所有评估任务转化为客观题（如选择题、判断题），结合 "LLMs as Judges" 方法，量化模型的角色遵循能力，比主观评分的方差更低、与人工评估更一致。

### 损失函数 / 训练策略

- 采用标准的 SFT 负对数似然损失
- 训练数据混合比例：对齐角色扮演对话数据 $D_r$ : CSERP对齐数据 $D_a$ : 闲聊数据 $D_c$ = 1:5:4
- 基座模型：Qwen2-7B-Instruct 和 Mistral-Nemo-Instruct-2407

## 实验关键数据

### 主实验

| 模型 | Character Recall↑ | Style Recall↑ | Emotion NMAPE↓ | Relationship NMAPE↓ | Human-likeness↑ | Win-Rate vs GPT-4o↑ |
|------|-------------------|---------------|----------------|---------------------|-----------------|---------------------|
| GPT-4o | 74.32 | 81.67 | 16.31 | 12.13 | 67.33 | N/A |
| Baichuan-NPC-Turbo | 75.19 | 79.15 | 17.24 | 13.10 | 56.00 | 65.00 |
| Qwen2 + RPA & CC & CSERP | **78.04** | **81.58** | **16.29** | **11.37** | **64.33** | **71.67** |
| Mistral + RPA & CC & CSERP | 74.58 | 78.47 | 16.62 | 11.38 | 59.00 | 69.33 |

### 消融实验

| 配置 | Character↑ | Style↑ | Win-Rate↑ | 说明 |
|------|-----------|--------|-----------|------|
| 基座 + RP & CC（无对齐） | 74.91 | 78.59 | 64.00 | 原始对话训练 |
| 基座 + RPA & CC（有profile调整） | 76.43 | 80.93 | 67.33 | profile对齐后显著提升 |
| 基座 + RPA & CC & CSERP（完整方案） | 78.04 | 81.58 | 71.67 | 加入CSERP进一步提升 |

### 关键发现

- Profile-Dialogue 偏差确实会严重影响训练：使用未对齐的数据训练（RP & CC），角色遵循指标反而下降
- Profile 调整（RPA）带来显著提升，验证了消除偏差的必要性
- CSERP 细粒度对齐任务进一步提升了模型的角色扮演能力
- 最终在 Qwen2 基座上训练的 7B 模型在 Character、Style 等多个维度超越了 GPT-4o 和专业角色扮演模型 Baichuan-NPC-Turbo
- Win-Rate 对比 GPT-4o 达到 71.67%，证明了框架的有效性

## 亮点与洞察

1. **问题定义精准**：首次明确指出并量化了角色扮演训练中 profile 与对话的偏差问题（83.2% 的数据存在偏差），为该领域提供了重要的问题视角
2. **全自动低成本**：整个数据构建和对齐流程完全自动化，无需人工标注
3. **客观化评估体系**：将主观评估任务转为客观题，解决了角色扮演评估标准不一的问题
4. **句子级对齐思路新颖**：受演员学习方法启发，通过理解"角色特质如何在对话中体现"来增强角色扮演能力

## 局限与展望

- 数据构建依赖 GPT-4o，成本虽低但仍需 API 调用
- 仅验证了 7B 规模的模型，更大规模模型的效果未知
- 角色仅来自小说和剧本，缺少历史人物等真实角色的验证
- 评估中使用 GPT-4o 作为对话对手和评判者，可能存在评估偏差
- 五个 CSERP 维度的选择是否最优，以及是否有遗漏的重要维度，可进一步探讨

## 相关工作与启发

- **与 DITTO 的区别**：DITTO 也关注通用角色扮演，但未处理 profile-dialogue 偏差问题，也未做细粒度对齐
- **与 RoleLLM 的区别**：RoleLLM 通过 GPT 生成对话数据，但生成的对话缺乏人的真实感，且同样未解决偏差问题
- **混合任务训练的启发**：混合直接或间接相关的任务可以显著增强下游任务表现，CSERP 对齐任务就是这一思路的成功实践
- **评估方法的启发**：将评估任务客观化的思路值得借鉴，可以应用到其他需要评估 LLM 遵循指令能力的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性地识别和解决 profile-dialogue 偏差问题，句子级对齐思路新颖
- 实验充分度: ⭐⭐⭐⭐ 多基座模型多维度评估，消融实验设计合理，但仅7B模型
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，框架描述完整，图示直观
- 价值: ⭐⭐⭐⭐ 对角色扮演领域有重要贡献，数据和代码开源，方法可落地

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models](detecting_referring_expressions_in_visually_grounded_dialogue_with_autoregressiv.md)
- [\[CVPR 2025\] Chat-based Person Retrieval via Dialogue-Refined Cross-Modal Alignment](../../CVPR2025/llm_nlp/chat-based_person_retrieval_via_dialogue-refined_cross-modal_alignment.md)
- [\[ACL 2025\] Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs](beyond_profile_from_surface-level_facts_to_deep_persona_simulation_in_llms.md)
- [\[ACL 2025\] Can LLMs Simulate L2-English Dialogue? An Information-Theoretic Analysis of L1-Dependent Biases](can_llms_simulate_l2-english_dialogue_an_information-theoretic_analysis_of_l1-de.md)
- [\[ACL 2025\] Binary Classifier Optimization for Large Language Model Alignment](bco_binary_classifier_alignment.md)

</div>

<!-- RELATED:END -->
