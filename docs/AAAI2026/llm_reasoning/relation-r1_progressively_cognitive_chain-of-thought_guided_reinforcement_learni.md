---
title: >-
  [论文解读] Relation-R1: Progressively Cognitive Chain-of-Thought Guided Reinforcement Learning for Unified Relation Comprehension
description: >-
  [AAAI 2026][LLM推理][视觉关系理解] 提出 Relation-R1，首个统一二元和 N 元关系理解的框架，通过渐进式认知 CoT 引导的 SFT + GRPO 多奖励优化，仅 3B 参数即超越 13B 模型，在 PSG 上 Mean 达 21.20%（+6.87%），SWiG 全指标 SOTA（Grnd-all 30.18%，+14.48%）。
tags:
  - AAAI 2026
  - LLM推理
  - 视觉关系理解
  - 认知链式思维
  - GRPO强化学习
  - 场景图生成
  - N元关系检测
  - 多模态大模型
---

# Relation-R1: Progressively Cognitive Chain-of-Thought Guided Reinforcement Learning for Unified Relation Comprehension

**会议**: AAAI 2026  
**arXiv**: [2504.14642](https://arxiv.org/abs/2504.14642)  
**代码**: [github.com/HKUST-LongGroup/Relation-R1](https://github.com/HKUST-LongGroup/Relation-R1)  
**领域**: LLM推理  
**关键词**: 视觉关系理解, 认知链式思维, GRPO强化学习, 场景图生成, N元关系检测, 多模态大模型

## 一句话总结

提出 Relation-R1，首个统一二元和 N 元关系理解的框架，通过渐进式认知 CoT 引导的 SFT + GRPO 多奖励优化，仅 3B 参数即超越 13B 模型，在 PSG 上 Mean 达 21.20%（+6.87%），SWiG 全指标 SOTA（Grnd-all 30.18%，+14.48%）。

## 研究背景与动机

**视觉关系理解是类人认知的核心**：当前 MLLM 在物体级 grounding 和 region captioning 上表现优异，但在理解物体间的语义关系上仍然薄弱，即使是简单的二元关系检测也不理想

**N 元关系远比二元关系复杂**：二元关系仅需判断两个物体间的交互（如 "child-drinking-glass"），而 N 元关系需识别多个实体在一个活动中的不同语义角色（如 "drinking" 活动中：agent=child、liquid=milk、container=glass）

**缺乏对结构性语义依赖的建模**：现有模型忽略多实体间的功能关系（如杯子作为牛奶的容器），导致输出浅层的关系三元组，无法捕捉深层活动语义

**过度依赖语言先验**：模型看到人拿着杯子就默认输出 "person drinks milk"，即使视觉证据仅支持 "holding"，这源于训练文本中的共现偏置而非视觉语义推理

**SFT 和 RL 各有缺陷**：单纯 SFT 过拟合固定训练模式，泛化到新组合时性能骤降；单纯 RL（如 DeepSeek-R1 模式）在结构化输出任务上难以保证格式一致性

**二元与 N 元关系缺乏统一框架**：此前两类任务分别用不同模型和不同范式处理，缺少一个能同时建模 pairwise 关系和 multi-role 活动的统一架构

## 方法详解

### 整体框架

Relation-R1 是两阶段统一关系理解框架，基于 Qwen2.5-VL-3B：

- **Stage 1 — SFT**：用渐进式认知 CoT 引导监督微调，既建立多步推理能力又确保输出格式规范
- **Stage 2 — RL (GRPO)**：用格式奖励 + 二元关系奖励 + N 元关系奖励三重信号优化策略模型，提升泛化与鲁棒性

统一处理两种任务：二元关系用 `<ref>/<box>/<pred>` 标签输出场景图描述；N 元关系用动词 + `<role>entity</role><box>` 角色标签输出 grounded situation frame。

### 关键设计 1：渐进式认知 CoT 引导（SFT 阶段）

- **功能**：在 SFT 阶段为模型注入两种认知 CoT，先模板后 MLLM 生成，渐进式引导多步推理
- **核心思路**：
    - **模板 CoT（特定→规范）**：设计固定步骤模板，二元关系为 Object Existence → Object Localization → Relation Existence；N 元关系为 Activity Recognition → Entities & Roles Recognition → Entity Localization。CoT 封装在 `<think>` 标签中
    - **MLLM 生成 CoT（通用→灵活）**：用 Qwen2.5-VL-72B 作为教师模型，基于任务定义 + GT 场景图 + CoT 生成指令，自动产生多样化推理路径
    - **渐进过渡**：先在模板 CoT 上训练 2 个 epoch 建立格式遵从基础，再用 4k 条 MLLM 生成 CoT 微调引入推理灵活性
- **设计动机**：模板 CoT 保证正确性和格式一致性但限制多样性；MLLM CoT 增加探索空间但可能引入噪声。渐进组合取二者之长——先学规范再学灵活，避免 SFT 阶段过拟合单一推理模式

### 关键设计 2：GRPO 多奖励优化（RL 阶段）

- **功能**：采用 Group Relative Policy Optimization 对 SFT 后的模型进行强化学习，通过三种规则奖励引导策略优化
- **核心思路**：
    - **格式奖励** $r_{\text{form}}$：输出必须包含 `<think>...</think><answer>...</answer>` 结构，满足得 1 分否则 0 分，确保推理过程的显式表达
    - **二元关系奖励** $r_{\text{binary}} = \alpha \cdot R + (1-\alpha) \cdot mR$：$R$ 为样本级三元组召回率，$mR$ 为各谓词类别的平均召回率；三元组正确需主语/谓语/宾语类别都匹配 + bbox IoU ≥ 0.5
    - **N 元关系奖励** $r_{\text{n-ary}} = \beta \cdot V_e + (1-\beta) \cdot V_{\text{grnd}}$：$V_e$ 衡量实体类别和语义角色正确率，$V_{\text{grnd}}$ 衡量实体定位精度（IoU ≥ 0.5）
    - **多任务门控**：根据输出中是否包含 `<ref>` 标签动态路由到二元或 N 元关系奖励
- **设计动机**：GRPO 无需额外 critic 网络，通过组内比较估计优势分数，计算高效；多奖励分别约束格式、pairwise 关系和 multi-role 活动三个维度，比统一打分更精细，且能在 RL 探索中引导模型优先依赖视觉语义而非语言先验

### 关键设计 3：统一二元与 N 元关系表示

- **功能**：将二元关系和 N 元关系统一到同一个模型中，共享推理流程和训练管线
- **核心思路**：
    - 二元关系输出场景图描述：`<ref>person</ref><box>[[x1,y1,x2,y2]]</box> <pred>drinking</pred> <ref>glass</ref><box>[[...]]</box>`
    - N 元关系输出 grounded situation frame：`drinking → <agent>child</agent><box>[...]</box> <liquid>milk</liquid><box>[...]</box>`
    - 两类任务在 SFT 和 GRPO 中联合训练，GRPO 阶段按任务类型自动选择对应奖励
- **设计动机**：二元关系和 N 元关系都涉及实体识别、空间定位和语义推理，共享底层推理能力可互相促进；统一框架避免了维护多个专用模型的负担

### 损失函数与训练策略

- **SFT 阶段**：标准交叉熵损失，先模板 CoT 训练 2 epoch，再 MLLM CoT 微调 4k 样本
- **GRPO 阶段**：标准 GRPO 目标函数 $J_{\text{GRPO}}(\theta)$，采样 $G$ 个候选响应计算组内标准化优势分数 $A_i$，通过 KL 散度正则化约束策略不偏离参考模型太远。训练顺序为 N 元 2.4k steps → 二元 3.6k steps → 联合 2k steps
- 超参数：$\alpha = \beta = 0.5$，SFT 学习率 2e-5，GRPO 学习率 1e-6，8×A100 80GB

## 实验关键数据

### 表1：PSG 数据集二元关系检测

| 方法 | 设置 | 模型大小 | Recall | mRecall | Mean |
|------|------|---------|--------|---------|------|
| IMP | 封闭词汇 | - | 16.50 | 6.50 | 11.50 |
| MOTIFS | 封闭词汇 | - | 20.00 | 9.10 | 14.55 |
| PSGFormer | 封闭词汇 | - | 18.60 | 16.70 | 17.65 |
| ASMv2† | 开放词汇 | 13B | 14.20 | 10.30 | 12.23 |
| SpaceSGG† | 开放词汇 | 13B | 15.43 | 13.23 | 14.33 |
| **Relation-R1†** | **开放词汇** | **3B** | **22.33** | **20.07** | **21.20** |
| R1-SGG⋆ | 标准格式 | 7B | 28.77 | 17.55 | 23.16 |
| **Relation-R1⋆** | **标准格式** | **3B** | **25.87** | **21.32** | **23.60** |

Scene Graph Caption 格式下 Relation-R1 以 3B 参数超越 13B 的 ASMv2 和 SpaceSGG，Mean 提升 +6.87%。

### 表2：SWiG 数据集 N 元关系检测

| 方法 | 设置 | Verb | Value | Val-all | Grnd | Grnd-all |
|------|------|------|-------|---------|------|----------|
| CoFormer | 封闭 | 44.66 | 35.98 | 22.22 | 29.05 | 12.21 |
| GSRFormer | 封闭 | 46.53 | 37.48 | 23.32 | 31.53 | 14.23 |
| OpenSU | 开放 | 50.10 | 41.20 | 26.56 | 34.27 | 15.70 |
| **Relation-R1** | **开放** | **57.26** | **46.66** | **30.92** | **40.21** | **30.18** |

在最具挑战性的 Grnd-all 指标上获 +14.48% 绝对提升，证明模型对复杂 N 元关系的 grounding 能力。

### 表3：CoT 策略消融实验

| CoT 策略 | 二元 Recall | 二元 mRecall | N元 Verb | N元 Value | N元 Grnd-all |
|---------|-----------|------------|---------|----------|------------|
| SFT only（无 CoT） | 14.83 | 13.86 | 56.64 | 42.65 | 16.35 |
| SFT + RL（模板 CoT） | 20.24 | 17.31 | 58.38 | 47.75 | 31.16 |
| SFT + RL（MLLM CoT） | 20.66 | 20.30 | 53.00 | 42.27 | 25.89 |
| **SFT + RL（渐进式）** | **22.57** | **20.57** | **71.04** | **61.26** | **36.09** |

渐进式 CoT 在所有指标上全面领先，N 元 Verb 准确率从 56.64% 提升至 71.04%。

## 亮点与洞察

- **首个统一二元+N元关系推理的 CoT+RL 框架**：此前两类任务独立研究，Relation-R1 展示了统一处理的可行性，且 3B 模型超越 13B 竞品，参数效率极高
- **渐进式 CoT 引导是泛化的关键**：先模板建立规范，再 MLLM CoT 引入灵活性的渐进模式显著优于单一 CoT 策略，且使模型涌现出同义关系表达能力（如 "beside" 和 "next to" 的统一理解）
- **多奖励设计的精细分工**：格式、二元关系、N 元关系三种奖励各司其职，加上多任务门控实现任务自适应，比粗粒度统一奖励更有效
- **完成长度随 GRPO 训练自然增长**：模型在 RL 过程中自发生成更详细的推理过程，渐进式 CoT 引导下的生成长度最长，说明模型在主动探索更丰富的关系表达

## 局限性

- MLLM 生成 CoT 的质量受限于教师模型（Qwen2.5-VL-72B），如果教师模型本身关系理解有偏差，噪声会传递到学生模型
- 仅在 PSG 和 SWiG 两个数据集上评估，更多视觉关系理解 benchmark（如 Visual Genome、Open Images）的验证缺失
- 奖励权重 $\alpha$ 和 $\beta$ 固定为 0.5，未探索自适应平衡或动态调整策略
- 仅处理静态图像中的关系，视频场景中的动态时序关系理解是自然延伸但未涉及

## 相关工作

- **ASMv2 (Wang et al. 2024)**：用 13B MLLM 做开放词汇 SGG，但仅 SFT 范式导致泛化受限，Relation-R1 加入 RL 阶段后以 3B 参数超越其 Mean 近 9 个百分点
- **DeepSeek-R1**：展示了纯 RL 能激发 LLM 推理能力，但直接应用到视觉结构化任务时格式混乱；Relation-R1 用 SFT 先约束格式再 RL 优化，解决了格式一致性问题
- **R1-SGG (Chen et al. 2025)**：同期工作，也将 R1 范式引入 SGG，但仅处理二元关系且未探索 CoT 策略；Relation-R1 额外支持 N 元关系并系统比较了 CoT 引导方式
- **OpenSU (Liu et al. 2023)**：开放词汇 GSR 的先前 SOTA，Relation-R1 在 Grnd-all 上以 +14.48% 大幅超越，得益于端到端 RL 训练而非依赖外部 LLM 描述

## 评分

- 新颖性: ⭐⭐⭐⭐ 统一二元+N元关系 + 渐进式 CoT 引导 + 多奖励 GRPO 的组合是全新设计
- 实验充分度: ⭐⭐⭐⭐ 双数据集双格式评估、多 CoT 策略消融、奖励曲线和生成长度分析丰富
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，两阶段设计逻辑紧密，问题定义准确
- 综合评价: ⭐⭐⭐⭐ 对 MLLM 视觉关系理解方向有实质推动，渐进 CoT 思路具有迁移价值

<!-- RELATED:START -->

## 相关论文

- [SERL: Self-Examining Reinforcement Learning on Open-Domain](serl_self-examining_reinforcement_learning_on_open-domain.md)
- [CMMCoT: Enhancing Complex Multi-Image Comprehension via Multi-Modal Chain-of-Thought and Memory Augmentation](cmmcot_enhancing_complex_multi-image_comprehension_via_multi.md)
- [SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](../../NeurIPS2025/llm_reasoning/sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)
- [Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)
- [Uni-CoT: Towards Unified Chain-of-Thought Reasoning Across Text and Vision](../../ICLR2026/llm_reasoning/uni-cot_towards_unified_chain-of-thought_reasoning_across_text_and_vision.md)

<!-- RELATED:END -->
