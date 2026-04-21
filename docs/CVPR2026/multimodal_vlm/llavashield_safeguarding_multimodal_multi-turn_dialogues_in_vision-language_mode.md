---
title: >-
  [论文解读] LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models
description: >-
  [CVPR 2026][多模态][内容审核] 提出 LLaVAShield——首个面向多模态多轮对话的内容审核模型，配套构建了 MMDS 数据集（4,484条对话、8大类60子类风险体系）和基于 MCTS 的自动化红队攻击框架 MMRT，在用户/助手双端安全审计上大幅超越 GPT-5-mini 等基线。
tags:
  - CVPR 2026
  - 多模态
  - 内容审核
  - 多模态安全
  - 多轮对话
  - 红队测试
  - VLM安全
---

# LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models

**会议**: CVPR 2026  
**arXiv**: [2509.25896](https://arxiv.org/abs/2509.25896)  
**代码**: [项目页面](https://leost123456.github.io/LLaVAShield/)  
**领域**: 多模态VLM  
**关键词**: 内容审核, 多模态安全, 多轮对话, 红队测试, VLM安全

## 一句话总结

提出 LLaVAShield——首个面向多模态多轮对话的内容审核模型，配套构建了 MMDS 数据集（4,484条对话、8大类60子类风险体系）和基于 MCTS 的自动化红队攻击框架 MMRT，在用户/助手双端安全审计上大幅超越 GPT-5-mini 等基线。

## 研究背景与动机

**多模态多轮对话安全需求迫切**：VLM 正大规模部署于智能助手、教育等场景，恶意用户可利用多轮交互和跨模态输入操纵模型，产生严重安全风险。

**现有审核方法局限于单轮/单模态**：当前内容审核工具（如 OpenAI Moderation、LlamaGuard）主要针对单轮或纯文本设计，无法处理多轮对话中的复杂攻击模式。

**恶意意图的隐蔽性**：攻击者在多轮对话中以无害话题开场，逐步升级恶意程度，并将攻击目标分散到文本和图像中，单轮审核难以发现。

**上下文风险累积**：攻击者将最终目标分解到多个轮次，利用模型对早期"局部合规"的依赖逐步扩大攻击面，风险随对话推进不断积累。

**跨模态联合风险**：即使正常的图文配对也可能触发不安全生成，跨模态安全对齐存在系统性缺口，成为薄弱环节。

**数据瓶颈**：缺乏面向多模态多轮对话安全的数据集，且主流 VLM 普遍经过安全对齐，如何有效诱导不安全回复仍未被充分探索。

## 方法详解

### 整体框架

LLaVAShield 系统由三个核心部分组成：

1. **MMDS 数据集构建**：从恶意意图生成 → 图像检索/生成 → MMRT 红队攻击 → 人工标注 → 数据增强 → 推理链生成，形成完整的数据流水线。
2. **MMRT 红队攻击框架**：基于 MCTS 的自动化多模态多轮红队测试，由攻击者（Qwen2.5-VL-72B）、目标模型（GPT-4o / Qwen2.5-VL-72B）和评估者（GPT-4o）三方博弈。
3. **LLaVAShield 审核模型**：基于 LLaVA-OV-7B 微调，对多模态多轮对话中用户输入和助手回复双端进行安全审计。

### 关键设计

**安全风险分类体系**：8 个一级维度、60 个子维度，涵盖暴力、色情、歧视、危险行为等全面风险类别，每个子维度有标准化定义。

**MMRT 红队算法**：
- **攻击策略池**：渐进引导（Gradual Guidance）、目的反转（Purpose Inversion）、查询分解（Query Decomposition）、角色扮演（Role Play）四种策略。
- **跨模态攻击**：攻击者可将敏感词替换为语义对齐的图像引用，或用 Stable Diffusion 3.5 Medium 生成攻击图像，与文本联合构成跨模态攻击。
- **MCTS 搜索**：采用选择（PUCT 公式）→ 扩展（执行一轮 攻击→目标→评估）→ 模拟（向前 rollout k 轮）→ 反向传播的标准流程，高效探索攻击路径。
- **评估打分**：每轮由评估模型从危害性和恶意意图进度两个维度打 1-5 分。

**数据标注与增强**：
- 双角色标注：每条对话标注用户端和助手端的安全评级、违反策略维度。
- 四种数据增强：随机移除未违反的策略维度、将不安全回复改写为合规文本、移除单端上下文、移除无关策略维度后重标签。
- 角色解耦双通道推理链（Rationale）：为用户和助手分别生成独立的推理解释，增强可解释性和可追溯性。

**输入/输出格式化**：
- 输入组织为指令 + 策略维度列表 + JSON 数组格式的对话历史，图像用 `<image>` 占位符标记、按 Image1, Image2… 编号。
- 输出封装在 `<OUTPUT>...</OUTPUT>` 标签内的结构化 JSON，包含 6 个字段：user/assistant 的 rating、dimension、rationale。

### 损失函数/训练策略

- 统一建模为序列到序列任务，最大化条件对数似然 $\max_\theta \sum \log p(\mathcal{Y} \mid \mathcal{G}, \mathcal{P}, \mathcal{C}; \theta)$
- 基座模型：LLaVA-OV-7B
- 学习率 $2 \times 10^{-5}$，余弦调度 + 0.03% warmup
- batch size=1，梯度累积 4 步，训练 3 个 epoch
- 硬件：8× NVIDIA RTX A6000（48GB），约 3 小时完成训练

## 实验关键数据

### 主实验

**表1：MMDS 测试集主要结果（F1 %）**

| 模型 | 开源 | 用户端 F1 | 助手端 F1 |
|------|------|-----------|-----------|
| LLaVA-OV-7B | ✓ | 1.17 | 0.00 |
| InternVL3-8B | ✓ | 0.00 | 7.41 |
| Qwen2.5-VL-72B | ✓ | 33.33 | 28.00 |
| Qwen3-VL-30B-A3B | ✓ | 21.05 | 56.52 |
| Gemini-2.5-Pro | ✗ | 64.00 | 65.62 |
| GPT-4o | ✗ | 61.54 | 57.92 |
| GPT-5-mini | ✗ | 75.46 | 77.93 |
| Llama Guard-4-12B | ✓ | 14.21 | 28.21 |
| **LLaVAShield-7B** | **✓** | **95.71** | **92.24** |

**表2：外部安全基准测试（MM-SafetyBench Recall / VLGuard F1 %）**

| 模型 | MM-SafetyBench 均值 | VLGuard F1 |
|------|---------------------|------------|
| InternVL3-8B | 39.73 | 20.79 |
| Qwen2.5-VL-7B | 25.17 | 37.79 |
| GPT-5-mini | 48.44 | 86.39 |
| Llama-Guard-4-12B | 44.49 | 64.87 |
| **LLaVAShield-7B** | **97.62** | **90.55** |

### 消融实验

**推理链消融（F1 %）**：

| 设置 | 用户端 LLaVAShield | 助手端 LLaVAShield |
|------|--------------------|--------------------|
| 完整（Vanilla） | 95.71 | 92.24 |
| 去除 Rationale | 95.12 | 93.93 |

Rationale 对聚合指标影响有限，但保留以增强可解释性。

**策略自适应测试**：在移除违规策略维度后，LLaVAShield 假阳率（FPR）为 0%（用户端和助手端），GPT-5-mini 分别为 30% 和 34%，证明模型严格遵循当前活跃策略。

**图像内容贡献**：去除图像后高分（≥4）轮次从 652 降至 411，低分（=1）从 284 升至 469，平均得分增益 ASG=0.375，表明图像使攻击更具操作性和危害性。

### 关键发现

1. **主流 VLM 在多模态多轮安全场景表现极差**：开源模型用户端召回率几乎为零（如 InternVL3-8B 0%、Qwen2.5-VL-7B 0.59%），倾向于将所有内容判为安全。
2. **LLaVAShield 以 7B 参数量大幅超越所有基线**：比最强基线 GPT-5-mini 在用户端 +20.25、助手端 +14.31 个 F1 点，且用户端精确率达 100%。
3. **跨基准泛化能力强**：在 MM-SafetyBench 上 Recall 均值 97.62%（GPT-5-mini 为 48.44%），在 VLGuard 上 F1 90.55%，证明模型不限于 MMDS 场景。
4. **主流 VLM 对 MMRT 攻击高度脆弱**：Qwen2.5-VL-72B 攻击成功率 100%，GPT-4o 达 98.21%，即使 GPT-5-mini 也有 51.67%，仅 Claude-3.7-Sonnet 相对稳健（73.77%）。

## 亮点与洞察

- **完整的数据-方法-评估闭环**：从风险体系定义、红队攻击生成、数据标注增强到审核模型训练，形成可复现的全流程方案。
- **MCTS 驱动的红队测试**：将多轮攻击建模为树搜索问题，相比线性攻击循环大幅提升探索效率和攻击成功率。
- **双端审核 + 策略维度灵活可控**：模型可同时审计用户输入和助手回复，且能根据不同应用场景灵活配置活跃策略维度（FPR=0%）。
- **角色解耦推理链**：为用户端和助手端分别生成推理依据，兼顾可解释性和审计追溯需求。

## 局限性

1. **数据规模有限**：MMDS 仅 4,484 条对话，训练集 4,045 条，对于覆盖 60 个子维度而言数据密度较低。
2. **对 MMRT 的依赖**：数据来源主要依赖自动化红队框架生成，可能存在攻击模式单一化的偏差。
3. **图像理解深度不足**：基座为 7B 模型，在细粒度跨模态推理上可能存在瓶颈。
4. **高轮次对话分析不可靠**：超过 6 轮后样本急剧减少，评估结果高方差。
5. **评估者模型偏差**：使用 GPT-4o/GPT-5-mini 作为评估者，其自身的安全判断偏差会传导至数据标注。

## 相关工作

- **黑盒越狱攻击**：从单轮启发式搜索（PAIR）→ 多轮子查询序列（Crescendo）→ 视觉链推理（VisualChain）→ 本文 MMRT 的多模态多轮 MCTS 搜索。
- **内容审核模型**：从单模态 API（OpenAI Moderation、Perspective）→ 开源模型（WildGuard、LlamaGuard、LLaVaGuard、ShieldVLM）→ 多图多轮扩展（LlamaGuard-4）→ 本文 LLaVAShield 专注多模态多轮场景。
- **安全对齐**：揭示了即使经过安全对齐的主流 VLM 在多模态多轮场景下仍高度脆弱，与 Red Queen、IDEATOR 等工作的发现一致。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统性地研究多模态多轮对话内容审核，MMRT 的 MCTS 攻击框架和角色解耦推理链设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 主实验+外部基准+策略自适应+消融+红队脆弱性分析+组件贡献分析，非常全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，问题定义和三大风险特征的归纳到位
- 价值: ⭐⭐⭐⭐ — 填补了多模态多轮对话安全审核的空白，数据集和方法具有实用价值

<!-- RELATED:START -->

## 相关论文

- [HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [Diagnosing and Repairing Unsafe Channels in Vision-Language Models via Causal Discovery and Dual-Modal Safety Subspace Projection](diagnosing_and_repairing_unsafe_channels_in_vision-language_models_via_causal_di.md)
- [GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [From Masks to Pixels and Meaning: A New Taxonomy, Benchmark, and Metrics for VLM Image Tampering](from_masks_to_pixels_and_meaning_a_new_taxonomy_benchmark_and_metrics_for_vlm_im.md)
- [GraphVLM: Benchmarking Vision Language Models for Multimodal Graph Learning](graphvlm_benchmarking_vision_language_models_for_multimodal_graph_learning.md)

<!-- RELATED:END -->
