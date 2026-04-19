---
title: >-
  [论文解读] Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering
description: >-
  [NeurIPS 2025][KB-VQA] 提出 Wiki-PRF，一套三阶段（处理-检索-过滤）的多模态 RAG 框架，通过强化学习训练 VLM 自主调用视觉工具和过滤检索结果，在 E-VQA 和 InfoSeek 上达到 SOTA。
tags:
  - NeurIPS 2025
  - KB-VQA
  - RAG
  - 多模态检索
  - 强化学习
  - 工具调用
---

# Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering

**会议**: NeurIPS 2025  
**arXiv**: [2510.14605](https://arxiv.org/abs/2510.14605)  
**代码**: [GitHub](https://github.com/cqu-student/Wiki-PRF) (有)  
**领域**: 强化学习 / 视觉问答 / 检索增强生成  
**关键词**: KB-VQA, RAG, 多模态检索, 强化学习, 工具调用

## 一句话总结

提出 Wiki-PRF，一套三阶段（处理-检索-过滤）的多模态 RAG 框架，通过强化学习训练 VLM 自主调用视觉工具和过滤检索结果，在 E-VQA 和 InfoSeek 上达到 SOTA。

## 研究背景与动机

**领域现状**：知识密集型视觉问答（KB-VQA）要求模型同时理解图像内容并检索外部知识。检索增强生成（RAG）方法已展示了在该任务上的显著进步。

**现有痛点**：
   - **检索粒度不足**：现有方法通常使用全图视觉特征做检索，在复杂场景中（如钟楼旁的小雕像），全局特征被主体对象主导，检索到大量无关信息
   - **过滤精度低**：段落级别的重排序无法过滤段落内部的无关内容，检索结果含大量噪声

**核心矛盾**：需要细粒度的信息提取以精准检索，同时又要过滤海量检索结果中的噪声。

**本文目标**：设计更精准的多模态检索和更有效的结果过滤机制。

**切入角度**：让 VLM 自主决定使用哪些视觉工具（captioning、grounding、flipping）来处理输入，并用 RL 训练 VLM 学会过滤检索结果中的无关信息。

**核心 idea**：训练一个 VLM-PRF 模型，用答案准确率和格式一致性作为奖励信号，通过 GRPO 让模型学会灵活调用工具+高效过滤信息——这是 RL 首次应用于多模态 RAG。

## 方法详解

### 整体框架

Wiki-PRF 包含三个核心阶段：

1. **处理阶段 (Processing)**：VLM-PRF 根据图像和问题自主选择并调用工具（captioning/grounding/flipping），生成高质量的多模态检索查询
2. **检索阶段 (Retrieval)**：基于视觉特征和文本描述进行多模态知识库检索
3. **过滤阶段 (Filtering)**：VLM-PRF 对检索结果做相关性过滤和信息浓缩，输出面向任务的精简知识

### 关键设计

1. **工具调用机制**：

    - **Captioning 工具**：VLM-PRF 输出初始描述 $C_{init}$，再由 VLM-base 细化为检索查询 $C_{query} = \text{VLM}_{\text{captioning}}(C_{init}, Q)$
    - **Grounding 工具**：VLM-PRF 指定目标对象，VLM-base 定位并裁剪图像区域 $I_{\text{grounding}} = \text{Crop}(I, \text{VLM}_{\text{grounding}}(\text{object}))$
    - **Flipping 工具**：对图像做左右翻转以缓解角度变化对检索的影响
    - VLM-PRF 在 `<think>` 标签中推理使用哪些工具及顺序，在 `<tool>` 标签中输出工具调用

2. **多模态检索**：使用 EVA-CLIP 8B 编码查询，通过 Faiss-GPU 做余弦相似度检索。检索结果拆分为段落级 sections，按与问题的相似度排序选取 Top-$k_s$。

3. **过滤阶段**：VLM-PRF 接收直接检索信息 $D$ 和工具检索结果 $\mathcal{S}_{\text{search}}$，在 `<think>` 中推理，在 `<answer>` 中输出过滤后的面向任务知识 $F$：
    $F = \text{VLM-PRF}(D, \mathcal{S}_{\text{search}}), \quad A = \text{VLM}(F, Q)$

4. **强化学习训练**：使用 GRPO（去除 KL 约束）训练 VLM-PRF，奖励函数：
    $r_\phi(x,y) = \alpha \cdot EM(a_{\text{pred}}, a_{\text{gt}}) + \beta \cdot M(a_{\text{tool}}, t_{\text{tool}}) + \gamma \cdot M(a_{\text{filter}}, t_{\text{filter}})$
    - 答案奖励 ($\alpha=1$)：精确匹配评分
    - 工具格式奖励 ($\beta=0.3$)：正则表达式验证格式合规性
    - 过滤格式奖励 ($\gamma=0.7$)：验证过滤输出格式
    - 使用 LoRA（rank=64, alpha=128）仅训练少量附加参数

### 训练策略

- 基础模型：Qwen2.5-VL-3B/7B
- 训练参数：8 个 generation 采样，温度 0.7，2 个 epoch，lr=1e-5
- 训练时长：约 15 小时（8 × A800 GPU）
- 检索设置：Top-1 图像直接检索 + Top-5 工具检索文章

## 实验关键数据

### 主实验

| 方法 | 模型 | E-VQA Single-Hop | E-VQA All | InfoSeek UQ | InfoSeek UE | InfoSeek All |
|------|------|------------------|-----------|-------------|-------------|-------------|
| GPT-4V | - | 26.9 | 28.1 | 15.0 | 14.3 | 14.6 |
| EchoSight | Mistral-7B | 19.4 | - | - | - | 27.7 |
| ReflectiVA | LLaMA-3.1-8B | 28.0 | 29.2 | 40.4 | 39.8 | 40.1 |
| MMKB-RAG | Qwen2-7B | 39.7 | 35.9 | 36.4 | 36.3 | 36.4 |
| **Wiki-PRF-7B** | **Qwen2.5-VL-7B** | **37.1** | **36.0** | **43.3** | **42.7** | **42.8** |
| **Wiki-PRF (InternVL3)** | **InternVL3-8B** | **40.1** | **39.2** | **43.5** | **42.1** | **42.5** |

Wiki-PRF-7B 在 InfoSeek 上达到 42.8 的新 SOTA，在 E-VQA 上达到 36.0。

### 检索召回率

| 模型 | 检索输入 | Recall |
|------|---------|--------|
| None | images | 45.56 |
| Qwen2.5-VL-7B | images + tools | 53.44 |
| **VLM-PRF-7B** | **images + tools** | **54.89** |

工具调用显著提升检索召回率（+9.3%），RL 训练进一步提升。

### 消融实验

**模块消融**（InfoSeek 10K 样本）：

| 配置 | VQA Accuracy |
|------|-------------|
| Baseline（无 RAG） | 34.22 |
| + Processing | 36.24 |
| + Processing + Filtering | 39.48 |

**RL vs SFT 对比**（InfoSeek 2K 样本）：

| 训练方式 | UQ | UE | All |
|---------|-----|-----|-----|
| Base (无微调) | 39.1 | 40.5 | 40.2 |
| SFT | 41.5 | 41.9 | 41.8 |
| **RL** | **46.6** | **46.2** | **46.3** |

RL 显著优于 SFT（+4.5%），因为 SFT 倾向于模仿表面模式，而 RL 让模型理解信息过滤的底层原则。

### Oracle 设置（给定正确文章）

| 方法 | VQA Accuracy |
|------|-------------|
| Wiki-LLaVA | 51.5 |
| ReflectiVA | 57.6 |
| **Wiki-PRF-7B** | **65.8** |

即使给定正确文章，Wiki-PRF 的过滤能力也使其定位关键信息的效率远超其他方法。

### 知识库规模影响

| 方法 | 10K | 50K | 100K |
|------|-----|-----|------|
| Vanilla-MRAG (7B) | 56.3 | 39.6 | 23.7 |
| **Wiki-PRF-7B** | **60.3** | **51.2** | **42.8** |

Wiki-PRF 在大规模知识库下的性能衰减明显更慢。

### 关键发现

- RL 训练后模型的工具调用组合数从 34 增加到 40-53 种，说明 RL 增强了工具使用的灵活性和多样性
- Captioning 工具被调用最频繁，是提升文章召回率的最有效工具
- OK-VQA 上达到 77.8 的新 SOTA，确认方法的跨数据集泛化能力
- 过滤阶段的贡献几乎与处理阶段相当（各约 +2-3%）

## 亮点与洞察

- **首次将 RL 应用于多模态 RAG**：用答案准确率作为奖励信号就足以让模型学会工具选择和信息过滤策略，无需标注中间步骤
- **VLM-PRF 双角色设计**：同一个 RL 训练的模型既做处理（工具规划）又做过滤（信息精炼），体现了"少量参数多功能"的效率
- **工具调用的灵活性**：VLM-PRF 自主推理使用哪些工具、以什么顺序调用，比硬编码管线更灵活
- **RL > SFT 的清晰实证**：RL 在缺乏中间步骤标注的情况下显著优于 SFT，验证了 RL 在 RAG 任务中的独特价值

## 局限与展望

- 工具集目前较小（captioning/grounding/flipping 三种），扩展更多工具（如 OCR、图表解析）可能进一步提升
- 检索器（EVA-CLIP 8B）是冻结的，联合训练检索器和 VLM-PRF 可能带来额外收益
- 知识库规模增大时性能仍有明显下降（100K 时约 43%），说明检索质量仍是瓶颈
- RL 训练需要 8 × A800 约 15 小时，对于资源有限的场景仍有成本
- Flipping 工具的用途不够通用（仅处理左右翻转），效果可能有限
- 缺少与最新的通用 RAG 方法（如 Self-RAG）的对比

## 相关工作与启发

- EchoSight 用视觉信息做文章级检索后文本重排序——Wiki-PRF 进一步引入工具处理和段落级过滤
- ReflectiVA 引入反思机制——Wiki-PRF 用 RL 替代手动设计的反思流程
- RL 训练 VLM 的方向（R1-OneVision、VisualThinker-R1-Zero）正在兴起——Wiki-PRF 将其扩展到 RAG 场景
- 工具增强的 LLM（如 Toolformer）的思路被成功迁移到多模态检索

## 评分

- 新颖性: ⭐⭐⭐⭐ 三阶段PRF框架和RL训练RAG的结合新颖，工具调用机制设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集验证，消融详细，RL vs SFT对比有说服力
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细，部分公式编号和符号可以更统一
- 价值: ⭐⭐⭐⭐ 解决了KB-VQA中检索和过滤的核心痛点，SOTA结果有说服力

<!-- RELATED:START -->

## 相关论文

- [ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering](../../CVPR2026/reinforcement_learning/reag_reasoning-augmented_generation_for_knowledge-based_visual_question_answerin.md)
- [Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](../../CVPR2026/reinforcement_learning/bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)
- [Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning](open_vision_reasoner_transferring_linguistic_cognitive_behavior_for_visual_reaso.md)
- [Continual Knowledge Adaptation for Reinforcement Learning](continual_knowledge_adaptation_for_reinforcement_learning.md)

<!-- RELATED:END -->
