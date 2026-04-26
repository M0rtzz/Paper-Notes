---
title: >-
  [论文解读] Taming Knowledge Conflicts in Language Models
description: >-
  [ICML 2025][LLM/NLP][知识冲突] 揭示了语言模型注意力头中"上下文信息与参数记忆的叠加"（CP Superposition）现象，提出 JuICE（Just Run Twice）方法，通过双次推理的注意力干预策略，在不微调的前提下灵活引导模型偏向参数知识或上下文知识，在 11 个数据集 × 6 种模型架构上达到 SOTA。
tags:
  - ICML 2025
  - LLM/NLP
  - 知识冲突
  - 注意力干预
  - 参数记忆
  - 上下文依赖
  - 推理时干预
---

# Taming Knowledge Conflicts in Language Models

**会议**: ICML 2025  
**arXiv**: [2503.10996](https://arxiv.org/abs/2503.10996)  
**代码**: [GaotangLi/JUICE](https://github.com/GaotangLi/JUICE)  
**领域**: LLM/NLP  
**关键词**: 知识冲突, 注意力干预, 参数记忆, 上下文依赖, 推理时干预

## 一句话总结

揭示了语言模型注意力头中"上下文信息与参数记忆的叠加"（CP Superposition）现象，提出 JuICE（Just Run Twice）方法，通过双次推理的注意力干预策略，在不微调的前提下灵活引导模型偏向参数知识或上下文知识，在 11 个数据集 × 6 种模型架构上达到 SOTA。

## 研究背景与动机

语言模型在预训练阶段将大量知识编码为**参数记忆**（parametric memory），推理时则同时利用参数记忆和**上下文知识**（contextual knowledge）生成输出。当两者发生矛盾时就产生了**知识冲突**（knowledge conflict），这在 RAG、LLM Agent、工具增强 LLM 等场景中极为常见。

现有工作存在三个核心局限：

**单一冲突类型**：之前的方法（如 PH3）主要针对句子级替换冲突（substitution conflict），在更难的段落级连贯冲突（coherent conflict）下效果大幅下降

**排他性假设有误**：先前研究假设存在排他的"记忆头"（memory head）和"上下文头"（context head），但本文发现同一注意力头可以**同时**贡献参数知识和上下文知识

**单向视角**：多数工作只关注增强上下文依赖（解决 RAG 幻觉），缺少统一的双向调控方法

## 方法详解

### 整体框架

JuICE 包含两个阶段：

1. **Head Identification（头识别阶段）**：使用极少量样本（仅 4 个）识别出两组注意力头——分别在正向/负向缩放时能**一致地**产生目标效果
2. **Dual-run Inference（双次推理阶段）**：模型运行两次——第一次保存已识别头的输出激活；第二次将这些激活的缩放版本注入对应模块

### 关键设计

#### 1. CP Superposition 的发现

本文核心发现是 **"上下文信息与参数记忆的叠加"（CP Superposition）**：高影响力的注意力头会**同时**编码参数记忆和上下文信息，其角色取决于接收到的残差流输入。

**观察 1：模型组件在不同冲突程度下行为不一致**

- 无冲突时：移除几乎所有组件导致参数答案概率**下降**
- 替换冲突时：移除不同组件分别导致参数答案概率上升或下降
- 连贯冲突时：移除几乎所有组件导致参数答案概率**上升**

在 Gemma-2b 的 26 层中，能在所有三种冲突类型下一致提升参数知识的注意力模块仅有 6 个，整层为 0 个。

更关键的是，在替换冲突下排名前 4 的"记忆头"中，有**一半**在连贯冲突下变成了"上下文头"（效果完全反转），例如 head (9,3) 在替换冲突下上下文概率 +0.13，但在连贯冲突下变为 -0.17。

**观察 2：多重干预的抵消效应**

直觉上，将多个单独有效的干预叠加应产生更大效果，但实验表明：

- Top-1 干预：目标概率 0.03 → 0.12 ✓
- Top-3 干预：0.03 → 0.24 ✓
- Top-10 干预：0.03 → 0.14 ✗（性能反而下降）

这是因为修改前层激活会改变下游组件接收到的残差流，导致其功能发生变化，产生抵消效应。

#### 2. Head Identification 策略

JuICE 的头识别阶段选取**跨多种冲突类型均一致有效**的注意力头：

- 对每个注意力头，在小规模选择集（|D|=4）上计算个体缩放后目标概率的期望变化
- **一致性约束**：仅保留在所有冲突类型下得分均非负的头
- 按聚合得分取 Top-K（默认 K=5）
- 支持跨域泛化：仅在 World Capital 域上识别的头可迁移到其他无关域

#### 3. Dual-run Inference 机制

这是 JuICE 区别于之前方法的核心设计：

- **第一次运行**：正常推理，保存已识别头的输出激活 $h_i^{(1)}$
- **第二次运行**：将缩放后的第一次激活 $\alpha \cdot h_i^{(1)}$ 加到对应模块的激活上

直觉是：**第一次运行的激活是更可靠的引导方向**，因为它们来自未被干预的原始模型状态。单次运行干预不稳定、易退化，而双次运行能有效缓解叠加效应导致的抵消问题。

#### 4. 消融变体：JuNe（Just Run Once）

JuNe 是 JuICE 去掉双次运行设计后的单次干预版本，用于验证 dual-run 的必要性。实验显示 JuICE 在 Gemma 上比 JuNe 平均高约 20%。

### 损失函数 / 训练策略

JuICE 是**推理时干预方法**（test-time intervention），不涉及微调或额外训练。核心超参数：

- **|D|**：头选择集大小，默认仅 4 个样本
- **K**：干预的头数量，默认 5
- **α**：缩放因子，通过验证集确定

理论分析部分，作者将知识冲突形式化为两层 Transformer 上的双任务学习：
- **事实回忆任务**（对应参数知识）：学习主体-答案映射 $\mathcal{G}^*: \mathcal{S} \to \mathcal{A}$
- **归纳任务**（对应上下文知识）：预测触发词后出现的 token

通过梯度下降训练的权重矩阵自然形成叠加结构（Proposition 5.3），标准交叉熵损失本身就**鼓励**叠加的产生，而模型输出偏好取决于系数 $C_1,...,C_4$ 的相对大小。

## 实验关键数据

### 主实验——增强参数记忆

| 数据集 | 冲突类型 | Original | Prompt | PH3_l | JuICE | 提升(vs PH3_l) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Athlete Sport (Gemma) | Coherent | 0.0 | 0.0 | 33.3 | **91.9** | +58.6 |
| World Capital (Gemma) | Coherent | 1.1 | 35.7 | 88.1 | **93.0** | +4.9 |
| Company HQ (Gemma) | Coherent | 0.0 | 0.0 | 30.6 | **59.3** | +28.7 |
| Average (Gemma) | 全部3类 | 0.2 / 11.3 / 78.1 | 10.5 / 29.6 / 78.1 | 42.4 / 41.1 / 61.2 | **73.4 / 75.3 / 79.1** | — |
| Average (Llama2) | 全部3类 | 0.2 / 25.9 / 82.5 | 19.8 / 55.0 / 82.5 | 54.5 / 82.0 / 80.6 | **83.0 / 82.5 / 82.2** | — |
| Average (Llama3) | 全部3类 | 0.4 / 11.0 / 78.7 | 3.7 / 70.1 / 78.7 | 39.2 / 78.1 / 80.7 | **84.7 / 84.2 / 84.5** | — |

### 主实验——增强上下文依赖

| 模型 | 方法 | NQ-Swap | Hate Speech | History QA | Proverb End | Proverb Trans | 平均 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Gemma | Original | 38.7 | 70.7 | 29.9 | 26.5 | 59.0 | 45.0 |
| Gemma | CAD | 56.9 | 81.7 | 16.9 | 37.1 | 62.9 | 51.1 |
| Gemma | JuICE | **58.4** | **84.1** | **47.0** | **74.6** | **66.8** | **66.2** |
| Llama2 | PH3_l | 48.2 | 63.4 | 20.4 | 68.7 | 58.8 | 51.9 |
| Llama2 | JuICE | **49.5** | **93.9** | **50.2** | **77.1** | **62.6** | **66.6** |
| Llama3 | PH3_l | 25.3 | 62.2 | 78.4 | 48.5 | 63.6 | 55.6 |
| Llama3 | JuICE | **35.3** | **78.4** | **74.2** | **75.4** | **70.7** | **66.8** |

### 消融实验

| 配置 | Gemma 平均准确率 (Coherent) | 说明 |
|:---|:---:|:---|
| Original | 0.2% | 模型几乎完全服从上下文 |
| JuNe (单次运行) | 52.7% | 有效但不稳定 |
| JuICE (双次运行) | **73.4%** | 双次运行带来 +20.7% 提升 |
| PH3_l (200样本) | 42.4% | 需要 50× 更多样本 |
| PH3_s (4样本) | 0.1% | 同等样本下几乎无效 |

### 关键发现

1. **JuICE 几乎完全逆转**模型在最困难连贯冲突下的上下文倾向：Gemma 从 0.2% → 73.4%，Llama2 从 0.2% → 83.0%
2. **极低数据需求**：仅需 4 个样本即可完成头识别，比 PH3 (200样本) 少 50 倍
3. **跨域泛化**：在 World Capital 上识别的头能有效迁移到 Athlete Sport、Company Founder 等无关领域
4. **双向调控**：同一框架既能增强参数记忆也能增强上下文依赖
5. **鲁棒性强**：对头选择集大小、干预头数量、缩放因子、输入改写均保持稳定

## 亮点与洞察

- **CP Superposition 是根本性发现**：打破了此前"排他性记忆头/上下文头"的假设，证明注意力头的功能是**输入依赖**的，同一头在不同冲突条件下可能角色互换
- **"运行两次"的设计极其简洁有效**：不需要修改架构、不需要微调、不需要对比解码，仅通过保存并重放注意力激活就能实现精准调控
- **理论与实验高度统一**：用双任务两层 Transformer 理论框架完美解释了叠加形成、知识冲突和 JuICE 有效性三个核心问题
- **实用价值高**：对 RAG 系统可灵活选择"信上下文"或"信自己"，4 样本即可部署

## 局限与展望

1. **计算开销翻倍**：双次推理意味着推理成本约为原来的 2 倍，对延迟敏感的场景可能受限
2. **仅针对 base model 测试**：未验证在 instruction-tuned 或 chat 模型上的效果
3. **缩放因子依赖验证集**：虽然头识别仅需 4 样本，但最优缩放因子仍需验证集调参
4. **理论分析基于简化模型**：两层线性注意力 + 正交嵌入假设与实际深层非线性模型有差距
5. **未考虑多跳推理冲突**：所有数据集都是单步事实回忆，复杂推理链中的冲突传播未涉及

## 相关工作与启发

- **PH3 (Jin et al., 2024)**：首个系统研究注意力头与知识冲突关系的工作，提出"记忆头/上下文头"概念，但排他性假设在连贯冲突下失效
- **CAD (Shi et al., 2024)**：基于对比解码增强上下文依赖，但在参数记忆增强方面无能力
- **Toy Models of Superposition (Elhage et al., 2022)**：在特征维度发现叠加现象，本文将叠加概念推广到知识维度（参数 vs 上下文）
- **启发：** 可探索将 JuICE 与 LoRA 结合，用少量训练识别更稳定的干预方向；也可将双次运行推广到 MLP 层干预

## 评分

| 维度 | 分数 (1-5) | 说明 |
|:---|:---:|:---|
| 新颖性 | ⭐⭐⭐⭐⭐ | CP Superposition 是全新发现，打破排他性假设 |
| 理论深度 | ⭐⭐⭐⭐ | 完整的理论框架但基于简化模型 |
| 实验充分度 | ⭐⭐⭐⭐⭐ | 11 数据集 × 6 模型 × 3 冲突类型，覆盖全面 |
| 实用性 | ⭐⭐⭐⭐ | 开箱即用但推理开销翻倍 |
| 写作质量 | ⭐⭐⭐⭐⭐ | 结构清晰，理论实验衔接紧密 |
| **综合** | **⭐⭐⭐⭐½** | 高质量工作，对理解和调控 LM 知识冲突有重要贡献 |

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models](../../ACL2025/llm_nlp/astute_rag_knowledge_conflicts.md)
- [\[ACL 2025\] KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan](../../ACL2025/llm_nlp/kazmmlu_evaluating_language_models_on_kazakh_russian_and_regional_knowledge_of_k.md)
- [\[ACL 2025\] Knowledge Boundary of Large Language Models: A Survey](../../ACL2025/llm_nlp/knowledge_boundary_survey.md)
- [\[ACL 2025\] Acquisition and Application of Novel Knowledge in Large Language Models](../../ACL2025/llm_nlp/acquisition_and_application_of_novel_knowledge_in_large_language_models.md)
- [\[NeurIPS 2025\] The Rise of Parameter Specialization for Knowledge Storage in Large Language Models](../../NeurIPS2025/llm_nlp/the_rise_of_parameter_specialization_for_knowledge_storage_in_large_language_mod.md)

<!-- RELATED:END -->
