---
title: >-
  [论文解读] Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning
description: >-
  [AAAI2026][音频/语音][Emotion Recognition in Conversation] 提出 PRC-Emo 框架，通过显式/隐式情感提示、专用检索库和课程学习策略三位一体地提升 LLM 在对话情感识别（ERC）任务上的表现，在 IEMOCAP 和 MELD 两个基准上取得 SOTA。
tags:
  - "AAAI2026"
  - "音频/语音"
  - "Emotion Recognition in Conversation"
  - "提示学习"
  - "Retrieval-Augmented Generation"
  - "Curriculum Learning"
  - "LLM Fine-tuning"
---

# Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning

**会议**: AAAI2026  
**arXiv**: [2511.07061](https://arxiv.org/abs/2511.07061)  
**代码**: [LiXinran6/PRC-Emo](https://github.com/LiXinran6/PRC-Emo)  
**领域**: 音频语音  
**关键词**: Emotion Recognition in Conversation, Prompt Engineering, Retrieval-Augmented Generation, Curriculum Learning, LLM Fine-tuning

## 一句话总结

提出 PRC-Emo 框架，通过显式/隐式情感提示、专用检索库和课程学习策略三位一体地提升 LLM 在对话情感识别（ERC）任务上的表现，在 IEMOCAP 和 MELD 两个基准上取得 SOTA。

## 背景与动机

对话情感识别（Emotion Recognition in Conversation, ERC）是实现自然人机交互的关键任务。随着 LLM 的兴起，基于生成式架构的 ERC 方法（如 InstructERC、BiosERC）已经大幅超越传统判别式模型（RNN、GNN、PLM 系列）。然而现有 LLM-ERC 方法仍存在三个核心不足：

1. **提示设计忽视隐式情感**：大多数工作仅基于文字表面含义预测情感，未区分说话者"表达出来的情感"（显式）与"内心真正感受的情感"（隐式）。例如一个人可能口头乐观，但内心其实感到焦虑。
2. **示范检索库质量有限**：现有方法的 demonstration 库仅使用训练集，覆盖场景单一，泛化能力受限。
3. **训练策略粗糙**：缺乏对样本难度的系统建模，对类别不均衡和复杂情感转换的处理不足。

## 核心问题

如何设计一个统一的训练框架，让 LLM 同时理解对话中的显式和隐式情感，并通过高质量检索和渐进式训练策略提升在类别不均衡场景下的情感识别准确率？

## 方法详解

### 整体架构：PRC-Emo

PRC-Emo 包含三个核心模块：**P**rompt（提示工程）、**R**etrieval（示范检索）、**C**urriculum learning（课程学习），整体分为两个阶段——外部知识提取和情感标签预测。

### 1. 外部补充知识提取（Prompt Engineering）

针对每条对话，利用外部 LLM（Qwen3-14B）生成三类辅助知识：

- **显式情感解读（Explicit Emotion Interpretation）**：基于历史对话，分析说话者通过语言直接表达出来的情感。
- **隐式情感解读（Implicit Emotion Interpretation）**：推断说话者可能真正感受到但未直接表达的内心情感状态。
- **说话者特征（Speaker Characteristic）**：基于完整对话归纳说话者的性格和行为模式。

这些知识以结构化文本的形式注入到后续的情感预测 prompt 中，引导模型关注多层次的情感表达。

### 2. 检索模板模块（Retrieval Template Module）

最终的情感预测 prompt 由五部分组成：

- **Instruction**：定义模型角色和任务目标。
- **Historical Content**：历史窗口 $w$ 内的对话上下文及说话者信息。
- **External Knowledge**：上一步提取的显式/隐式情感解读和说话者特征。
- **Demonstration Retrieval**：从检索库中用 SBERT 余弦相似度检索 top-3 最相似的示范对。
- **Label Statement**：约束输出为预定义情感标签集合。

**ERC 专用示范检索库**：首次为 ERC 构建专用检索库，包含 36,712 条语句：

| 来源 | 数量 |
|------|------|
| 自建数据集（GPT-4o 生成 + 人工验证） | 14,009 |
| IEMOCAP 训练集 | 5,163 |
| MELD 训练集 | 9,989 |
| EmoryNLP 训练集 | 7,551 |

自建数据集覆盖医疗、职场、教育、家庭、社交、娱乐六大场景，采用"标签遮蔽 + 双人独立标注"策略保证质量，生成-过滤迭代三轮以实现类别均衡。

### 3. 课程学习（Curriculum Learning）

**对话难度度量**：提出基于加权情感转移频率的难度函数，同时考虑同一说话者和不同说话者之间的情感变化（此前方法仅考虑同一说话者）。

- 情感相似度基于二维 arousal-valence 情感轮计算：$s_{ij} = \max(\cos(\theta_{ij}), 0)$
- 加权情感转移：$N^{WES} = k \times s_{ij} + b$
- 对话难度：$DIF(c_i) = \frac{WES_{same}(c_i) + WES_{diff}(c_i) + N_{sp}(c_i)}{N_u(c_i) + N_{sp}(c_i)}$

其中 $WES_{same}$ 和 $WES_{diff}$ 分别为同说话者和跨说话者的加权情感转移总和，$N_{sp}$ 为说话者数量（平滑因子），$N_u$ 为话语总数。

**训练调度器**：按难度将数据划分为 $n$ 个桶（easy → hard），训练前期仅用简单桶，逐步引入更难的桶，最终在全部数据上继续训练。

### 微调方式

基于 LoRA 高效微调。IEMOCAP 使用 Qwen2.5-7B-Instruct，MELD 使用 Qwen3-8B。所有实验在单张 NVIDIA 4090D GPU 上完成。

## 实验关键数据

### 主实验（Weighted F1）

| 方法 | IEMOCAP | MELD |
|------|---------|------|
| InstructERC | 71.39 | 69.15 |
| BiosERC | 71.19 | 69.83 |
| **PRC-Emo（本文）** | **71.95** | **70.44** |

相比最佳基线 InstructERC / BiosERC，IEMOCAP 提升 +0.76%，MELD 提升 +0.61%。

### 消融实验

| 配置 | IEMOCAP | MELD |
|------|---------|------|
| PRC-Emo（完整） | 71.95 | 70.44 |
| w/o Curriculum | 71.52（↓0.43） | 70.07（↓0.37） |
| w/o Retrieval + Curriculum | 70.74（↓1.21） | 69.62（↓0.82） |
| w/o Prompt + Retrieval + Curriculum | 68.54（↓3.41） | 68.72（↓1.72） |

Prompt 模块贡献最大（去掉后 IEMOCAP 下降 2.20），说明显式/隐式情感解读对模型理解情感状态至关重要。

### Prompt 设计消融

去除隐式情感解读（w/o I + R）后 IEMOCAP 下降 1.68，验证了隐式情感建模的重要性。

## 亮点

1. **显式 + 隐式双通道情感解读**：首次在 ERC 中系统区分并建模显式与隐式情感，从"表面表达"和"内心感受"两个层面理解说话者状态，思路清晰且有实际意义。
2. **首个 ERC 专用示范检索库**：融合多数据集 + LLM 生成 + 人工验证的 36K 语句库，跨六大生活场景，显著提升 few-shot 检索泛化能力。
3. **跨说话者情感转移建模**：在课程学习难度设计中同时考虑同说话者和不同说话者间的情感变化，比现有方法更全面。
4. **工程友好**：单卡 4090D 即可训练，代码已开源。

## 局限与展望

1. **仅使用文本模态**：尽管引入了丰富的文本辅助知识，但 ERC 本质上是多模态任务（语音语调、面部表情），未利用声学/视觉信息是明显短板。
2. **外部 LLM 依赖**：情感解读的生成依赖 Qwen3-14B，推理成本较高，且解读质量受限于外部模型能力。
3. **数据集规模有限**：仅在 IEMOCAP（双人对话）和 MELD（多人对话）两个英文数据集上验证，对中文/其他语言以及更大规模数据集的适用性待考察。
4. **检索库人工成本**：双人标注 + 三轮迭代过滤虽保证了质量，但扩展到新领域或新语言时成本较高。
5. **课程学习超参敏感**：桶数量 $n$、线性变换参数 $k, b$ 等需要针对数据集调整，缺乏自适应机制。

## 与相关工作的对比

- **vs InstructERC**：InstructERC 首次将生成式架构引入 ERC，但其检索库仅用训练集，且未区分显式/隐式情感。PRC-Emo 在提示设计和检索库质量上均有实质提升。
- **vs BiosERC**：BiosERC 引入说话者背景信息丰富 prompt，PRC-Emo 在此基础上进一步加入显式/隐式情感解读和课程学习。
- **vs HybridCL / LSDGNN**：这些方法在课程学习难度设计上仅考虑同说话者情感转移，PRC-Emo 扩展到跨说话者维度，更全面地衡量对话复杂度。

## 启发与关联

- **检索增强 + 课程学习**的组合思路可迁移到其他对话理解任务（意图识别、对话摘要等）。
- **显式/隐式双通道**的理念类似于"表层语义 vs 深层语义"的分离，可推广到讽刺检测、立场检测等需要理解言外之意的任务。
- 检索库的构建方法（LLM 生成 + 人工验证 + 多轮迭代）为低资源场景下的数据增强提供了可复用范式。

## 评分

- 新颖性: 3.5/5（各模块单独看不算全新，但组合方式和显式/隐式情感的系统建模有新意）
- 实验充分度: 4/5（消融细致，但仅两个数据集，缺少跨语言/跨场景验证）
- 写作质量: 4/5（结构清晰，图表丰富，动机阐述充分）
- 价值: 3.5/5（ERC 领域扎实的工程改进，检索库有复用价值，但提升幅度较小）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition](let_the_model_learn_to_feel_mode-guided_tonality_injection_f.md)
- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)
- [\[ICML 2025\] Teaching Physical Awareness to LLMs through Sounds](../../ICML2025/audio_speech/teaching_physical_awareness_to_llms_through_sounds.md)
- [\[ICLR 2026\] Dynamic Parameter Memory: Temporary LoRA-Enhanced LLM for Long-Sequence Emotion Recognition in Conversation](../../ICLR2026/audio_speech/dynamic_parameter_memory_temporary_lora-enhanced_llm_for_long-sequence_emotion_r.md)
- [\[ICML 2026\] Do Audio LLMs Listen or Read? Analyzing and Mitigating Paralinguistic Failures with VoxParadox](../../ICML2026/audio_speech/do_audio_llms_listen_or_read_analyzing_and_mitigating_paralinguistic_failures_wi.md)

</div>

<!-- RELATED:END -->
