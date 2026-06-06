---
title: >-
  [论文解读] Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length
description: >-
  [ICML 2025][机器人][前摄干扰] 借鉴认知科学中的前摄干扰（Proactive Interference）范式，发现LLM的信息检索准确率随干扰信息量呈对数线性下降至零，揭示了一种独立于上下文长度的"工作记忆"容量瓶颈，且提示工程无法有效缓解。
tags:
  - "ICML 2025"
  - "机器人"
  - "前摄干扰"
  - "工作记忆"
  - "信息检索"
  - "上下文长度"
  - "LLM评测"
---

# Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length

**会议**: ICML 2025  
**arXiv**: [2506.08184](https://arxiv.org/abs/2506.08184)  
**代码**: [有](https://github.com/zhuangziGiantfish/Unable-to-Forget)  
**领域**: 机器人  
**关键词**: 前摄干扰, 工作记忆, 信息检索, 上下文长度, LLM评测

## 一句话总结

借鉴认知科学中的前摄干扰（Proactive Interference）范式，发现LLM的信息检索准确率随干扰信息量呈对数线性下降至零，揭示了一种独立于上下文长度的"工作记忆"容量瓶颈，且提示工程无法有效缓解。

## 研究背景与动机

LLM的信息检索评测中，输入长度通常被当作任务难度的主要衡量指标。现有长上下文基准（如Needle-in-a-Haystack、MRCR等）主要通过增加prompt长度来提高难度。然而，这些研究将**搜索难度**（在大量上下文中定位目标）和**干扰**（在相似但错误的条目中正确识别目标）混为一谈。

**核心矛盾**：当前研究隐式地将"区分相似信息的困难"归因于输入长度，忽视了干扰作为独立因素的影响。实际上，语义相似的干扰信息在很多数据处理任务中极为常见——例如血压连续记录中追踪最新读数。

**认知科学启发**：前摄干扰（PI）是经典的认知心理学范式——先前学到的信息阻碍对新信息的回忆。在人类中，PI抵抗能力与工作记忆容量负相关。人类表现出"平台效应"——超过一定阈值后，额外干扰不再产生显著影响，这归因于人类能主动"解绑"过时的关联。

**本文切入**：将PI范式适配到LLM测试中，通过固定检索目标位置（始终为最后一次更新）来最小化搜索难度，纯粹隔离和量化干扰效应。

## 方法详解

### 整体框架

设计了**PI-LLM评测**，核心是合成键值对检索任务：
- **输入**：一系列键值对更新流，固定若干键，每个键经历多次值更新（随机交错排列）
- **查询**：要求模型返回每个键的最新值（即最后出现的值）
- **控制**：搜索难度天然很低（目标总是最后更新），错误主要归因于干扰

### 关键设计

#### 1. 基础干扰实验（Experiment 1）

**功能**：固定46个唯一键，系统性增加每个键的更新次数（3→400），测量检索准确率。

**核心发现**：所有测试模型的检索准确率随更新次数呈**对数线性下降**：

$$\text{Accuracy} \propto -\alpha \cdot \log(\text{update count}) + \beta$$

模型大小影响下降斜率——大模型（>150B）下降更慢，小模型快速降至接近零。

**错误分析三阶段**：
- **低干扰**：错误集中在正确值附近的位置（局部混淆）
- **中干扰**：错误分散到更早的更新值，少量幻觉出现
- **高干扰**：大量返回从未出现的值（幻觉），同时有强烈的"首因效应"偏向最早的几次更新

#### 2. 干扰独立于输入长度（Experiment 2）

**功能**：设计两个子实验来解耦干扰和输入长度：
- **Exp A**：固定每键更新次数（125或350），变化更新的键数量（1→46）
- **Exp B**：固定总输入长度（更新次数和键数都固定），仅变化需要追踪的键数

**核心发现**：两个实验呈现几乎相同的对数线性下降曲线——即使Exp B的**输入长度完全不变**，准确率仍随追踪键数增加而对数线性下降。这直接证明干扰是独立于输入长度的限制因素。

#### 3. 统一干扰容量瓶颈（Experiment 3）

**功能**：固定之前三个干扰源，变化每个值的长度（通过拼接多个词）。

**核心发现**：准确率同样呈对数线性下降，且斜率最陡——值长度从1增到10个词，所有模型准确率降至40%以下。这表明**所有形式的干扰共享同一个统一的容量限制**，类似于人类工作记忆的统一资源。

#### 4. 干扰耐受分数（IES）

引入**Interference Endurance Score**量化模型的抗干扰能力：

$$\text{IES} = \text{AUC}(\text{accuracy vs. log(update count)})$$

回归分析表明：**模型大小是IES的显著预测因子**（$p = 0.005$），而**上下文窗口长度无显著效果**（$p = 0.886$）。MoE架构表现不如同等总参数的dense模型（因为激活参数远少于标称总参数）。

### 损失函数 / 训练策略

本文为评测/分析工作，不涉及训练。实验覆盖了0.6B到637B参数的广泛模型，包括GPT、Claude、Gemini、Grok、DeepSeek、Qwen等主流模型。

## 实验关键数据

### 主实验（干扰缓解策略效果）

| 策略 | 效果 | 说明 |
|------|------|------|
| Per-key forget | 无效/负效果 | 错误反而聚集在forget指令的位置 |
| Forward focus | 边际改善 | <10百分点改善 |
| Relevance meta-prompt | 无效 | 模型能正确说出答案位置但仍检索错误 |
| Soft session reset | 无效 | 自然语言重置信号无法改变检索行为 |
| Mock QA reset（hack） | **有效** | 模拟对话轮次边界，大幅改善准确率 |

### 消融实验（模型大小 vs 上下文长度）

| 因素 | 对IES的影响 | $p$值 |
|------|-----------|-------|
| 模型大小类别 | 显著正相关 | 0.005 |
| 上下文窗口长度 | 无显著影响 | 0.886 |
| 128k-131k模型中大小vs IES | Spearman $\rho^2 = 0.673$ | 0.0016 |

### 关键发现

1. **普遍的对数线性衰减**：所有测试模型（从0.6B到637B）无一例外地展现干扰导致的准确率对数线性下降，包括GPT-4.1、Claude、Gemini 2.5等最新模型
2. **干扰独立于输入长度**：在固定输入长度条件下，增加追踪键数仍导致相同的下降模式
3. **统一容量限制**：更新次数、键数量、值长度三个正交维度都产生同样的对数线性下降，指向统一的抗干扰资源
4. **CoT不改善检索**：推理模型（如DeepSeek-R1）在干扰检索任务上表现不优于甚至差于基础模型——"知道答案在哪"不等于"能正确检索"
5. **Mock QA reset是唯一有效策略**：通过模拟对话边界"欺骗"模型弃置先前信息，但这是一种hack而非系统性解决方案
6. **顺序更新vs随机更新**：顺序更新模式下表现为阶梯式崩溃（在模型特定阈值处突然降至零），而随机交错下为渐进对数线性下降

## 亮点与洞察

- 极其精妙的实验设计——通过固定检索目标位置完美隔离了干扰效应，控制变量一流
- 认知科学与AI评测的跨学科融合：PI范式已有数十年的研究积累，移植到LLM评测中产生了深刻洞见
- 揭示了一个反直觉的结论：LLM的"工作记忆"不等于上下文窗口长度，真正的瓶颈在于抗干扰能力
- "知道但无法执行"的分离现象（top-down vs bottom-up）：模型能分析出正确策略但无法在检索中执行，这对CoT/推理模型的局限性有重要启示
- Per-key forget的失败模式极具启发性——forget指令反而成为新的干扰"锚点"，暗示注意力机制的根本局限

## 局限与展望

- 合成键值对任务与现实NLP任务存在差距，需要在更自然的场景中验证结论
- IES指标基于单一实验设置（46键），不同设置下的IES排名可能变化
- 未深入分析干扰效应的机制层面原因（如注意力头的行为模式）
- Mock QA reset的成功暗示了架构层面的改进方向（如显式的门控/遗忘机制），但本文未做探索
- 仅测试了文本模型，多模态模型是否有类似的干扰瓶颈未知
- 未考虑RAG等外部记忆增强方案对干扰问题的缓解效果

## 相关工作与启发

- 与"Lost in the Middle"互补：该工作侧重位置效应，本文侧重干扰效应——二者是影响检索的两个正交因素
- 对长上下文模型的实际启示：仅增加上下文窗口不能解决干扰问题，需要新的架构设计思路
- 与人类工作记忆的类比具有启发性但需谨慎——Transformer的自注意力与人类工作记忆的神经机制有本质差异
- 为LLM评测提供了新维度：除了上下文长度、推理能力、知识量外，抗干扰能力应成为独立的评估轴
- 对RAG系统设计的启示：检索结果中的相似但过时信息可能严重干扰LLM的处理

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 认知科学PI范式到LLM评测的迁移极具原创性，揭示了此前被忽视的根本限制
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖30+模型(0.6B-637B)，多个正交实验维度，统计分析严谨
- 写作质量: ⭐⭐⭐⭐⭐ 实验设计和结论层层递进,图表直观易读,认知科学背景介绍清晰
- 价值: ⭐⭐⭐⭐⭐ 揭示了Transformer架构的根本局限,对模型评测和架构设计都有重要指导价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Context](../../ACL2025/robotics/dice_idiomaticity.md)
- [\[NeurIPS 2025\] Understanding Prompt Tuning and In-Context Learning via Meta-Learning](../../NeurIPS2025/robotics/understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)
- [\[ICCV 2025\] Beyond Losses Reweighting: Empowering Multi-Task Learning via the Generalization Perspective](../../ICCV2025/robotics/beyond_losses_reweighting_empowering_multi-task_learning_via_the_generalization_.md)
- [\[NeurIPS 2025\] Beyond Parallelism: Synergistic Computational Graph Effects in Multi-Head Attention](../../NeurIPS2025/robotics/beyond_parallelism_synergistic_computational_graph_effects_in_multi-head_attenti.md)
- [\[CVPR 2026\] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation](../../CVPR2026/robotics/profocus_proactive_perception_and_focused_reasoning_in_vision-and-language_navig.md)

</div>

<!-- RELATED:END -->
