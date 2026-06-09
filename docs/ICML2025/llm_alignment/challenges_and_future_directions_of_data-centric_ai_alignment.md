---
title: >-
  [论文解读] Challenges and Future Directions of Data-Centric AI Alignment
description: >-
  [ICML 2025][LLM对齐][data-centric alignment] 本文是一篇 position paper，倡导将 AI 对齐的研究重心从算法设计转向数据质量，通过对 Anthropic-HH 数据集的定性分析揭示了人类反馈中的六大不可靠来源，并提出了改进数据收集、清洗和验证的未来方向。
tags:
  - "ICML 2025"
  - "LLM对齐"
  - "data-centric alignment"
  - "human feedback"
  - "RLHF"
  - "preference data"
  - "feedback reliability"
---

# Challenges and Future Directions of Data-Centric AI Alignment

**会议**: ICML 2025  
**arXiv**: [2410.01957](https://arxiv.org/abs/2410.01957)  
**代码**: 无  
**领域**: 对齐RLHF  
**关键词**: data-centric alignment, human feedback, RLHF, preference data, feedback reliability

## 一句话总结

本文是一篇 position paper，倡导将 AI 对齐的研究重心从算法设计转向数据质量，通过对 Anthropic-HH 数据集的定性分析揭示了人类反馈中的六大不可靠来源，并提出了改进数据收集、清洗和验证的未来方向。

## 研究背景与动机

**领域现状**：当前 AI 对齐方法主要集中在算法层面——RLHF 学习奖励函数、DPO 直接优化偏好等——这些方法依赖设计精巧的优化算法和损失函数来引导模型行为。

**现有痛点**：算法中心的对齐方法隐含地假设训练数据能准确反映真实人类偏好，但这一前提在实践中往往不成立。人类判断复杂且不可靠，即使算法设计得再好，如果训练数据本身有缺陷，对齐效果依然会大打折扣。

**核心矛盾**：当前研究过度关注"如何优化"（算法端），忽略了"优化什么"（数据端）。算法中心方法的瓶颈在于：它假设参与训练的偏好数据是完美的，而现实中人类反馈充满噪声、偏差和不一致。

**本文目标** (1) 系统性识别人类反馈数据中不可靠性的具体来源；(2) 分析 AI 生成反馈面临的限制；(3) 提出数据中心对齐的未来研究方向。

**切入角度**：作者对 Anthropic-HH 数据集进行了深入的定性分析，通过人工标注重新审视偏好标签的质量，从中发现了系统性的问题模式。

**核心 idea**：AI 对齐需要从"算法中心"转向"数据中心"，数据质量和代表性与算法设计同等重要甚至更为关键。

---

## 方法详解

### 整体框架

本文构建了一个数据中心对齐的分析框架，将反馈数据来源分为人类反馈和 AI 反馈两大类，系统分析了各自面临的挑战，并提出了覆盖数据收集、数据清洗、反馈验证三个维度的改进方向。

### 关键设计

1. **人类反馈不可靠性的六大来源分析**:

    - 功能：通过定性标注揭示偏好数据中噪声的具体成因
    - 核心思路：对 Anthropic-HH 数据子集进行重新标注，将低标注者间一致性（low IAA）和"两个都差"的样本分别聚类分析。识别出六类问题：(1) 人工标注错误——被拒绝的回答实际上更好；(2) 高主观性和缺乏上下文——旅行推荐等主观问题无客观好坏；(3) 不同的偏好标准——有人偏好直接回答，有人偏好追问澄清；(4) 不同的标准阈值——对"够好"的门槛不同；(5) 两个回答都包含有害建议；(6) 两个回答都含错误/无关信息
    - 设计动机：揭示偏好数据噪声不是随机的而是系统性的，仅靠算法层面的鲁棒化不足以解决

2. **AI 反馈的三大挑战**:

    - 功能：分析用 AI 替代人类标注的局限性
    - 核心思路：识别 AI 反馈面临的三个核心问题：(1) 对底层模型的依赖——AI 反馈受限于训练数据的多样性和偏差；(2) 无法真正反映人类价值观——AI 优化可量化指标但遗漏伦理推理的微妙之处，且存在呈现偏差、社会偏差、内容偏差和认知偏差；(3) 一致性不足——GPT-4 在评估微妙差异的回答时，多次试验的选择接近随机
    - 设计动机：说明简单地用 AI 替代人类标注并非万能解决方案，需要人机协作

3. **数据中心对齐的七个未来方向**:

    - 功能：为该领域描绘研究路线图
    - 核心思路：提出三大方向类别共七个具体方向：**数据收集改进**包括（方向1）全方位反馈收集——从标注者多样性、提示多样性、回答多样性三个维度确保覆盖面；（方向2）动态纵向偏好收集——追踪人类价值观随时间的漂移；（方向3）验证数据收集协议——引入"两者都好"/"两者都差"等选项。**数据清洗**包括（方向4）人机协作减少不可靠性——用奖励模型委员会识别人类标注错误并翻转标签；（方向5）优先数据质量而非数量——仅用5%数据训练可超越全量数据。**反馈验证**包括（方向6）为 AI 反馈引入人类监督；（方向7）标准化反馈验证流程
    - 设计动机：弥合理论分析与实践改进之间的差距

### 损失函数

本文为 position paper，未提出新的损失函数设计。但讨论了标注格式变化（如增加"both are bad"选项）如何影响奖励建模和对齐算法设计。

## 实验关键数据

### 主实验表格

本文的核心"实验"是定性标注分析。对低标注者间一致性（Low IAA）样本的分布：

| 不可靠来源 | Low IAA 数据占比 | "Both are bad" 数据占比 |
|---|---|---|
| 人工标注错误 | 2% | 0% |
| 高主观性 | 28% | 0% |
| 不同偏好标准 | 29% | 25% |
| 不同标准阈值 | 37% | 0% |
| 有害建议 | 0% | 39% |
| 错误/无关信息 | 4% | 36% |

### 消融表格

对比数据中心vs算法中心对齐的关键差异：

| 方面 | 数据中心对齐 | 算法中心对齐 |
|---|---|---|
| 关注点 | 反馈数据的质量和代表性 | 奖励模型和优化算法 |
| 核心挑战 | 数据偏差、反馈可靠性、多样性 | 奖励黑客、鲁棒性、偏好聚合 |
| 主要目标 | 确保数据反映真实人类价值 | 创建理论保证或奖励结构 |

### 关键发现

- Low IAA 样本中，65%的分歧来自主观性（28%）和不同的偏好标准/阈值（29%+37%），人工标注错误仅占2%
- "Both are bad"样本中，75%由有害建议（39%）和错误信息（36%）导致，本可通过提供"两者都差"选项避免强制选择
- 数据清洗文献表明：仅用5%精选数据训练可超越100%全量数据训练的效果（Li et al., 2024d）

## 亮点与洞察

- 将对齐问题从算法视角转向数据视角，这一视角转变颇具启发性——好比机器学习中"garbage in, garbage out"的朴素道理，在对齐领域被长期忽视
- 六大不可靠性来源的分类既全面又具操作性，为后续研究提供了清晰的问题拆分
- 引用社会科学领域关于问卷设计（如"both are bad"选项的研究 Olsen 1999）的成果，体现了跨学科视角

## 局限性

- 定性分析仅基于 Anthropic-HH 一个数据集的子集，分析规模较小，结论的普适性待验证
- 作为 position paper 缺乏具体的算法设计和定量实验验证
- 提出的未来方向大多停留在概念层面，缺少具体实现路径和可行性分析
- 没有充分讨论数据中心和算法中心方法如何协同，二者并非对立关系
- 对 AI 反馈一致性问题的讨论偏浅，如 position bias（Wang et al., 2024c）仅一笔带过
- 未涉及多语言、多文化背景下偏好数据收集的具体挑战和解决路径

## 相关工作与启发

- 与 RLHF（Ouyang et al., 2022）和 DPO（Rafailov et al., 2023）等算法中心方法形成互补
- PRISM 数据集（Kirk et al., 2024）是数据中心方向的先驱工作，收集了来自75个国家1500名参与者的偏好
- 弱 LLM 也能提供与人类标注匹敌的反馈（Tao & Li, 2025），预示了可扩展的对齐策略
- 启发：在实际对齐工作中，与其追求最先进的对齐算法，不如先审查和改善偏好数据的质量
- 数据清洗领域的 human-AI 协作方案（如奖励模型委员会，Yeh et al., 2024a）值得在更大规模上验证
- 偏好收集中增加"both are good/bad"选项的启示可迁移到其他标注任务设计中
- 人口统计多样性（Kirk et al., 2024, PRISM）对 prompt 分布的影响提示我们注意标注者选择偏差

## 评分

⭐⭐⭐ （6/10）

position paper 视角新颖，六大不可靠性来源的分析有价值，但作为研究贡献较为有限——缺乏具体算法和定量验证，更多是对现有问题的文献综述和讨论。未来方向的提出也相对宽泛，可操作性有限。适合作为该领域的入门阅读和问题定义参考。

值得注意的是，数据质量优先于数据规模的发现具有广泛适用性——Li et al. (2024d) 仅用 5% Alpaca 数据训练即超越全量数据，Lu et al. (2024) 用 6K 数据超越 50K 数据，这些发现对降低对齐成本有重要意义。本文最大的引领性贡献在于将"数据中心 AI"的理念系统性地引入对齐领域，为后续更具体的技术方案奠定了问题框架。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](../../ICLR2026/llm_alignment/skywork-reward-v2_scaling_preference_data_curation_via_human-ai_synergy.md)
- [\[ACL 2025\] LLMs Caught in the Crossfire: Malware Requests and Jailbreak Challenges](../../ACL2025/llm_alignment/llms_caught_in_the_crossfire_malware_requests_and_jailbreak_challenges.md)
- [\[ICML 2025\] Layer-wise Alignment: Examining Safety Alignment Across Image Encoder Layers in Vision Language Models](layer-wise_alignment_examining_safety_alignment_across_image_encoder_layers_in_v.md)
- [\[ICML 2025\] On the Robustness of Reward Models for Language Model Alignment](on_the_robustness_of_reward_models_for_language_model_alignment.md)
- [\[ICML 2025\] PoisonBench: Assessing Large Language Model Vulnerability to Data Poisoning](poisonbench_assessing_large_language_model_vulnerability_to_data_poisoning.md)

</div>

<!-- RELATED:END -->
