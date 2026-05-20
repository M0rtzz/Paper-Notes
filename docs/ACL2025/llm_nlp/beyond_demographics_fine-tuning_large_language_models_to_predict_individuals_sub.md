---
title: >-
  [论文解读] Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions
description: >-
  [ACL 2025][LLM/NLP][社会人口学提示] 本文系统研究了 LLM 能否通过社会人口学属性（年龄、性别、种族、教育）来预测个体标注者的主观文本感知，发现微调后的改进主要来自学习个体标注者行为而非社会人口学模式，质疑了用 LLM 模拟社会人口学差异的可行性。
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "社会人口学提示"
  - "标注者建模"
  - "主观文本感知"
  - "个体差异"
  - "LLM微调"
---

# Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions

**会议**: ACL 2025  
**arXiv**: [2502.20897](https://arxiv.org/abs/2502.20897)  
**代码**: [https://github.com/morlikowski/beyond-demographics](https://github.com/morlikowski/beyond-demographics)  
**领域**: LLM/NLP  
**关键词**: 社会人口学提示、标注者建模、主观文本感知、个体差异、LLM微调

## 一句话总结
本文系统研究了 LLM 能否通过社会人口学属性（年龄、性别、种族、教育）来预测个体标注者的主观文本感知，发现微调后的改进主要来自学习个体标注者行为而非社会人口学模式，质疑了用 LLM 模拟社会人口学差异的可行性。

## 研究背景与动机

**领域现状**：主观 NLP 任务（如情感、冒犯性、亲密度评判）中标注者的自然差异与其社会人口学特征相关。近期研究尝试通过向 LLM 提示社会人口学属性来模拟不同群体的标注行为。

**现有痛点**：多项研究表明 LLM 在零样本社会人口学提示下表现不佳，即用 "你是一个 30 岁的白人女性" 这样的提示并不能使模型准确预测该群体的标注偏好。但此前无人系统研究微调是否能改善这一状况。

**核心矛盾**：社会人口学属性对标注行为确实有影响，但 LLM 似乎无法有效利用这些信息。问题的根源是 LLM 缺乏社会人口学知识，还是这种知识本身就不足以预测个体行为？

**本文目标**：回答四个研究问题：（RQ1）LLM 能否通过社会人口学或个体 ID 更好地建模标注者？（RQ2）能否泛化到新标注者？（RQ3）从社会人口学中学到了什么信息？（RQ4）个体建模能否改善争议预测？

**切入角度**：构建统一的多任务、多数据集评测框架 DeMo，系统比较四种输入格式：纯文本、+属性、+ID、+ID+属性。

**核心 idea**：通过对比"社会人口学属性"和"标注者 ID"两种个体化信息对 LLM 微调的影响，揭示 LLM 从社会人口学中学到的主要是将属性组合作为个体身份的代理，而非真正的社会人口学-标注关联。

## 方法详解

### 整体框架
输入是文本（推文/Reddit评论/邮件/对话）加上可选的标注者信息（社会人口学属性和/或唯一 ID），输出是标注者对文本的评分预测（3 或 5 级分类）。使用 Llama 3 8B 作为基础模型，添加预测头进行微调。

### 关键设计

1. **DeMo 统一评测数据集**:

    - 功能：提供五个主观任务的标准化评测框架
    - 核心思路：整合五个已有数据集（亲密度 MinT、冒犯性 Popquorn、礼貌度 Popquorn、安全性 DICES-350、情感 Díaz et al.），统一社会人口学属性为四维（年龄、性别、种族、教育），共 21,632 文本、2,614 标注者、147,297 条标注
    - 设计动机：现有研究各自使用不同数据集和属性定义，缺乏可比性

2. **四种输入格式的对比实验设计**:

    - 功能：分离社会人口学信息和个体身份信息的贡献
    - 核心思路：设计四种输入格式——Content-Only（纯文本基线）、+Attributes（加社会人口学属性）、+ID（加标注者唯一编号）、+ID+Attributes（两者都加）。模板极简，如 "Annotator: hispanic/latino, 40 to 44 years old, a woman, a college degree\n Text: ..."
    - 设计动机：通过控制变量精确测量每种信息的贡献，特别是区分属性提供的群体信息和ID提供的个体信息

3. **实例分割 vs 标注者分割的双评估策略**:

    - 功能：分别评估"对已知标注者的建模"和"对新标注者的泛化"
    - 核心思路：实例分割中标注者在训练/测试集中都出现（不同文本），测试模型能否学习个体偏好；标注者分割中测试集标注者从未在训练中出现，测试模型能否从属性泛化到新个体
    - 设计动机：两种场景对应不同的应用需求——已知用户的个性化 vs 陌生用户的群体画像

### 损失函数 / 训练策略
使用预测头（类似奖励模型架构）进行分类，交叉熵损失。采用 LoRA 微调，学习率通过 10 次运行的网格搜索选择，主实验每个配置运行 30 个不同随机种子。使用 macro-average F1 作为评估指标。

## 实验关键数据

### 主实验（实例分割 — RQ1）

| 输入格式 | Intimacy F1 | Offensiveness F1 | Politeness F1 | Safety F1 | Sentiment F1 |
|---------|-------------|-----------------|---------------|-----------|-------------|
| Zero-shot | ~0.22 | ~0.28 | ~0.24 | ~0.25 | ~0.26 |
| Content-Only | ~0.30 | ~0.25 | ~0.32 | ~0.38 | ~0.35 |
| +Attributes | ~0.33 | ~0.28 | ~0.35 | ~0.42 | ~0.38 |
| +ID | **~0.42** | **~0.35** | **~0.44** | **~0.48** | **~0.45** |
| +ID+Attr | ~0.42 | ~0.35 | ~0.44 | ~0.48 | ~0.45 |

### 标注者分割结果（RQ2）

| 输入格式 | 平均表现 | 说明 |
|---------|---------|------|
| Content-Only | 基线 | 仅基于文本 |
| +Attributes | ≈基线 | 属性对新标注者无帮助 |
| +ID | ≈基线 | 新ID无信息，符合预期 |
| +ID+Attributes | ≈基线 | 均无泛化能力 |

### 关键发现
- **RQ1**：微调后属性确实带来一致性提升，但 ID 的提升远大于属性。+ID+Attributes 不比 +ID 更好，说明属性提供的信息被 ID 完全包含
- **RQ2**：对新标注者，任何额外信息都不能超过纯文本基线，说明模型并未学到可泛化的社会人口学-标注关联
- **RQ3**：通过分析"唯一属性组合"和"高频属性组合"标注者的表现差异，发现属性提升主要来自唯一组合（属性等效于 ID），高频组合几乎无提升——模型将属性当作 ID 的代理
- **RQ4**：使用 ID 的模型在高分歧（high label entropy）样本上的 Wasserstein 距离显著改善，说明个体建模有助于捕捉标注争议

## 亮点与洞察
- **实验设计精妙**：通过唯一/高频属性组合的对比分析（RQ3），巧妙地揭示了"看似有效的社会人口学建模实际上是个体身份识别"。这种分析方法可以迁移到其他个体化建模研究中，帮助区分群体效应和个体效应
- **对 LLM 模拟人类行为的警示**：结论对用 LLM 做社会科学模拟（如用 LLM 替代调查受访者）提出了重要警告——LLM 并不真正理解社会人口学如何影响主观判断

## 局限与展望
- 数据集仅包含美国标注者，跨文化泛化能力未验证
- 主要使用 Llama 3 一个模型家族（附录中 Mistral 7B 补充实验显示类似趋势）
- 四维社会人口学属性可能不够丰富，但这已是五个数据集的最大交集
- 未来可以探索更细粒度的个体信息（如历史标注模式）来替代简单的 ID 嵌入

## 相关工作与启发
- **vs Orlikowski et al. (2023)**: 之前工作发现社会人口学不能超越 ID 的贡献（生态学谬误），本文在更大规模上确认了这一结论
- **vs Fleisig et al. (2023)**: 该工作发现属性优于 ID，与本文结论相反，可能源于数据集和架构差异
- **vs Beck et al. (2024)**: 本文微调结果远超零样本社会人口学提示，但提升主要来自个体记忆

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统性研究微调 LLM 的社会人口学建模能力
- 实验充分度: ⭐⭐⭐⭐⭐ 30个种子、5个任务、4种格式、2种分割、详尽分析
- 写作质量: ⭐⭐⭐⭐⭐ 研究问题层层递进，分析深入
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 社会模拟和标注者建模领域有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] HFT: Half Fine-Tuning for Large Language Models](hft_half_fine-tuning_for_large_language_models.md)
- [\[ACL 2025\] EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models](editext_diffusion_text_editing.md)
- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)
- [\[ACL 2025\] Perspective Transition of Large Language Models for Solving Subjective Tasks](perspective_transition_of_large_language_models_for_solving_subjective_tasks.md)
- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)

</div>

<!-- RELATED:END -->
