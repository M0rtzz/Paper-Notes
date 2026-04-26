---
title: >-
  [论文解读] CAP: Evaluation of Persuasive and Creative Image Generation
description: >-
  [ICCV 2025][图像生成][广告图像生成] 针对广告图像生成任务，提出三个新评估指标（创意性、对齐度、说服力），并用LLM扩展隐式消息为显式视觉描述来改善T2I模型的广告生成效果，在人类标注一致性上显著优于CLIPScore等基线指标。
tags:
  - ICCV 2025
  - 图像生成
  - 广告图像生成
  - 创意评估
  - 说服力评估
  - 文本-图像对齐
  - 隐式消息
---

# CAP: Evaluation of Persuasive and Creative Image Generation

**会议**: ICCV 2025  
**arXiv**: [2412.10426](https://arxiv.org/abs/2412.10426)  
**代码**: https://aysanaghazadeh.github.io/CAP/  
**领域**: Image Generation  
**关键词**: 广告图像生成, 创意评估, 说服力评估, 文本-图像对齐, 隐式消息

## 一句话总结
针对广告图像生成任务，提出三个新评估指标（创意性、对齐度、说服力），并用LLM扩展隐式消息为显式视觉描述来改善T2I模型的广告生成效果，在人类标注一致性上显著优于CLIPScore等基线指标。

## 研究背景与动机
广告图像的生成是一个未被充分研究的文本到图像（T2I）生成子领域。有效的广告需同时满足三个条件：**清晰传达信息、以创意方式呈现、有效影响受众**。然而，当前T2I模型（如Stable Diffusion、DALL-E 3）虽然能从详细显式描述生成高质量图像，但面对"隐式文本提示"（即不直接描述图像应包含什么对象，而是描述意图和消息）时，往往只能生成相关但缺乏创意和说服力的图像。

现有评估指标（FID、CLIPScore、VQAScore等）只关注显式文本-图像对齐，无法衡量创意性和说服力。例如，Fig.1中显示Gatorade广告，基线T2I只显示瓶子和罐头，而人类创作的广告包含运动员奔跑的画面和标语。现有指标无法区分这种质量差异。

核心切入点：**将修辞学中的说服理论（Aristotle的Pathos/Ethos/Logos）引入计算化评估**，并利用LLM的推理能力对广告图像进行多维度的创意和说服力打分。

## 方法详解

### 整体框架
给定广告消息（action-reason格式，如"I should drink Gatorade because it would help me win"），评估生成广告图像的三个维度：对齐度（AIM）、创意性（C_obj）、说服力（P_comp+AIM）。同时提出用LLM扩展隐式消息为显式视觉描述再输入T2I模型的生成策略。

### 关键设计

1. **AIM（Alignment of Image and Message）对齐度指标**:

    - 功能：评估生成图像与隐式广告消息的语义对齐度
    - 核心思路：
      1. 用MLLM（如InternVL-V2-26B）生成图像描述
      2. 用CPO（Contrastive Preference Optimization）微调LLM（如LLAMA3-8B），使其能从图像描述推断action-reason声明
      3. 分别计算生成声明的action和reason与原始消息的语义相似度，加权平均: $AIM = \frac{Sim(A_{gen}, A_m) + \alpha \cdot Sim(R_{gen}, R_m)}{1+\alpha}$，$\alpha=4$
    - 设计动机：直接使用CLIPScore等指标无法处理隐式文本；微调LLM可使其更准确地推断广告的深层含义而非简单的对象匹配

2. **C_obj（创意性指标）**:

    - 功能：量化生成图像的创意程度
    - 核心思路：创意 = 高对齐度 + 低对象相似度（即不只是简单展示消息中提到的对象）
    - 公式: $C_{obj} = \frac{AIM(I_{gen}, AR_m)}{\frac{1}{n}\sum_{obj} Sim(I_{gen}, obj) + 0.01}$
    - 设计动机：有创意的广告应以非典型方式传达消息，而非仅罗列产品图片

3. **P_comp+AIM（说服力指标）**:

    - 功能：多维度评估广告的说服效果
    - 核心思路：结合7个修辞学维度（受众定位AU、利益转化B、诉求类别AP、细节描述E、原创性O、想象力I、综合性S），让LLM对图像描述逐维度打分，取平均后与AIM结合
    - 设计动机：单一维度无法捕捉说服力的复杂性，多维度组合像集成模型，各维度的小误差相互抵消

### 损失函数 / 训练策略
AIM中LLM微调使用CPO损失（对比偏好优化），训练数据250张图像，3500个数据点（描述+正确/错误action-reason对），batch size=4，lr=5e-5，3000步。

## 实验关键数据

### 主实验

| 指标 | 与人类标注一致性（Krippendorff's Alpha） |
|---|---|
| CLIPScore | 0.17 |
| VQAScore | 0.31 |
| ImageReward | 0.11 |
| AIM (零样本LLM) | 0.18 |
| **AIM (InternVL+LLAMA3, 微调)** | **0.68** |
| 人类标注者间一致性 | 0.86 |

LLM扩展对T2I生成的改善（InternVL+LLAMA3+AuraFlow）:

| 配置 | AIM(COM) | C_obj(COM) | P_c+A(COM) | AIM(PSA) | C_obj(PSA) | P_c+A(PSA) |
|---|---|---|---|---|---|---|
| I_AR (直接) | 0.50 | 2.12 | 0.64 | 0.31 | 1.36 | 0.42 |
| I_LLAMA3 (扩展) | 0.53 | 2.25 | 0.70 | 0.43 | 1.87 | 0.60 |
| I_QwenLM (扩展) | 0.55 | 2.34 | 0.59 | 0.48 | 2.05 | 0.54 |

### 消融实验

| 指标配置 | 与人类在COM广告的一致性 | 与人类在PSA广告的一致性 |
|---|---|---|
| C_LLM（直接问LLM创意度） | -0.03 | 0.15 |
| **C_obj（提出的创意指标）** | **0.57** | **0.53** |
| P_LLM（直接问LLM说服力） | 0.27 | 0.26 |
| P_comp（多组件无AIM） | 0.83 | 0.54 |
| **P_comp+AIM（完整说服力）** | **0.85** | **0.75** |
| 人类标注者间一致性 | 0.80 | 0.56 |

说服力指标P_comp+AIM在商业广告上的人-AI一致性（0.85）甚至超过了人-人一致性（0.80）。

### 关键发现
- PSA（公益广告）比商业广告更难生成和评估，因为其消息更隐式
- T2I模型在隐式提示下严重缺乏创意和说服力
- 简单的LLM扩展策略在PSA广告上改善尤其显著（AIM提升39%，创意性提升38%，说服力提升43%）
- 多组件说服力评估优于直接让LLM打分，因为子问题引导了更细粒度的推理

## 亮点与洞察
- 将修辞学理论（Pathos/Ethos/Logos）引入计算化评估，跨学科思维令人耳目一新
- AIM指标的设计思路巧妙：先让LLM理解图像，再反推消息，最后与原始消息对比
- 人-AI一致性超过人-人一致性的结果说明多组件集成评估的优势
- LLM扩展策略简单但极其有效，揭示了T2I模型在理解隐式意图方面的根本缺陷

## 局限与展望
- 创意性指标C_obj与人类一致性仍有提升空间（0.54 vs 0.73人-人一致性）
- 依赖PittAd数据集，广告类型和消息格式有限
- LLM扩展策略虽然有效但偏向文字描述，无法捕捉纯视觉创意
- 未考虑文化差异对广告创意和说服力感知的影响
- 评估流程涉及多个模型串联（MLLM+LLM+CLIP），计算开销大

## 相关工作与启发
- 与说服性文本生成领域的NLP工作形成互补，但从文本延伸到了图像
- 隐式消息对齐问题也存在于MEME生成、海报设计等任务中
- 多组件评估范式可推广到其他需要多维度评估的生成任务

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次提出广告图像生成的创意和说服力计算评估指标，问题定义新
- 实验充分度: ⭐⭐⭐⭐ 多种MLLM/LLM/T2I组合 + 人类标注实验 + 商业/PSA分组分析
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，示例直观，但公式符号较多需要仔细阅读
- 价值: ⭐⭐⭐⭐ 为广告和创意内容生成提供了有价值的评估工具和改进方向

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Redefining <Creative> in Dictionary: Towards an Enhanced Semantic Understanding of Creative Generation](../../CVPR2025/image_generation/redefining_creative_in_dictionary_towards_an_enhanced_semantic_understanding_of_.md)
- [\[ICCV 2025\] ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation](adiee_automatic_dataset_creation_and_scorer_for_instruction_guided_image_editing_evaluation.md)
- [\[CVPR 2025\] Enhancing Creative Generation on Stable Diffusion-based Models](../../CVPR2025/image_generation/enhancing_creative_generation_on_stable_diffusion-based_models.md)
- [\[ICCV 2025\] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](holistic_unlearning_benchmark_a_multi-faceted_evaluation_for_text-to-image_diffu.md)
- [\[NeurIPS 2025\] OVERT: A Benchmark for Over-Refusal Evaluation on Text-to-Image Models](../../NeurIPS2025/image_generation/overt_a_benchmark_for_over-refusal_evaluation_on_text-to-image_models.md)

<!-- RELATED:END -->
