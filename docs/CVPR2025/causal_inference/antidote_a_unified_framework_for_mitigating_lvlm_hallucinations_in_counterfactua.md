---
title: >-
  [论文解读] Antidote: A Unified Framework for Mitigating LVLM Hallucinations in Counterfactual Presupposition and Object Perception
description: >-
  [CVPR 2025][LVLM幻觉] 提出Antidote统一后训练框架和CP-Bench基准，通过合成数据驱动的自校正偏好优化，同时缓解LVLM在反事实预设问题和物体感知上的幻觉。
tags:
  - CVPR 2025
  - LVLM幻觉
  - 反事实预设
  - 偏好优化
  - CP-Bench
  - DPO
---

# Antidote: A Unified Framework for Mitigating LVLM Hallucinations in Counterfactual Presupposition and Object Perception

**会议**: CVPR 2025  
**arXiv**: [2504.20468](https://arxiv.org/abs/2504.20468)  
**代码**: 无  
**领域**: LVLM 幻觉缓解  
**关键词**: LVLM幻觉, 反事实预设, 偏好优化, CP-Bench, 自校正

## 一句话总结
提出Antidote——合成数据驱动的统一后训练框架，通过将事实先验注入提示实现模型自校正，将幻觉缓解解耦为偏好优化问题，在LLaVA系列上CP-Bench提升超50%，POPE提升1.8-3.3%，CHAIR/SHR降低30-50%，且无灾难性遗忘。

## 研究背景与动机

**领域现状**：LVLM幻觉缓解主要聚焦"物体感知"——包括物体存在性（POPE评估）和图像描述准确性（CHAIR/SHR评估）。已有方法包括指令微调（LRV）、对比解码（VCD）和后训练（HA-DPO）等。

**现有痛点**：现有方法都关注模型在回答生成阶段的幻觉，却忽略了问题本身可能包含错误前提。当用户问"图中那辆车是什么品牌？"而图中没有车时，即使最新的InternVL-2和Qwen2-VL也会盲目接受错误预设并编造答案。这种"反事实预设问题"（CPQ）代表了更深层的幻觉，现有缓解方法对此几乎无效。

**核心矛盾**：LVLM在指令微调过程中过度学习了遵循指令的模式，导致即使问题前提与图像内容矛盾，模型也倾向于"配合"回答。同时，物体共现的统计偏差使模型在看到相似场景时生成常见但不存在的共现物体。

**本文目标**：设计统一框架同时缓解两种幻觉：（1）反事实预设幻觉——识别问题中的错误前提并拒绝；（2）物体感知幻觉——减少不存在物体的描述。

**切入角度**：作者认为可以通过合成数据控制共现物体的解耦——生成"统计上应共现但实际不存在"的物体缺失图像，然后针对这些不存在的物体构造CPQ。关键洞察是：数据合成过程中的事实先验（哪些物体存在/不存在）可以无成本地融入提示，使模型自校正，将问题转化为偏好优化。

**核心 idea**：合成共现物体解耦的图像+对应的CPQ/存在性/描述查询，用事实先验提示模型自校正获得"正样本"，原始幻觉回答作为"负样本"，通过DPO训练模型学会区分。

## 方法详解

### 整体框架
输入为LVLM基线模型和合成数据流水线生成的偏好训练对。流水线分三步：（1）从CC3M构建标题池并用DeepSeek-V2去噪过滤；（2）场景理解——识别场景中物体和共现幻觉候选物体；（3）用Stable Diffusion 3生成解耦图像并用Grounding-DINO验证事实。自校正阶段将事实先验注入提示生成正样本，与原始幻觉回答组成偏好对做DPO后训练。

### 关键设计

1. **合成数据流水线（Data Synthesis Pipeline）**:

    - 功能：自动生成包含共现物体解耦的图像和对应查询
    - 核心思路：（a）从CC3M收集标题，用DeepSeek-V2重写过滤并用MinHash+LSH去重；（b）用DeepSeek-V2从标题中提取场景物体 $\mathcal{O}_{pre}$ 和幻觉候选物体 $\mathcal{O}_{hallu}$；（c）以标题为正提示、$\mathcal{O}_{hallu}$为负提示用SD3生成图像，再用Grounding-DINO做事实验证——检查 $\mathcal{O}_{pre}$ 确实存在且 $\mathcal{O}_{hallu}$ 确实不存在
    - 设计动机：通过控制生成过程解耦共现物体，获得"场景正常但特定物体缺失"的图像，为构造有针对性的幻觉训练数据提供精确的事实标注，无需人工干预

2. **自校正偏好对构建（Self-Correction via Preference Alignment）**:

    - 功能：无需外部专家模型，利用模型自身能力构建高质量偏好训练对
    - 核心思路：对三种查询类型分别构建偏好对。CPQ类：问不存在物体的属性，原始回答（幻觉）为负，加事实先验后的自校正回答（如"图中没有该物体"）为正。同时构建TPQ（真预设问题）防止模型过度谨慎。物体存在性查询和图像描述查询类似处理。用BGE-m3计算正负回答的余弦相似度过滤无幻觉样本。还维护key-value记忆库避免查询重复
    - 设计动机：相比依赖GPT-4V生成偏好样本，自校正完全利用合成数据中的事实信息，零额外成本；相比SFT只增加正样本概率，DPO的对比学习能更好区分幻觉vs真实回答

3. **CP-Bench基准**:

    - 功能：首个专门评估LVLM处理反事实预设能力的基准
    - 核心思路：包含dev集（自动合成）和test集（人工标注），各1000个样本（500 CPQ + 500 TPQ）。test集的CPQ覆盖四类日常场景：物品、知识、场景、活动。幻觉候选物体选自语义相关的高共现物体（如火车场景中的"铁路"），增加挑战性。用GPT-4o将开放式回答转化为二分类评估
    - 设计动机：现有幻觉基准（POPE、CHAIR）只评估回答生成中的幻觉，缺乏对问题预设判断能力的评估

### 损失函数 / 训练策略
使用DPO损失：$\mathcal{L}_{dpo} = -\mathbb{E}_\mathcal{D}[\log\sigma(\beta\log\frac{\pi_\theta(y_{pos}|[x_T,x_I])}{\pi_{ref}(y_{pos}|[x_T,x_I])} - \beta\log\frac{\pi_\theta(y_{neg}|[x_T,x_I])}{\pi_{ref}(y_{neg}|[x_T,x_I])})]$。训练数据：5000 CPQ + 5000 TPQ + 2000物体存在性 + 8000图像描述，共20K样本。LoRA微调（r=64, α=128），$\beta=0.1$。

## 实验关键数据

### 主实验（CP-Bench Test F1-Score）

| 模型 | 原始 | + Antidote | 提升 |
|------|------|-----------|------|
| LLaVA-1.5-7B | 5.7 | 78.4 | +72.7 |
| LLaVA-1.5-13B | 17.3 | 83.5 | +66.2 |
| LLaVA-Next-Mistral-7B | 26.7 | 76.8 | +50.1 |

### 物体感知幻觉（CHAIR_s↓ / SHR↓）

| 模型 | CHAIR_s | +Antidote | SHR | +Antidote |
|------|---------|-----------|-----|-----------|
| LLaVA-1.5-7B | 19.4 | 9.4 (-51%) | 36.7 | 18.1 (-51%) |
| LLaVA-1.5-13B | 30.0 | 12.6 (-58%) | 37.2 | 21.3 (-43%) |

### 消融实验

| 配置 | CP-Bench | POPE | SHR↓ | MMBench |
|------|----------|------|------|---------|
| Baseline | 12.4 | 86.07 | 36.7 | 64.3 |
| SFT替代DPO | 67.8 | 85.14 | 25.9 | 59.6 |
| Antidote (DPO) | 82.9 | 87.89 | 18.1 | 65.4 |

### 关键发现
- Antidote使开源模型在CP-Bench上接近闭源模型水平（78.4 vs Claude-3.5的86.0）
- DPO显著优于SFT——SFT在POPE和MMBench上退化，有灾难性遗忘；DPO不仅不退化，通用能力还有轻微提升
- 模型规模不是反事实预设能力的决定因素——MiniCPM-V2.5(8B)在CP-Bench上recall比Qwen2-VL(72B)高8.5%
- 注意力可视化显示Antidote训练后模型在回答物体相关词时更准确地关注图像中对应区域

## 亮点与洞察
- **新型幻觉类型的定义与评估**：首次系统化定义和评估CPQ幻觉，揭示了即使是最强开源LVLM也在此上严重失败——InternVL-2-Pro的F1仅64.2%
- **自校正偏好优化的零成本设计**：巧妙利用合成数据中已有的事实信息驱动自校正，无需GPT-4V等外部专家，是一个可推广到其他后训练场景的范式
- **合成数据控制共现解耦**：用负提示控制生成图像中的物体存在/缺失，为训练数据提供精确标注——这种"先验可控的合成训练数据"思路可推广到其他需要精确标注的任务

## 局限与展望
- 合成数据的多样性受限于CC3M标题池和SD3生成能力
- CP-Bench的GPT-4o评估引入了额外偏差
- LoRA rank的选择对效果影响大——r=128开始灾难性遗忘，r=256模型崩溃，需要仔细调参
- 未在InternVL-2/Qwen2-VL等更强基线上验证Antidote效果

## 相关工作与启发
- **vs HA-DPO**: HA-DPO依赖GPT-4V生成偏好样本，且对CPQ无效（F1仅4.7%）。Antidote通过合成数据自校正避免了外部依赖
- **vs VCD/OPERA等解码策略**: 解码策略只改善推理时行为，不改变模型内部。Antidote通过后训练从根本上增强模型的预设判断能力
- **vs SeVa**: SeVa使用负样本数据自监督，CPQ效果一般（F1=24.1）。Antidote的正负对比更有效

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统化解决CPQ幻觉，合成数据+自校正+DPO的组合有效且优雅
- 实验充分度: ⭐⭐⭐⭐⭐ CP-Bench+POPE+CHAIR+SHR+4个通用基准，消融全面，对比方法多
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法动机充分，实验分析深入
- 价值: ⭐⭐⭐⭐⭐ CPQ是LVLM在实际应用中面临的核心安全问题，Antidote提供了实用的解决方案

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding](seeing_far_and_clearly_mitigating_hallucinations_in_mllms_with_attention_causal_.md)
- [\[ACL 2026\] Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate](../../ACL2026/causal_inference/dialectic-med_mitigating_diagnostic_hallucinations_via_counterfactual_adversaria.md)
- [\[CVPR 2026\] Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression](../../CVPR2026/causal_inference/fighting_hallucinations_with_counterfactuals_diffusion-guided_perturbations_for_.md)
- [\[ACL 2025\] FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation](../../ACL2025/causal_inference/fitcf_a_framework_for_automatic_feature_importance-guided_counterfactual_example.md)
- [\[ACL 2025\] IRIS: An Iterative and Integrated Framework for Verifiable Causal Discovery](../../ACL2025/causal_inference/iris_an_iterative_and_integrated_framework.md)

<!-- RELATED:END -->
