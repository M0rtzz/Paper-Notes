---
title: >-
  [论文解读] On the Out-of-Distribution Generalization of Multimodal Large Language Models
description: >-
  [CVPR 2025][医学图像][多模态大语言模型] 本文系统评估了14个MLLM在20个数据集上的分布外泛化能力，发现MLLM在医学/分子等领域特定数据上性能近似随机，通过三假设分析确定"语义-视觉映射缺陷"为主因，并证明上下文学习（ICL）能显著缓解该问题但对标签偏移和伪相关偏移敏感。
tags:
  - CVPR 2025
  - 医学图像
  - 多模态大语言模型
  - 分布外泛化
  - 上下文学习
  - 领域特定任务
  - 映射缺陷
---

# On the Out-of-Distribution Generalization of Multimodal Large Language Models

**会议**: CVPR 2025  
**arXiv**: [2402.06599](https://arxiv.org/abs/2402.06599)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 多模态大语言模型, 分布外泛化, 上下文学习, 领域特定任务, 映射缺陷

## 一句话总结

本文系统评估了14个MLLM在20个数据集上的分布外泛化能力，发现MLLM在医学/分子等领域特定数据上性能近似随机，通过三假设分析确定"语义-视觉映射缺陷"为主因，并证明上下文学习（ICL）能显著缓解该问题但对标签偏移和伪相关偏移敏感。

## 研究背景与动机

多模态大语言模型（MLLM）在通用视觉理解任务上展现了令人瞩目的能力。GPT-4V、Gemini、LLaVA等模型在常见物体识别、多模态推理等基准测试上取得了优异表现。然而，这些公共基准测试的优异性能是否真正反映了模型的泛化能力？

现有研究缺乏对MLLM在分布外（OOD）场景下的系统评估。一些零散的案例研究表明GPT-4V在遇到分布偏移时容易产生错误输出，但缺乏深入分析。**MLLM泛化能力的边界在哪里？导致泛化失败的根本原因是什么？如何改善？**这些关键问题尚未得到充分回答。

本文的动机是：(1) 全面划定MLLM泛化能力的边界，(2) 系统分析泛化失败的原因，(3) 探索可行的缓解方案。通过对比合成图像、自然分布偏移、医学影像和分子图像等不同场景，揭示了MLLM在训练数据分布之外的脆弱性。

## 方法详解

### 整体框架

本文是一项评估与分析工作，包含三个递进步骤：
1. **零样本泛化评估**：在合成、自然偏移、领域特定三类数据上评估14个MLLM
2. **失败分析**：提出并检验三个假设来解释泛化失败的原因
3. **ICL泛化探索**：研究上下文学习对弥补映射缺陷的效果及其在分布偏移下的鲁棒性

### 关键设计

1. **三假设失败分析框架**:
    - 功能：系统诊断MLLM泛化失败的根本原因
    - 核心思路：提出三个可能原因——(a) 语义误解：MLLM无法理解领域特定的科学概念；(b) 视觉特征提取不足：数据的高维度或复杂特征超出模型编码能力；(c) 映射缺陷：训练数据不足导致语义与视觉特征间的映射不完善
    - 设计动机：通过分别控制每个因素进行实验，可以精确定位瓶颈所在

    **假设(a)排除**：设计包含领域说明、专业知识引导和类别详细解释的增强提示，结果几乎无改善（如GPT-4V的HAM10000从53%下降到29.1%），排除了语义理解是主要瓶颈。

    **假设(b)排除**：使用CLIP作为特征提取器+线性探测分类器（linear probing），在相同数据集上表现远优于零样本MLLM（如COVID-CT: 83% vs GPT-4V 43.2%）。由于CLIP的视觉特征提取能力弱于大多数MLLM，这说明视觉编码不是瓶颈。

    **确定假设(c)为主因**：以上两个假设排除后，映射缺陷（即缺乏领域数据导致语义到视觉的映射不足）被确认为主要障碍。

2. **零样本泛化的Scaling Law分析**:
    - 功能：评估扩大模型规模能否改善OOD泛化
    - 核心思路：使用不同ViT规模的CLIP模型，在5个OOD数据集上测试零样本性能
    - 设计动机：如果扩大规模能持续改善OOD性能，则问题可通过规模化解决
    - **关键发现**：与ID任务上的典型scaling law不同，OOD泛化性能随模型规模增大并不持续改善，甚至在TerraInc和SVIRO上呈下降趋势。这表明简单扩大规模无法解决OOD泛化问题

3. **上下文学习（ICL）泛化探索**:
    - 功能：研究ICL作为弥补映射缺陷的低成本方案
    - 核心思路：系统评估ICL在三种分布关系下的效果：(a) ICE来自目标分布（理想情况），(b) ICE与测试数据存在域偏移，(c) ICE与测试数据存在标签/伪相关偏移
    - 设计动机：ICL无需更新模型参数，通过在输入中添加示例引导模型适应新任务

    **ICL实验设计**：将数据的领域和类别均分为两组，确保类别平衡。系统变化ICE数量（0, 2, 4, 8），ICE按类别均匀采样。

### 损失函数 / 训练策略

本文为评估工作，不涉及模型训练。评估策略包括：
- 零样本评估：使用精心设计的提示模板直接评估
- 增强提示评估：加入领域上下文信息的提示
- Linear probing：冻结CLIP视觉编码器，训练线性分类头
- ICL评估：在输入中添加不同数量和分布的示例

## 实验关键数据

### 主实验

零样本泛化——常见OOD数据集（11个）：

| 模型 | PACS | VLCS | DomainNet | iWildCam | 平均 |
|------|------|------|-----------|----------|------|
| GPT-4V | 96.9% | 87.2% | 74.8% | 52.3% | 69.1% |
| Gemini | 98.7% | 83.2% | 75.3% | 68.2% | 76.9% |
| LLaVA | 98.0% | 97.5% | 48.0% | 5.4% | 64.4% |

零样本泛化——领域特定数据集（9个）：

| 模型 | Camelyon17 | HAM10000 | NIH-Chest | DrugOOD_A | 平均 |
|------|-----------|----------|-----------|-----------|------|
| GPT-4V | 46.2% | 53.0% | 5.7% | 42.1% | 38.3% |
| Gemini | 50.2% | 41.0% | 7.8% | 52.2% | 43.3% |
| CLIP(LP) | 50.2% | **84.0%** | **74.0%** | **76.0%** | - |

ICL增强效果（GPT-4V，从0到8个示例）：

| 数据集 | 0-shot | 2-shot | 8-shot | 提升 |
|--------|--------|--------|--------|------|
| iWildCam | 63.4% | 78.0% | ~100% | +36.6% |
| HAM10000 | 53.9% | 66.4% | 74.2% | +20.3% |
| Camelyon17 | 48.4% | 52.0% | 57.4% | +9.0% |
| CT-XCOV | 44.3% | - | - | 平稳 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 标准提示 vs 增强提示 | GPT-4V HAM10000: 53%→29.1% | 增强提示反而可能有害 |
| CLIP零样本 vs CLIP线性探测 | HAM10000: 21.9%→84.0% | 证明视觉特征足够好 |
| ViT-B/16 → ViT-L/14 | TerraInc: 下降趋势 | OOD不满足典型scaling law |
| ICL目标域 vs ICL域偏移 | 域偏移ICL仍有显著增益 | 映射缺陷可通过相关域示例部分弥补 |

### 关键发现

- **MLLM泛化的二元性**：在PACS(>96%)、VLCS(>80%)等常见数据集上性能优异，但在医学/分子数据上近似随机猜测（二分类任务~50%）
- **映射缺陷是主因**：排除语义误解和视觉提取不足后，训练数据中缺乏领域映射知识被确认为泛化失败的核心原因
- **Scale不能解决OOD**：OOD泛化不遵循典型的scaling law，更大模型不一定更好
- **ICL有效但有限**：ICL可带来显著提升（iWildCam +36.6%），但在分子数据集上无效——需要深层领域知识的任务ICL不足以弥补
- **ICL对分布偏移敏感**：标签偏移和伪相关偏移会导致ICL性能下降和不稳定
- **不同MLLM的错误模式不一致**：无共享的系统性偏差，暗示模型集成可能有潜力

## 亮点与洞察

- **全面的评估基准**：14个MLLM × 20个数据集的系统评估，是该领域最全面的OOD泛化基准之一
- **假设驱动的分析范式**：通过提出-排除-确认的假设检验方法论诊断泛化失败原因，方法论规范且有说服力
- **映射缺陷概念**：将泛化失败归因于"语义-视觉映射"的缺失而非能力不足，为后续研究提供了清晰的改进方向
- **ICL作为低成本适应方案**：证明了无需微调，仅通过少量示例即可显著增强MLLM在新领域的性能
- **打破Scale迷信**：OOD不遵循scaling law的发现对"大力出奇迹"的思路提出了警示

## 局限与展望

- 评估受限于API可访问模型的上下文窗口大小（ICE数量有限）
- 仅考虑了图像分类任务，未涉及更复杂的视觉推理任务下的OOD泛化
- 领域特定数据集数量有限（主要为医学和分子），结论是否推广到其他专业领域需验证
- ICL策略较简单（随机采样），更智能的示例选择策略可能带来更大提升
- 未探索检索增强生成（RAG）等其他适应方案
- 未来可研究：最优ICE选择策略、领域自适应微调、多MLLM集成、以及将分析扩展到VQA/图像描述等任务

## 相关工作与启发

- **vs 现有MLLM基准(MMBench/SEED等)**: 现有基准主要评估通用能力，本文专注于OOD场景，揭示了通用基准高分不等于真正泛化
- **vs 领域泛化方法(DG/OOD)**: 传统OOD方法主要针对小模型+有监督设定,本文评估大模型零样本泛化，发现规模化不是解决方案
- **vs ICL for VLM**: 先前ICL工作主要关注ID任务性能，本文首次系统研究ICL在OOD场景下的效果和脆弱性
- **vs GPT-4V评估工作**: 先前工作多为案例展示，本文提供了定量系统评估和原因分析

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统评估MLLM OOD泛化并通过假设分析定位根因
- 实验充分度: ⭐⭐⭐⭐⭐ 14个模型×20个数据集，失败分析和ICL分析全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，假设-验证-结论的逻辑链完整
- 价值: ⭐⭐⭐⭐ 为MLLM在实际专业领域的部署提供了重要参考和改进方向

<!-- RELATED:START -->

## 相关论文

- [Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy](playing_the_fool_jailbreaking_llms_and_multimodal_llms_with_out-of-distribution_.md)
- [COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts](counts_benchmarking_object_detectors_and_multimodal_large_language_models_under_.md)
- [LAION-C: An Out-of-Distribution Benchmark for Web-Scale Vision Models](../../ICML2025/multimodal_vlm/laion-c_an_out-of-distribution_benchmark_for_web-scale_vision_models.md)
- [FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection](../../ICCV2025/multimodal_vlm/fa_forced_prompt_learning_of_vision-language_models_for_out-of-distribution_dete.md)
- [Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](../../NeurIPS2025/multimodal_vlm/revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)

<!-- RELATED:END -->
