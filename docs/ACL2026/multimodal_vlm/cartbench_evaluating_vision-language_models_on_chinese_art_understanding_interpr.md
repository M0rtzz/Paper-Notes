---
title: >-
  [论文解读] CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity
description: >-
  [ACL 2026][多模态][中国艺术] 本文构建了 CArtBench——一个基于故宫博物院藏品的多任务基准，评估 VLM 在中国艺术理解中的四种能力（证据问答、结构化鉴赏、可辩护重解读、真伪辨别），发现即使最强模型在证据关联和风格-年代推理上也存在显著性能下降，而真伪辨别接近随机水平。
tags:
  - ACL 2026
  - 多模态
  - 中国艺术
  - 博物馆基准
  - 视觉语言模型
  - 鉴赏能力
  - 真伪辨别
---

# CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity

**会议**: ACL 2026  
**arXiv**: [2604.11632](https://arxiv.org/abs/2604.11632)  
**代码**: [https://github.com/Big-Sid/CARTBENCH-Chinese-Artwork-Benchmark](https://github.com/Big-Sid/CARTBENCH-Chinese-Artwork-Benchmark)  
**领域**: 多模态VLM/文化理解  
**关键词**: 中国艺术, 博物馆基准, 视觉语言模型, 鉴赏能力, 真伪辨别

## 一句话总结

本文构建了 CArtBench——一个基于故宫博物院藏品的多任务基准，评估 VLM 在中国艺术理解中的四种能力（证据问答、结构化鉴赏、可辩护重解读、真伪辨别），发现即使最强模型在证据关联和风格-年代推理上也存在显著性能下降，而真伪辨别接近随机水平。

## 研究背景与动机

**领域现状**：VLM 越来越多地被用作通用多模态助手，但其评估主要由网络图像和西方中心概念主导。中文和文化聚焦的基准虽有扩展，但主要集中在短文本识别和问答上。

**现有痛点**：(1) 现有基准缺乏面向专家的解释能力评估——需要文化锚定和明确视觉证据支持的深度理解；(2) 中国艺术的许多视觉惯例是时代敏感的，策展级理解需要将可观察线索与历史背景联系；(3) 真伪判断是文化遗产的核心工作流程，但当前 VLM 在此方面的能力从未被评测。

**核心矛盾**：VLM 可能在短文本问答上表现良好，但其高准确率可能掩盖在证据关联、结构化鉴赏和鉴真等深层能力上的严重不足。

**本文目标**：构建一个统一的基准来全面评估 VLM 在中国艺术理解中的策展级能力。

**切入角度**：将故宫博物院藏品的 Wikidata 实体与权威图录页面对齐，构建跨多朝代、五大艺术类别的博物馆基准。

**核心 idea**：从短文本 QA 扩展到证据锚定问答、结构化鉴赏、可辩护解读和真伪辨别四个递进任务层次，揭示 VLM 在文化理解中的系统性失败模式。

## 方法详解

### 整体框架

CArtBench 通过三阶段管道构建：(1) 从 Wikidata 检索故宫博物院的图像藏品；(2) 将藏品与官方图录描述对齐；(3) 专家指导下的筛选和分类。基于构建的数据，实例化四个互补任务。

### 关键设计

1. **CuratorQA（策展级问答）**:

    - 功能：评估 VLM 的证据锚定识别和推理能力
    - 核心思路：14,421 个问题覆盖 1,589 件艺术品，分为 P1（仅需视觉证据）和 P2（需结合艺术知识）两种难度，6 种题型包括主题识别、场景分类、构图格式、技法风格、图像学检测和风格-年代推理。使用 GPT-5.2 生成问答对，专家审核 1000 条错误率仅 0.47%
    - 设计动机：P1/P2 难度分层和 6 种题型分类使评估可以精确定位模型的能力短板

2. **CatalogCaption（结构化鉴赏）**:

    - 功能：评估 VLM 生成四段式专家级鉴赏文本的能力
    - 核心思路：86 件艺术品，要求模型生成包含基本信息、技法分析、历史背景和美学评价的结构化鉴赏文本，与权威图录描述比较
    - 设计动机：长文本生成是比 QA 更具挑战性的任务，要求模型综合视觉理解和文化知识

3. **ConnoisseurPairs（真伪辨别）**:

    - 功能：评估 VLM 在视觉相似的真伪对中进行鉴别的能力
    - 核心思路：10 对视觉相似的真品-仿品对，要求模型基于整体一致性和细微线索判断哪件为真品。这是诊断性压力测试
    - 设计动机：真伪辨别是鉴赏家的核心技能，测试 VLM 是否能超越表面识别进行深层推理

### 损失函数 / 训练策略

不涉及模型训练。评估使用统一协议结合自动指标、格式合规检查和专家评分。

## 实验关键数据

### 主实验

**CuratorQA 总体准确率（9 种 VLM）**

| 模型 | 总体准确率 | QA6(风格-年代推理) |
|------|----------|------------------|
| Qwen3-VL-235B | 0.84 | 0.56 |
| Qwen3-VL-30B | 0.80 | 0.42 |
| Qwen2.5-VL-72B | 0.81 | 0.53 |
| Qwen2.5-VL-32B | 0.80 | 0.53 |

### 消融实验

- 高整体准确率掩盖了在证据关联（QA5）和风格-年代推理（QA6）上的显著性能下降
- 长文本鉴赏（CatalogCaption）远未达到专家参考水平
- 真伪辨别（ConnoisseurPairs）在所有模型上接近随机水平，凸显鉴赏家级推理的极端困难

### 关键发现

- VLM 在短文本识别上的高分可能掩盖在证据关联和文化推理上的严重不足
- 风格-年代推理是最困难的子任务，最强模型也仅达到 56%
- 真伪辨别接近随机表现，说明当前 VLM 缺乏鉴赏家级别的视觉推理能力
- 不同艺术类别间存在显著性能差异

## 亮点与洞察

- 首个博物馆级中国艺术 VLM 基准，跨越识别、鉴赏、解读和鉴真四个层次
- 与故宫博物院权威图录对齐确保了数据的权威性
- 真伪辨别任务设计独特，直指 VLM 深层推理的盲区
- 评估协议设计严谨，结合自动指标和专家评分

## 局限与展望

- ReInterpret 和 ConnoisseurPairs 规模较小（25/10），属于诊断性评估
- 数据主要来自故宫博物院，可能存在收藏偏差
- 真伪辨别任务的专家标注成本极高，难以大规模扩展
- 未来可扩展到更多博物馆和更多艺术传统

## 相关工作与启发

- 与 CVLUE、CulturalVQA 等文化感知基准互补，但深入到专家级评估层次
- 与 ArtEmis（情感）、MuseumQA（事实）形成任务互补
- 为文化遗产领域的 AI 应用提供了更严格的评测标准

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个涵盖真伪辨别的博物馆级中国艺术 VLM 基准
- 实验充分度: ⭐⭐⭐⭐ 9 种 VLM、四种任务的全面评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，任务设计动机充分

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](../../ACL2025/multimodal_vlm/alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[ACL 2026\] Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning](enhancing_multimodal_large_language_models_for_ancient_chinese_character_evoluti.md)
- [\[ACL 2026\] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)
- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[CVPR 2026\] Understanding Task Transfer in Vision-Language Models](../../CVPR2026/multimodal_vlm/understanding_task_transfer_in_vision-language_models.md)

<!-- RELATED:END -->
