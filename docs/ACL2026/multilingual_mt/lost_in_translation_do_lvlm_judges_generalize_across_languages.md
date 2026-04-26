---
title: >-
  [论文解读] Lost in Translation: Do LVLM Judges Generalize Across Languages?
description: >-
  [ACL 2026][多语言评估] 本文提出 MM-JudgeBench，首个大规模多语言多模态评判模型基准（25 种语言、60K+ 偏好实例），评估 22 个 LVLM 发现当前 LVLM 评判器存在显著的跨语言性能差异——模型大小和架构不能预测多语言鲁棒性，即使最先进的评判器也表现不一致，突显了多语言多模态评估基准的必要性。
tags:
  - ACL 2026
  - 多语言评估
  - LVLM评判
  - 奖励模型
  - 跨语言泛化
  - 视觉语言基准
---

# Lost in Translation: Do LVLM Judges Generalize Across Languages?

**会议**: ACL 2026  
**arXiv**: [2604.19405](https://arxiv.org/abs/2604.19405)  
**代码**: [https://github.com/tahmedge/mm-judgebench](https://github.com/tahmedge/mm-judgebench)  
**领域**: 多语言 / 模型评估  
**关键词**: 多语言评估, LVLM评判, 奖励模型, 跨语言泛化, 视觉语言基准

## 一句话总结

本文提出 MM-JudgeBench，首个大规模多语言多模态评判模型基准（25 种语言、60K+ 偏好实例），评估 22 个 LVLM 发现当前 LVLM 评判器存在显著的跨语言性能差异——模型大小和架构不能预测多语言鲁棒性，即使最先进的评判器也表现不一致，突显了多语言多模态评估基准的必要性。

## 研究背景与动机

**领域现状**：自动评估器（奖励模型/LLM-as-Judge）在 LVLM 开发中扮演核心角色，从训练对齐到模型选择和基准测试。然而，现有评估几乎完全基于英语。

**现有痛点**：(1) VL-RewardBench 和 Multimodal RewardBench 仅覆盖英语；(2) 多语言扩展（如 M-RewardBench）仅限文本模态；(3) 没有现有基准能统一研究跨语言和跨模态的奖励模型行为。

**核心矛盾**：LVLM 评判器被期望在多语言多模态设定中使用，但其可靠性仅在英语上验证。同一模型在英语上表现优秀但在法语上可能选择错误答案。

**本文目标**：(1) 构建首个多语言多模态评判基准；(2) 大规模评估 22 个 LVLM 的跨语言评判一致性；(3) 揭示当前奖励建模的多语言局限。

**切入角度**：使用高质量翻译模型（Gemini-3-Pro）将 VL-RewardBench 和 OpenCQA 翻译到 24 种语言（加英语共 25 种），严格质量过滤后构建控制实验。

**核心 idea**：通过固定视觉输入仅变化语言来隔离跨语言评估效应，揭示 LVLM 评判器在语言维度上的脆弱性。

## 方法详解

### 整体框架

三阶段方法：(1) 翻译模型选择——对比 Gemini 系列的翻译质量（LaBSE 和 CometKiwi 指标），选择 Gemini-3-Pro；(2) 数据构建——翻译 VL-RewardBench（视觉语言偏好判断）和 OpenCQA（图表问答判断）到 24 种语言，经质量过滤得到 60K+ 实例；(3) 模型评估——22 个 LVLM 的配对准确率、位置偏差、长度偏差分析。

### 关键设计

1. **MM-JudgeBench 数据集构建**:

    - 功能：提供首个多语言多模态评判基准
    - 核心思路：两个互补子集——M-VL-RewardBench（通用视觉语言偏好评估）和 M-OpenCQA（图表中心的视觉文本推理评估）。25 种类型学多样化的语言覆盖阿拉伯语到越南语。每个提示将查询和两个候选答案翻译为目标语言。质量过滤：LaBSE 和 CometKiwi < 0.75 的样本经人工回译检查并重翻或删除
    - 设计动机：现有基准无法揭示 LVLM 评判器的多语言脆弱性。单一提示翻译所有 24 种语言（减少 24 倍 API 调用）保证了成本可控

2. **多维度评估协议**:

    - 功能：超越准确率，揭示偏差和指令遵循失败
    - 核心思路：(1) 配对准确率——正确识别偏好响应的比例；(2) 位置偏差——正序和反序呈现答案时判断准确率的差异；(3) 长度偏差——模型是否倾向于选择更长但错误的答案。两次呈现每对答案（正序+反序）以检测位置偏差
    - 设计动机：仅看准确率会隐藏系统性偏差。位置偏差和长度偏差在评判模型的实际部署中可能导致严重的系统性误差

3. **多语言训练集 M-MM-RewardBench**:

    - 功能：支持开源模型的多语言领域适应
    - 核心思路：将 MM-RewardBench 翻译到 24 种语言，得到 100K+ 偏好实例训练集，与评估数据不重叠。用于微调开源模型以改善多语言评判性能
    - 设计动机：开源模型在多语言评判上表现较差，提供训练数据支持领域适应性微调

### 损失函数 / 训练策略

评估为零样本提示，要求 LVLM 选择更好的答案并提供理由。领域适应微调使用标准 SFT 在 M-MM-RewardBench 上进行。评估指标为配对准确率。

## 实验关键数据

### 主实验

**22 个 LVLM 在 MM-JudgeBench 上的平均准确率和方差**

| 模型 | 平均准确率 | 方差 | 说明 |
|------|----------|------|------|
| GPT-5 | 81.3% | 0.2 | 最稳定 |
| Gemini-2.5-Flash | ~78% | 低 | 接近 GPT-5 |
| Qwen3-VL-32B | ~77% | 低 | 开源最佳 |
| Gemma-3-27B | ~74% | 中 | 部分语言下降明显 |
| InternVL-3.5-8B | ~70% | 高 | 跨语言变异大 |
| LLaVA-Critic-7B | ~55% | 高 | 专用评判模型但仅英语训练 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 英语评估 | 最高 | 所有模型在英语上最强 |
| 低资源语言（哈萨克语等） | 下降最大 | 训练数据覆盖不足 |
| 效率优化变体 | 多语言崩溃 | 如 Gemini-Flash-Lite 英语强但多语言差 |
| + 推理增强 | 提升 | 要求提供理由改善评判 |
| + 多语言微调 | 显著提升 | 领域适应有效 |

### 关键发现

- 模型大小不能预测多语言鲁棒性——小模型 Qwen3-VL 在多语言上比许多更大模型更一致
- 效率优化变体（如 Flash-Lite）在英语上接近全尺寸版本，但多语言上严重退化
- LLaVA-Critic（专门训练的评判模型）因仅在英语上训练，多语言表现极差
- 位置偏差和长度偏差在非英语语言中更严重
- 领域适应微调和推理增强评判都能改善多语言性能

## 亮点与洞察

- 揭示了 LVLM 评判器的多语言"盲区"——整体平均分掩盖了语言间的巨大差异
- 效率优化变体的多语言崩溃是重要的实用警告——降低成本可能以牺牲公平性为代价
- 训练集 M-MM-RewardBench 的发布为社区改善多语言评判提供了直接支持

## 局限与展望

- 翻译可能引入系统性偏差（所有翻译来自同一模型）
- 25 种语言仍未覆盖世界上多数语言
- 未分析翻译质量如何影响评估结果
- 未来需要原生多语言（非翻译）的评估数据

## 相关工作与启发

- **vs VL-RewardBench**: 仅英语；MM-JudgeBench 扩展到 25 种语言
- **vs M-RewardBench**: 仅文本模态；MM-JudgeBench 增加视觉模态
- **vs Multimodal RewardBench**: 英语多模态；MM-JudgeBench 同时多语言和多模态

## 评分

- 新颖性: ⭐⭐⭐⭐ 填补了多语言多模态评判评估的空白
- 实验充分度: ⭐⭐⭐⭐⭐ 22 个模型、25 种语言、60K+ 实例
- 写作质量: ⭐⭐⭐⭐ 结构清晰，发现的实践含义阐述充分
- 价值: ⭐⭐⭐⭐⭐ 基准和训练集的发布对社区有持续价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Semantic and Expressive Variation in Image Captions Across Languages](../../CVPR2025/multilingual_mt/semantic_and_expressive_variations_in_image_captions_across_languages.md)
- [\[NeurIPS 2025\] DCAD-2000: A Multilingual Dataset across 2000+ Languages with Data Cleaning as Anomaly Detection](../../NeurIPS2025/multilingual_mt/dcad-2000_a_multilingual_dataset_across_2000_languages_with_data_cleaning_as_ano.md)
- [\[NeurIPS 2025\] HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages](../../NeurIPS2025/multilingual_mt/helpsteer3-preference_open_human-annotated_preference_data_across_diverse_tasks_.md)
- [\[ACL 2026\] Just Use XML: Revisiting Joint Translation and Label Projection](just_use_xml_revisiting_joint_translation_and_label_projection.md)
- [\[ACL 2026\] Syntax as a Rosetta Stone: Universal Dependencies for In-Context Coptic Translation](syntax_as_a_rosetta_stone_universal_dependencies_for_in-context_coptic_translati.md)

<!-- RELATED:END -->
