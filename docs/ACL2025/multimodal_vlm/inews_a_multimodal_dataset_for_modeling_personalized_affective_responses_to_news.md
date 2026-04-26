---
title: >-
  [论文解读] iNews: A Multimodal Dataset for Modeling Personalized Affective Responses to News
description: >-
  [ACL 2025][多模态][个性化情感响应] 构建了一个包含 291 位英国标注者对 2,899 条 Facebook 多模态新闻帖子的个性化情感标注数据集 iNews，标注者特征（人口统计、人格、媒体信任等）可解释 15.2% 的标注方差，结合 persona 信息的 LLM 零样本预测准确率提升最高 7%。
tags:
  - ACL 2025
  - 多模态
  - 个性化情感响应
  - 新闻感知
  - 多模态数据集
  - LLM人格提示
  - VAD情感模型
  - 标注者特征
---

# iNews: A Multimodal Dataset for Modeling Personalized Affective Responses to News

**会议**: ACL 2025  
**arXiv**: [2503.03335](https://arxiv.org/abs/2503.03335)  
**代码**: 已开源（数据集公开）  
**作者**: Tiancheng Hu, Nigel Collier
**机构**: University of Cambridge
**领域**: 多模态 / 情感计算 / 个性化建模  
**关键词**: 个性化情感响应, 新闻感知, 多模态数据集, LLM人格提示, VAD情感模型, 标注者特征

## 一句话总结

构建了一个包含 291 位英国标注者对 2,899 条 Facebook 多模态新闻帖子的个性化情感标注数据集 iNews，标注者特征（人口统计、人格、媒体信任等）可解释 15.2% 的标注方差，结合 persona 信息的 LLM 零样本预测准确率提升最高 7%。

## 研究背景与动机

**领域现状**：情感检测是 NLP 长期研究方向，但现有数据集几乎都使用聚合标签（gold labels），忽略了人类情感反应中的个体差异。心理学大量研究表明人格、人口统计、文化背景等个体因素深刻影响情感感知。

**现有方法的不足**：
   - 多数情绪检测数据集仅关注文本内容本身的情感倾向，而非读者个体的真实情感反应
   - 少量包含标注者信息的数据集（如 Diaz et al. 2018）仅覆盖简单情感维度，缺乏多模态、多面向标注
   - LLM 个性化能力的研究缺乏兼具行为数据+个体特征的基准数据集

**核心动机**：新闻情感反应是研究个体差异的理想场景——情感反应有成熟的测量框架（VAD、Ekman），个体差异在传播学和心理学中有充分文献支持，且新闻是人们日常接触的真实刺激。iNews 旨在同时采集丰富的标注者画像信息和多面向情感标注，为 LLM 个性化、主观现象建模、情感计算等方向提供数据支持。

## 方法详解

### 整体数据采集框架

iNews 数据集的采集包括三个阶段（见论文 Figure 1）：

1. **标注者招募**：通过 Prolific 平台招募 291 名英国标注者，使用配额抽样确保性别、年龄、政治倾向、地理区域的均衡分布（覆盖 124 个英国邮编区中的 97 个）
2. **人格画像调查**（Stage 1）：每位标注者填写包含 5 大类 47 个变量的问卷：
    - **人口统计与意识形态**：年龄、性别、政治倾向
    - **新闻消费与信任**：消费习惯和对主要英国新闻媒体的信任评分
    - **认知特征**：认知反思测试（CRT）
    - **人格特质**：大五人格量表（BFI-10）
    - **情感特征**：Perth 情绪反应量表（PERS）和正负情感量表（PANAS）
3. **新闻帖子标注**（Stage 2）：每位标注者为约 50 个 Facebook 新闻帖子截图提供以下 5 类标注：
    - **维度情绪评分**：效价（Valence）、唤醒度（Arousal）、支配感（Dominance），使用 SAM 自我评估人偶，7 级量表
    - **离散情绪分类**：Ekman 6 种基本情绪
    - **模态影响**：图片 vs 文字对其情绪反应的相对影响
    - **个人相关性**：帖子与个人的相关程度
    - **分享意愿**：是否愿意分享该帖子

### 新闻帖子采样策略

数据采集在 2024 年英国大选前后三个时间段进行（4月、6月、7月），确保时间多样性和新闻事件覆盖：
- 第一阶段：随机抽样
- 第二、三阶段：按媒体关注者数按比例分层抽样，提升生态效度
- 使用 CrowdTangle 采集 Facebook 新闻帖子，展示截图而非纯文本，保留互动数据但排除评论

### 质量控制

- 指导页强制停留至少 2 分钟（平均 4.67 分钟）
- 理解力测试题过滤不理解任务的标注者
- 两道注意力检查题（仅 2.4% 标注者未通过）
- 3 个来自 ANET 的标准化校准项目，与原始规范对比验证标注者使用 SAM 量表的一致性（平均偏差 < 0.5）

## 实验关键数据

### 回归分析：个体差异解释力

论文使用线性混合效应模型分析唤醒度评分的方差来源：

| 模型 | 固定效应 | 随机效应 | 边际 R² | 条件 R² |
|------|----------|----------|---------|---------|
| Null 模型 | 无 | 文本 | 0.000 | 0.131 |
| Persona 模型 | 47 个画像变量 | 文本 | 0.152 | 0.286 |
| User 模型 | 无 | 文本 + 用户 | 0.000 | 0.317 |

**核心发现**：
- 新闻内容仅解释 13.1% 的唤醒度方差
- 加入画像变量后解释力提升至 28.6%，画像变量单独解释 **15.2%** 的方差——高于此前所有 NLP 数据集
- User 模型（含未观测因素）解释 31.7%，说明仍存在额外个体差异因素

### 零样本 LLM 预测实验

在 7 个前沿 LLM 上测试 4 种输入条件（文本/图像 × 有/无 persona）：

| 模型 | 输入 | MAE↓ | 精确准确率↑ | ±1 准确率↑ |
|------|------|------|-----------|-----------|
| Gemini 1.5 Pro | 图像+Persona | **0.84** | **39.55%** | **82.04%** |
| Gemini 1.5 Pro | 文本+Persona | 0.91 | 36.44% | 78.76% |
| Gemini 1.5 Pro | 图像 | 0.94 | 36.96% | 77.03% |
| Llama-3.1-405B | 文本+Persona | 0.89 | 38.00% | 81.17% |
| GPT-4o | 图像+Persona | 0.89 | 36.79% | 79.45% |

**关键发现**：
- Persona 信息一致性提升所有模型性能，Gemini 1.5 Pro 的 MAE 降低 10.1%（图像）和 11.6%（文本）
- 图像输入在零样本中普遍优于文本输入（心理学一致：视觉刺激引发更强情感反应）
- 不同模型对 persona 提示的可操纵性差异大：Gemini/Grok/Llama 家族响应明显，GPT-4o 相对稳定

### Few-shot 实验（Gemini 1.5 Pro）

- 观察到 **"早期上升"现象**：从 0-shot 到 4-shot 时性能先下降，8-16 shot 后逐渐恢复并超越零样本
- 32-shot 达到最佳精确准确率 **44.4%**（文本+Persona）
- 即使在 32-shot 下，persona 信息仍提供额外增益（MAE: 0.812→0.782）
- 图像 few-shot 扩展效果不如文本——尽管零样本中图像更优，但 few-shot 学习中图像恢复更慢

## 亮点与洞察

1. **画像变量解释力最高的 NLP 数据集**：15.2% 的方差解释量显著高于已有工作，验证了采集丰富个体特征的价值
2. **多维标注设计完善**：同时覆盖 VAD 连续维度 + Ekman 离散情绪 + 模态影响 + 相关性 + 分享意愿，适用于多种下游任务
3. **定性分析补充**：通过 20 位标注者的开放式问卷发现了超出结构化数据的个体差异模式（如冷战成长经历、工作阶级背景对情感麻木的影响）
4. **Persona + 示例的互补性**：显式画像信息和行为示例提供互补信号，类似推荐系统中基于用户画像和基于行为的推荐方法的混合
5. **早期上升现象**：在情感预测中首次系统性验证了 few-shot ICL 中的性能先降后升模式

## 局限性

1. **样本局限**：仅覆盖英国标注者和英国新闻媒体，不代表全球文化和媒体生态
2. **平台局限**：仅使用 Facebook 帖子，可能存在平台特定效应
3. **情绪测量框架有限**：依赖自我报告的情绪，可能受社会期望偏差影响；VAD 和 Ekman 框架不能完全捕捉情绪反应的复杂性
4. **数据质量**：虽有多重质控，但无法完全排除 AI 生成的回答
5. **精确准确率仍然较低**：最佳模型仅 ~44%，说明个体级别的情感预测仍是极具挑战的任务

## 相关工作

- **新闻情感与个体差异**：Soroka et al. (2019) 的负面偏差研究、Oliver (2002) 等关于个体暴露差异的传播学文献
- **NLP 情感检测**：Strapparava & Mihalcea (2007)、Plaza-del Arco et al. (2024) 综述了该领域资源，但多数使用聚合标签
- **标注者背景建模**：Diaz et al. (2018) 提供了在线社区的人口统计+情感标注，但维度单一
- **LLM 个性化**：Rescala et al. (2024)、Dong et al. (2024) 探索了 persona prompting 的有效性

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性** ⭐⭐⭐⭐：首个同时包含丰富个体画像和多面向情感标注的多模态新闻数据集，填补了重要空白
- **实验充分性** ⭐⭐⭐⭐：回归分析 + 7 个 LLM 的零样本 + few-shot 实验全面系统
- **实用价值** ⭐⭐⭐⭐⭐：对 LLM 个性化、情感计算、人类行为模拟等多个方向有直接推动
- **局限性** ⭐⭐⭐：仅限英国+Facebook，精确准确率仍较低，说明该问题本身的难度

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] AkaCE: A Multimodal Multi-party Dataset for Emotion Recognition in Movie Dialogues](akan_cinematic_emotions_ace_a_multimodal_multi-party_dataset_for_emotion_recogni.md)
- [\[ACL 2025\] Aligning VLM Assistants with Personalized Situated Cognition](aligning_vlm_assistants_with_personalized_situated.md)
- [\[ACL 2025\] ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding](vigil3d_a_linguistically_diverse_dataset_for_3d_visual_grounding.md)
- [\[ICML 2025\] Universal Retrieval for Multimodal Trajectory Modeling](../../ICML2025/multimodal_vlm/universal_retrieval_for_multimodal_trajectory_modeling.md)
- [\[ACL 2025\] Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents](agent_rewardbench.md)

<!-- RELATED:END -->
