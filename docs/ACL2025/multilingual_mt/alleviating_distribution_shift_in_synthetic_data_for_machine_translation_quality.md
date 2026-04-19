---
title: >-
  [论文解读] Alleviating Distribution Shift in Synthetic Data for Machine Translation Quality Estimation
description: >-
  [ACL 2025][Quality Estimation] 提出 DCSQE 框架，通过约束波束搜索生成更真实的合成翻译、利用独立的标注模型纠正标签偏差、以及 SPCE 算法将 token 级标签聚合为短语级标签，有效缓解合成 QE 数据的分布偏移问题，在有监督和无监督设置下均超越 CometKiwi 等 SOTA 基线。
tags:
  - ACL 2025
  - Quality Estimation
  - 分布偏移
  - 合成数据
  - MQM标注
  - 翻译质量评估
---

# Alleviating Distribution Shift in Synthetic Data for Machine Translation Quality Estimation

**会议**: ACL 2025  
**arXiv**: [2502.19941](https://arxiv.org/abs/2502.19941)  
**代码**: [https://github.com/NJUNLP/njuqe](https://github.com/NJUNLP/njuqe)  
**领域**: NLP Generation / Machine Translation  
**关键词**: Quality Estimation, 分布偏移, 合成数据, MQM标注, 翻译质量评估

## 一句话总结

提出 DCSQE 框架，通过约束波束搜索生成更真实的合成翻译、利用独立的标注模型纠正标签偏差、以及 SPCE 算法将 token 级标签聚合为短语级标签，有效缓解合成 QE 数据的分布偏移问题，在有监督和无监督设置下均超越 CometKiwi 等 SOTA 基线。

## 研究背景与动机

翻译质量评估（QE）旨在不依赖参考译文的情况下评估机器翻译质量，可被视为翻译任务的奖励模型。MQM（多维质量度量）标注是当前 QE 的主流标准，能提供细粒度的错误跨度和严重程度信息，但人工标注成本极高，数据集规模小且语种覆盖有限。

为解决数据稀缺问题，已有工作尝试从平行语料生成合成 MQM 数据。然而，现有方法面临严重的**分布偏移**问题：

**MQMQE** 方法：通过随机遮蔽参考译文中的片段并用翻译模型负采样替换，生成的合成翻译流畅度差；且使用同一翻译模型标注自身输出时存在过度自信问题，标签与人类偏好不对齐。

**InstructScore** 方法：利用 GPT-4 提示生成错误，虽然翻译流畅但生成的错误不自然，且使用闭源 LLM 成本高昂。

这些问题不仅降低 QE 性能，还影响下游的人类偏好优化效果。

## 方法详解

### 整体框架

DCSQE（Distribution-Controlled Data Synthesis for QE）框架包含两个核心环节：**生成更真实的合成翻译**和**标注更准确的合成标签**。

首先训练两个独立的翻译模型：Generator（生成器）和 Annotator（标注器）。Generator 用约束波束搜索生成合成翻译；Annotator 用生成概率对标签进行细粒度修正。

### 关键设计

**1. 约束波束搜索（CBS）生成合成翻译**

与标准波束搜索不同，CBS 在生成时保留参考译文中生成概率超过阈值的 token，避免生成同义替换，使错误更自然。CBS 保留了参考译文的主体结构，产生的翻译错误更接近真实翻译模型产生的错误。

**2. 模型多样性增强**

为提高合成翻译多样性，作者在不同的平行语料子集上训练多个 Generator（如 L 和 L'），使其产生风格不同但性能相近的翻译输出。实验中两个生成器在 Flores-200 上的 BLEU 仅为 80.06，说明存在明显多样性。

**3. 粗粒度标签：TER 对齐**

使用 TER 工具将合成翻译与参考译文进行词级对齐，匹配部分标记为"OK"，不匹配部分标记为"BAD"。

**4. 细粒度标签：Annotator 修正**

对 TER 标记为"BAD"的 token，使用独立的 Annotator 模型的生成概率进行重新判定。通过设定三个有序阈值（tMINOR, tMAJOR, tCRITICAL），将生成概率映射为 MINOR/MAJOR/CRITICAL/OK 四个严重级别。

关键洞察：**同一模型不能准确标注自身输出**。翻译模型对自己的输出过度自信，导致错误标签比例极低（仅 0.11%-1.60%）。因此必须使用独立的 Annotator。

**5. 利用监督信号增强 Annotator**

确保 Annotator 在用于合成数据生成的平行语料上训练过，使其对这些数据具有专业级标注能力。

**6. SPCE 算法（最短短语覆盖错误）**

人类标注者倾向于标注完整短语而非零散 token。SPCE 算法通过依存句法树实现 token 级到短语级的聚合：
- 对连续 BAD token 构建候选集
- 在依存树上找最低公共祖先（LCA）
- 补全路径上的 token 和中间 token
- 迭代直到候选集稳定
- 短语的严重程度取候选 token 中最严重的级别

### 损失函数 / 训练策略

QE 模型基于 XLM-R-Large 骨干网络，训练目标结合句子级 MSE 回归损失和词级交叉熵分类损失。有监督设置下先在合成数据上预训练再在真实数据上微调；无监督设置仅使用合成数据训练。

推理时通过比较"OK"概率与不同阈值来确定错误严重程度，连续"BAD" token 组成跨度，严重程度取跨度内最严重等级。

## 实验关键数据

### 主实验

在 WMT QE Shared Task 数据集上评估，涵盖 EN-DE、ZH-EN、HE-EN 三个语言方向：

**有监督设置：**
- DCSQE 在 WMT23 EN-DE 上达到 Spearman 43.17（CometKiwi 40.47），MCC 27.11 vs 21.50
- ZH-EN 上 Spearman 46.41 vs CometKiwi 40.35
- 平均比 CometKiwi 高 4.38（Spearman）、3.41（MCC），尽管参数量更少
- 显著超越基于 GPT-4 的 GEMBA-MQM

**无监督设置：**
- MQMQE 和 InstructScore 性能平均下降 15.74 和 7.64
- DCSQE 仅下降 6.64，鲁棒性最佳
- HE-EN 上无监督 DCSQE（56.46 Spearman）超越有监督 CometKiwi（55.00）

### 消融实验

在 WMT23 EN-DE 无监督设置下：
- 完整 DCSQE：Spearman 35.78, MCC 18.00
- 移除 SPCE：Spearman 30.99, MCC 15.70（下降明显）
- 同时移除 SPCE 和 Annotator：Spearman 进一步下降

模型自标注 vs 独立标注实验：
- M 标注 M（自标注）：错误率仅 1.60%，Spearman 25.91
- M 生成 + L 标注（独立标注）：错误率 19.23%，Spearman 35.78

### 关键发现

1. **分布偏移是合成 QE 数据的核心问题**：DCSQE 从翻译和标签两方面缓解分布偏移
2. **模型不能公正标注自身输出**：自标注导致过度自信和大量假阴性
3. **Generator 多样性有益**：L+L' 双生成器比单 L 提升约 1 Spearman
4. **Generator 能力需要平衡**：太强（错误少）或太弱（错误不真实）都不好，中等能力 M 最优
5. **Annotator 能力越强越好**：利用监督信号和扩大训练语料均有效
6. **生成效率远超 InstructScore**：DCSQE 速度是 InstructScore 的 14.29 倍

## 亮点与洞察

- 将 QE 模型定位为翻译任务的奖励模型，对分布偏移问题的分析视角独到
- Generator 和 Annotator 分离的设计思路可推广到其他合成数据场景
- SPCE 算法巧妙利用依存句法实现 token→phrase 聚合，符合人类标注习惯
- 系统的控制变量实验（固定 Similarity 变 Error Rate / 固定 Error Rate 变 Similarity）提供了清晰的因果分析

## 局限与展望

- 未探索使用 LLM（如 GPT-4）作为 Annotator 的效果（受算力限制）
- 在极端数据稀缺场景（无平行语料）下的鲁棒性需验证
- 合成 QE 数据的洞察对通用奖励模型的迁移价值有待进一步探索
- Generator 能力平衡的最优点需要针对具体语言对调优

## 相关工作与启发

- 与 CometKiwi 的比较说明合成数据方法在参数更少的情况下也能超越跨语种标注迁移
- SPCE 算法与结构化预测中的 CRF 思路有异曲同工之处
- Generator-Annotator 分离的范式对 RLHF 中奖励模型构建有启发意义
- 多样性增强策略可推广到其他数据增强场景

## 评分

- **新颖性**: 7/10 — 将分布偏移问题系统化分解为翻译分布和标签分布两个维度
- **技术深度**: 8/10 — 实验设计精细，控制变量分析充分
- **实用性**: 8/10 — 方法实用且效率高，已开源
- **写作质量**: 8/10 — 逻辑清晰，分析全面

<!-- RELATED:START -->

## 相关论文

- [Watching the Watchers: Exposing Gender Disparities in Machine Translation Quality Estimation](watching_the_watchers_exposing_gender_disparities_in_machine_translation_quality.md)
- [Data Quality Issues in Multilingual Speech Datasets: The Need for Sociolinguistic Awareness and Proactive Language Planning](multilingual_speech_data_quality.md)
- [Machine Translation Models are Zero-Shot Detectors of Translation Direction](machine_translation_models_are_zero-shot_detectors_of_translation_direction.md)
- [Exploring In-context Example Generation for Machine Translation](exploring_in-context_example_generation_for_machine_translation.md)
- [Exploring In-Image Machine Translation with Real-World Background](exploring_in-image_machine_translation_with_real-world_background.md)

<!-- RELATED:END -->
