---
title: >-
  [论文解读] Evaluation of Vision-LLMs in Surveillance Video
description: >-
  [NeurIPS 2025][3D视觉][Vision-LLM] 提出一个无训练的两阶段框架，利用小型 Vision-LLM 生成视频文本描述 + NLI 分类器零样本评分，系统评估了提示策略和隐私保护滤镜对监控视频异常行为识别的影响。
tags:
  - NeurIPS 2025
  - 3D视觉
  - Vision-LLM
  - 零样本异常检测
  - 监控视频
  - 隐私保护
  - 自然语言推理
---

# Evaluation of Vision-LLMs in Surveillance Video

**会议**: NeurIPS 2025  
**arXiv**: [2510.23190](https://arxiv.org/abs/2510.23190)  
**代码**: [GitHub](https://github.com/pascalbenschopTU/VLLM_AnomalyRecognition)  
**领域**: 3d_vision  
**关键词**: Vision-LLM, 零样本异常检测, 监控视频, 隐私保护, 自然语言推理  

## 一句话总结

提出一个无训练的两阶段框架，利用小型 Vision-LLM 生成视频文本描述 + NLI 分类器零样本评分，系统评估了提示策略和隐私保护滤镜对监控视频异常行为识别的影响。

## 研究背景与动机

**监控数据量远超人力监控能力**：摄像头的广泛部署产生了海量视频数据，人工实时监控不现实，需要自动化异常检测手段。

**传统方法依赖大量标注数据**：有监督方法（如 MIL、GNN）需要精细的事件边界标注，成本高且难以推广到新类别的异常行为。

**公开异常检测数据集覆盖有限**：UCF-Crime、XD-Violence、RWF-2000 等数据集在规模、标签多样性上均有局限，训练在其上的模型泛化能力不足。

**Vision-LLM 在异常识别上缺乏系统评估**：尽管 VLM 在常规动作识别上表现出色，但对罕见/犯罪行为的零样本能力尚未被系统验证。

**隐私保护是实际部署的硬约束**：安防场景中需要对视频进行匿名化处理（模糊、GAN 替换等），但这些操作对 VLM 性能的影响不明。

**零样本灵活性具有实际价值**：如果 VLM 能通过自然语言提示识别新类型异常，无需重新训练，将极大提升系统的适应性和可扩展性。

## 方法详解

### 整体框架

两阶段流水线：(1) 冻结的 Vision-LLM 将视频帧序列转化为自然语言描述；(2) 冻结的 NLI 分类器（BART-large-MNLI）对描述与候选标签的文本蕴含关系打分，取最高分标签作为预测类别。全过程无需梯度更新。

### 关键设计 1：视频到文本的描述生成

- **功能**：将视频采样为帧序列输入 VLM，生成 ≤40 词的简洁文本描述。
- **核心思路**：利用预训练 VLM 内嵌的世界知识进行语义推理，将像素到标签的映射问题转化为语言推理问题。
- **设计动机**：VLM 通过大规模预训练已具备丰富的视觉-语言对齐能力，无需任务特定微调即可生成有意义的视频描述。

### 关键设计 2：基于 NLI 的零样本分类

- **功能**：将生成的文本描述作为前提，每个候选异常标签作为假设，用 NLI 模型计算蕴含分数。
- **核心思路**：将多分类问题转化为文本蕴含任务，利用预训练 NLI 模型的语义匹配能力实现零样本分类。
- **设计动机**：新增异常类别只需添加文本标签到候选集，无需修改模型参数，实现真正的零样本灵活性。

### 关键设计 3：多层提示策略

- **功能**：设计三种提示方案——无引导提示（自由描述）、引导提示（提供候选类别列表）、引导+少样本提示（额外提供示例图像和描述）。
- **核心思路**：通过结构化提示约束 VLM 的输出空间，使其生成与分类任务更相关的描述。
- **设计动机**：开放式描述可能偏离异常检测的关键信息，引导提示可以聚焦模型注意力到任务相关的语义上。

### 关键设计 4：隐私保护滤镜评估

- **功能**：在 RWF-2000 上测试三种隐私保护方案——局部头部模糊、GAN 人脸匿名化（DeepPrivacy2）、GAN 全身匿名化。
- **核心思路**：预生成匿名化数据集，在相同评估条件下对比不同隐私滤镜对 VLM 异常检测性能的影响。
- **设计动机**：实际部署中隐私保护不可或缺，需量化其对模型性能的具体代价，为工程决策提供依据。

## 损失函数与训练策略

本方法无需训练——VLM 和 NLI 分类器均使用冻结参数。推理时采用保守解码策略：temperature 0.05–0.1，最大新 token 数 64–128，重复惩罚系数 1.5。对于长视频按时间窗口分批处理，只要任一窗口预测正确即视为该视频预测正确。

## 实验

### 实验 1：提示策略对 UCF-Crime 的影响

| 模型 | 无引导 Top-1(%) | 引导 Top-1(%) | 引导+少样本 Top-1(%) |
|------|----------------|--------------|---------------------|
| Gemma-3 (4B) | 26.29 | 33.85 | 29.80 |
| NVILA-8B | 13.39 | 27.00 | **45.05** |
| Qwen-2.5-VL-7B | 25.31 | 34.69 | — |
| VideoLLaMA-3-7B | 19.94 | 34.16 | — |

- 引导提示普遍提升准确率（+7–14 pp）。
- 少样本示例对 NVILA 提升显著（+18 pp），但对 Gemma-3 反而下降，且普遍增加假阳性率。

### 实验 2：隐私保护滤镜对 RWF-2000 的影响

| 模型 | 无滤镜 Acc/FP(%) | 模糊 ΔAcc/ΔFP | GAN 人脸 ΔAcc/ΔFP | GAN 全身 ΔAcc/ΔFP |
|------|-----------------|---------------|-------------------|-------------------|
| Gemma-3 (4B) | 86.25/20.50 | –5.0/+10.5 | –2.8/+7.0 | –4.0/+7.0 |
| NVILA-8B | 82.50/14.00 | –1.8/+2.0 | –1.8/+5.0 | –11.3/+7.5 |
| Qwen-2.5-VL-7B | 82.25/24.50 | –4.8/+9.0 | –1.0/+2.0 | –6.5/+11.0 |
| VideoLLaMA-3-7B | 83.25/8.50 | –2.5/+2.0 | –4.5/–5.5 | –8.8/–6.5 |

- 隐私滤镜普遍导致准确率下降 2–11 pp，假阳性率上升。
- GAN 全身匿名化影响最大，因为 GAN 在帧间生成不一致的外观导致运动线索失真。
- VideoLLaMA-3 在 GAN 滤镜下假阳性反而降低，显示不同模型对隐私处理的敏感性差异。

## 亮点

- **完全无训练**：整个流水线无需梯度更新，真正的零样本——添加新标签即可检测新类型异常。
- **模块化架构**：VLM 和 NLI 分类器解耦，可独立升级替换。
- **首次系统评估隐私保护对 VLM 异常检测的影响**：填补了该方向的实验空白。
- **实验设计严谨**：少样本示例来自训练集避免泄露，单一变量对比，控制变量清晰。

## 局限性

- **准确率整体偏低**：UCF-Crime 上最高仅 45%，距实际部署差距大。
- **仅评估小型模型（≤8B）**：更大的模型（如 GPT-4V、Gemini）可能表现更好，但未涉及。
- **单次运行**：由于计算成本限制，每个实验只跑一次，缺乏统计显著性分析。
- **数据集有限**：仅用两个数据集，未涵盖更多场景（如 XD-Violence 的多模态数据）。
- **GAN 时间不一致性**：论文指出但未解决 GAN 全身匿名化帧间不一致的问题。

## 相关工作

- **有监督异常检测**：UCF-Crime 的 MIL 范式 [Sultani et al. 2018]、REWARD [Karim et al. 2024]、AnomalyCLIP [Zanella et al. 2024a]、MissionGNN [Yun et al. 2025]。
- **VLM 用于异常检测**：LAVAD [Zanella et al. 2024b] 用 LLM 时序聚合字幕异常分数、Holmes-VAD [Zhang et al. 2024] 指令微调多模态 LLM、TEVAD [Chen et al. 2023] 利用文本改进异常评分。
- **VLM 基座**：Gemma-3 [Team et al. 2025]、Qwen-2.5-VL [Bai et al. 2025]、VideoLLaMA-3 [Zhang et al. 2025]、NVILA [Liu et al. 2024]。

## 评分

- 新颖性: ⭐⭐⭐ — 框架思路（VLM+NLI）不算新颖，但对隐私保护的系统评估有价值
- 实验充分度: ⭐⭐⭐⭐ — 多模型×多提示×多隐私滤镜的全面消融，但缺统计检验
- 写作质量: ⭐⭐⭐⭐ — 条理清晰，形式推导完整
- 价值: ⭐⭐⭐⭐ — 为隐私保护下的零样本监控异常检测提供了实用基线和工程指导

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content](can_llms_write_faithfully_an_agent-based_evaluation_of_llm-generated_islamic_con.md)
- [\[NeurIPS 2025\] NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods](nerfbaselines_consistent_and_reproducible_evaluation_of_novel_view_synthesis_met.md)
- [\[NeurIPS 2025\] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations](openlex3d_a_tiered_evaluation_benchmark_for_open-vocabulary_3d_scene_representat.md)
- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)
- [\[NeurIPS 2025\] Instant Video Models: Universal Adapters for Stabilizing Image-Based Networks](instant_video_models_universal_adapters_for_stabilizing_image-based_networks.md)

<!-- RELATED:END -->
