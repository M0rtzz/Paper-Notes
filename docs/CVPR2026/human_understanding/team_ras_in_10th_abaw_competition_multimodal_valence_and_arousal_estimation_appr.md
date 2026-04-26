---
title: >-
  [论文解读] Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach
description: >-
  [CVPR 2026][人体理解][情感计算] 提出一种结合人脸视觉特征、VLM行为描述嵌入和音频特征的多模态方法用于连续效价-唤醒（VA）估计，通过两种融合策略（DCMMOE 和 RAAV）在 Aff-Wild2 数据集上取得了竞争力的结果。
tags:
  - CVPR 2026
  - 人体理解
  - 情感计算
  - 效价-唤醒估计
  - 多模态融合
  - 混合专家
  - VLM行为描述
---

# Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach

**会议**: CVPR 2026  
**arXiv**: [2603.13056](https://arxiv.org/abs/2603.13056)  
**代码**: [GitHub](https://github.com/SMIL-SPCRAS/CVPRW-26)  
**领域**: 人体理解  
**关键词**: 情感计算, 效价-唤醒估计, 多模态融合, 混合专家, VLM行为描述

## 一句话总结

提出一种结合人脸视觉特征、VLM行为描述嵌入和音频特征的多模态方法用于连续效价-唤醒（VA）估计，通过两种融合策略（DCMMOE 和 RAAV）在 Aff-Wild2 数据集上取得了竞争力的结果。

## 研究背景与动机

**领域现状**: 在自然场景（in-the-wild）条件下的连续情感识别是人工智能领域的重要问题。效价（valence）反映情绪的愉悦程度，唤醒（arousal）反映情绪的强度，二者构成情感的连续维度表征。ABAW 挑战赛系列是该领域的标杆竞赛，已举办至第 10 届。

**现有痛点**: 现有方法主要依赖面部视觉特征与音频特征的融合，但对行为层面的上下文信息（如姿态、手势、场景语境）利用不足。尽管 VLM（视觉语言模型）已在多个视觉任务上表现出色，但尚未被充分用于连续 VA 估计。

**核心矛盾**: 单一模态信息不足以准确捕捉复杂的情感表达；现有多模态方法的融合策略对模态间不对称交互建模不够充分。

**本文目标**: 利用 VLM 提取行为描述嵌入，并设计有效的多模态融合策略来整合人脸、音频和行为信息进行连续 VA 估计。

**切入角度**: 引入 Qwen3-VL-4B-Instruct 提取视频中的行为语义嵌入，结合 GRADA 人脸特征和 WavLM 音频特征，探索两种不同的融合方案。

**核心 idea**: 将 VLM 行为描述嵌入作为第三模态，配合有向交叉模态混合专家融合和可靠性感知音视觉融合策略进行 VA 估计。

## 方法详解

### 整体框架

三个模态分支独立提取特征后通过融合策略整合：
- **人脸分支**: YOLO 人脸检测 → GRADA (EfficientNet-B1) 提取 256 维帧级情感嵌入 → Transformer 时序回归
- **行为描述分支**: Qwen3-VL-4B-Instruct 处理 16 帧视频段 + 情感导向文本提示 → 段级嵌入 → Mamba 时序建模
- **音频分支**: WavLM-Large 处理 4 秒音频段 → 注意力统计池化 → 跨模态嘴部动态过滤

### 关键设计

1. **GRADA 人脸特征提取器**: 基于 EfficientNet-B1，在 10 个公开情感数据集上多任务预训练（表情分类 + VA 回归），使用判别性学习率和渐进解冻策略。仅 7.9M 参数，兼顾泛化性和效率。256 维嵌入层作为帧级特征。

2. **Qwen3-VL 行为描述模型**: 使用多模态 VLM 处理视频帧和情感导向提示语（关注面部表情、头部运动、手势、姿态、场景等），提取最后隐藏层的最终有效 token 表示作为段级行为嵌入。分为仅视觉和多模态两种提取模式，实验表明多模态模式（CCC=0.5385）显著优于仅视觉模式（CCC=0.4007）。

3. **跨模态音频过滤**: 基于 MediaPipe 提取的嘴部张开动态来近似语音存在检测，过滤掉不含语音的音频段，降低噪声和不可靠段的影响。WavLM-Large 仅微调顶部 4 层 Transformer。

### 损失函数 / 训练策略

- 使用基于 CCC 的混合损失，部分实验加入 MAE 项提升训练稳定性。
- 不同模态使用不同的超参搜索：人脸模型 L=400, S=150; Mamba 模型窗口长度 16, 步幅 8。
- 音频模型主干学习率 5e-6, 头部学习率 2e-4，差异化学习率策略。

## 实验关键数据

### 主实验

各模型配置在 Aff-Wild2 上的 CCC 结果：

| 模型配置 | Valence CCC | Arousal CCC | 平均 CCC (dev) | 平均 CCC (test) |
|---------|-------------|-------------|---------------|----------------|
| 人脸 GRADA + Transformer | 0.5869 | 0.6508 | 0.6189 | 0.54 |
| 视觉 Qwen3 + Mamba | 0.2499 | 0.5515 | 0.4007 | — |
| 多模态 Qwen3 + Mamba | 0.4290 | 0.6480 | 0.5385 | — |
| 音频 WavLM | 0.3415 | 0.4636 | 0.4025 | — |
| 人脸+音频 DCMMOE | 0.6252 | 0.6671 | 0.6461 | 0.58 |
| 人脸+多模态Qwen3+音频 DCMMOE | 0.6100 | 0.6875 | 0.6487 | 0.61 |
| 人脸+多模态Qwen3+音频 RAAV | **0.6078** | **0.7073** | **0.6576** | **0.62** |

### 消融实验

Qwen3 嵌入提取模式对比：

| 提取模式 | Valence CCC | Arousal CCC | 平均 CCC |
|---------|-------------|-------------|----------|
| 仅视觉（Visual） | 0.2499 | 0.5515 | 0.4007 |
| 多模态（Multimodal） | 0.4290 | 0.6480 | 0.5385 |
| **提升** | +0.1791 | +0.0965 | +0.1378 |

### 关键发现

- **多模态融合一致优于单模态**: 三模态 RAAV 融合（CCC=0.6576）比最佳单模态（人脸模型 CCC=0.6189）提升约 6%。
- **VLM 行为嵌入的价值**: Qwen3 多模态嵌入明显优于仅视觉嵌入（+13.8% CCC），验证了视觉语言联合理解在情感分析中的优势。
- **RAAV 优于 DCMMOE**: RAAV 通过帧中心融合和音频作为辅助上下文的非对称设计取得最佳结果（arousal CCC=0.7073）。
- **芳香度（Arousal）普遍比效价（Valence）更易估计**: 所有单模态模型中 arousal CCC 均高于 valence。
- **从开发集到测试集的一致提升**: 测试集上从单模态 0.54 → 双模态 0.58 → 三模态 0.62。

## 亮点与洞察

- **首次将 VLM 行为描述引入 VA 估计**: 利用 Qwen3 提取行为语义嵌入作为独立模态，弥补了传统面部/音频特征对高层行为语义的不足。
- **有向交叉模态 MoE**: DCMMOE 显式建模有向跨模态交互（从 A→B 和 B→A 是不同的专家），并通过门控机制自适应加权。
- **跨模态音频过滤**: 利用视觉模态（嘴部动态）来过滤不可靠的音频段，跨模态协作预处理的巧妙设计。
- **极度轻量的行为建模**: Mamba 替代 Transformer 进行时序建模，参数效率更高。

## 局限与展望

- 这是一个竞赛方案论文，方法论创新深度有限，更多是工程组合优化。
- Qwen3 行为描述嵌入的提取成本较高（4B 参数模型推理），实际部署可能受限。
- 跨模态音频过滤依赖嘴部动态近似语音存在，并非完全可靠。
- 仅在 Aff-Wild2 单一数据集上验证，泛化性未知。
- 未探索更先进的端到端多模态融合架构。

## 相关工作与启发

- 与前几届 ABAW 挑战赛方法（如 Zhang et al. 的 Masked AE + VGGish、Yu et al. 的 ResNet + TCN + 跨模态注意力）相比，本文的核心区别在于引入 VLM 行为嵌入。
- RAAV 的非对称融合设计（视觉决定时间分辨率，音频提供窗口级证据）是一种值得借鉴的模态角色分配思路。
- Mamba 用于情感时序建模是一个新方向。

## 评分

- **新颖性**: ⭐⭐⭐ — VLM 行为嵌入引入有新意，但整体是模块组合方案
- **实验充分度**: ⭐⭐⭐ — 消融较为系统，但仅单一数据集，缺少跨数据集验证
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，技术细节充分，竞赛论文质量较好
- **价值**: ⭐⭐⭐ — 作为竞赛方案有参考价值，VLM 用于情感计算的探索值得关注

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] A Two-Stage Dual-Modality Model for Facial Expression Recognition](a_two_stage_dual_modality_model_for_facial_expression_recognition.md)
- [\[CVPR 2026\] TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size](teamhoi_learning_a_unified_policy_for_cooperative_human-object_interactions_with.md)
- [\[CVPR 2026\] Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](graph2eval_multimodal_task_generation_agents.md)
- [\[CVPR 2026\] See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles](see_think_act_teaching_multimodal_agents_to_effectively_interact_with_gui_by_ide.md)

<!-- RELATED:END -->
