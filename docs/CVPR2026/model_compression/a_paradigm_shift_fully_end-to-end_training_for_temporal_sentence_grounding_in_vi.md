---
title: >-
  [论文解读] A Paradigm Shift: Fully End-to-End Training for Temporal Sentence Grounding in Videos
description: >-
  [CVPR 2026][模型压缩][时序语句定位] 提出首个完全端到端的时序语句定位(TSGV)框架，通过语句条件适配器(SCADA)将语句嵌入注入视频backbone的中间层来动态调制视觉特征，配合视频中心学习策略加速训练，在Charades-STA和ActivityNet上超越SOTA。
tags:
  - CVPR 2026
  - 模型压缩
  - 时序语句定位
  - 端到端训练
  - 语句条件适配器
  - 视觉-语言对齐
  - TSGV
---

# A Paradigm Shift: Fully End-to-End Training for Temporal Sentence Grounding in Videos

**会议**: CVPR 2026  
**arXiv**: [2604.02860](https://arxiv.org/abs/2604.02860)  
**代码**: 即将开源  
**领域**: 视频理解 / 跨模态定位  
**关键词**: 时序语句定位, 端到端训练, 语句条件适配器, 视觉-语言对齐, TSGV

## 一句话总结
提出首个完全端到端的时序语句定位(TSGV)框架，通过语句条件适配器(SCADA)将语句嵌入注入视频backbone的中间层来动态调制视觉特征，配合视频中心学习策略加速训练，在Charades-STA和ActivityNet上超越SOTA。

## 研究背景与动机

**领域现状**：TSGV旨在根据自然语言查询定位未剪辑视频中的对应时间段。现有方法大多采用预训练的视频编码器（如C3D/I3D）冻结提取特征，然后仅训练定位模块。

**现有痛点**：(1) 视频backbone训练用于视觉分类，但用于TSGV——存在任务不匹配；(2) 预训练模型仅学习了短语级的物体/动作，难以理解复杂自然语言语义；(3) 部分方法在定位阶段未利用语句特征，跨模态对齐不充分。

**核心矛盾**：冻结backbone→特征无法适配TSGV任务→定位精度受限。但直接微调大backbone→内存开销大且易灾难性遗忘。

**切入角度**：设计轻量适配器，在微调极少参数的前提下实现backbone的语句条件化。

**核心idea**：SCADA通过内外双分支将语句嵌入注入backbone各层，实现语句导向的视觉特征提取；视频中心学习策略让同一视频的多个查询共享特征提取。

## 方法详解

### 整体框架
语句编码(DistilBERT)→视频编码(ViT/I3D/C3D)+**SCADA适配器**→跨模态融合→检测头(BiLSTM+语句调制)→时间边界预测。

### 关键设计

1. **语句条件适配器(SCADA)**:

    - 功能：在视频backbone各层间插入，用语句嵌入动态调制视觉特征
    - **内分支(Inner)**：降维→语句乘法调制→深度可分离1D卷积(捕捉时间上下文)→升维+残差。输出传入下一backbone层
    - **外分支(Outer)**：3D卷积降维+空间压缩→语句调制→3D卷积升维→空间池化。输出跳过后续backbone层直接聚合
    - 最终：$F = \text{Normalize}(x_b + \sum_{i=1}^{n} x_{outer}^i)$
    - 设计动机：内分支让每层backbone感知语句语义；外分支提取查询引导的多尺度特征。仅训练少量适配参数，冻结backbone主体，解决内存和遗忘问题

2. **视频中心学习策略**:

    - 功能：加速端到端训练
    - 核心思路：同一视频的所有查询分组到同一mini-batch，backbone只需提取一次视频特征共享给多个查询
    - 设计动机：标准采样中同一视频反复通过backbone→计算冗余严重。视频中心采样同时让网络在单次迭代中对齐同一视频与多种语言上下文

3. **检测头的语句融合**:

    - BiLSTM架构+残差连接捕捉时间依赖
    - 在检测过程中反复用元素乘法融合语句嵌入，深度集成两模态

### 损失函数 / 训练策略
$\mathcal{L} = \mathcal{L}_b + \mathcal{L}_{iou} + \mathcal{L}_{offset}$：边界概率损失(正负平衡BCE)+ IoU损失(分类+L2回归)+ 偏移损失(Smooth L1)。

## 实验关键数据

### 主实验

| Backbone | 方法 | Charades R1@0.5 | Charades R1@0.7 | ActivityNet R1@0.5 | ActivityNet mIoU |
|----------|------|-----------------|-----------------|--------------------|----|
| C3D | MS-2D-TAN | 41.10 | 23.25 | 46.16 | - |
| C3D | APGN | 48.20 | 29.37 | - | - |
| C3D | **Ours** | **50.44** | - | - | - |
| I3D | PGSR等 | ~53 | ~30 | ~48 | ~48 |
| I3D | **Ours** | **Rank1最优** | **Rank1最优** | **领先** | **领先** |

Charades-STA: R1@0.5 = **48.1%**(ViT), ActivityNet: R1@0.5 = **30.5%**。

### 消融实验

| 配置 | Charades R1@0.5 | 说明 |
|------|-----------------|------|
| 冻结backbone(基线) | 基准 | 标准冻结范式 |
| 端到端全微调 | +大幅提升 | 验证E2E的有效性 |
| + SCADA | +进一步提升 | 语句条件化的价值 |
| + 视频中心学习 | 训练加速 | 减少冗余计算 |
| 无外分支 | 有下降 | 多尺度特征重要 |
| 无内分支 | 有下降 | 逐层调制重要 |

### 关键发现
- 端到端训练相比冻结baseline带来**平均16%的提升**（跨不同backbone和数据集一致）
- SCADA在I3D backbone上将Charades R1@0.5从~38提升到~53，提升显著
- ViT作为视频编码器在TSGV中的潜力首次被充分挖掘
- 视频中心学习让训练速度提升数倍（具体取决于每视频查询数）

## 亮点与洞察
- **端到端范式的系统性验证**：首次跨C3D/I3D/ViT-S/B/g多种backbone系统验证了端到端训练对TSGV的巨大价值，推翻了"冻结backbone足够"的默认假设
- **SCADA的设计巧妙**：双分支结构让语句信息既影响backbone内部（内分支）又产生跳跃连接（外分支），在极少参数下实现深度跨模态融合
- **视频中心学习的实用性**：利用TSGV数据集"一视频多查询"的天然特性，是一个低开销高回报的工程优化

## 局限与展望
- 当前仅在Charades-STA和ActivityNet两个数据集上验证，更多数据集（如TACoS、DiDeMo）有待探索
- SCADA的插入位置和数量是手动设定的，能否自动搜索最优配置？
- 与Video LLM方法(如D2VLM R1@0.5=50.30)的对比还不够全面
- ViT backbone的训练时间和显存开销未详细报告

## 相关工作与启发
- **vs 2D-TAN/APGN等**: 它们冻结backbone仅训练定位模块，本文证明E2E训练是更优范式
- **vs Video LLMs**: LLM方法通过时间敏感的指令微调预测时间戳，本文不依赖LLM但在部分指标上可比
- **vs 其他adapter方法**: LoRA等通用适配器不考虑语句条件化，SCADA专为跨模态任务设计

## 评分
- 新颖性: ⭐⭐⭐⭐ SCADA设计有新意，E2E范式虽非首创但系统验证有价值
- 实验充分度: ⭐⭐⭐⭐ 多backbone、消融充分，但数据集覆盖可更广
- 写作质量: ⭐⭐⭐⭐ 动机清晰、方法描述完整
- 价值: ⭐⭐⭐⭐ 为TSGV领域提供了新的训练范式参考

<!-- RELATED:START -->

## 相关论文

- [GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance](../../ICML2025/model_compression/guidedquant_large_language_model_quantization_via_exploiting_end_loss_guidance.md)
- [QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [Grounding and Enhancing Informativeness and Utility in Dataset Distillation](../../ICLR2026/model_compression/grounding_and_enhancing_informativeness_and_utility_in_dataset_distillation.md)
- [Partial Forward Blocking: A Novel Data Pruning Paradigm for Lossless Training Acceleration](../../ICCV2025/model_compression/partial_forward_blocking_a_novel_data_pruning_paradigm_for_lossless_training_acc.md)
- [Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge](critical_patch-aware_sparse_prompting_with_decoupled_training_for_continual_lear.md)

<!-- RELATED:END -->
