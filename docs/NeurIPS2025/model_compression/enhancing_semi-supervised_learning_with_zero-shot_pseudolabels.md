---
title: >-
  [论文解读] Enhancing Semi-supervised Learning with Zero-shot Pseudolabels
description: >-
  [NeurIPS 2025][模型压缩][半监督学习] ZeroMatch 提出两阶段框架将基础模型的零样本伪标签与半监督学习相结合：先用知识蒸馏初始化学生模型，再以辅助 KD loss 防止灾难性遗忘的方式执行 SSL 训练，在 6 个视觉/NLP 基准上一致超越标准 SSL 和零样本增强方法。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 半监督学习
  - 知识蒸馏
  - 零样本伪标签
  - 基础模型
  - 小模型训练
---

# Enhancing Semi-supervised Learning with Zero-shot Pseudolabels

**会议**: NeurIPS 2025  
**arXiv**: [2502.12584](https://arxiv.org/abs/2502.12584)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 半监督学习, 知识蒸馏, 零样本伪标签, 基础模型, 小模型训练

## 一句话总结
ZeroMatch 提出两阶段框架将基础模型的零样本伪标签与半监督学习相结合：先用知识蒸馏初始化学生模型，再以辅助 KD loss 防止灾难性遗忘的方式执行 SSL 训练，在 6 个视觉/NLP 基准上一致超越标准 SSL 和零样本增强方法。

## 研究背景与动机

**领域现状**：半监督学习 (SSL) 通过利用未标注数据降低标注成本，FixMatch/AdaMatch 等方法通过一致性正则化和伪标签取得了优异效果。同时，基础模型（FM）的零样本能力提供了额外的监督信号来源。

**现有痛点**：(a) 直接微调大型 FM 对资源受限用户不可行（如只有单卡的个人设备）；(b) 天真地将 FM 零样本预测作为伪标签训练小模型可能因伪标签噪声或领域不匹配导致性能退化；(c) 现有 KD 方法要么只用教师输出要么只用标注数据，错失互补。

**核心矛盾**：标注数据提供准确但稀少的监督，FM 伪标签提供广泛但有噪声的监督，如何在资源受限场景下同时利用两者？

**本文目标**：设计一个统一框架，联合利用标注数据、未标注数据和 FM 伪标签训练紧凑的学生模型。

**切入角度**：KD 和 SSL 从不同来源改进未标注数据上的预测——KD 利用教师预测，SSL 利用标注数据。两者互补，可以联合使用。

**核心 idea**：两阶段训练——先 KD 建立初始高置信度预测，再 SSL + 辅助 KD 联合训练防遗忘。

## 方法详解

### 整体框架
输入：标注集 $\mathcal{D}_L$，未标注集 $\mathcal{D}_U$，FM 生成的伪标签 $\hat{y}^L, \hat{y}^U$。输出：训练好的紧凑学生模型 $f$。分两个阶段：Stage 1 做知识蒸馏预热，Stage 2 做带辅助 KD 的半监督学习。

### 关键设计

1. **Stage 1: 知识蒸馏预热**:

    - 功能：用 FM 伪标签作为教师输出训练学生模型
    - 核心思路：对所有数据（含标注集输入）的伪标签做标准 KD：$\mathcal{L}_{KD} = \frac{1}{N}(\sum_{i=1}^{N_L}\mathcal{H}(\hat{y}_i^L, \mathbf{p}(y|x_i)) + \sum_{i=1}^{N_U}\mathcal{H}(\hat{y}_i^U, \mathbf{p}(y|u_i)))$
    - 设计动机：为 SSL 阶段提供高质量初始预测。标准 SSL 训练初期因模型弱导致利用率低，KD 预热可以让更多未标注样本从一开始就超过置信阈值被利用

2. **Stage 2: 带辅助 KD 的 SSL 训练**:

    - 功能：在标准 SSL 训练的同时保持从教师学到的知识
    - 核心思路：学生由编码器 $g(\cdot)$ + 主头 $h(\cdot)$ + 辅助头 $h_p(\cdot)$ 组成。主头跑 SSL 目标（$\mathcal{L}_s + \mathcal{L}_u$），辅助头跑 KD：$\mathcal{L}_{KD_2} = \frac{1}{B}(\sum \mathcal{H}(\hat{y}_i^L, \mathbf{q}(y|x_i)) + \sum \mathcal{H}(\hat{y}_i^U, \mathbf{q}(y|u_i)))$
    - 总损失：$\mathcal{L}_{KD\text{-}SSL} = \mathcal{L}_s + \mathcal{L}_u + \alpha_t \cdot \lambda_p \mathcal{L}_{KD_2}$
    - 设计动机：低标注场景下 SSL 易产生不准确伪标签，覆盖 Stage 1 学到的知识（灾难性遗忘）。辅助 KD 头共享编码器但独立于主头，确保教师知识持续回流而不直接干扰 SSL 决策

3. **退火机制 (Annealing)**:

    - 功能：$\alpha_t$ 从 0 线性增长到 1，控制辅助 KD loss 权重
    - 核心思路：训练初期让 SSL 目标主导以充分利用标注数据，后期逐渐引入 KD 稳定训练
    - 设计动机：避免低质量伪标签在一开始就主导训练，实现 "soft start" 的知识融入

### 损失函数 / 训练策略
- 与 AdaMatch 完全相同的超参数，确保公平对比
- $\alpha_p = 1$（启用退火）, $\lambda_p = 1$（所有实验统一）
- 视觉用 ViT-Small，NLP 用 BERT-Base，单卡 A5000 24GB
- 辅助头架构与主头相同（MLP 分类头）

## 实验关键数据

### 主实验
NLP 数据集（使用 GPT-4o 伪标签）：

| 数据集 | 标注量 | AdaMatch | Zero-shot | Pseudo-sup | ZeroMatch |
|--------|--------|----------|-----------|------------|-----------|
| Yahoo | 250 | 64.81 | 68.81 | 67.68 | **70.90** |
| Yahoo | 2000 | 69.42 | 68.81 | 67.56 | **72.09** |
| AG News | 40 | 85.21 | 86.25 | 86.33 | **88.70** |
| Amazon | 250 | 52.39 | 59.14 | 56.65 | **59.82** |

视觉数据集（使用 GPT-4.1 伪标签）：

| 数据集 | 标注量 | AdaMatch | Zero-shot | Pseudo-sup | ZeroMatch |
|--------|--------|----------|-----------|------------|-----------|
| CIFAR100 | 100 | 71.43 | 83.25 | 84.84 | **88.01** |
| Flowers102 | 204 | 86.71 | 88.37 | 85.40 | **95.17** |
| Resisc45 | 90 | 78.87 | 79.28 | 79.59 | **87.83** |

### 消融实验
ZeroMatch 各组件效果（CIFAR100, 100 标注）：

| 配置 | Accuracy | 说明 |
|------|---------|------|
| AdaMatch (无伪标签) | 71.43 | SSL baseline |
| Zero-shot 直接用 | 83.25 | FM 推理无训练 |
| Pseudo-supervision | 84.84 | 伪标签当真标签用 |
| PL feature input | 72.81 | 伪标签作额外特征 |
| ZeroMatch (完整) | **88.01** | 两阶段 KD+SSL |

### 关键发现
- ZeroMatch 在所有 6 个数据集上一致超越所有基线，标注极少时（CIFAR100 仅 1 样本/类）优势更大
- 对低质量伪标签鲁棒：使用 FLAN-T5 等弱教师时仍不退化
- Pseudo-supervision 在 NLP 上甚至不如 AdaMatch，说明噪声伪标签直接训练有害
- 视觉提升更大（CIFAR100: +16.6, Flowers102: +8.5），因为 GPT-4.1 视觉伪标签质量高

## 亮点与洞察
- **优雅的两阶段设计**：Stage 1 建立初始知识、Stage 2 辅助头防遗忘的思路简洁有效，不需要修改底层 SSL 算法，通用性好
- **辅助头设计**巧妙解耦了 KD 和 SSL 两个目标：主头专注下游任务，辅助头保持教师知识，通过共享编码器传递而不冲突
- **实用价值**：仅需 FM 推理 API（一次调用），无需微调 FM 本身，适合闭源模型；也避免数据泄露风险

## 局限与展望
- 伪标签格式限于分类任务的 one-hot，未探索生成式或回归任务
- 辅助 KD head 训练后被丢弃，存在参数浪费
- 未考虑 FM 对不同样本的置信度差异（所有伪标签等权处理）
- 退火策略是简单线性增长，可能不是最优调度
- 仅与 AdaMatch 集成，未验证与 FreeMatch 等更新 SSL 方法的兼容性

## 相关工作与启发
- **vs FixMatch/AdaMatch**: 标准 SSL 不利用外部知识，ZeroMatch 通过 FM 伪标签拓展了监督来源
- **vs GRIP/CPL/FineSSL**: 这些方法需微调 FM 本身，计算开销大；ZeroMatch 仅需 FM 推理
- **vs Pseudo-supervision**: 简单将伪标签当标签用无法处理噪声，ZeroMatch 的渐进式融入更鲁棒

## 评分
- 新颖性: ⭐⭐⭐ 思路是 KD + SSL 的自然组合，辅助头防遗忘虽有效但不算全新
- 实验充分度: ⭐⭐⭐⭐ 6 个数据集，多种教师质量，多种标注量，对比全面
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法阐述流畅，实验安排合理
- 价值: ⭐⭐⭐⭐ 对低资源场景利用 FM 知识有明确实用价值，方法简洁可复现

<!-- RELATED:START -->

## 相关论文

- [Revisiting Semi-Supervised Learning in the Era of Foundation Models](revisiting_semi-supervised_learning_in_the_era_of_foundation_models.md)
- [Boomerang Distillation Enables Zero-Shot Model Size Interpolation](../../ICLR2026/model_compression/boomerang_distillation_enables_zero-shot_model_size_interpolation.md)
- [Improving Zero-Shot Generalization for CLIP with Variational Adapter](../../ECCV2024/model_compression/improving_zero-shot_generalization_for_clip_with_variational_adapter.md)
- [FastLongSpeech: Enhancing Large Speech-Language Models for Efficient Long-Speech Processing](fastlongspeech_enhancing_large_speech-language_models_for_efficient_long-speech_.md)
- [VESSA: Video-based objEct-centric Self-Supervised Adaptation for Visual Foundation Models](vessa_video-based_object-centric_self-supervised_adaptation_for_visual_foundatio.md)

<!-- RELATED:END -->
