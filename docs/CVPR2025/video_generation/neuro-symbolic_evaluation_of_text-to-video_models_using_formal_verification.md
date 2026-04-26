---
title: >-
  [论文解读] NeuS-V: Neuro-Symbolic Evaluation of Text-to-Video Models using Formal Verification
description: >-
  [CVPR 2025][文本到视频评估] 提出 NeuS-V，首个用形式化验证（时序逻辑+概率模型检验）评估文本到视频（T2V）模型时序一致性的框架——将文本提示转为时序逻辑规范，用 VLM 评分原子命题，构建视频自动机后形式化验证满足概率，在 Gen-3 上与人类标注 Pearson 相关 0.71（VBench 仅 0.47）。
tags:
  - CVPR 2025
  - 文本到视频评估
  - 时序逻辑
  - 模型检验
  - 形式化验证
  - VLM
---

# NeuS-V: Neuro-Symbolic Evaluation of Text-to-Video Models using Formal Verification

**会议**: CVPR 2025  
**arXiv**: [2411.16718](https://arxiv.org/abs/2411.16718)  
**代码**: https://utaustin-swarmlab.github.io/NeuS-V (有)  
**领域**: 自动驾驶 / 视频生成评估  
**关键词**: 文本到视频评估, 时序逻辑, 模型检验, 形式化验证, VLM

## 一句话总结

提出 NeuS-V，首个用形式化验证（时序逻辑+概率模型检验）评估文本到视频（T2V）模型时序一致性的框架——将文本提示转为时序逻辑规范，用 VLM 评分原子命题，构建视频自动机后形式化验证满足概率，在 Gen-3 上与人类标注 Pearson 相关 0.71（VBench 仅 0.47）。

## 研究背景与动机

### 领域现状

**领域现状**：T2V 模型（如 Gen-3、Pika、CogVideoX）的评估主要依赖 VBench 等基于 CLIP 相似度的指标。这些指标通过逐帧相似度求平均，无法捕捉时序动态——"先A后B"和"先B后A"会得到相同分数。

**现有痛点**：VBench 对时序对齐的评估与人类判断相关性低（Pearson 仅 0.47），因为它不理解时序逻辑（ALWAYS/EVENTUALLY/UNTIL 等）。单帧 CLIP 无法区分"一直存在"和"偶尔出现"。

**核心矛盾**：视频的语义不仅是帧级的（每帧正确）还是序列级的（时序关系正确），但现有评估只做前者。

**切入角度**：用计算机科学中的形式化验证技术——将提示转为时序逻辑（TL），将视频建模为离散时间马尔可夫链（DTMC），用概率模型检验器 STORM 计算满足概率。

**核心 idea**：提示→时序逻辑 → 视频→自动机 → 形式化验证满足概率 = 严格的时序一致性评估。

## 方法详解

### 关键设计

1. **PULS（Prompt→TL 转换）**：用 GPT-4o 将自然语言提示分解为原子命题+时序逻辑规范。如"A cat walks then sits"→ EVENTUALLY(walk) AND EVENTUALLY(sit) AND walk UNTIL sit

2. **VLM 原子命题评分**：用 LLaMA-3.2 对每帧打分（0/1），判断每个原子命题是否满足

3. **视频自动机 + STORM 验证**：将帧级评分构建为 DTMC 转移矩阵 $\delta(q,q') = \prod_i (C_i^*)^{1_{q'_i=1}}(1-C_i^*)^{1_{q'_i=0}}$，用 STORM 模型检验器计算时序逻辑规范的满足概率

### 损失函数 / 训练策略

无训练——纯评估框架。满足概率 $\mathbb{P}[\mathcal{A}_\mathcal{V} \models \Phi]$ 由 STORM 精确计算。

## 实验关键数据

| 模型 | NeuS-V Pearson | VBench Pearson |
|------|---------------|----------------|
| Gen-3 | **0.71** | 0.47 |
| Pika | 0.62 | 0.70 |
| T2V-Turbo | 0.55 | 更低 |
| 错配字幕检测 | **0.52差值** | 0.38差值 |

### 消融实验
- 无 TL/无验证（纯 VLM）：Pearson 仅 0.30-0.40
- 时序算子（ALWAYS/EVENTUALLY/UNTIL）对区分对齐/不对齐视频至关重要
- LLaVA-Video-7B 过度自信，逐帧 LLaMA-3.2 更可靠
- PULS转换的准确性直接影响最终评分：错误的TL规范会导致误判，但GPT-4o的转换质量在测试提示上达到了实用水平
- STORM模型检验器的计算开销在简单规范（≤3个算子）下可接受，复杂规范下需要优化

### 关键发现
- **NeuS-V 在时序对齐上比 VBench 好 5×**——形式化验证真正理解了"先后顺序"
- **错配检测能力更强**：对齐-不对齐的分数差 0.52 vs VBench 0.38

## 亮点与洞察
- **形式化方法× AI 的优雅结合**——将理论计算机科学的模型检验引入视频生成评估
- **时序逻辑是时序对齐的正确抽象**——ALWAYS/EVENTUALLY/UNTIL 精确描述了人类对时序的判断

## 局限与展望
- 依赖 VLM 准确度——误差会传播
- TL 规范依赖 GPT-4o 转换（可能有语义偏差）
- 仅限英文提示
- 复杂规范（>3 个 TL 算子）计算开销非平凡
- VLM帧级判断的时间粒度受限于采样帧率，快速动作变化可能被遗漏
- 时序逻辑的表达能力虽强但不涵盖所有人类判断维度（如美学、物理合理性），仅评估时序对齐
- 基准数据集规模（360提示）相对较小，覆盖的时序模式类型可能不够全面
- 未来可将NeuS-V扩展到评估视频的物理合理性和因果一致性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将形式化验证引入 T2V 评估
- 实验充分度: ⭐⭐⭐⭐ 4个模型+360提示+人类标注相关
- 写作质量: ⭐⭐⭐⭐⭐ 跨领域概念解释清晰
- 价值: ⭐⭐⭐⭐⭐ 可能改变 T2V 评估范式

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] VideoDirector: Precise Video Editing via Text-to-Video Models](videodirector_precise_video_editing_via_text-to-video_models.md)
- [\[CVPR 2025\] Mimir: Improving Video Diffusion Models for Precise Text Understanding](mimir_improving_video_diffusion_models_for_precise_text_understanding.md)
- [\[CVPR 2025\] ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models](shotadapter_text-to-multi-shot_video_generation_with_diffusion_models.md)
- [\[ICCV 2025\] ETVA: Evaluation of Text-to-Video Alignment via Fine-Grained Question Generation and Answering](../../ICCV2025/video_generation/etva_evaluation_of_text-to-video_alignment_via_fine-grained_question_generation_.md)
- [\[CVPR 2026\] SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation](../../CVPR2026/video_generation/slvmeval_synthetic_meta_evaluation_benchmark_for_text-to-long_video_generation.md)

<!-- RELATED:END -->
