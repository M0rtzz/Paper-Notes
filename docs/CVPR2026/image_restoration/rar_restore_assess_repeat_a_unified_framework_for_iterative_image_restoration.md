---
title: >-
  [论文解读] RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration
description: >-
  [CVPR 2026][图像恢复][图像修复] RAR 将图像质量评估（IQA）与图像修复（IR）深度集成为统一端到端模型，在潜在空间中迭代执行"评估-修复-验证"循环，在复合退化场景下 PSNR 提升 +2.71 dB 且速度比 AgenticIR 快 11.27×。
tags:
  - CVPR 2026
  - 图像恢复
  - 图像修复
  - 图像复原
  - 迭代修复
  - 复合退化
  - 流匹配
---

# RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration

**会议**: CVPR 2026  
**arXiv**: [2603.26385](https://arxiv.org/abs/2603.26385)  
**代码**: [https://restore-assess-repeat.github.io/](https://restore-assess-repeat.github.io/)  
**领域**: 图像修复  
**关键词**: 图像修复, 图像质量评估, 迭代修复, 复合退化, 流匹配

## 一句话总结

RAR 将图像质量评估（IQA）与图像修复（IR）深度集成为统一端到端模型，在潜在空间中迭代执行"评估-修复-验证"循环，在复合退化场景下 PSNR 提升 +2.71 dB 且速度比 AgenticIR 快 11.27×。

## 研究背景与动机

真实场景中图像退化复杂且未知，可能同时包含模糊、噪声、雨雾等多种退化。现有方案分两类：All-in-one 模型（统一模型处理多种退化，但性能受限）和 Agentic 模型（智能体迭代选择专用工具，效果好但极慢）。

**核心矛盾**：All-in-one 模型缺乏对退化的精准识别能力，而 Agentic 模型的 IQA 和 IR 模块完全割裂——需要图像反复编解码、LLM 规划决策，流程臃肿且信息损失大。

**本文切入**：取两者之长——使用 VLM-based 自由文本 IQA（不限于预定义退化类别）+ 将 IQA 和 IR 深度集成到同一潜在空间中，实现端到端可训练的迭代修复。

## 方法详解

### 整体框架

RAR 基于 DepictQA（IQA模块）和 SD3.5（IR模块），通过适配器将两者的潜在空间对齐。退化图像编码到潜在空间 → LQA 评估当前质量 → 其输出 logits 直接作为 IR 的条件（无需文本解码）→ 修复后再评估 → 重复直到满足停止条件。

### 关键设计

1. **潜在质量评估 (LQA)**:

    - 功能：将 IQA 模块集成到修复模型的潜在空间中，实现端到端训练
    - 核心思路：通过输入适配器 $\mathcal{A}_I$ 将 IR 的潜在编码桥接到 IQA 的输入空间（避免图像解码），通过输出适配器 $\mathcal{A}_Q$ 将 IQA 的输出 logits 直接对齐到 IR 的条件嵌入空间（跳过文本解码和文本条件分支）。两阶段微调：先训适配器，再全参数微调
    - 设计动机：消除 IQA→文本→IR 的信息瓶颈，同时可以移除 IR 的文本条件分支节省参数和延迟

2. **基于流匹配的直接映射**:

    - 功能：从退化分布直接映射到高质量分布，支持迭代更新
    - 核心思路：传统扩散模型从噪声出发，中间状态含噪导致 LQA 无法准确评估。RAR 改用 Flow Matching，学习退化分布 $\rho_{deg}$ 到高质量分布 $\rho_{hq}$ 的直接映射：$\mathcal{L}_v = \mathbb{E}\|v_\theta(\mathbf{z}_t^n, Q_{deg}^n, t) - (\mathbf{z}_{hq} - \mathbf{z}_{deg}^n)\|^2$。中间状态是退化图像和目标的线性插值，始终是"有意义的图像"，LQA 可以准确评估
    - 设计动机：要实现迭代评估-修复循环，中间表示必须对 IQA 模块有意义，这排除了加噪声的扩散方案

3. **RAR 停止准则**:

    - 功能：自动判断何时结束迭代修复
    - 核心思路：每 $T$ 步用 LQA 比较修复前后的图像质量。LQA 基于 DepictQA 天然支持图像对比较，输出二元决策：CONTINUE（新图质量更好）或 STOP（质量不再改善）。STOP 时使用上一轮结果
    - 设计动机：自适应停止避免过度修复或不必要的计算，不同退化程度的图像自然获得不同的迭代次数

### 损失函数 / 训练策略

流匹配速度场损失 + LQA 的 IQA 训练损失。训练时 RAR 的迭代过程无缝集成到标准流匹配训练中——LQA 可以在任意时间步被调用更新条件。

## 实验关键数据

### 主实验

| 方法 | 复合退化PSNR↑ | MANIQA↑ | CLIP-IQA↑ | 速度 |
|------|-------------|---------|-----------|------|
| AgenticIR | 21.04 | 0.3071 | 0.4474 | 1× |
| AutoDIR | 19.64 | 0.2500 | 0.3767 | — |
| MiOIR | 20.84 | 0.2451 | 0.3933 | — |
| **RAR** | **20.46** | **0.4659** | **0.6566** | **11.27×** |

RAR 在感知指标（MANIQA、CLIP-IQA）上大幅领先，速度比 AgenticIR 快 11.27×。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无 LQA（固定文本条件） | 明显下降 | 动态评估条件至关重要 |
| 噪声扩散替代流匹配 | IQA 失效 | 中间表示含噪无法评估 |
| 无迭代（单次修复） | 复合退化处理不完整 | 迭代对复合退化必要 |
| 完整 RAR | 最优 | 所有组件协同增效 |

### 关键发现

- 流匹配的直接映射对于迭代修复是关键设计选择——扩散模型的加噪过程与迭代 IQA 评估根本不兼容
- 迭代过程自然地"先去噪再去雾"等，符合退化从易到难的处理顺序
- 端到端集成相比 pipeline 方案在延迟和性能上都有巨大优势

## 亮点与洞察

- **IQA-IR 深度集成**：从"两个独立模型协作"变成"一个模型的两个能力"，这种思路可迁移到其他需要评估-执行循环的任务
- **流匹配的意外优势**：流匹配相比扩散模型的一个被忽视的优势是中间表示有物理意义，使得迭代评估成为可能
- **停止准则的设计**：用 IQA 的对比能力做停止判断，优雅地解决了"修多少次够了"的问题

## 局限与展望

- PSNR 在某些设置下不如 AgenticIR（保真度 vs 感知质量的权衡）
- 依赖 DepictQA 的评估能力，对其不擅长的退化类型可能失效
- 停止准则可能在某些边界情况下不够鲁棒
- 未来可探索与更大的生成模型结合

## 相关工作与启发

- **vs AgenticIR**: AgenticIR 用 LLM 做规划+专用工具，功能强但极慢；RAR 将评估和修复融为一体，快 11× 且端到端可训练
- **vs AutoDIR**: AutoDIR 用 CLIP 做退化分类（闭集），RAR 用 DepictQA 做自由文本评估（开集），泛化能力更强
- **vs PromptIR/MiOIR**: All-in-one 方法无动态评估能力，对复合退化处理不够

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ IQA-IR 潜在空间集成+流匹配迭代的组合非常创新
- 实验充分度: ⭐⭐⭐⭐⭐ 复合/单一/未知退化全覆盖，保真和感知指标齐全
- 写作质量: ⭐⭐⭐⭐⭐ 架构清晰，逻辑严密，图示精美
- 价值: ⭐⭐⭐⭐⭐ 对图像修复领域有范式性贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](real_iisr_infrared_image_super_resolution_autoregressive.md)
- [\[CVPR 2026\] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)
- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)
- [\[CVPR 2026\] UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization](unirain_unified_image_deraining_rag_dataset_distillation.md)
- [\[CVPR 2026\] UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution](ucan_unified_convolutional_attention_lightweight_sr.md)

</div>

<!-- RELATED:END -->
