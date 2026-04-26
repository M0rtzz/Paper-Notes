---
title: >-
  [论文解读] Blink: Dynamic Visual Token Resolution for Enhanced Multimodal Understanding
description: >-
  [CVPR 2026][图像恢复][视觉token分辨率] 提出 Blink 框架，通过在 MLLM 不同 Transformer 层动态扩展和丢弃视觉 token（模拟人类"快速眨眼"式扫描），在单次前向传播中自适应增强视觉感知能力，在多个多模态基准上提升 LLaVA-1.5 性能。
tags:
  - CVPR 2026
  - 图像恢复
  - 视觉token分辨率
  - 动态注意力
  - 多模态大语言模型
  - 显著性引导
  - token超分辨率
---

# Blink: Dynamic Visual Token Resolution for Enhanced Multimodal Understanding

**会议**: CVPR 2026  
**arXiv**: [2512.10548](https://arxiv.org/abs/2512.10548)  
**代码**: 无  
**领域**: 多模态大语言模型 / 图像修复（视觉感知增强）  
**关键词**: 视觉token分辨率, 动态注意力, 多模态大语言模型, 显著性引导, token超分辨率

## 一句话总结
提出 Blink 框架，通过在 MLLM 不同 Transformer 层动态扩展和丢弃视觉 token（模拟人类"快速眨眼"式扫描），在单次前向传播中自适应增强视觉感知能力，在多个多模态基准上提升 LLaVA-1.5 性能。

## 研究背景与动机
**领域现状**: 多模态大语言模型（MLLMs）在视觉-语言任务上取得显著进展（LLaVA、Qwen-VL 等），但视觉感知能力仍然不足，容易出现幻觉。

**现有痛点**: 现有 MLLMs 采用传统 LLM 架构处理视觉输入，缺乏对显著视觉区域的显式利用；后处理方法（如先识别显著区域再裁剪二次推理）效率低且只能聚焦单一区域。

**核心矛盾**: 人类通过"扫描-聚焦-转移"的动态过程感知视觉场景，但 MLLMs 对所有视觉 token 一视同仁，无法模拟跨层的注意力转移。

**本文要解决**: 如何在单次前向传播中动态增强 MLLM 的视觉感知能力？

**切入角度**: 先做 pilot study 发现两个关键洞察——(a) 不同层关注不同视觉区域，(b) 对高注意力 token 增加计算量可提升感知能力——然后据此设计动态框架。

**核心idea**: 利用注意力图的非均匀分布，在每层动态决定是否扩展（超分辨率增强）或丢弃视觉 token，模拟人类的"扫描-聚焦-转移"认知过程。

## 方法详解

### 整体框架
MLLM 正常前向传播 → 在选定层：计算显著性图 → 判断是否超过扩展/丢弃阈值 → 若超过则用 TokenSR 模块扩展显著区域的 token → 在后续层若注意力转移则丢弃扩展的 token → 最终输出。

### 关键设计

1. **Saliency-Guided Scanning（显著性引导扫描）**:

    - 在每个参与层 $L$，计算最后一个文本 token 对所有视觉 token 的注意力：
    $S_v^{(L)} = q_{t_n}^{(L)} (k_v^{(L)})^\top$
    - 将视觉 token 重塑为 $H \times W$ 网格，分成 $p \times p$ patch，计算每个 patch 的聚合显著性
    - 定义显著性比率：$\rho^{(L)} = \frac{\mathcal{S}_{r_{\max}}^{(L)}}{\sum_i \mathcal{S}_{r_i}^{(L)}}$，反映注意力的集中程度
    - **设计动机**: pilot study 发现不同层的注意力分布差异大，集中度高表示模型"确信"关注某区域，适合增强

2. **Dynamic Token Resolution（动态 Token 分辨率）**:

    - **Token 扩展**: 当 $\rho^{(L)} > \tau_{\text{exp}}$ 时，用 TokenSR 模块超分辨率增强显著 patch
    $hs_{SR}^{(L)} = \text{TokenSR}^{(L)}(hs_{LR}^{(L)})$
      然后将增强 token 插入序列: $[hs_s; hs_v; hs_{SR}; hs_t]$
    - **Token 丢弃**: 当 $\rho^{(L)} < \tau_{\text{drop}}$ 时，移除之前扩展的 token，恢复原始序列
    - **设计动机**: 扩展增加对显著区域的计算投入，丢弃防止低信息量的 token 干扰后续推理

3. **Token Super-Resolution Module (TokenSR)**:

    - 由三层 2D 卷积 + ReLU 组成的轻量模块
    - 训练时：将完整图像的显著区域 token 放大，用对应裁剪图像的 token 作为参考，最小化 KL 散度
    - MLLM 骨干冻结，只训练 TokenSR 参数
    - **设计动机**: 模拟图像超分辨率的思路，用轻量网络从低分辨率 token 恢复细节，保持语义一致性

### 损失函数 / 训练策略
- TokenSR 训练: 最小化增强 token 与裁剪参考 token 之间的 KL 散度
- 训练数据: LLaVA-1.5 训练集（COCO + GQA + OCR-VQA + TextVQA + VisualGenome）
- 所有操作在层归一化之前执行，确保 Transformer 正常处理扩展/剪裁的序列

## 实验关键数据

### 主实验（LLaVA-1.5-7B）

| 基准 | Vanilla | Blink-interp | Blink | 提升 |
|------|---------|-------------|-------|------|
| MME Perception | 1505.72 | 1514.08 | **1519.74** | +14.02 |
| MME Cognition | 357.86 | 353.21 | **361.79** | +3.93 |
| GQA | 61.93 | 61.93 | **61.98** | +0.05 |
| MMBench | 64.60 | **64.69** | **64.69** | +0.09 |
| MMBench-CN | 58.08 | 58.51 | **58.59** | +0.51 |
| POPE | 85.17 | 85.17 | **85.23** | +0.06 |
| ScienceQA | 69.46 | 69.51 | **69.66** | +0.20 |
| MM-Vet | 32.20 | 31.70 | **33.40** | +1.20 |

### 消融实验

| 配置 | MME Total | 变化 | 说明 |
|------|-----------|------|------|
| Blink 完整 | **1881.53** | — | 最优 |
| w/o SGS（随机选择） | 1879.38 | -2.15 | 显著性引导必要 |
| w/o DTR（固定周期） | 1840.46 | -41.07 | 动态分辨率调整至关重要 |
| w/o Drop | 1884.03 | +2.50 | 不丢弃在 Blink 下略有提升 |
| High $\tau_{\text{exp}}$ | 1865.54 | -15.99 | 过高阈值限制有效扩展 |

### 关键发现
- DTR 模块移除后性能下降最大（-41.07），是框架核心
- Blink-interp（无训练插值）也能提升 MME Perception 8.36 分，证明动态推理管线本身有价值
- 完整训练的 Blink 在所有基准上一致优于或持平于基线
- 层范围选择（12-18 层）对应 pilot study 发现的"正确注意力中间层"区间

## 亮点与洞察
- Pilot study 的两个发现（跨层注意力转移 + 增加显著 token 计算量有效）为方法设计提供了扎实的实证基础
- "动态扫描-聚焦"模拟人类视觉认知过程，思路优雅
- 即插即用设计——只需训练轻量 TokenSR 模块，骨干完全冻结
- Blink-interp 变体证明即使不训练，仅靠推理管线也有收益

## 局限与展望
- 绝对提升幅度不大（MME 总分 +17.95），但方向正确
- 仅在 LLaVA-1.5-7B 上验证，更大模型和更新架构待测试
- 阈值 $\tau_{\text{exp}}$ 和 $\tau_{\text{drop}}$ 需手动调节，可考虑自适应学习
- 当前每层只选择一个显著 patch，多显著区域的场景可能需要扩展

## 相关工作与启发
- 后处理放大方法（LLaVA-HR 等）需要多次前向传播，效率低
- 视觉 token 剪枝（FastV、LLaVA-PruMerge）是互补思路——Blink 是"增强重要的"而非"移除不重要的"
- 启示：MLLM 内部注意力分布包含丰富的视觉感知信号，值得进一步挖掘

## 评分
- 新颖性: ⭐⭐⭐⭐ 动态 token 分辨率调整的 idea 新颖，pilot study 提供了良好的动机
- 实验充分度: ⭐⭐⭐⭐ 7 个基准 + 详细消融 + 可视化分析
- 写作质量: ⭐⭐⭐⭐ 从发现到方法的逻辑链清晰
- 价值: ⭐⭐⭐⭐ 为增强 MLLM 视觉感知提供了新思路，即插即用

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Beyond the Ground Truth: Enhanced Supervision for Image Restoration](beyond_the_ground_truth_enhanced_supervision_for_image_restoration.md)
- [\[CVPR 2026\] ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration](shiftlut_spatial_shift_enhanced_look-up_tables_for_efficient_image_restoration.md)
- [\[CVPR 2026\] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](polishing_the_sky_wide-field_and_high-dynamic_range_interferometric_image_recons.md)
- [\[ECCV 2024\] Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution](../../ECCV2024/image_restoration/contourlet_residual_for_prompt_learning_enhanced_infrared_image_super-resolution.md)
- [\[AAAI 2026\] TMDC: A Two-Stage Modality Denoising and Complementation Framework for Multimodal Sentiment Analysis](../../AAAI2026/image_restoration/tmdc_a_two-stage_modality_denoising_and_complementation_framework_for_multimodal.md)

<!-- RELATED:END -->
