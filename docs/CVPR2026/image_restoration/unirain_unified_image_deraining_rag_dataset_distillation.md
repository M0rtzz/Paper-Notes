---
title: >-
  [论文解读] UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization
description: >-
  [CVPR 2026][图像恢复][统一去雨] 提出 UniRain 统一图像去雨框架，通过 RAG 驱动的数据蒸馏从百万级公开数据集筛选高质量样本，结合非对称 MoE 架构和多目标重加权优化策略，在雨条纹和雨滴（白天/夜间）四种退化类型上实现一致优异性能。
tags:
  - "CVPR 2026"
  - "图像恢复"
  - "统一去雨"
  - "RAG数据蒸馏"
  - "多目标优化"
  - "混合专家"
  - "白天夜间"
---

# UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization

**会议**: CVPR 2026  
**arXiv**: [2603.03967](https://arxiv.org/abs/2603.03967)  
**代码**: [https://github.com/QianfengY/UniRain](https://github.com/QianfengY/UniRain)  
**领域**: 图像修复 / 图像去雨  
**关键词**: 统一去雨, RAG数据蒸馏, 多目标优化, 混合专家, 白天夜间

## 一句话总结

提出 UniRain 统一图像去雨框架，通过 RAG 驱动的数据蒸馏从百万级公开数据集筛选高质量样本，结合非对称 MoE 架构和多目标重加权优化策略，在雨条纹和雨滴（白天/夜间）四种退化类型上实现一致优异性能。

## 研究背景与动机

1. **领域现状**：现有去雨方法通常针对特定退化类型（雨条纹、雨滴、夜间雨等），在其他类型上性能显著下降。
2. **现有痛点**：直接合并所有公开数据集（>200 万对）会引入数据质量不均的问题——部分数据集背景质量差、合成不真实，干扰模型训练。同一优化目标训练不同退化类型导致学习不平衡。
3. **核心矛盾**：简单增大数据量不等于更好的泛化性。不同退化类型难度不同，统一训练时模型倾向过拟合容易的类型（如夜间雨条纹）而忽略困难类型（如白天雨滴）。
4. **本文目标**：统一处理四种雨退化类型的高质量去雨模型。
5. **切入角度**：数据端用 RAG 蒸馏筛选可靠样本，模型端用非对称 MoE 和多目标优化平衡不同类型。
6. **核心 idea**：数据质量比数据量更重要；不同退化类型需要动态平衡的优化策略。

## 方法详解

### 整体框架

两大模块：(1) RAG 数据蒸馏管道：从百万级数据中筛选 52,869 对高质量训练样本（仅 2.6%）；(2) 非对称 MoE + 多目标重加权优化的统一去雨模型。

### 关键设计

1. **RAG 数据蒸馏管道**:

    - 功能：从大规模公开数据集中筛选高质量训练样本
    - 核心思路：检索阶段：构建真实雨图数据库（BLIP 生成文本 + CLIP 提取视觉特征）。对每张候选图片，进行三级相似度匹配：语义相似度（CLIP 文本编码器 L2 距离） → 视觉相似度（CLIP 特征余弦相似度） → 结构相似度（SSIM）。生成阶段：将检索到的真实参考图和候选图输入 VLM 评估质量，三个 VLM 投票决策。
    - 设计动机：利用真实雨图作为参考基准评估合成数据质量，比无参考评估更可靠。最终仅保留 2.6% 的数据。

2. **多目标重加权优化策略**:

    - 功能：动态平衡不同退化类型的学习速度
    - 核心思路：三个指标协同：(1) TBS（类型平衡分数）：对收敛快的类型降权，慢的类型升权（基于损失斜率）；(2) TSS（类型稳定性分数）：惩罚发散的类型避免不稳定；(3) AF（自适应因子）：训练早期 TBS 主导（促进平衡），后期 TSS 主导（保证稳定）。最终权重 $\omega_i(t) = \text{AF}(t) \cdot \text{TBS}(t) + (1-\text{AF}(t)) \cdot \text{TSS}(t)$。
    - 设计动机：简单固定权重无法适应训练过程中不同类型收敛速度的动态变化。

3. **非对称 MoE 架构**:

    - 功能：编码器和解码器使用不同的 MoE 策略以适应不同角色
    - 核心思路：编码器用 Soft-MoE（所有专家连续加权组合）保留多样退化线索；解码器用 Hard-MoE（Top-k 路由选择性激活）增强精细纹理重建。
    - 设计动机：编码器需要广泛探索不同退化模式，解码器需要精确重建细节——角色不同要求不同的专家选择策略。

### 损失函数 / 训练策略

4 × RTX 4090，AdamW，128×128 crop，batch size 8，30 万次迭代。

## 实验关键数据

### 主实验

| 数据集/类型 | 指标 | UniRain | MSDT (之前SOTA) | 提升 |
|------------|------|---------|----------------|------|
| RainRAG 平均 | PSNR | 28.93 | 27.94 | +0.99 |
| RealRain-1k-H | PSNR | 33.74 | 30.91 | +2.83 |
| RainDS-real-RD | PSNR | 22.07 | 20.72 | +1.35 |
| WeatherBench | PSNR | 34.25 | 33.56 | +0.69 |

### 消融实验

| 配置 | PSNR | SSIM | 说明 |
|------|------|------|------|
| 仅 VLM (无 RAG) | 27.73 | 0.8358 | 缺少真实参考 |
| 无生成阶段 | 28.36 | 0.8425 | 不做 VLM 质量评估 |
| 完整管道 | 28.93 | 0.8515 | RAG 数据蒸馏完整有效 |
| Soft-MoE 编码+解码 | 27.91 | 0.8465 | 纯 soft 不足 |
| 非对称 MoE | 28.93 | 0.8515 | 最优组合 |

### 关键发现

- 直接合并所有数据训练反而不如蒸馏后的 2.6% 数据
- 蒸馏数据集的特征分布更宽更多样
- 多目标优化使四种类型的损失曲线收敛更稳定
- 模型可扩展到全天候修复（雨+雪+雾），PSNR 26.01 超越 TransWeather 24.70

## 亮点与洞察

- **"少即是多"的数据哲学**：仅用 2.6% 的数据就超越使用全部数据的训练效果
- **RAG 在低级视觉的首次应用**：将 RAG 技术创新性地用于数据集蒸馏而非模型推理
- **三指标协同的动态优化**：TBS+TSS+AF 的设计兼顾了类型平衡和训练稳定性

## 局限与展望

- RAG 管道依赖 VLM 的评估质量，VLM 本身可能存在偏差
- 非对称 MoE 中专家数量和 Top-k 值需要手动调优
- 模型复杂度（FLOPs 126.5G）虽低于部分方法但仍不算轻量

## 相关工作与启发

- **vs URIR**: URIR 是首个统一去雨网络但仅在驾驶场景验证，UniRain 更通用
- **vs NeRD-Rain**: NeRD-Rain 用隐式神经表征做去雨但未做统一多类型训练

## 评分

- 新颖性: ⭐⭐⭐⭐ RAG 数据蒸馏和多目标优化的组合新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集+多场景+全面消融+天候扩展
- 写作质量: ⭐⭐⭐⭐ 动机图示清晰，消融系统化
- 价值: ⭐⭐⭐⭐ 统一去雨的实用框架，数据蒸馏思路可广泛迁移

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] EVLF: Early Vision-Language Fusion for Generative Dataset Distillation](evlf_early_vision-language_fusion_for_generative_dataset_distillation.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](real_iisr_infrared_image_super_resolution_autoregressive.md)
- [\[CVPR 2026\] UniBlendNet: Unified Global, Multi-Scale, and Region-Adaptive Modeling for Ambient Lighting Normalization](uniblendnet_unified_global_multi_scale_and_region_adaptive_modeling_for_ambient_lighting_normalization.md)
- [\[NeurIPS 2025\] MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization](../../NeurIPS2025/image_restoration/mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)
- [\[CVPR 2026\] RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration](rar_restore_assess_repeat_a_unified_framework_for_iterative_image_restoration.md)

</div>

<!-- RELATED:END -->
