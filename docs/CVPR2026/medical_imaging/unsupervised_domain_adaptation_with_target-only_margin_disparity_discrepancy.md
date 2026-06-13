---
title: >-
  [论文解读] Unsupervised Domain Adaptation with Target-Only Margin Disparity Discrepancy
description: >-
  [CVPR 2026][医学图像][无监督域自适应] 针对 CT→CBCT 肝脏分割的无监督域自适应问题，发现经典 MDD 优化目标中存在矛盾项（源域上特征提取器被优化为最大化 $f$ 和 $f'$ 的差异），提出 Target-Only MDD 改进，去除矛盾项并在两域上统一最小化预测差异，在 2D 和 3D 实验中均取得 UDA SOTA。
tags:
  - "CVPR 2026"
  - "医学图像"
  - "无监督域自适应"
  - "Margin Disparity Discrepancy"
  - "CBCT"
  - "肝脏分割"
  - "介入影像"
---

# Unsupervised Domain Adaptation with Target-Only Margin Disparity Discrepancy

**会议**: CVPR 2026  
**arXiv**: [2603.09932](https://arxiv.org/abs/2603.09932)  
**领域**: 医学图像  
**关键词**: 无监督域自适应, Margin Disparity Discrepancy, CBCT, 肝脏分割, 介入影像  
**arXiv**: [2603.09932](https://arxiv.org/abs/2603.09932)  
**代码**: 无  
**领域**: 医学图像  
**关键词**: 无监督域自适应, Margin Disparity Discrepancy, CBCT, 肝脏分割, 介入影像  

## 一句话总结

针对 CT→CBCT 肝脏分割的无监督域自适应问题，发现经典 MDD 优化目标中存在矛盾项（源域上特征提取器被优化为最大化 $f$ 和 $f'$ 的差异），提出 Target-Only MDD 改进，去除矛盾项并在两域上统一最小化预测差异，在 2D 和 3D 实验中均取得 UDA SOTA。

## 研究背景与动机

- **临床背景**：介入放射中 CBCT 可提供术中实时三维引导，自动肝脏分割对手术规划至关重要
- **数据困境**：CT 有大量公开标注数据集，但介入 CBCT 数据稀缺且无标注
- **域差异来源**：
  1. CBCT 的散射伪影、有限动态范围、重建几何差异
  2. 动脉内造影增强导致肝脏内出现高亮区域
  3. CBCT 视野有限，与 CT 视野不同
- **现有方法局限**：
    - 基础模型（SAM-MED 2D/3D、MA-SAM）：主要在自然图像上训练，泛化到 CBCT 效果有限
    - 图像对齐方法（SIFA）：依赖相同视野假设，不适用于 CT/CBCT
    - 自训练方法（BDCL）：在大域偏移下伪标签质量差
- **MDD 的问题**：经典 MDD 在实际优化中，特征提取器 $\psi$ 在源域上被优化为**最大化** $f$ 和 $f'$ 的差异（公式 3 红色框项），这与域对齐目标矛盾

## 方法详解

### 整体框架

这篇论文做 CT→CBCT 肝脏分割的无监督域自适应：CT 有大量公开标注、CBCT 稀缺又无标注，目标是把 CT 上学到的分割能力迁到 CBCT。它沿用 MDD 的框架，把 U-Net 拆成特征提取器 $\psi$ + 分割头 $f$ + 对抗分割头 $f'$，但核心贡献不是堆模块，而是发现经典 MDD 的优化目标里藏着一个和域对齐相悖的矛盾项，去掉它、重写成三步交替优化，就把性能顶到了 SOTA。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["CT（有标注） + CBCT（无标注）"] --> B["特征提取器 ψ<br/>提取共享特征 z"]
    B --> C["分割头 f"]
    B --> D["对抗分割头 f'"]
    C --> E["步骤①更新 f<br/>源域学好分割任务"]
    D --> F["步骤②更新 f'<br/>源域模仿 f、目标域偏离 f"]
    E --> G["步骤③更新 ψ<br/>两域统一最小化 f/f' 差异<br/>修正矛盾项符号"]
    F --> G
    G -->|交替迭代| B
    G --> H["Few-shot：丢弃 f'<br/>少量 CBCT 标注微调"]
```

### 关键设计

**1. 诊断经典 MDD 的矛盾项：源域上特征被逼着放大分歧**

经典 MDD 的优化目标是

$$\min_{f,\psi} \max_{f'} \mathcal{L}^{\text{task}}(f(z^S), y^S) + \alpha \mathcal{L}_{CE}(f'(z^T), f(z^T)) - \gamma \mathcal{L}_{CE}(f'(z^S), f(z^S))$$

问题出在最后一项：实际优化里 $f$ 并不参与这个差异项，于是特征提取器 $\psi$ 在源域上被优化成**最大化** $f$ 和 $f'$ 的差异（符号为负）。这与“在两个域上对齐特征”的初衷直接矛盾——一边想拉近、一边却在源域把两个头推开。看清这一点是后面所有改动的出发点。

**2. Target-Only MDD：去掉矛盾项，改成三步交替优化**

把上面那个矛盾项的符号翻正，整个目标拆成三步轮流优化。先更新分割头 $f$，只在源域上学好任务：

$$\min_f L^{\text{task}}(f(z^S), y^S)$$

再更新对抗头 $f'$，让它在源域模仿 $f$、在目标域偏离 $f$：

$$\min_{f'} \left[ \mathcal{L}_{CE}(f'(z^S), f(z^S)) - \gamma \mathcal{L}_{CE}(f'(z^T), f(z^T)) \right]$$

最后更新特征提取器 $\psi$，关键变化是源域差异项符号从减变加，让 $\psi$ 在**两个域上都最小化** $f$ 和 $f'$ 的差异：

$$\min_\psi \left[ L^{\text{task}}(f(z^S), y^S) + \alpha \mathcal{L}_{CE}(f'(z^S), f(z^S)) + \gamma \mathcal{L}_{CE}(f'(z^T), f(z^T)) \right]$$

矛盾消除后，特征对齐方向在两域上一致、不再自相拆台，这也是它无标注就能超过 5-shot 目标域训练的根由。

**3. Few-shot 扩展：UDA 收敛后用极少标注再顶一截**

UDA 训练完后保留 $f \circ \psi$、丢掉对抗头 $f'$，再用少量目标域标注样本微调即可。因为特征已经在两域对齐，少量标注就能把性能推到接近全监督的水平（如 3D 上 5 例标注即达 90.9%）。

### 实现细节

- 骨干：U-Net，5 阶段，首阶段 64 通道
- 超参：$\alpha = 7.5 \times 10^{-2}$，$\gamma = 3 \times 10^{-1}$
- 数据：573 例 CBCT + 678 例 CT，患者级分割

## 实验关键数据

### 2D CT→CBCT 肝脏分割

| 类型 | 方法 | F1 (%) |
|------|------|--------|
| Source Only | U-Net | 54.1 |
| Foundation | SAM-MED 2D (5pt) | 67.7 |
| Self-Training | BDCL | 60.0 |
| Feature Align | DANN | 68.3 |
| Feature Align | MDD | 70.0 |
| **Feature Align** | **Ours** | **74.4** |
| Few-shot | Ours + 50 vol | 84.6 |
| 上界 | Target Only (100%) | 85.5 |

### 3D CT→CBCT 肝脏分割

| 类型 | 方法 | F1 (%) |
|------|------|--------|
| Source Only | U-Net | 80.1 |
| Foundation | SAM-MED 3D (5pt) | 65.3 |
| Foundation | MA-SAM | 61.8 |
| Image Align | SIFA | 64.7 |
| Feature Align | DANN | 84.6 |
| **Feature Align** | **Ours** | **86.6** |
| Few-shot | Ours + 5 vol | 90.9 |
| 上界 | Target Only (100%) | 93.7 |

### 关键发现

1. **UDA 无标注即超越 5-shot 目标域训练**：Ours (86.6%) > Target Only 5-vol (84.7%)
2. **Ours + 5 vol (90.9%) ≈ Target Only 20-vol (89.6%)**
3. **超参数鲁棒**：不同 $\alpha, \gamma$ 组合下性能稳定
4. **方差最低**：Ours 标准差 9.4%，远低于 MA-SAM (18.3%) 和 SAM-MED 3D (28.8%)

## 亮点与洞察

1. **理论驱动的方法改进**：通过分析 MDD 优化目标的矛盾项，提出简洁有效的修正，而非堆叠复杂模块
2. **对基础模型的冷静审视**：SAM-MED 在 CBCT 上表现远不及 UDA，说明医学域偏移问题仍需专门处理
3. **CBCT 特殊挑战的深入分析**：造影增强导致的高亮区域使模型欠分割肝脏，本方法通过 3D 上下文信息有效缓解
4. **Few-shot 扩展自然**：UDA 后少量标注即可达到接近全监督的性能

## 局限性

- 仅在肝脏分割单一任务上验证，未扩展到其他器官或分割任务
- 数据集为私有数据，无法复现
- 2D 实验中未对比常见的 CycleGAN 等图像翻译方法
- MDD 理论框架主要针对分类，在分割任务上的理论保证仍不完善

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐ |
| 实验 | ⭐⭐⭐⭐ |
| 写作 | ⭐⭐⭐⭐ |
| 价值 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [\[CVPR 2026\] Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](adaptation_of_weakly_supervised_localization_in_histopathology_by_debiasing_pred.md)
- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](apex_adaptive_visual_prompting.md)
- [\[CVPR 2026\] CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection](chips_efficient_clip_adaptation_via_curvature-aware_hybrid_influence-based_data_.md)

</div>

<!-- RELATED:END -->
