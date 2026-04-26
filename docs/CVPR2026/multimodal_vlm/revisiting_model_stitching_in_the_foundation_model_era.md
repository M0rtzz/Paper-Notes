---
title: >-
  [论文解读] Revisiting Model Stitching In the Foundation Model Era
description: >-
  [CVPR2026][多模态][模型拼接] 提出针对异构视觉基础模型(VFM)的两阶段拼接训练方法(Final Feature Matching + Task Loss Training)，证明异构VFM可以可靠拼接且融合互补知识，并设计VFM Stitch Tree (VST)架构实现多VFM系统的可控精度-效率权衡。
tags:
  - CVPR2026
  - 多模态
  - 模型拼接
  - 视觉基础模型
  - 表征兼容性
  - 特征匹配
  - 多VFM融合
  - 效率优化
---

# Revisiting Model Stitching In the Foundation Model Era

**会议**: CVPR2026  
**arXiv**: [2603.12433](https://arxiv.org/abs/2603.12433)  
**代码**: 待确认  
**领域**: 多模态VLM  
**关键词**: 模型拼接, 视觉基础模型, 表征兼容性, 特征匹配, 多VFM融合, 效率优化

## 一句话总结

提出针对异构视觉基础模型(VFM)的两阶段拼接训练方法(Final Feature Matching + Task Loss Training)，证明异构VFM可以可靠拼接且融合互补知识，并设计VFM Stitch Tree (VST)架构实现多VFM系统的可控精度-效率权衡。

## 背景与动机

1. **VFM百花齐放但表征兼容性未知**：当前CLIP、DINOv2、SigLIP 2等VFM在训练目标（对比学习 vs 自监督重建）、数据（LAION vs WebLI vs LVD-142M）和模态组合（视觉-语言 vs 纯视觉）上差异巨大，但它们的内部表征是否兼容尚不清楚。

2. **模型拼接作为表征兼容性探针**：先前工作表明同数据集训练的小模型（如ResNet-18在CIFAR-10上）即使初始化/目标不同仍可拼接（精度几乎无损），但这一结论能否推广到异构VFM尚未验证。

3. **现有拼接策略在VFM上失效**：传统的Layer Feature Matching（在拼接点匹配特征）和Task Loss Training（直接优化下游损失）在VFM场景下表现糟糕，尤其在浅层拼接点。

4. **浅层拼接的梯度传播困境**：当拼接点较浅时，目标模型后续所有层被冻结，梯度需穿过长链冻结层才能更新拼接层，导致优化困难。

5. **多VFM系统的效率瓶颈**：现代多模态LLM（如MoF-LLaVA、Cambrian-1）部署多个VFM以捕获互补视觉信息，但带来线性增长的计算和显存开销（k个VFM = k倍成本）。

6. **从诊断工具到实用方案的需求**：需要将模型拼接从纯粹的表征分析工具提升为融合互补VFM优势的实用方案。

## 方法详解

### 整体框架

给定源模型 $f_\theta$ 和目标模型 $f_\phi$，在第 $n$ 层拼接：冻结源模型前 $n$ 层和目标模型后 $N-n$ 层，仅训练一个轻量拼接层 $S$。拼接模型表示为 $F(x) = T_\phi^N \circ S \circ R_\theta^n(x)$。

### 核心设计：两阶段训练

**阶段一：Final Feature Matching (FFM)**

- 不在拼接点匹配中间特征，而是匹配目标模型**最终层输出特征**
- 损失函数：$\mathcal{L}_{FFM} = \frac{1}{M}\sum_{i=1}^{M}\|T_\phi^N(S(R_\theta^n(x_i))) - T_\phi^N(R_\phi^n(x_i))\|_2^2$
- 关键发现：FFM虽然直接匹配最终特征，但同时也隐式保持了拼接点的局部对齐（层特征距离与LFM相当）
- 此阶段**无需标签**，仅需无标注图像

**阶段二：Task Loss Training (TLT)**

- 用阶段一的FFM初始化拼接层参数，避免随机初始化导致的优化困难
- 在下游任务上微调：$\mathcal{L}_{task} = \frac{1}{M}\sum_{i=1}^{M}\ell(F(x_i), y_i)$
- FFM初始化将拼接层放置在更好的损失景观中，后续微调将良好初始化转化为强拼接精度

### 拼接层设计

- **Linear**：按token独立处理，表达能力最弱
- **MLP**（默认）：两层感知机+ReLU，表现最佳
- **LoRA**：在源模型第 $n$ 层加LoRA，允许token间交互，表达能力最强但反而不如MLP——适度的不匹配有助于互补信息融合

### VFM Stitch Tree (VST)

- **核心思想**：多个VFM共享浅层计算，仅在深层保留各自特化层，通过拼接层连接
- **架构**：树形结构——主干为一个VFM的浅层，分支为各VFM的深层
- 以Cambrian-1（4个VFM）为例，在第14层拼接可减少 54% GPU显存和计算量

## 实验关键数据

### 两阶段训练 vs 朴素TLT（fMoW分类，准确率%）

| 方向 | 预训练 | L2 | L6 | L10 | L14 | L18 | L22 |
|------|--------|------|------|------|------|------|------|
| DINOv2→SigLIP2 | 无 | 25.1 | 39.4 | 52.6 | 62.3 | 68.6 | 68.6 |
| DINOv2→SigLIP2 | FFM | **51.7** | **55.8** | **59.3** | **68.0** | **72.0** | **71.8** |
| SigLIP2→DINOv2 | 无 | 38.7 | 56.7 | 58.3 | 64.4 | 70.4 | 70.1 |
| SigLIP2→DINOv2 | FFM | **53.8** | **53.8** | **61.9** | **69.6** | **70.4** | **72.2** |

FFM初始化在浅层拼接点（L2）带来高达 **+26.6%** 的提升，深层也有稳定增益。

### 跨数据集/任务一致性（分类准确率%/分割mIoU%）

| 方向 | fMoW (L6/14/22) | iNaturalist (L6/14/22) | Aircraft (L6/14/22) | ADE20K (L14/22) |
|------|-----------------|------------------------|---------------------|-----------------|
| Self-Stitch DINOv2 | 41.5/59.7/69.9 | 56.9/81.5/91.2 | 37.8/79.3/91.2 | 35.4/50.9 |
| Self-Stitch SigLIP2 | 50.5/62.0/68.9 | 71.2/88.5/87.3 | 67.9/88.1/89.3 | 44.5/50.5 |
| DINOv2→SigLIP2 | **55.8/68.0/71.8** | **75.9/89.1/92.8** | **77.8/87.6/92.4** | **44.9/51.2** |
| SigLIP2→DINOv2 | **53.8/69.6/72.2** | **86.3/88.9/91.9** | **80.7/89.0/91.0** | **49.0/51.4** |

跨模型拼接**一致性地超越自拼接基线**，分类提升 +0.7%~+5.5%，分割提升 +0.5~+0.7 mIoU。

### VST性能-效率权衡（MoF-LLaVA）

| 配置 | 额外资源 | 增益恢复比 |
|------|----------|------------|
| 单VFM baseline | 0% | 0% |
| VST-22（1层特化） | 4.3% | 45% |
| VST-14（9层特化） | 39% | 84% |
| 完整双VFM | 100% | 100% |

### 消融：拼接层类型对比（fMoW准确率%）

| 方法 | L2 | L6 | L10 | L14 | L18 | L22 |
|------|------|------|------|------|------|------|
| D→S Linear | 26.1 | 54.3 | 59.5 | 66.5 | 69.1 | 69.6 |
| D→S MLP | **51.7** | **55.8** | **59.3** | **68.0** | **72.0** | **71.8** |
| D→S LoRA | 49.1 | 49.4 | 57.4 | 61.7 | 67.7 | 67.3 |

MLP在所有拼接点上一致优于Linear和LoRA，LoRA尽管表达能力更强反而次于MLP。

## 亮点

- **问题洞察深刻**：精准分析了LFM在浅层失效的原因（小误差经冻结层累积放大）和TLT在浅层的梯度传播困难，FFM方案简洁有效
- **自拼接基线设计巧妙**：通过Self-Stitch控制实验排除了"拼接层容量带来的增益"这一混淆因素，证明互补知识融合是真实存在的
- **从理论到应用的完整闭环**：从表征分析出发，发现VFM可拼接性，进而设计VST架构解决实际多VFM系统的效率问题
- **实验全面系统**：覆盖4种VFM、4个数据集、2类任务（分类+分割）、3种拼接层、6个拼接深度

## 局限与展望

- **弱源模型限制**：当CLIP作为源模型时拼接效果不佳，说明弱编码器可能丢失关键信息导致目标模型无法恢复
- **VST评估有限**：仅在LLaVA框架和少量VQA基准上验证，未覆盖更多MLLM架构和更广泛的基准测试
- **拼接层设计空间未充分探索**：仅尝试Linear/MLP/LoRA三种，更复杂的跨注意力等机制未探索
- **仅限ViT架构**：所有VFM均为Transformer架构，CNN或混合架构的可拼接性未验证
- **静态拼接点**：拼接点固定，未探索输入自适应的动态拼接策略

## 与相关工作的对比

- **vs Bansal et al. (2021)**：先前仅在同数据集小模型上验证拼接性，本文首次系统扩展到异构VFM并发现朴素方法失效
- **vs Collins et al. (2025)**：该工作指出TLT可能创建分布外表征，本文FFM初始化正好缓解此问题——先保持表征保真度再做任务适配
- **vs Smith et al. (2025)**：该工作质疑拼接成功仅反映表征聚类而非语义相似性，本文自拼接控制实验直接回应了这一质疑
- **vs SN-Net (Pan et al.)**：SN-Net关注同家族模型跨尺度的可拼接性训练，本文关注独立训练的异构VFM的事后拼接

## 评分

- 新颖性: ⭐⭐⭐⭐ — FFM训练策略和VST架构有新意，但核心思想基于已有模型拼接框架的改进
- 实验充分度: ⭐⭐⭐⭐⭐ — 覆盖多VFM、多数据集、多任务、多拼接层、多拼接深度，控制实验设计优秀
- 写作质量: ⭐⭐⭐⭐⭐ — 逻辑清晰，问题-分析-方案-验证的叙事流畅，自拼接基线的引入令人信服
- 价值: ⭐⭐⭐⭐ — VST为多VFM系统效率优化提供了实用方案，但弱源模型的限制和有限的MLLM评估稍减实际影响力

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)
- [\[CVPR 2026\] AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](anomalyvfm_--_transforming_vision_foundation_models_into_zero-shot_anomaly_detec.md)
- [\[CVPR 2026\] World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training](rehearsevla_simulated_post-training_for_vlas_with_physically-consistent_world_mo.md)
- [\[CVPR 2026\] Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks](do_vision-language_models_leak_what_they_learn_adaptive_token-weighted_model_inv.md)
- [\[CVPR 2026\] Revisiting Multimodal KV Cache Compression: A Frequency-Domain-Guided Outlier-KV-Aware Approach](revisiting_multimodal_kv_cache_compression_a_frequency-domain-guided_outlier-kv-.md)

<!-- RELATED:END -->
