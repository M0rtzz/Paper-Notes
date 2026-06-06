---
title: >-
  [论文解读] Revisiting Model Stitching in the Foundation Model Era
description: >-
  [CVPR 2026][多模态VLM][模型拼接] 本文系统研究视觉基础模型（VFM）之间的拼接可行性，发现传统方法在VFM上失效，提出"Final Feature Matching + Task Loss"两阶段训练策略使异构VFM可靠拼接，拼接模型甚至能超越两个单独VFM…
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "模型拼接"
  - "视觉基础模型"
  - "表示相容性"
  - "VFM Stitch Tree"
  - "多模态LLM"
---

# Revisiting Model Stitching in the Foundation Model Era

**会议**: CVPR 2026  
**arXiv**: [2603.12433](https://arxiv.org/abs/2603.12433)  
**代码**: 无  
**领域**: 多模态VLM / 模型融合  
**关键词**: 模型拼接, 视觉基础模型, 表示相容性, VFM Stitch Tree, 多模态LLM

## 一句话总结
本文系统研究视觉基础模型（VFM）之间的拼接可行性，发现传统方法在VFM上失效，提出"Final Feature Matching + Task Loss"两阶段训练策略使异构VFM可靠拼接，拼接模型甚至能超越两个单独VFM，进而提出VFM Stitch Tree（VST）架构为多VFM系统提供可控的精度-效率权衡方案。

## 研究背景与动机

1. **领域现状**：视觉基础模型（如CLIP、DINOv2、SigLIP 2）在不同目标函数、数据集和模态组合下预训练，已成为各类下游任务的默认backbone。多模态系统（如MoF-LLaVA、Cambrian-1）越来越多地同时使用多个VFM以捕获互补视觉信息。
2. **现有痛点**：
    - 模型拼接（model stitching）作为测量表示兼容性的探针工具，已有研究表明同数据集训练的小模型（如ResNet-18 on CIFAR-10）可以拼接，但对异构VFM是否可拼接是未知的。
    - 传统拼接训练方法（Layer Feature Matching和Task Loss Training）在VFM上失效——前者在浅层拼接时中间层匹配误差会累积放大导致最终特征偏差大，后者在浅层拼接时梯度需穿过长链冻结层导致优化困难。
    - 使用多个VFM带来线性计算/内存开销（k个VFM就是k倍），缺乏高效的共享机制。
3. **核心矛盾**：VFM在预训练数据（LAION vs LVD-142M vs WebLI）、目标函数（对比学习 vs 自监督重建）、模态组合（纯视觉 vs 视觉-语言）上差异巨大，直接用简单变换桥接它们的中间表示是不够的。
4. **本文要解决什么？** ①探清异构VFM是否可拼接；②找到可靠的拼接训练方法；③将拼接从诊断工具升级为实用的VFM融合方案。
5. **切入角度**：系统分析拼接失败的原因（中间匹配≠最终对齐、梯度衰减），提出对症下药的两阶段方法。
6. **核心idea一句话**：用Final Feature Matching在目标VFM的倒数第二层对齐特征作为初始化，再用Task Loss微调，使异构VFM变得可靠可拼接且能融合互补知识。

## 方法详解

### 整体框架
给定源VFM $f_\theta$ 和目标VFM $f_\phi$（均为$N$层Transformer），在第$n$层拼接：保留源模型前$n$层 $R_\theta^n$ 和目标模型后$N-n$层 $T_\phi^N$，中间插入可训练拼接层$S$。拼接模型 $F(x) = T_\phi^N \circ S \circ R_\theta^n(x)$，仅$S$可训练，源和目标层全部冻结。

### 关键设计

1. **Final Feature Matching（FFM）**

    - 功能：为拼接层提供高质量初始化，确保最终输出特征与目标VFM对齐。
    - 核心思路：不在拼接点$n$处匹配中间特征，而是直接最小化经过拼接后在最终层$N$处的特征差异：$\mathcal{L}_{FFM} = \frac{1}{M}\sum_{i=1}^M \|T_\phi^N(S(R_\theta^n(x_i))) - T_\phi^N(R_\phi^n(x_i))\|_2^2$。虽然优化目标在最终层，但实验发现FFM同时也隐式地在中间层保持了低特征距离，且最终特征距离远小于Layer Feature Matching方法。
    - 设计动机：Layer Feature Matching虽然在拼接点处误差极小（$10^{-3}$量级），但这个小误差经过后续冻结层时会被累积放大，导致最终特征严重偏移（尤其浅层拼接）。FFM直接优化最终结果，从根源解决这个问题。且FFM无需标签，可以纯无监督方式训练。

2. **两阶段训练方案（FFM + Task Loss Training）**

    - 功能：先用FFM建立良好的loss landscape初始化，再用任务损失微调实现下游性能最大化。
    - 核心思路：Stage 1用FFM预训练拼接层（无标签），Stage 2用下游任务loss（如交叉熵分类）微调拼接层（有标签）。这个流程特别解决了Task Loss Training在浅层拼接时的优化困难——随机初始化+弱监督信号（从pooled token来的梯度穿过长冻结链）导致loss landscape条件差。FFM initialization将拼接层放到好的起点。
    - 设计动机：直接用Task Loss Training在浅层拼接DINOv2→SigLIP2时仅25.1%精度，远低于两个模型各自的linear probing（46.7%和53.5%）。FFM初始化后提升到51.7%，FFM+TLT进一步到55.8%（Layer 6）。

3. **Self-Stitch基线（严格控制实验）**

    - 功能：区分拼接增益是来自拼接层容量还是真正的VFM知识融合。
    - 核心思路：在同一个VFM内部自拼接（如SigLIP2→SigLIP2），使用相同的拼接层、拼接点、训练损失和下游数据。如果跨VFM拼接超越自拼接，说明增益来自真正的互补知识融合而非额外参数/微调带来的容量增加。
    - 设计动机：VFM在大规模异构数据上预训练后在下游数据上评估，改善可能只来自拼接层对下游数据的适配（相当于额外的微调参数），Self-Stitch基线排除了这种解释。实验证实跨VFM拼接一致性超越自拼接（+2.3%到+2.6%），确认了真正的互补融合。

### 损失函数 / 训练策略
- Stage 1: FFM loss（无标签数据），可以预提取源和目标特征加速训练
- Stage 2: 下游任务交叉熵loss（有标签数据）
- 拼接层：默认使用2层MLP with ReLU（同LLaVA-1.5的特征投影器）
- 评估VFM对：DINOv2-L, SigLIP2-L, CLIP, DINOv3（均24层Transformer）
- 拼接点：$n \in [2, 6, 10, 14, 18, 22]$

## 实验关键数据

### 主实验：两阶段方法 vs 原始Task Loss Training

| 拼接 | 初始化 | L2 | L6 | L10 | L14 | L18 | L22 |
|------|--------|-----|-----|------|------|------|------|
| DINOv2→SigLIP2 | 无 | 25.1 | 39.4 | 52.6 | 62.3 | 68.6 | 68.6 |
| DINOv2→SigLIP2 | FFM | **51.7** | **55.8** | **59.3** | **68.0** | **72.0** | **71.8** |
| SigLIP2→DINOv2 | 无 | 38.7 | 56.7 | 58.3 | 64.4 | 70.4 | 70.1 |
| SigLIP2→DINOv2 | FFM | **53.8** | **53.8** | **61.9** | **69.6** | **70.4** | **72.2** |

### 跨数据集/任务一致性验证

| 配置 | fMoW(L6/14/22) | iNaturalist(L6/14/22) | Aircraft(L6/14/22) | ADE20K seg(L14/22) |
|------|------------|-----------|----------|---------|
| DINOv2→DINOv2(自拼接) | 41.5/59.7/69.9 | 56.9/81.5/91.2 | 37.8/79.3/91.2 | 35.4/50.9 |
| SigLIP2→SigLIP2(自拼接) | 50.5/62.0/68.9 | 71.2/88.5/87.3 | 67.9/88.1/89.3 | 44.5/50.5 |
| **DINOv2→SigLIP2** | **55.8/68.0/71.8** | **75.9/89.1/92.8** | **77.8/87.6/92.4** | **44.9/51.2** |
| **SigLIP2→DINOv2** | **53.8/69.6/72.2** | **86.3/88.9/91.9** | **80.7/89.0/91.0** | **49.0/51.4** |

### 消融：拼接层类型

| 拼接层 | L2 | L6 | L10 | L14 | L18 | L22 |
|--------|-----|-----|------|------|------|------|
| Linear | 26.1/50.3 | 54.3/56.4 | 59.5/60.0 | 66.5/65.7 | 69.1/69.6 | 69.6/71.9 |
| **MLP** | **51.7/53.8** | **55.8/53.8** | **59.3/61.9** | **68.0/69.6** | **72.0/70.4** | **71.8/72.2** |
| LoRA | 49.1/48.3 | 49.4/56.2 | 57.4/62.4 | 61.7/65.3 | 67.7/66.2 | 67.3/65.0 |

### 关键发现
- FFM初始化对浅层拼接效果最显著（L2: 25.1→51.7），在深层拼接也有稳定增益（L22: 68.6→71.8）。
- 跨VFM拼接一致性超越自拼接（+0.7%到+5.5%），在分类和语义分割上都成立，确认了真正的互补知识融合。
- MLP拼接层整体最优，LoRA虽然表达力更强但反而不如MLP——可能因为适度的mismatch有助于互补信息融合。
- CLIP作为源模型时拼接效果差（弱编码器丢失了任务关键信息），但作为目标模型时效果好，类似encoder-decoder架构中升级encoder的效果。
- VST-22仅用4.3%额外资源即可获得双VFM45%的性能增益，VST-14用39%额外资源获得84%增益。

## 亮点与洞察
- **"FFM同时实现了隐式局部对齐"**这个意外发现非常有insight：虽然只在最终层匹配，但监督信号可以隐式传导到中间层促进局部对齐，说明深层的匹配可以有效约束浅层的表示。
- **Self-Stitch基线**的实验设计非常严谨，彻底排除了"只是多了点参数"的替代解释，是负责任的实验方法论典范。
- **VFM Stitch Tree的accuracy-latency旋钮**思想非常实用：不再是"用不用第二个VFM"的二选一，而是可以在4.3%-100%额外开销之间连续调节，适合不同部署预算。
- 将Model Stitching从纯诊断工具升级为实用融合方案，是一个有意义的范式转变。

## 局限性 / 可改进方向
- VST的评估仅在VQAv2和MME上进行了初步验证（称为"early exploration"），后续需扩展到更多多模态benchmark（如SEED-Bench、MMVet等）以全面衡量融合增益。
- 当前仅测试了ViT-L规模的VFM，对更大规模（如ViT-G）或不同架构的VFM的拼接性待验证。
- 拼接层训练需要无标签数据上的VFM前向推理（FFM阶段），对于非常大的VFM可能计算成本不低。
- 可以探索自适应拼接点选择（而非手动选择哪层拼接），以及多于两个VFM的拼接树设计。
- FFM loss is label-free但仍需在下游domain的数据上训练，zero-shot场景下效果未知。

## 相关工作与启发
- **vs SN-Net [35]**: SN-Net在训练时显式设计可拼接性做模型压缩，本文是post-hoc地拼接独立训练的异构VFM，场景完全不同。
- **vs [2] (Bansal et al.)**: 原始stitching工作在同数据集同架构下发现可拼接性（Anna Karenina假说），本文将其扩展到异构数据/目标/模态的VFM，发现naive方法失败但定制方法可行。
- **vs [7] (Collins et al.)**: 该工作argue TLT优于LFM，本文在VFM上发现两者都有问题，FFM是更好的替代方案。

## 评分
- 新颖性: ⭐⭐⭐⭐ FFM和两阶段方案虽然简洁但有效，VST应用有新意，但整体属于careful engineering
- 实验充分度: ⭐⭐⭐⭐⭐ 多VFM对、多数据集、多任务、多拼接层类型的系统验证，Self-Stitch控制实验设计精巧
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑推导清晰，从诊断到处方再到应用层层推进，示范性的研究论文写法
- 价值: ⭐⭐⭐⭐ 对理解VFM表示兼容性有重要贡献，VST为多VFM部署提供了实用方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] VL-RouterBench: A Benchmark for Vision-Language Model Routing](vl-routerbench_a_benchmark_for_vision-language_model_routing.md)
- [\[CVPR 2026\] HouseMind: Tokenization Allows MLLMs to Understand, Generate and Edit Architectural Floor Plans](housemind_tokenization_mllm_floor_plan.md)
- [\[CVPR 2026\] MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](masquant_modality-aware_smoothing_quantization_for_multimodal_large_language_mod.md)
- [\[CVPR 2026\] Medic-AD: Towards Medical Vision-Language Model's Clinical Intelligence](medic-ad_towards_medical_vision-language_models_clinical_intelligence.md)

</div>

<!-- RELATED:END -->
