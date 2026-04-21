---
title: >-
  [论文解读] MoETTA: Test-Time Adaptation Under Mixed Distribution Shifts with MoE-LayerNorm
description: >-
  [AAAI 2026][test-time adaptation] 本文提出 MoETTA，一种将 LayerNorm 重参数化为多个结构解耦专家分支的测试时自适应框架，通过路由机制为不同域的样本选择不同的适应方向，解决了混合分布偏移下单一适应路径的局限性，并提出 potpourri/potpourri+ 两个更真实的评估基准，在所有设定下取得 SOTA。
tags:
  - AAAI 2026
  - test-time adaptation
  - 混合分布偏移
  - Mixture-of-Experts
  - LayerNorm
  - Transformer
---

# MoETTA: Test-Time Adaptation Under Mixed Distribution Shifts with MoE-LayerNorm

**会议**: AAAI 2026  
**arXiv**: [2511.13760](https://arxiv.org/abs/2511.13760)  
**代码**: [GitHub](https://github.com/AnikiFan/MoETTA)  
**领域**: 测试时自适应 / 域适应  
**关键词**: test-time adaptation, 混合分布偏移, Mixture-of-Experts, LayerNorm, Vision Transformer

## 一句话总结

本文提出 MoETTA，一种将 LayerNorm 重参数化为多个结构解耦专家分支的测试时自适应框架，通过路由机制为不同域的样本选择不同的适应方向，解决了混合分布偏移下单一适应路径的局限性，并提出 potpourri/potpourri+ 两个更真实的评估基准，在所有设定下取得 SOTA。

## 研究背景与动机

测试时自适应（TTA）旨在推理阶段利用无标签数据调整模型参数，以缓解分布偏移带来的性能下降。早期方法（如 Tent、EATA）主要关注单域分布偏移，即所有测试样本来自同一目标分布。但真实部署场景中，推理批次往往包含来自多个不同域的样本——例如边缘设备的并发异构请求——形成**混合分布偏移**。

现有方法的核心局限在于：它们强制所有测试样本共享**同一个适应路径**（同一组梯度更新方向）。然而，实验发现不同域之间的累积梯度方向余弦相似度仅为 0.69（ImageNet-C 的 15 个域），某些极端情况下甚至接近 0。理论分析也表明，当域特定参数服从高斯分布时，累积梯度间的期望余弦相似度收敛到 $0.5 + \mathcal{O}(1/d)$，这意味着梯度方向的不一致是内在的、不可避免的。

本文的核心 idea：与其强制单一适应方向，不如利用多个专家来表示模型内的多条适应路径，让不同域的样本被路由到不同的专家，从而实现解耦的、多方向的参数更新。

## 方法详解

### 整体框架

MoETTA 将 ViT 中的 LayerNorm 模块替换为 MoE-LayerNorm 模块。每个 MoE-LayerNorm 包含一组结构解耦的专家分支（LayerNorm 参数）和一个路由器。在推理时，路由器根据输入嵌入的均值为每个样本选择一个专家，该专家的参数与冻结的预训练 LayerNorm 参数相加，形成样本特定的归一化参数。整个框架通过熵最小化损失和负载均衡损失联合优化。

### 关键设计

1. **MoE-LayerNorm 模块**:

    - 功能：将标准 LayerNorm 替换为包含多个专家的 MoE 结构，每个专家是一组 LayerNorm 仿射参数（初始化为零）。
    - 核心思路：对于输入嵌入，先沿 token 维度计算均值，送入路由器得到路由概率，选择概率最高的专家。有效 LayerNorm 参数 = 选中的专家参数 + 冻结的预训练参数（共享专家）。采用 top-1 路由策略，每个样本仅激活一个专家。
    - 设计动机：(a) LayerNorm 参数作为专家极其轻量，适合 TTA 的低开销要求；(b) top-1 路由避免了多专家输出合并导致的参数干扰；(c) 共享专家提供域不变知识，减少专家间冗余，鼓励互补行为。

2. **路由机制与梯度传播**:

    - 功能：实现可训练的路由器，使其在熵损失下也能获得梯度信号。
    - 核心思路：路由器是一个线性投影层。前向传播时仅激活最大概率 $p$ 对应的专家，但用 $p / p.\text{detach}()$ 缩放专家输出（不影响前向值，但允许梯度流向路由器）。加上负载均衡损失 $\mathcal{L}_\text{load balancing} = N \times \sum_{i=1}^{N} \bm{F}_i \times \bm{P}_i$，鼓励专家利用均衡。
    - 设计动机：路由决策需要与模型预测联合优化，使路由器学会将相似特征的样本分配到同一专家，让每个专家沿不同方向适应。

3. **样本筛选与熵重加权**:

    - 功能：过滤高熵（不可靠）样本，对可靠样本按熵值重加权。
    - 核心思路：动态阈值 $E_\text{max}^t$ 根据历史平均熵自适应调整。总损失为 $\frac{1}{|\mathcal{S}_t|} \sum_{\bm{x} \in \mathcal{S}_t} \exp[E_0 - \text{Ent}(\bm{x})] \cdot \text{Ent}(\bm{x}) + \alpha_t \sum_{i=1}^{M} \mathcal{L}^i_\text{load balancing}$，其中 $\alpha_t$ 动态平衡两个损失项。
    - 设计动机：混合分布中不可靠样本（高熵）会产生噪声梯度，过滤和重加权可提高梯度信号质量。

4. **Potpourri / Potpourri+ 基准**:

    - 功能：提出两个更真实的混合分布偏移评估基准。
    - Potpourri：混合 ImageNet-C（合成腐蚀）、ImageNet-R（风格渲染）、ImageNet-A（对抗样本）和 ImageNet-Sketch（语义抽象），涵盖噪声、模糊、天气、数字变换、自然、艺术和对抗扰动。
    - Potpourri+：在 Potpourri 基础上加入 ImageNet 验证集样本，评估方法在同时处理 ID 和 OOD 数据时的灾难性遗忘问题。
    - 设计动机：现有基准（仅 ImageNet-C 混合）无法反映真实部署中多种多样的分布偏移类型。

### 损失函数 / 训练策略

总损失由三部分组成：
- **熵损失**：对通过筛选的样本计算后验分布的熵，并用指数重加权（低熵样本权重更高）
- **负载均衡损失**：鼓励专家利用均衡，防止路由坍塌到少数专家
- **动态系数** $\alpha_t$：根据平均熵变化自适应调整两个损失项的权重比例

仅更新路由器和专家 LayerNorm 参数，其余模型参数冻结。

## 实验关键数据

### 主实验

| 模型 | 设定 | 无适应 | Tent | EATA | SAR | DeYO | MGTTA | BECoTTA | MoETTA |
|------|------|--------|------|------|-----|------|-------|---------|--------|
| ViT-B/16 | Classical | 55.52 | 63.20 | 64.28 | 60.76 | 63.97 | 66.20 | 61.57 | **67.20** |
| ViT-B/16 | Potpourri | 54.18 | 60.99 | 61.99 | 58.71 | 61.66 | 62.98 | 59.08 | **65.12** |
| ViT-B/16 | Potpourri+ | 55.92 | 62.28 | 63.17 | 59.99 | 62.90 | 64.35 | 58.87 | **66.15** |
| ConvNeXt-B | Classical | 54.81 | 58.88 | 64.50 | 61.67 | 64.32 | - | 50.16 | **67.40** |
| ConvNeXt-B | Potpourri | 53.91 | 58.23 | 62.69 | 61.16 | 62.46 | - | 28.28 | **65.70** |
| ConvNeXt-B | Potpourri+ | 55.69 | 59.69 | 63.94 | 62.72 | 63.57 | - | 48.92 | **66.68** |

MoETTA 在全部 6 个设定上取得最佳性能，在 Potpourri 设定（更具挑战性）上比次优方法 MGTTA 高出 2.14%。值得注意的是 MoETTA 不需要额外预训练样本，而 MGTTA 需要额外的 OOD 和 ID 样本。

### 消融实验

| 配置 | Classical | Potpourri | Potpourri+ | 平均 |
|------|-----------|-----------|------------|------|
| 完整方法 | 67.25 | 65.14 | 66.21 | 66.20 |
| 去掉样本筛选 | 67.04 | 64.01 | 57.61 | 62.89 |
| 去掉熵重加权 | 62.86 | 60.51 | 61.79 | 61.72 |
| 去掉负载均衡损失 | 26.27 | 16.27 | 21.29 | 21.28 |
| 去掉路由器梯度 | 65.17 | 62.80 | 63.92 | 63.96 |
| 去掉样本级路由 | 28.69 | 28.60 | 24.96 | 27.42 |
| 去掉 MoE-LayerNorm | 22.38 | 17.94 | 26.93 | 22.42 |
| 去掉层级路由 | 17.40 | 27.09 | 15.18 | 19.89 |

### 关键发现

- **负载均衡损失至关重要**：去掉后性能从 66.20% 暴跌至 21.28%，路由完全坍塌
- **样本级路由是核心**：强制同一批次使用同一专家导致性能崩溃（27.42%），说明输入自适应路由的必要性
- **专家参数自然分化**：适应过程中同一 MoE-LayerNorm 内专家参数的余弦相似度逐渐降低，尤其在浅层更明显
- **浅层应冻结**：冻结前 5 个 LayerNorm 层（保留低级域不变特征）效果最佳
- **计算效率适中**：运行时间为无适应基线的 247%，与 Tent（226%）、EATA（239%）接近

## 亮点与洞察

- **动机分析扎实**：通过梯度方向余弦相似度的实证分析和理论推导，清晰揭示了单一适应路径在混合偏移下的内在缺陷
- **设计极其轻量**：仅使用 LayerNorm 参数作为专家，每样本激活参数仅 0.23M，远低于 CoTTA 的 86.42M
- **Potpourri 基准设计合理**：结合合成、自然、艺术和对抗多种偏移类型，比现有基准更接近实际部署
- **专家多样性自然涌现**而非预先设定，来源于结构解耦和路由机制的联合优化

## 局限与展望

- 仅在分类任务上验证，在检测、分割等密集预测任务上的表现有待探索
- 专家数量（9个最佳）和替换策略需要针对不同架构重新调优
- 路由器使用简单的线性投影，更复杂的路由策略可能进一步提升性能
- 未考虑持续学习场景（域逐渐演变），与 BECoTTA 等方法的结合值得研究
- Potpourri 基准中各数据集的类别空间不完全对齐（如 ImageNet-A 仅 200 类），可能引入评估偏差

## 相关工作与启发

本文将 MoE 思想引入 TTA 领域，与 BECoTTA（使用 LoRA 作为专家，针对持续 TTA）形成对比。MoETTA 选择 LayerNorm 作为专家单元——比 LoRA 更轻量，且直接对应 TTA 中最常被更新的模块（Tent 也仅更新 BN/LN 参数）。这种"在已有适应目标上构建 MoE"的策略值得在其他自适应问题中推广。负载均衡损失借鉴自大语言模型中的 Switch Transformer，在 TTA 中同样发挥关键作用。

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries](graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)
- [Test-Time Adaptation by Causal Trimming](../../NeurIPS2025/llm_evaluation/test-time_adaptation_by_causal_trimming.md)
- [Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams](../../ECCV2024/llm_evaluation/distribution_alignment_for_fully_test-time_adaptation_with_dynamic_online_data_s.md)
- [When and Where to Reset Matters for Long-Term Test-Time Adaptation](../../ICLR2026/llm_evaluation/when_and_where_to_reset_matters_for_long-term_test-time_adaptation.md)
- [Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](../../NeurIPS2025/llm_evaluation/leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)

<!-- RELATED:END -->
