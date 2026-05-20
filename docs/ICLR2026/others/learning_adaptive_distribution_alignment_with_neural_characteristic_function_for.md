---
title: >-
  [论文解读] Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation
description: >-
  [ICLR 2026][图域适应] 提出ADAlign框架，利用神经特征函数在谱域自适应对齐源/目标图分布——无需手动选择对齐标准，自动识别每个迁移场景中最显著的分布差异。在10个数据集16个迁移任务上达SOTA，同时降低内存和训练时间。
tags:
  - "ICLR 2026"
  - "图域适应"
  - "特征函数"
  - "谱域对齐"
  - "自适应频率采样"
  - "minimax优化"
---

# Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation

**会议**: ICLR 2026  
**arXiv**: [2602.10489](https://arxiv.org/abs/2602.10489)  
**代码**: [https://github.com/gxingyu/ADAlign](https://github.com/gxingyu/ADAlign)  
**领域**: 其他 / 图神经网络  
**关键词**: 图域适应, 特征函数, 谱域对齐, 自适应频率采样, minimax优化

## 一句话总结
提出ADAlign框架，利用神经特征函数在谱域自适应对齐源/目标图分布——无需手动选择对齐标准，自动识别每个迁移场景中最显著的分布差异。在10个数据集16个迁移任务上达SOTA，同时降低内存和训练时间。

## 研究背景与动机
图域适应(GDA)旨在将有标签源图的知识迁移到无标签目标图。分布偏移的来源复杂多样——节点属性差异、度分布差异、同质性差异等往往交织在一起。现有方法依赖人工设计的图滤波器提取特定特征（如属性或结构统计量）再对齐，但不同迁移场景中主导差异不同，固定策略难以适应。

如Figure 1可视化所示，三个Airport迁移任务中最大KL散度对应的特征维度完全不同——B-E中feature 2,3最大，U-E中feature 1,2,4最大。固定对齐某几个特征无法捕获所有场景的完整偏移。

核心创新：用特征函数(CF)在谱域统一表示分布差异——CF唯一确定概率分布(Thm 2)且可自适应地在频域中寻找最信息量的频率成分进行对齐(NSD + learnable frequency sampler)。

## 方法详解

### 整体框架
GNN编码器 → 特征函数变换 → Neural Spectral Discrepancy (NSD) → Adaptive Frequency Sampler → minimax优化。

### 关键设计

1. **特征函数变换**：将源/目标图嵌入Z^S, Z^T的经验分布转换到频域：Ψ(t) = E[exp(it^T z)]。CF唯一确定分布(Thm 2)且convergence保证(Thm 1)。

2. **Neural Spectral Discrepancy (NSD)**：NSD = ∫ |Ψ^S(t) - Ψ^T(t)|² dF_T(t)。分解为amplitude差异(全局结构变化)和phase差异(关系对齐偏移)，系数κ控制平衡。

3. **Adaptive Frequency Sampler**：用normal scale mixture参数化采样分布 p_T(t;φ)。通过minimax训练：φ最大化NSD（找到最大差异的频率），δ最小化NSD（对齐分布）。

4. **minimax优化 (Eq 14)**：min_δ max_φ [L_source + λ·L_align]。GNN参数δ优化分类+对齐，采样参数φ对抗性寻找最大差异频率。

### 损失函数 / 训练策略
L = L_source(CE) + λ·L_align(NSD)。L_align通过Monte Carlo采样M个频率点近似。reparameterization trick保证采样可微。

## 实验关键数据

### 主实验（部分展示）

| 任务 | GAT | GCN | UDAGCN | DEAL | **ADAlign** | 说明 |
|------|-----|-----|--------|------|----------|------|
| A→C (Citation) | 62.8 | 69.2 | 72.1 | 74.3 | **76.8** | +2.5 |
| C→D (Citation) | 67.1 | 68.1 | 71.5 | 73.2 | **75.4** | +2.2 |
| B1→B2 (Blog) | 21.2 | 20.5 | 23.1 | 24.8 | **28.3** | +3.5 |

### 消融实验

| 组件 | 效果 | 说明 |
|------|------|------|
| 去掉adaptive sampler（fixed频率） | 显著下降 | 自适应是关键 |
| 去掉phase alignment | 下降 | 两者都重要 |
| 去掉amplitude alignment | 下降 | 互补信息 |
| κ=0 (仅phase) vs κ=1 (仅amplitude) | 都不如κ=0.5 | 需要平衡 |

### 效率比较

| 方法 | 内存(MB) | 训练时间(s) | 说明 |
|------|---------|-----------|------|
| DEAL | 1,245 | 892 | 重型GNN对齐 |
| FLAN | 987 | 756 | 滤波器设计 |
| **ADAlign** | **423** | **312** | 轻量谱域操作 |

### 关键发现
- ADAlign在16/16个迁移任务上达到最优或接近最优。
- 内存和训练时间分别降低2-3倍——CF操作比GNN-based对齐更轻量。
- 自适应频率采样在不同场景自动聚焦不同谱成分——验证了设计初衷。
- PAC-Bayesian分析(Thm 3 + Prop 1)为NSD提供了泛化理论支持。

## 亮点与洞察
- 特征函数为图分布对齐提供了统一、完备的理论工具——不需要手动选择对齐什么。
- 振幅/相位分解有直觉意义：振幅≈全局统计量差异，相位≈关系结构差异。
- minimax中的frequency sampler是"对抗性搜索最大差异"的自然表达。
- 效率优势使框架实用性更强。

## 局限与展望
- Monte Carlo近似的频率采样引入方差，M的选择需要权衡。
- 仅在节点分类任务验证，图级任务待探索。
- κ的选择目前是超参，自适应κ可能更优。
- 对极端domain gap的处理能力需进一步测试。

## 相关工作与启发
- 将特征函数从生成模型/知识蒸馏引入GDA，开辟了新的方法空间。
- 自适应谱域对齐的思路可推广到其他domain adaptation任务。

## 评分
- 新颖性: ⭐⭐⭐⭐ 特征函数+谱域对齐+自适应采样
- 实验充分度: ⭐⭐⭐⭐⭐ 10数据集16任务+消融+效率分析
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰
- 价值: ⭐⭐⭐⭐ GDA方法论的有意义贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation](learning_structure-semantic_evolution_trajectories_for_graph_domain_adaptation.md)
- [\[ICLR 2026\] Distributionally Robust Classification for Multi-Source Unsupervised Domain Adaptation](distributionally_robust_classification_for_multi-source_unsupervised_domain_adap.md)
- [\[ICLR 2026\] OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection](owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection.md)
- [\[ICLR 2026\] Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning](noisy-pair_robust_representation_alignment_for_positive-unlabeled_learning.md)
- [\[ICLR 2026\] Probabilistic Kernel Function for Fast Angle Testing](probabilistic_kernel_function_for_fast_angle_testing.md)

</div>

<!-- RELATED:END -->
