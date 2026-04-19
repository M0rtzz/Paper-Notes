---
title: >-
  [论文解读] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation
description: >-
  [CVPR 2026][医学图像][半监督分割] 本文提出 SCDL（Semantic Class Distribution Learning），一个即插即用模块，通过类别分布双向对齐（CDBA）学习结构化的类条件特征分布并与可学习类代理双向对齐，结合语义锚点约束（SAC）利用标注数据引导代理学习正确语义，缓解了半监督医学图像分割中的监督偏差和特征表示偏差，在尾类器官上取得了显著提升。
tags:
  - CVPR 2026
  - 医学图像
  - 半监督分割
  - 类别不平衡
  - 分布学习
  - 代理分布
  - 语义锚点
---

# Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation

**会议**: CVPR 2026  
**arXiv**: [2603.05202](https://arxiv.org/abs/2603.05202)  
**代码**: [GitHub](https://github.com/Zyh55555/SCDL)  
**领域**: 医学图像  
**关键词**: 半监督分割、类别不平衡、分布学习、代理分布、语义锚点

## 一句话总结

本文提出 SCDL（Semantic Class Distribution Learning），一个即插即用模块，通过类别分布双向对齐（CDBA）学习结构化的类条件特征分布并与可学习类代理双向对齐，结合语义锚点约束（SAC）利用标注数据引导代理学习正确语义，缓解了半监督医学图像分割中的监督偏差和特征表示偏差，在尾类器官上取得了显著提升。

## 研究背景与动机

1. **领域现状**：半监督医学图像分割（SSMIS）利用少量标注数据+大量无标注数据进行训练，主流方法包括一致性正则化、对比学习和伪标签等。然而，医学图像数据集普遍存在严重的类别不平衡——大器官（如肝脏）占据大量像素，小器官（如食管、肾上腺）像素极少。

2. **现有痛点**：类别不平衡与半监督机制的结合导致两个层面的偏差。（1）监督信号偏差：大类占主导的像素级梯度、伪标签的自增强效应都使监督偏向头部类。（2）特征表示偏差：现有方法（重加权、输出校准）仅在损失层或输出层操作，缺乏对类条件特征分布的直接约束，导致头部类特征紧凑而尾部类发散，特征空间中尾类被头类"吞噬"。

3. **核心矛盾**：无标注数据主要用于局部一致性正则化，很少被用来显式纠正类条件特征分布的倾斜——因此无标注数据并未帮助少数类建立良好的特征表示，不平衡持续存在。

4. **本文目标**：如何在特征空间层面直接缓解类别不平衡导致的表示偏差，而非仅在损失或输出层面（治标不治本）。

5. **切入角度**：为每个语义类学习一个代理分布（高斯分布），通过双向对齐约束使嵌入靠近对应代理、代理远离非目标嵌入，同时用标注区域的语义锚点为代理提供正确的语义监督。

6. **核心 idea**：通过学习类条件代理分布并双向对齐，在特征空间中直接重塑类别分布结构，使少数类也能获得稳定的表示学习信号。

## 方法详解

### 整体框架

SCDL 作为即插即用模块集成到现有半监督分割网络中。它包含两个核心组件：CDBA 在编码器输出的嵌入空间中为每个类建立可学习代理分布，并双向对齐嵌入与代理；SAC 从标注区域提取语义锚点来监督代理学习。最终，通过代理采样构建的先验信息注入解码器各层，增强尾类的特征表示。

### 关键设计

1. **类别分布双向对齐（Class Distribution Bidirectional Alignment, CDBA）**

    - 功能：学习结构化的类条件特征分布，缓解表示偏差
    - 核心思路：每个语义类 $c$ 用一个可学习的高斯代理分布建模：$p(u|c) = \mathcal{N}(\mu_c, \text{diag}(\sigma_c^2))$。计算每个 token 嵌入到各代理的软分配 $P(c|z_{i,l}) = \text{softmax}_c(\cos(z_{i,l}, \mu_c))$。双向对齐包括：（1）Embedding-to-Proxy (E2P)：$\mathcal{L}_{E2P} = \sum P(c|z) \cdot [1 - \cos(z, \mu_c)]$，将嵌入推向软分配的代理；（2）Proxy-to-Embedding (P2E)：$\mathcal{L}_{P2E} = \frac{1}{C}\sum \exp(-(\mathcal{E}_c^+ - \mathcal{E}_c^-))$，鼓励每个代理区分属于和不属于该类的嵌入
    - 设计动机：软分配使每个嵌入可以影响多个代理的梯度更新，消除了类别频率差异的影响——即使少数类像素少，其代理仍能从软分配中获得学习信号。双向对齐确保代理既有吸引力（E2P拉近）又有区分力（P2E排斥）

2. **代理采样与特征增强**

    - 功能：利用学到的代理分布为下游解码器提供结构化的语义先验
    - 核心思路：构建三种先验：（1）分布加权先验 $\mathbf{r}^{dist}$：从代理分布中采样 $S$ 个样本，计算每个嵌入与采样点的平均余弦相似度作为权重，加权组合代理均值；（2）中心相似度先验 $\mathbf{r}^{center}$：直接用嵌入与各代理均值的余弦相似度加权组合；（3）token 采样先验 $\mathbf{z}^{sam}$：对每个 token 做局部扰动采样增强鲁棒性。三种先验拼接后通过轻量投影层注入解码器各阶段
    - 设计动机：分布加权先验保留了方差信息（不确定性感知），中心先验提供互补的确定性信号，两者结合使头部和尾部类都能有效贡献

3. **语义锚点约束（Semantic Anchor Constraints, SAC）**

    - 功能：为随机初始化的代理分布提供真实类语义指导
    - 核心思路：从标注数据中，对每个类提取标注区域的类感知嵌入（用 ground-truth 掩码遮蔽非目标区域后过编码器），计算均值作为语义锚点 $\text{anchor}_c = \frac{1}{|\mathcal{Z}_c|}\sum_{z \in \mathcal{Z}_c} z$。然后用余弦相似度损失对齐每个代理均值与对应锚点：$\mathcal{L}_{SAC} = \frac{1}{C}\sum [1 - \cos(\mu_c, \text{anchor}_c)]$。锚点在反向传播中被 detach，确保 SAC 只更新代理而不影响编码器
    - 设计动机：代理随机初始化，没有语义约束可能学到错误的类对应关系。SAC 利用少量标注数据的"确定性信号"来锚定代理，即使标注很少也足够——重要的是锚点方向正确，精确度可以在训练中进一步优化

### 损失函数 / 训练策略

总损失 = 基线分割损失 + $\mathcal{L}_{E2P}$ + $\mathcal{L}_{P2E}$ + $\mathcal{L}_{SAC}$。SCDL 模块的权重衰减设为 1e-4。其他配置随基线方法不同而变化（如 GenericSSL、DHC、GA-CPS 等）。batch size 为 4，在 NVIDIA A40 GPU 上训练。

## 实验关键数据

### 主实验

Synapse（20% 标注）和 AMOS（5% 标注）数据集结果：

| 方法 | Synapse DSC↑ | Synapse ASD↓ | AMOS DSC↑ | AMOS ASD↓ |
|------|-------------|-------------|----------|----------|
| GenericSSL 基线 | 55.94 | 6.14 | 35.73 | 45.82 |
| SCDL-GenericSSL | **58.90 (+2.96)** | **5.79** | **47.35 (+11.62)** | **22.84** |
| DHC 基线 | 46.16 | 10.04 | 40.11 | 40.65 |
| SCDL-DHC | **49.17 (+3.01)** | 10.59 | **49.28 (+9.17)** | **17.47** |
| GA-CPS 基线 | 66.29 | 5.44 | 50.90 | 13.77 |
| SCDL-GA-CPS | **67.50 (+1.21)** | **3.32** | **61.57 (+10.67)** | 10.08 |
| GA-MagicNet 基线 | 66.00 | 3.42 | 59.15 | 8.66 |
| SCDL-GA-MagicNet | **66.75 (+0.75)** | 3.65 | **62.16 (+3.01)** | **5.65** |

尾类器官的显著提升（Synapse, SCDL-DHC 相比 DHC）：

| 器官 | DHC | SCDL-DHC | 提升 |
|------|-----|----------|------|
| 门静脉和脾静脉 (PSV) | 30.7 | 42.6 | +11.9 |
| 食管 (Es) | 14.7 | 23.5 | +8.8 |
| 右肾上腺 (RAG) | 27.9 | 36.7 | +8.8 |

AMOS 上更极端的恢复（SCDL-DHC）：右肾上腺 0%→33.9%，左肾上腺 0%→30.3%。

### 消融实验

在 Synapse 上（GA-CPS 基线）：

| 配置 | DSC↑ | ASD↓ | 说明 |
|------|------|------|------|
| 基线 | 66.29 | 5.44 | GA-CPS |
| + CDBA | 66.77 (+0.48) | 6.24 | DSC 提升但 ASD 上升 |
| + CDBA + SAC | **67.50 (+1.21)** | **3.32** | SAC 加入后 ASD 骤降 2.92 |

### 关键发现

- CDBA 单独使用能提升 DSC 但可能损害 ASD（边界质量），SAC 的加入至关重要——它不仅进一步提升 DSC，还大幅改善边界精度
- SCDL 的增益主要集中在尾类/小器官上：在 AMOS 5% 标注下，DHC 的右/左肾上腺 Dice 从 0% 恢复到 33.9%/30.3%，说明 SCDL 有效防止了极度少数类被完全忽视
- 在强基线（如 GA-MagicNet DSC=66.00）上提升幅度适中（+0.75%），但在弱基线上提升显著（GenericSSL AMOS +11.62%），说明 SCDL 更善于纠正严重的类别偏差
- ASD 的改善在加入 SAC 后尤为显著（从 6.24 降至 3.32），表明语义锚点约束有助于改善边界几何质量

## 亮点与洞察

- **即插即用的设计**：SCDL 可以无缝集成到任何现有 SSMIS 方法中，无需修改基线架构，这极大提升了其实用价值
- **软分配机制消除类频率偏差**：与硬分配不同，每个嵌入按软权重影响所有代理的学习，少数类代理即使在极少样本下也能持续接收梯度信号
- **三种先验的互补设计很有想法**：分布加权先验考虑方差（不确定性），中心先验考虑均值（确定性），token 采样先验增加鲁棒性
- **将无标注数据用于分布级学习**而非仅用于一致性正则化，是一个重要的范式转变——无标注数据参与了全局类分布的建模

## 局限与展望

- 代理使用各向同性高斯假设（对角协方差矩阵），可能不够灵活表示复杂的类边界形状
- SAC 的语义锚点取均值简单但粗糙，对于多模态分布（如一个器官在不同切面的外观差异大）可能不足
- 消融中 CDBA 单独使用时 ASD 反而上升，说明仅有分布对齐而缺乏语义监督可能引入不稳定性
- 仅在 CT 多器官分割上验证，缺少 MRI、病理、视网膜等其他模态的实验

## 相关工作与启发

- **vs DHC**: DHC 使用动态混合课程学习处理半监督不平衡，SCDL-DHC 在其基础上 DSC 提升 3%+，且对尾类改善更大
- **vs GA-MagicNet/GA-CPS**: GA 系列使用几何感知增强处理不平衡，SCDL 提供了正交的分布级解决方案，二者可以叠加
- **vs CLD**: CLD 使用对比分布学习但主要在输出层面操作，SCDL 在嵌入空间直接约束类条件分布

## 评分

- 新颖性: ⭐⭐⭐⭐ 类代理分布的双向对齐+语义锚点约束是新的组合
- 实验充分度: ⭐⭐⭐⭐ 两个数据集四种基线方法的系统验证，但缺少非 CT 模态
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，三层偏差分析（监督/表示/分布）有深度
- 价值: ⭐⭐⭐⭐ 即插即用模块对半监督医学分割社区有直接价值

<!-- RELATED:START -->

## 相关论文

- [SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)
- [SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)

<!-- RELATED:END -->
