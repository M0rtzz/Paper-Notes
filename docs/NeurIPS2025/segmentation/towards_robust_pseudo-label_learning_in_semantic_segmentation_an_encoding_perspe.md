---
title: >-
  [论文解读] Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective
description: >-
  [NeurIPS 2025][语义分割][伪标签学习] 提出 ECOCSeg，用纠错输出码（ECOC）替代 one-hot 编码来表示语义类别，将 N 类分类分解为 K 个二分类子任务，配合 bit 级伪标签去噪和定制优化损失，显著提升 UDA 和 SSL 语义分割中伪标签学习的鲁棒性。 领域现状：语义分割在标签稀缺场景下（…
tags:
  - "NeurIPS 2025"
  - "语义分割"
  - "伪标签学习"
  - "纠错输出码"
  - "无监督域适应"
  - "半监督学习"
---

# Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective

**会议**: NeurIPS 2025  
**arXiv**: [2512.06870](https://arxiv.org/abs/2512.06870)  
**代码**: [https://github.com/Woof6/ECOCSeg](https://github.com/Woof6/ECOCSeg)  
**领域**: 分割  
**关键词**: 伪标签学习, 语义分割, 纠错输出码, 无监督域适应, 半监督学习

## 一句话总结

提出 ECOCSeg，用纠错输出码（ECOC）替代 one-hot 编码来表示语义类别，将 N 类分类分解为 K 个二分类子任务，配合 bit 级伪标签去噪和定制优化损失，显著提升 UDA 和 SSL 语义分割中伪标签学习的鲁棒性。

## 研究背景与动机

**领域现状**：语义分割在标签稀缺场景下（无监督域适应 UDA 和半监督学习 SSL）广泛依赖伪标签学习。主流方法分为自训练（用 EMA 教师模型生成伪标签）和一致性正则化（对同一样本不同扰动要求预测一致），两者本质上都是伪标签学习范式。

**现有痛点**：伪标签不可避免地存在错误，而现有方法使用 one-hot 编码通过 argmax 硬分配类别标签，一旦预测错误就会完全误导训练。阈值过滤策略丢弃低置信度样本导致模型偏向简单样本；加权策略需要精心调参，泛化性差。关键是：现有方法都聚焦在伪标签的**选择策略**上，几乎没有人考虑过**编码形式**本身的影响。

**核心矛盾**：相似类别（如 sheep/cow/horse）共享视觉属性导致频繁互相混淆，one-hot 编码下一旦混淆就完全错误（互信息为零），但实际上这些类别之间存在共享属性——如果编码形式能利用这种共享关系，即使分类错误也能提供部分正确的监督信号。

**本文目标** 从编码形式的角度重新审视伪标签噪声问题，设计一种能容忍部分 bit 错误的类别编码，使得错误伪标签仍能提供有意义的监督。

**切入角度**：作者借鉴通信领域的纠错输出码（ECOC），将每个类别表示为多 bit 的二进制编码而非 one-hot 向量。类似类别在编码中共享某些 bit，即使分类错误，共享的 bit 仍然正确，从而实现"错中有对"的容错监督。

**核心 idea**：用纠错输出码替代 one-hot 编码来表示语义类别，让伪标签即使分类错误也能在 bit 层面提供部分正确的监督信号。

## 方法详解

### 整体框架

ECOCSeg 可以作为插件集成到现有伪标签学习框架中。输入图像经过编码器提取像素特征，然后不再送入 N 类分类器，而是送入 K 个二分类器（每个预测编码的一个 bit），输出 K 维概率向量后通过软 Hamming 距离在码本中查找最近邻来确定类别。伪标签生成环节引入 bit 级去噪机制，在 bit-wise 和 code-wise 两种伪标签形式之间取长补短。训练时使用三个定制损失函数联合优化。

### 关键设计

1. **ECOC 密集分类范式**:

    - 功能：将 N 类语义分割转化为 K 个二分类问题，天然具备纠错能力
    - 核心思路：构建一个 $N \times K$ 的二值码本矩阵，每个类别对应一个长度为 K 的 codeword。分类器变为 K 个独立的 sigmoid 二分类器，每个预测一个 bit。分类时计算预测向量与码本中各 codeword 的软 Hamming 距离 $d_{SH}(\mathbf{c}_n, \mathbf{p}^i) = \frac{1}{K}\sum_{k=1}^K \|p(k|\mathbf{z}_i) - \mathbf{c}_{nk}\|_1$，选最近邻确定类别。码本设计提供两种策略：max-min 距离编码（最大化类间最小 Hamming 距离，保证纠错能力）和基于文本的编码（利用类名语义关系生成编码，保证语义一致性）
    - 设计动机：one-hot 编码下类间 Hamming 距离恒为 2，纠错能力为零。ECOC 编码可以让类间距离远大于 2，理论上能容忍 $\lfloor(d-1)/2\rfloor$ 个 bit 错误。论文还从 NTK 理论证明了在全监督下 ECOC 与 one-hot 性能等价（Theorem 4.1），在伪标签噪声下 ECOC 能获得更紧的误分类上界（Theorem 4.2）

2. **可靠 bit 挖掘算法（Reliable Bit Mining）**:

    - 功能：在 bit 级别从噪声伪标签中挖掘可靠的监督信号
    - 核心思路：ECOCSeg 自然引出两种伪标签形式——bit-wise（将每个 bit 的 sigmoid 输出直接量化）和 code-wise（查最近邻码字）。两者各有优劣：bit-wise 较软但每个 bit 独立噪声；code-wise 在分类正确时完全准确但一旦错误引入整体噪声。算法的核心是查询 C-最近邻码字，找出候选集中所有码字共享的 bit 位置（这些 bit 无论哪个候选是真值都是对的），标记为"可靠 bit"，最终混合伪标签：可靠位置用 code-wise 值，不可靠位置用 bit-wise 值。C 的大小通过置信度阈值 T 自适应确定
    - 设计动机：单一伪标签形式无法同时获得高精度和高覆盖率。通过挖掘候选类别间的共享 bit，能在保证正确性的同时最大化可用监督信号

3. **定制优化目标**:

    - 功能：在 bit 级 BCE 基础上引入结构化表示约束，加速收敛并增强判别力
    - 核心思路：三个损失组合——(1) 二值交叉熵 $\mathcal{L}_{bce}$ 独立优化每个 bit 分类器；(2) 像素-码距离 $\mathcal{L}_{pcd} = 1 - \cos(\hat{\mathbf{p}}^i, \hat{\mathbf{c}}^i)$ 鼓励类内紧凑性，将 logits 向对应码字的方向拉近；(3) 像素-码对比 $\mathcal{L}_{pcc}$ 作对比学习，将预测推离非目标码字，且只在区分性 bit 位（$P_d$）上计算，忽略共享 bit 位
    - 设计动机：单独的 BCE 忽略 bit 之间的结构关系，缺乏类内紧凑和类间分离的约束。PCD 提供类内约束，PCC 提供类间约束，三者互补

### 损失函数 / 训练策略

总损失为 $\mathcal{L}_{total} = \mathcal{L}_{bce} + \lambda_1 \mathcal{L}_{pcd} + \lambda_2 \mathcal{L}_{pcc}$，同时用于有标签的监督损失和无标签的伪标签损失。整体训练流程保持 self-training/consistency regularization 的标准范式不变，只替换编码形式、伪标签生成和损失函数。

## 实验关键数据

### 主实验

| 基线方法 | 架构 | 原始 mIoU | +ECOCSeg mIoU | 提升 |
|----------|------|-----------|---------------|------|
| DACS (GTA→CS) | CNN | 52.1 | 54.5 | +2.4 |
| DAFormer (GTA→CS) | Trans. | 68.3 | 70.5 | +2.2 |
| MIC (GTA→CS) | Trans. | 75.9 | 76.9 | +1.0 |
| DACS (SYN→CS) | CNN | 48.3 | 52.1 | +3.8 |
| DAFormer (SYN→CS) | Trans. | 60.9 | 63.3 | +2.4 |
| MIC (SYN→CS) | Trans. | 68.7 | 69.8 | +1.1 |

### 消融实验

| 组件 | GTA→CS mIoU | 说明 |
|------|-------------|------|
| 基线 (DAFormer) | 68.3 | one-hot + CE |
| + ECOC 编码 | 69.0 | 仅换编码形式 |
| + bit-wise PL | 69.5 | bit 级伪标签 |
| + code-wise PL | 69.3 | 码字级伪标签 |
| + 混合 PL (RBM) | 70.0 | 可靠 bit 挖掘 |
| + PCD + PCC 损失 | 70.5 | 完整 ECOCSeg |

### 关键发现

- ECOCSeg 对不同基线（DACS/DAFormer/MIC）和不同架构（CNN/Transformer）一致有效，说明其正交于已有改进
- SYNTHIA→Cityscapes 上提升更大（+3.8），因为跨域差距更大时伪标签噪声更严重，ECOC 的容错优势更明显
- 混合伪标签（RBM）比单独的 bit-wise 或 code-wise 都好，验证了两种形式互补的假设
- ECOC 编码在全监督设置下不掉点，验证了 Theorem 4.1 的理论预测
- 论文还展示了 ECOC 能改善模型校准（calibration），间接提高后续迭代中伪标签的质量

## 亮点与洞察

- **从编码视角切入伪标签噪声问题**：这是一个全新的正交方向，与现有的过滤、加权等策略完全不冲突，可以叠加使用。这种思维的转换——不优化选什么标签，而是优化标签的表示形式——非常巧妙
- **类间共享属性的利用**：sheep 和 cow 都有角和蹄，即使分错类别，共享属性对应的 bit 仍然正确。这个洞察简单但深刻，将通信领域的纠错思想自然地迁移到语义分割
- **理论保证**：从 NTK 框架出发证明了 ECOC 在全监督下等价性和在噪声下的优越性，不仅是启发式方法而有严格的理论支撑

## 局限与展望

- 码本设计（K 值选择和编码策略）的最优配置可能因数据集而异，缺乏自适应选择机制
- K 个二分类器比 1 个 N 类分类器引入更多参数，当类别数非常大时可能有效率问题
- 目前只在语义分割上验证，但 ECOC 编码思想理论上可以迁移到任何伪标签学习场景（如目标检测、实例分割），值得探索
- 可靠 bit 挖掘的阈值 T 是固定超参，自适应阈值策略可能进一步提升效果

## 相关工作与启发

- **vs 阈值过滤方法**（如 FixMatch）：阈值过滤丢弃不确定样本导致难样本缺失，ECOCSeg 在 bit 级保留所有样本的部分监督，覆盖更全面
- **vs 加权策略方法**（如 FlexMatch）：加权策略需要精心设计权重函数，ECOCSeg 通过编码形式本身实现容错，无需额外的权重设计
- **vs 负学习方法**：负学习通过告诉模型"不是什么"来避免噪声，ECOCSeg 通过告诉模型"部分地是什么"来利用噪声，二者可能互补

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 从编码形式角度审视伪标签噪声是全新方向，将通信领域 ECOC 引入分割是首次
- 实验充分度: ⭐⭐⭐⭐ 覆盖 UDA/SSL 多个基线和多个 benchmark，消融详细，但缺少更多任务的验证
- 写作质量: ⭐⭐⭐⭐⭐ 问题形式化清晰，三个组成部分（编码/伪标签策略/优化目标）的分析框架很优雅
- 价值: ⭐⭐⭐⭐⭐ 正交于现有方法、即插即用、有理论保证，实用价值和启发价值都很高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Learning Camouflaged Object Detection from Noisy Pseudo Label](../../ECCV2024/segmentation/learning_camouflaged_object_detection_from_noisy_pseudo_label.md)
- [\[NeurIPS 2025\] Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation](towards_unsupervised_domain_bridging_via_image_degradation_in_semantic_segmentat.md)
- [\[NeurIPS 2025\] OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation](omnisegmentor_a_flexible_multi-modal_learning_framework_for_semantic_segmentatio.md)
- [\[NeurIPS 2025\] Diffusion-Driven Two-Stage Active Learning for Low-Budget Semantic Segmentation](diffusion-driven_two-stage_active_learning_for_low-budget_semantic_segmentation.md)
- [\[CVPR 2026\] EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection](../../CVPR2026/segmentation/erecu_pseudolabel_evolution_unsupervised_camouflage.md)

</div>

<!-- RELATED:END -->
