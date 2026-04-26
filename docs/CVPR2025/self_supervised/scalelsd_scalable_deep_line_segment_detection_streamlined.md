---
title: >-
  [论文解读] ScaleLSD: Scalable Deep Line Segment Detection Streamlined
description: >-
  [CVPR 2025][自监督学习][线段检测] ScaleLSD 通过精简线段检测架构（引入 HAT 诱导的提案验证）和设计高效伪标签生成管线（LSD-Rectifier），首次实现了在1000万无标注图像上的大规模自监督线段检测训练，在零样本评测中全面超越经典非深度 LSD 方法。
tags:
  - CVPR 2025
  - 自监督学习
  - 线段检测
  - HAT场
  - 大规模训练
  - 伪标签
---

# ScaleLSD: Scalable Deep Line Segment Detection Streamlined

**会议**: CVPR 2025  
**arXiv**: [2506.09369](https://arxiv.org/abs/2506.09369)  
**代码**: https://github.com/ant-research/scalelsd  
**领域**: 自监督学习 / 底层视觉  
**关键词**: 线段检测, 自监督学习, HAT场, 大规模训练, 伪标签

## 一句话总结

ScaleLSD 通过精简线段检测架构（引入 HAT 诱导的提案验证）和设计高效伪标签生成管线（LSD-Rectifier），首次实现了在1000万无标注图像上的大规模自监督线段检测训练，在零样本评测中全面超越经典非深度 LSD 方法。

## 研究背景与动机

**领域现状**：线段检测（LSD）是图像几何表征的基础任务，广泛用于消失点估计、双视图匹配和多视图3D重建等下游任务。深度学习 LSD 方法主要依赖 Wireframe 数据集（仅5K标注图像）进行有监督训练，但泛化能力受限。自监督 LSD 方法（SOLD2、HAWPv3、DeepLSD）尝试解决泛化问题，但训练规模仍受限于几千张图像。

**现有痛点**：(1) 有监督方法在跨域场景上泛化差；(2) 自监督方法的伪标签生成管线存在扩展性问题——SOLD2/HAWPv3 的 homographic adaptation 策略导致低召回率，DeepLSD 依赖经典 LSD 的局部对齐方案但继承了其局部性问题；(3) 经典 LSD 方法在短线段上产生虚假结果，但深度方法的检测完整性又不如经典 LSD。

**核心矛盾**：现有自监督方法的伪标签生成策略不具备大规模扩展能力。Homographic adaptation 以牺牲完整性来过滤错误检测，而经典 LSD 的局部对齐方案虽然召回率高但有局部性缺陷。

**本文目标**：设计一个能在1000万+无标注图像上进行自监督训练的线段检测器，在零样本场景下全面超越经典 LSD。

**切入角度**：作者重新审视了深度和非深度 LSD 方法的基本设计，发现三个关键洞察：(1) HAT 场表示在自监督学习中有巨大潜力；(2) 经典 LSD 的图像梯度信息在方向估计上鲁棒可靠，可用于校正伪标签；(3) 表达能力强的 Transformer backbone 对消化大规模数据至关重要。

**核心 idea**：精简架构（用 HAT 诱导验证替代 LOI 验证），并用经典 LSD 的方向场校正 HAT 场预测来高效生成高质量伪标签，从而在 SA1B 的1000万图像上实现大规模自监督训练。

## 方法详解

### 整体框架

ScaleLSD 的元架构基于 HAWPv3 精简而来。输入图像经 ViT-Base backbone 提取特征，然后通过 DPT head 预测 HAT 场（4通道：距离 $d$、方向 $\theta$、端点角度 $\alpha, \beta$）和节点热力图。从 HAT 场中解码出线段提案后，使用 HAT 诱导的提案验证来筛选可靠线段。伪标签生成管线使用 LSD-Rectifier 将经典 LSD 的方向信息注入 HAT 场预测中，实现高效高质量的伪标签生成。

### 关键设计

1. **HAT 场表示与稀疏解码**:

    - 功能：将稀疏的线段集合表示为密集的像素级场，并从中解码出唯一的线段集合
    - 核心思路：HAT 场将每个前景像素分配到其垂直最近的线段，用4个分量 $(d, \theta, \alpha, \beta)$ 编码距离、方向和端点位置。解码时，将每个像素预测的端点绑定到最近的节点（junction），通过节点索引对的唯一化来得到稀疏线段集。索引距离超过 $\tau_{\text{dist}}=10$ 像素的被剪枝。GPU 内置实现确保唯一化操作几乎无延迟
    - 设计动机：HAT 场的密集表示使得学习目标更加明确，稀疏解码方案利用节点信息消除重复提案

2. **HAT 诱导的提案验证**:

    - 功能：替代传统 LOI（Line-of-Interest）验证方案，用白盒几何方式度量线段可信度
    - 核心思路：对每个节点索引对 $(\imath_\alpha^k, \imath_\beta^k)$，计算其在 HAT 场预测中的支持度 $\text{Deg}(\imath_\alpha^k, \imath_\beta^k) = \sum \mathbb{1}[(\imath_\alpha(\mathbf{p}), \imath_\beta(\mathbf{p})) \sim (\imath_\alpha^k, \imath_\beta^k)]$，即有多少像素的预测指向该线段。支持像素数越多，线段越可信。默认使用10像素阈值过滤
    - 设计动机：LOI 验证需要学习置信分数，在含噪伪标签的自监督学习中可靠性差。HAT 诱导验证基于几何一致性度量，具有更好的可解释性和鲁棒性，且无需额外标签

3. **LSD-Rectifier 伪标签生成**:

    - 功能：高效生成高质量伪标签，支持千万级图像规模的自监督训练
    - 核心思路：先用合成数据训练种子模型，然后对真实图像同时产生两组结果：种子模型预测的 HAT 场（主源）和经典 LSD 的方向场（辅助源）。关键操作是用经典 LSD 的 $\theta$ 分量替换主源的 $\theta$ 分量，形成校正后的 HAT 场，再从中解码线段作为伪标签。经典 LSD 在方向估计上局部准确且泛化性强，该校正操作消除了需要代价高昂的 homographic adaptation 的必要性
    - 设计动机：经典 LSD 的图像梯度方向信息跨域鲁棒，但端点定位不精确；种子模型的 HAT 场端点准确但方向可能有偏差（特别在合成到真实的迁移阶段）。LSD-Rectifier 结合两者优势。

### 损失函数 / 训练策略

- **训练方案**："合成到真实"的两阶段训练。先在16K合成图像上训练10 epoch 得到种子模型；然后用种子模型+LSD-Rectifier 在真实数据上生成伪标签，从头在 Wireframe（20K）训练30 epoch 或在 SA1B（10M）训练6 epoch
- **优化器**：ADAM，合成阶段 lr=4e-4 在第7 epoch 衰减10倍；SA1B 阶段使用 linear warmup（2000步从2e-4到1e-3）+ cosine annealing
- **Backbone**：ViT-Base + DPT head

## 实验关键数据

### 主实验 — 零样本重复性评测

| 方法 | YorkUrban Rep-5(S)↑ | HPatches Rep-5(S)↑ | COCO Rep-5(S)↑ | 平均检测数/图 |
|------|---------------------|---------------------|-----------------|--------------|
| LSD（经典） | 0.419 | 0.275 | 0.456 | 493-591 |
| HAWPv3 | 0.711 | 0.322 | 0.644 | 99-225 |
| DeepLSD | 0.514 | 0.241 | 0.423 | 207-310 |
| **ScaleLSD@SA1B** | **0.725** | **0.367** | **0.666** | 540-708 |

### 消失点估计

| 方法 | YUD+ VP Error↓ | YUD+ AUC↑ |
|------|---------------|-----------|
| LSD | 2.05 | 82.9 |
| DeepLSD | 1.63 | 85.6 |
| **ScaleLSD@SA1B** | **1.47** | **87.2** |

### 关键发现

- ScaleLSD 是首个在所有评测维度上全面超越经典 LSD 的深度方法
- 从 Wireframe（20K）扩展到 SA1B（10M）带来了一致且显著的提升，验证了大规模训练的价值
- HAT 诱导验证不仅简化了架构，还在自监督场景下提供了更可靠的线段筛选
- ScaleLSD 检测到的线段数量远多于其他深度方法（接近甚至超过经典 LSD），同时保持更高精度

## 亮点与洞察

- **大规模自监督的有效性验证**：证明了"简单方法 + 大规模数据"在底层视觉任务上同样有效，与NLP/高层视觉的趋势一致
- **经典方法的新角色**：经典 LSD 不再是竞争对手而是互补工具——其梯度方向信息用于校正深度模型的伪标签
- **白盒验证的价值**：HAT 诱导验证用几何支持度替代学习的置信分数，在噪声伪标签环境下更鲁棒
- **极简设计哲学**：整体架构精简，去掉了 homographic adaptation、LOI 验证、边缘图学习等复杂组件

## 局限与展望

- 伪标签质量仍受经典 LSD 方向估计精度的制约，特别是在曲线边界附近
- 在 RDNIM 等极具挑战性的数据集上，HAWPv3 的定位误差仍低于 ScaleLSD（但检测数量远少）
- 未探索更大的 backbone（如 ViT-Large）或更多训练数据的效果上限
- 可考虑将 LSD-Rectifier 迭代应用以逐步提升伪标签质量

## 相关工作与启发

- **HAWPv3**：ScaleLSD 的直接前身，ScaleLSD 精简了其架构并扩展了训练规模
- **DeepLSD**：同样使用经典 LSD 辅助自监督训练，但将其用于局部对齐而非方向校正，导致性能收敛到经典 LSD 水平
- **SAM-1B 数据集**：为 ScaleLSD 提供了1000万无标注图像，展示了大规模数据集在自监督底层视觉中的潜力

## 评分

- **新颖性**: 7/10 — 核心贡献在于工程精简和规模扩展，单个技术点的新颖性适中
- **实验充分度**: 9/10 — 零样本评测覆盖4个数据集、4个下游任务，对比全面
- **写作质量**: 8/10 — 逻辑清晰，观察和设计选择的推导过程详实
- **价值**: 8/10 — 首次证明深度LSD可全面超越经典LSD，对底层视觉领域有重要意义

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)
- [\[ICML 2025\] Neighbour-Driven Gaussian Process Variational Autoencoders for Scalable Structured Latent Modelling](../../ICML2025/self_supervised/neighbour-driven_gaussian_process_variational_autoencoders_for_scalable_structur.md)
- [\[ECCV 2024\] Rethinking Unsupervised Outlier Detection via Multiple Thresholding](../../ECCV2024/self_supervised/rethinking_unsupervised_outlier_detection_via_multiple_thresholding.md)
- [\[ECCV 2024\] FlowCon: Out-of-Distribution Detection using Flow-Based Contrastive Learning](../../ECCV2024/self_supervised/flowcon_out-of-distribution_detection_using_flow-based_contrastive_learning.md)
- [\[ICML 2025\] Deep Learning is Not So Mysterious or Different](../../ICML2025/self_supervised/deep_learning_is_not_so_mysterious_or_different.md)

<!-- RELATED:END -->
