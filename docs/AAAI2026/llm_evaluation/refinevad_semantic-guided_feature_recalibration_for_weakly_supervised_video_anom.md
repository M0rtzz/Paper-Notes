---
title: >-
  [论文解读] RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection
description: >-
  [AAAI 2026][弱监督视频异常检测] 提出 RefineVAD 框架，通过运动感知时序注意力重校准（MoTAR）和类别导向特征精炼（CORE）两个模块，联合建模时序运动动态与异常类别语义，在弱监督视频异常检测任务上实现了对异常事件的精准定位与可解释检测。
tags:
  - AAAI 2026
  - 弱监督视频异常检测
  - 多实例学习
  - 语义引导
  - 时序建模
  - 类别原型
---

# RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection

**会议**: AAAI 2026  
**arXiv**: [2511.13204](https://arxiv.org/abs/2511.13204)  
**代码**: [GitHub](https://github.com/VisualScienceLab-KHU/RefineVAD)  
**领域**: 视频异常检测 / 弱监督学习  
**关键词**: 弱监督视频异常检测, 多实例学习, 语义引导, 时序建模, 类别原型

## 一句话总结

提出 RefineVAD 框架，通过运动感知时序注意力重校准（MoTAR）和类别导向特征精炼（CORE）两个模块，联合建模时序运动动态与异常类别语义，在弱监督视频异常检测任务上实现了对异常事件的精准定位与可解释检测。

## 研究背景与动机

弱监督视频异常检测（WVAD）仅依赖视频级标签来识别异常事件，在标注效率与实际应用之间取得平衡。现有方法通常基于多实例学习（MIL）范式，将视频视为片段的"包"，假设异常视频中至少存在一个异常片段。

然而，现有 WVAD 方法存在两个关键局限性：

**时序建模浅层且僵化**：大多依赖固定池化或简单聚合方案，无法适应真实异常中多样的运动特征。许多异常由动态、非均匀或上下文相关的运动模式定义，时序刚性严重限制了定位精度。

**语义多样性被忽视**：大多数框架将所有异常事件视为单一通用类别，忽略了不同异常类型之间的语义差异。例如打架涉及突然的双向运动，而爆炸表现为突然闪光和空间爆裂——忽略这些差异限制了模型学习判别性特征的能力。

人类感知异常时会同时利用两个互补维度：（1）上下文运动动态的时序演变，（2）对异常类型的先验知识。RefineVAD 正是模拟这种双过程推理，联合建模"运动如何演变"和"语义类别是什么"。

## 方法详解

### 整体框架

RefineVAD 遵循 MIL 设置，将输入视频分为 $T$ 个固定长度片段。每个片段通过预训练的视觉编码器（CLIP ViT-L/14）和文本编码器（InternVideo2.5）独立编码，拼接形成联合多模态表示。该表示依次经过 MoTAR 和 CORE 两个核心模块处理，最终通过轻量分类器计算片段级异常分数。

### 关键设计

1. **MoTAR（运动感知时序注意力重校准）**：核心思路是根据运动强度自适应调整时序特征的通道偏移比例。给定输入序列 $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_T] \in \mathbb{R}^{T \times D}$，首先计算相邻帧特征差异 $\Delta_t = \mathbf{x}_t - \mathbf{x}_{t-1}$，再求方差 $\mathbf{v}_t = \text{Var}(\Delta_t)$ 来衡量局部运动强度。方差越高说明运动越显著，需要更广泛的时序上下文聚合。方差向量通过轻量 MLP 预测偏移比例 $r_t = \sigma(W_3 \cdot \phi(W_2 \cdot \phi(W_1 v_t)))$，然后计算偏移通道数 $s_t = \lfloor r_t \cdot D/K \rfloor$。动态构造的偏移输出为 $\mathbf{y}_t = [\mathbf{x}_{t-1}^{(1:s_t)}, \mathbf{x}_{t+1}^{(s_t:2s_t)}, \mathbf{x}_t^{(2s_t:D)}]$。最后用轻量 Transformer 编码长程时序依赖。设计动机：传统 TSM 使用固定偏移比例，无法适应不同运动强度的帧；MoTAR 通过方差驱动的自适应偏移解决了这一问题。

2. **CORE（类别导向精炼）**：分为软类别分类和类别原型注入两个阶段。首先，MoTAR 输出的时序上下文特征经过 FC 层粗略打分，归一化后聚合为视频级特征，再通过软类别分类器得到 logits $\mathbf{z} \in \mathbb{R}^{C \times 2}$（$C$ 为异常类别数），每行对应"正常"/"异常"得分。对每个类别 $c$ 计算异常概率 $p_c^a$，并通过 softmax 得到类别权重 $w_c$。然后，利用这些权重对可学习类别原型嵌入 $\mathbf{E} \in \mathbb{R}^{C \times d_{\text{emb}}}$ 进行加权求和，得到软类别嵌入 $\mathbf{v} = \sum_{c=1}^{C} w_c \mathbf{e}_c$。该嵌入通过交叉注意力机制注入到片段级特征中：$\mathbf{x}_t^{\text{ca}} = \text{CrossAttn}(\mathbf{v}, \mathbf{x}_t^{\text{tc}}, \mathbf{x}_t^{\text{tc}})$。设计动机：将"正常"状态定义为缺乏强异常特征（而非作为第一个异常类别），避免表示空间的扭曲；使用软分类而非硬分类可以捕获模糊或重叠的异常特征。

### 损失函数 / 训练策略

总损失由三部分组成：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MIL}} + \lambda_1 \mathcal{L}_{\text{GMM}} + \lambda_2 \mathcal{L}_{\text{cat}}$$

- $\mathcal{L}_{\text{MIL}}$：Top-$k$ MIL 排序损失，促使异常片段的得分高于正常片段
- $\mathcal{L}_{\text{GMM}}$：改进的 GMM 平滑损失，注入加权类别嵌入以更清晰地反映类别特定特征
- $\mathcal{L}_{\text{cat}}$：类别分类损失，使用 BCE 鼓励语义嵌入保持类别判别性

训练时将真实类别嵌入加入软类别嵌入 $\mathbf{v}_{\text{train}} = \mathbf{v} + \mathbf{e}_y$ 以促进类别感知特化。损失权重 $\lambda_1 = 0.1$, $\lambda_2 = 0.2$。使用 AdamW 优化器，批大小 64，最大 30 epochs，单块 NVIDIA A5000 GPU。

## 实验关键数据

### 主实验

| 数据集 | 指标 | RefineVAD | 之前SOTA | 提升 |
|--------|------|-----------|----------|------|
| UCF-Crime | AUC (%) | 88.92 | 90.33 (π-VAD) | -1.41 |
| XD-Violence | AP (%) | **88.66** | 86.52 (Ex-VAD) | **+2.14** |
| UCF-Crime | mAP@0.1 (%) | **20.90** | 16.51 (Ex-VAD) | **+4.39** |

在 XD-Violence 上超过所有弱监督方法取得 SOTA（88.66% AP）。在 UCF-Crime 的细粒度 mAP@0.1 上也显著领先（20.90%）。UCF-Crime AUC 略低于 π-VAD 但仍具竞争力。

### 消融实验

| 配置 | AUC (%) | 说明 |
|------|---------|------|
| Base (MLP + MIL) | 84.60 | 基线模型 |
| + MoTAR | 85.43 | 运动感知时序调整带来+0.83 |
| + Category-Injection | 87.28 | 类别注入带来最大提升+1.85 |
| + Category-Injection + Soft-Classification | 87.85 | 软分类进一步提升+0.57 |
| + MoTAR + CORE (完整) | **88.89** | 所有模块协同最优+4.29 |

### 关键发现

1. **类别注入是最大贡献者**：单独加入 Category-Injection 就带来了 +2.68% 的 AUC 提升（84.60→87.28），是所有模块中增益最大的，证明语义类别信息对异常检测至关重要。
2. **跨数据集语义迁移能力强**：在 UCF-Crime 上训练的 CORE 模块直接迁移到 XD-Violence 仍能达到 87.52% AP（完整训练为 88.66%）；零样本跨域评估仍有 77.56% AP。
3. **t-SNE 可视化显示语义聚类合理**：纵火/爆炸/交通事故聚为一簇（场景级突变），逮捕/袭击/打架聚为一簇（多人交互），入店行窃/抢劫聚为一簇（单人行为）。

## 亮点与洞察

- 将"正常"定义为缺乏异常特征而非独立类别，避免了表示空间的扭曲，这是一个巧妙且有实际意义的设计选择。
- 软分类机制而非硬分类，能同时捕获多个类别的语义线索，处理模糊边界异常更灵活。
- MoTAR 中基于方差的运动强度估计是无参的、噪声鲁棒的，计算开销极低，适合实时应用。
- 框架的模块化设计使得各组件可独立验证贡献，工程上也便于替换升级。

## 局限与展望

- 在 UCF-Crime AUC 上未超过 π-VAD（90.33% vs 88.92%），说明视频级类别预测在高 IoU 阈值下对帧级精确边界定位存在结构性劣势。
- 类别原型数量需预先设定且与数据集绑定，面对开放世界的新异常类型扩展性有限。
- 仅在 UCF-Crime 和 XD-Violence 两个数据集上验证，未涉及更大规模或更多样化的场景。
- 文本编码器 InternVideo2.5 的引入增加了模型复杂度，未充分讨论其单独贡献。

## 相关工作与启发

- 与 VadCLIP、PEMIL 等利用 CLIP 的方法相比，RefineVAD 不依赖离散类别标签或手工 prompt，而是通过可学习原型实现软语义注入，适应性更强。
- 软原型 + 交叉注意力的设计思路可迁移到其他弱监督任务（如弱监督动作检测、弱监督目标检测），将类别先验注入特征空间来引导定位。
- 跨数据集迁移实验表明，学习到的语义空间具有泛化性，为标签高效的跨域部署提供了新思路。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 双过程推理框架模拟人类认知，MoTAR 的自适应偏移和 CORE 的软原型注入均有新意
- **技术深度**: ⭐⭐⭐⭐ — 公式推导完整，模块设计合理，损失函数设计有细节考量
- **实验充分性**: ⭐⭐⭐⭐ — 消融、跨域迁移、可视化分析丰富，但数据集偏少
- **实用性**: ⭐⭐⭐⭐ — 单 GPU 训练，计算开销低，有开源代码
- **总体**: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](../../CVPR2026/llm_evaluation/weakly_supervised_video_anomaly_detection_with_anomaly-connected_components_and_.md)
- [Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [Normal-Abnormal Guided Generalist Anomaly Detection](../../NeurIPS2025/llm_evaluation/normal-abnormal_guided_generalist_anomaly_detection.md)
- [Scalable Vision-Guided Crop Yield Estimation](scalable_vision-guided_crop_yield_estimation.md)
- [DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning](dicap_distribution-calibrated_pseudo-labeling_for_semi-supervised_multi-label_le.md)

<!-- RELATED:END -->
