---
title: >-
  [论文解读] ACAM-KD: Adaptive and Cooperative Attention Masking for Knowledge Distillation
description: >-
  [ICCV 2025][模型压缩][知识蒸馏] 提出 ACAM-KD，一种自适应学生-教师协作注意力掩码知识蒸馏方法，通过跨注意力特征融合（STCA-FF）和自适应空间-通道掩码（ASCM）动态调整蒸馏焦点，在 COCO 检测上超越 SOTA 最高 1.4 mAP，在 Cityscapes 分割上提升 3.09 mIoU。
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "知识蒸馏"
  - "注意力掩码"
  - "跨注意力融合"
  - "空间-通道选择"
  - "目标检测"
---

# ACAM-KD: Adaptive and Cooperative Attention Masking for Knowledge Distillation

**会议**: ICCV 2025  
**arXiv**: N/A (CVF OpenAccess)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 知识蒸馏, 注意力掩码, 跨注意力融合, 空间-通道选择, 目标检测

## 一句话总结

提出 ACAM-KD，一种自适应学生-教师协作注意力掩码知识蒸馏方法，通过跨注意力特征融合（STCA-FF）和自适应空间-通道掩码（ASCM）动态调整蒸馏焦点，在 COCO 检测上超越 SOTA 最高 1.4 mAP，在 Cityscapes 分割上提升 3.09 mIoU。

## 研究背景与动机

**为什么现有特征蒸馏方法的注意力选择存在问题？** 现有方法的核心问题可归结为三点：

**静态性**：对于同一张图像，蒸馏的关注区域在整个训练过程中保持不变。即使学生在 Epoch 12 已经很好地学会了某些区域的特征，到 Epoch 24 仍然被强制关注同样的区域——这是低效的，甚至可能有害。

**单向性**：知识选择完全由教师驱动（FKD、MasKD）或人工启发式（Ground-truth 边界框、RPN 区域）。教师认为重要的区域未必是学生当前学习阶段最需要的。例如 Figure 1 展示：学生在 Epoch 12 的注意力定位甚至**优于教师**，但到 Epoch 24 却退化为模仿教师的次优注意力。

**空间-通道割裂**：大多数方法仅做空间维度的特征选择，忽略了不同通道的贡献差异。

**为什么需要"协作"？** 知识蒸馏不应是学生被动接收教师知识的过程，而应该是双方**动态互动**的过程——教师提供方向，学生根据自身学习状态反馈需求，掩码随之调整。

## 方法详解

### 整体框架

ACAM-KD 包含两个核心模块，依次作用：

1. **STCA-FF**（Student-Teacher Cross-Attention Feature Fusion）：通过跨注意力融合教师和学生的特征，产生交互特征
2. **ASCM**（Adaptive Spatial-Channel Masking）：在融合特征上生成自适应的空间掩码和通道掩码

关键理念：掩码不再是固定的或由教师单方面决定的，而是从**教师-学生交互特征**中动态生成，随着学生的学习状态实时变化。

### 关键设计

#### Student-Teacher Cross-Attention Feature Fusion (STCA-FF)

给定教师和学生的特征图 $F_T, F_S \in \mathbb{R}^{C \times H \times W}$：

$$Q = W_q F_T, \quad K = W_k F_S, \quad V = W_v F_S$$

其中 $W_q \in \mathbb{R}^{C_q \times C}$，$W_k \in \mathbb{R}^{C_q \times C}$ 将通道降维到 $C_q = C/2$，$W_v \in \mathbb{R}^{C \times C}$ 保持原始维度。

**为什么教师做 Query、学生做 Key/Value？** 教师作为更有知识的模型，通过 Query 引导"关注哪里"；学生提供 Key/Value，代表其当前的特征状态。这样注意力矩阵反映的是"教师关注的位置在学生特征空间中的对应"。消融实验（Table 9）证实这种配置比反过来高 0.2 mAP。

注意力矩阵计算：

$$A = \text{softmax}\left(\frac{QK}{\sqrt{C_q}}\right) \in \mathbb{R}^{HW \times HW}$$

融合特征：$F_{fused} = AV \in \mathbb{R}^{C \times H \times W}$

#### Adaptive Spatial-Channel Masking (ASCM)

在融合特征上引入**可学习的选择单元**，动态生成两种掩码：

**通道掩码**：$M^c = \sigma(m_c \cdot v)$
- $m_c \in \mathbb{R}^{M \times 1}$：M 个通道选择单元
- $v \in \mathbb{R}^{1 \times C}$：$F_{fused}$ 的空间平均池化向量

**空间掩码**：$M^s = \sigma(m_s \cdot z)$
- $m_s \in \mathbb{R}^{M \times C}$：M 个空间选择单元
- $z \in \mathbb{R}^{C \times HW}$：$F_{fused}$ 的空间展平

**为什么用 M 个掩码而非单一掩码？** 多个掩码可以捕捉不同的特征重要性模式。检测任务中 M=6；分割任务中 M=19（等于类别数，每个掩码对应一个语义类别）。

**掩码多样性损失**：为防止所有掩码收敛到相似模式，引入基于 Dice 系数的多样性损失：

$$L_{div} = \frac{2 \sum_{i=1}^M \sum_{j=1, j \neq i}^M M_i \cdot M_j}{\sum_{i=1}^M M_i^2 + \sum_{j=1}^M M_j^2}$$

### 损失函数 / 训练策略

**通道蒸馏损失**：

$$L^c_{distill} = \frac{1}{M} \sum_{m=1}^M \frac{1}{HW} \sum_{k=1}^C M^c_{m,k} \| M^c_m \odot (F_T - f_{align}(F_S)) \|_2^2$$

**空间蒸馏损失**：

$$L^s_{distill} = \frac{1}{M} \sum_{m=1}^M \frac{1}{C} \sum_{p=1}^{H \times W} M^s_{m,p} \| M^s_m \odot (F_T - f_{align}(F_S)) \|_2^2$$

**总损失**：

$$L = L_{task} + \alpha (L^c_{distill} + L^s_{distill}) + \lambda L_{div}$$

其中 $\alpha = 1$，$\lambda = 1$。检测任务在 FPN neck 层做特征蒸馏，分割任务在预测分割图上做蒸馏。

训练细节：
- 检测：SGD，momentum=0.9，weight decay=0.0001，采用 inheritance 策略稳定早期训练
- 分割：SGD，weight decay=5e-4，polynomial annealing 学习率调度，初始 LR=0.02，40K 迭代

## 实验关键数据

### 主实验

**COCO 目标检测——ResNet-101 教师（Table 1）：**

| 方法 | RetinaNet mAP | Faster R-CNN mAP | RepPoints mAP |
|------|--------------|-----------------|---------------|
| Student baseline (R50) | 37.4 | 38.4 | 38.6 |
| FGD | 39.6 | 40.4 | 41.0 |
| MasKD | 39.8 | 40.8 | 41.1 |
| FreeKD | 39.9 | 40.8 | - |
| **ACAM-KD (Ours)** | **41.2** (+1.3) | **41.4** (+0.6) | **42.5** (+1.4) |

**COCO 目标检测——ResNeXt-101 教师（Table 2）：**

| 方法 | RetinaNet mAP | Faster R-CNN mAP | RepPoints mAP |
|------|--------------|-----------------|---------------|
| Student baseline (R50) | 37.4 | 38.4 | 38.6 |
| MasKD | 40.9 | 42.4 | 41.8 |
| FreeKD | 41.0 | 42.4 | 42.4 |
| **ACAM-KD (Ours)** | **41.5** | **42.6** | **42.8** |

从 R50 学生到 X101 教师，ACAM-KD 使学生 mAP 提升高达 **4.2 点**。

**Cityscapes 语义分割（Tables 3-5）：**

| 学生模型 | Baseline mIoU | MasKD | FreeKD | **ACAM-KD** |
|---------|-------------|-------|--------|-------------|
| DeepLabV3-R18 | 72.96 | 77.00 | 76.45 | **77.53** (+0.53) |
| DeepLabV3-MBV2 | 73.12 | 75.26 | - | **76.21** (+0.95) |
| PSPNet-R18 | 72.55 | 75.34 | - | **75.99** (+0.65) |

### 消融实验

**空间 vs 通道掩码（Table 8）：**

| 掩码策略 | mAP | AP_s | AP_m | AP_l |
|---------|-----|------|------|------|
| Baseline (R50) | 37.4 | 20.0 | 40.7 | 49.7 |
| 仅空间 | 40.9 | **25.4** | 44.3 | 52.3 |
| 仅通道 | 40.4 | 24.5 | 44.1 | 52.3 |
| **空间 + 通道** | **41.2** | 24.6 | **45.5** | **54.1** |

空间掩码对小物体（AP_s）帮助最大（+5.4），二者结合在中大物体上进一步提升。

**固定 vs 自适应掩码（Table 10）：**

| 掩码策略 | mAP | AP_s | AP_m | AP_l |
|---------|-----|------|------|------|
| 无掩码 | 37.4 | 20.8 | 40.8 | 50.9 |
| 教师离线固定掩码（MasKD 风格） | 39.8 | 21.5 | 43.9 | 54.0 |
| 教师自适应掩码 | 39.9 | 21.7 | 43.7 | 53.9 |
| **ACAM-KD（协作自适应）** | **41.2** | **24.6** | **45.5** | **54.1** |

关键发现：仅让掩码可学习（39.8 → 39.9）提升微乎其微；但加入学生-教师交互后（39.9 → 41.2）提升巨大，说明**协作机制才是性能提升的根本来源**。

**Cross-attention 中 Query 来源（Table 9）：**

| Query 来源 | mAP | AP_s | AP_m | AP_l |
|-----------|-----|------|------|------|
| 来自学生 | 41.0 | 24.4 | 45.1 | 54.0 |
| **来自教师** | **41.2** | **24.6** | **45.5** | **54.1** |

### 关键发现

1. **协作 > 教师主导**：学生-教师协作掩码比纯教师驱动掩码高 1.3+ mAP
2. **空间 + 通道互补**：二者结合优于任一单独使用
3. **自适应 ≠ 可学习**：仅让教师掩码可学习提升极小，关键在于引入学生的参与
4. **跨架构泛化**：在 RetinaNet、Faster R-CNN、RepPoints 三种检测器上一致有效
5. **跨任务泛化**：检测和分割任务上均达到 SOTA

## 亮点与洞察

- **重新思考蒸馏中的主动性**：学生不应只是被动接受者，而应主动参与知识选择
- **掩码动态随训练演化的设计**：不同 epoch 对同一图像生成不同的蒸馏掩码
- **简洁的消融验证**：Table 10 清晰地隔离了"可学习"和"协作"各自的贡献
- **FPN neck 蒸馏 + 分割图蒸馏**：对两种任务采用不同的蒸馏位置是合理的适配
- **多样性损失避免掩码退化**：简单有效的正则化，Figure 3 可视化展示了掩码的互补性

## 局限与展望

- Cross-attention 模块增加了计算开销（额外的 Q/K/V 投影和注意力计算），文中未单独报告训练时间增加
- 仅在 CNN 架构（ResNet 系列）上验证，未涉及 ViT/Swin 等 Transformer 架构
- M 的选择（检测 M=6，分割 M=19）看起来是经验设定，缺乏敏感性分析
- $\alpha$ 和 $\lambda$ 固定为 1，未探索不同超参的影响
- 仅在 COCO 和 Cityscapes 上验证，未涉及其他密集预测任务（如深度估计、全景分割）

## 相关工作与启发

与 MasKD（ICLR'23，可学习 receptive tokens 但离线固定）和 FreeKD（CVPR'24，频域 prompt 引导）相比，ACAM-KD 的核心创新在于**在线协作**——掩码随训练过程实时更新。与 CWD（通道蒸馏）相比，ACAM-KD 同时优化空间和通道维度。这一工作的启发在于：**让学生模型在蒸馏中有更多"发言权"** 可能是提升蒸馏效率的关键方向。

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SDKD: Frequency-Aligned Knowledge Distillation for Lightweight Spatiotemporal Forecasting](frequency-aligned_knowledge_distillation_for_lightweight_spatiotemporal_forecast.md)
- [\[ICCV 2025\] PBCAT: Patch-Based Composite Adversarial Training against Physically Realizable Attacks on Object Detection](pbcat_patch-based_composite_adversarial_training_against_physically_realizable_a.md)
- [\[ICCV 2025\] Where, What, Why: Towards Explainable Driver Attention Prediction](where_what_why_towards_explainable_driver_attention_prediction.md)
- [\[ICCV 2025\] Passing the Driving Knowledge Test](passing_the_driving_knowledge_test.md)
- [\[ICCV 2025\] AdaDrive: Self-Adaptive Slow-Fast System for Language-Grounded Autonomous Driving](adadrive_self-adaptive_slow-fast_system_for_language-grounded_autonomous_driving.md)

</div>

<!-- RELATED:END -->
