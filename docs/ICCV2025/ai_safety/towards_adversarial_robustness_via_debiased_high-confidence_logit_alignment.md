---
title: >-
  [论文解读] Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment
description: >-
  [ICCV 2025][AI安全][对抗训练] 揭示了逆向对抗攻击（inverse adversarial attack）在对抗训练中导致模型注意力偏移至背景特征的虚假相关性问题，提出 DHAT 方法通过去偏高置信度 logit 正则化（DHLR）和前景 logit 正交增强（FLOE）两个组件来消除这种偏差…
tags:
  - "ICCV 2025"
  - "AI安全"
  - "对抗训练"
  - "对抗鲁棒性"
  - "逆向对抗攻击"
  - "虚假相关性"
  - "logit 对齐"
  - "去偏"
  - "注意力偏移"
---

# Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment

**会议**: ICCV 2025  
**arXiv**: [2408.06079](https://arxiv.org/abs/2408.06079)  
**代码**: [KejiaZhang-Robust/DHAT](https://github.com/KejiaZhang-Robust/DHAT)  
**作者**: Kejia Zhang, Juanjuan Weng, Shaozi Li, Zhiming Luo (厦门大学, 暨南大学)
**领域**: AI安全  
**关键词**: 对抗训练, 对抗鲁棒性, 逆向对抗攻击, 虚假相关性, logit 对齐, 去偏, 注意力偏移

## 一句话总结

揭示了逆向对抗攻击（inverse adversarial attack）在对抗训练中导致模型注意力偏移至背景特征的虚假相关性问题，提出 DHAT 方法通过去偏高置信度 logit 正则化（DHLR）和前景 logit 正交增强（FLOE）两个组件来消除这种偏差，在 CIFAR-10/100 和 ImageNet-1K 上取得了 SOTA 的对抗鲁棒性。

## 研究背景与动机

### 对抗训练的发展

对抗训练（Adversarial Training, AT）是目前公认最有效的对抗防御方法，通过在训练过程中加入对抗样本来增强模型鲁棒性。其核心是一个 min-max 优化问题：外层最小化模型参数以降低对抗损失，内层最大化扰动以生成最强攻击。

### 逆向对抗攻击的引入与隐患

近年来，UIAT 和 ACR 等方法引入了**逆向对抗攻击**来生成比自然样本置信度更高的样本，用于将对抗样本的分布对齐到正确类别的高置信区域。然而本文发现了一个关键现象：

**逆向对抗样本的高置信输出源于偏置的特征激活**。通过 Grad-CAM 可视化发现，逆向对抗攻击虽然提高了预测置信度，但模型的注意力系统性地从前景目标（如羊）转移到无关的背景区域（如草地）。这就像"通过识别草来判断是否为羊"，是一种**虚假相关性偏差**。

### 量化验证

作者通过 IoU 度量在 ImageNet 子集上统计分析：UIAT 模型在增加逆向攻击强度后，注意力与前景的 IoU 并未提升，反而与背景的 IoU 显著增加。这种偏差导致模型过拟合背景特征，降低了鲁棒泛化性能。

### 核心洞察

逆向对抗训练不成比例地将注意力偏向背景特征，产生虚假相关性偏差——模型过度依赖与目标标签无因果关系的上下文特征。如表 5 所示，这会同时损害鲁棒性和泛化能力。

## 方法详解

### 整体框架：DHAT

DHAT（Debiased High-Confidence Adversarial Training）包含两个核心组件：

1. **DHLR** (Debiased High-Confidence Logit Regularization)：量化并消除背景特征偏差，将对抗样本的 logit 与去偏后的高置信 logit 对齐
2. **FLOE** (Foreground Logit Orthogonal Enhancement)：通过仿射空间中的正交投影减少高置信 logit 与背景特征 logit 的相关性，恢复模型对前景的关注

总损失函数：

$$\mathcal{L}_{DHAT} = \mathcal{L}_{AT}(\hat{z}, y) + \lambda_1 \cdot \mathcal{L}_{DHLR}(\check{z}^*, \hat{z}) + \lambda_2 \cdot \mathcal{L}_{FLOE}(\check{z}, \check{z}_{(B)})$$

其中 $\lambda_1 = \lambda_2 = 1.0$。

### 关键设计一：去偏高置信度 Logit 正则化 (DHLR)

**Step 1 — 分离背景特征**：利用自然样本 $x$ 的 Grad-CAM 注意力图 $M$，从逆向对抗样本 $\check{x}$ 中提取背景部分：

$$[\check{x}_{(B)}]_{(i,j)} = \mathbb{I}_{(M_{i,j} < \omega)} \cdot \check{x}_{(i,j)}$$

其中 $\omega$ 为预定义阈值，$\mathbb{I}$ 为指示函数。低注意力区域被判定为背景。

**Step 2 — 量化偏差程度**：计算背景特征的 logit：$\check{z}_{(B)} = f_\theta(\check{x}_{(B)})$，这反映了网络推理中背景激活引入的偏差程度。

**Step 3 — 去偏校准**：用逆向对抗 logit 减去背景 logit 得到去偏的高置信 logit：

$$\check{z}^* = \check{z} - \check{z}_{(B)}$$

**Step 4 — KL 散度对齐**：将对抗样本 logit 与去偏 logit 对齐：

$$\mathcal{L}_{DHLR}(\check{z}^*, \hat{z}) = \mathcal{L}_{KL}(\phi(\check{z}^*) \| \phi(\hat{z}))$$

### 关键设计二：前景 Logit 正交增强 (FLOE)

DHLR 校准了对齐目标，但未直接解决模型在处理逆向对抗样本时持续偏向背景的问题。FLOE 通过减小高置信 logit $\check{z}$ 在背景 logit $\check{z}_{(B)}$ 方向上的投影，使 $\check{z}$ 更少被 $\check{z}_{(B)}$ 解释：

$$\mathcal{L}_{FLOE}(\check{z}, \check{z}_{(B)}) = -\left|\check{z} - \frac{\check{z} \cdot \check{z}_{(B)}}{|\check{z}_{(B)}|^2} \cdot \check{z}_{(B)}\right|_p$$

直觉：最大化 $\check{z}$ 的正交分量（即非背景解释部分），迫使模型更多依赖前景特征进行预测。

### 训练流程

1. 用 PGD 生成对抗样本 $\hat{x}$
2. 用逆向 PGD 生成逆向对抗样本 $\check{x}$
3. 用 Grad-CAM 生成注意力图 $M$，分离背景特征
4. 计算三部分损失并联合优化

## 实验关键数据

### 主实验：CIFAR-10 (WRN28-10, ε=8/255)

| 方法 | Clean↑ | PGD-10↑ | C&W↑ | AA↑ | Robust Gap↓ |
|------|--------|---------|------|-----|-------------|
| MART | 82.99 | 56.25 | 52.26 | 50.67 | 9.52 |
| AWP | 82.67 | 57.80 | 54.82 | 51.90 | 6.90 |
| UIAT | 82.94 | 58.66 | 54.11 | 52.17 | 7.92 |
| SGLR | 85.76 | 57.53 | 54.28 | 52.07 | 9.38 |
| **DHAT** | 83.95 | **60.49** | 55.27 | 53.10 | **3.51** |
| **DHAT-CFA** | 84.49 | **62.67** | 55.95 | **54.05** | 6.33 |

### 主实验：CIFAR-100 (WRN28-10, ε=8/255)

| 方法 | Clean↑ | PGD-10↑ | AA↑ | Robust Gap↓ |
|------|--------|---------|-----|-------------|
| AWP | 57.94 | 34.01 | 28.90 | 7.87 |
| UIAT | 57.65 | 34.27 | 29.03 | 11.70 |
| **DHAT** | 59.14 | **35.82** | **30.17** | **4.24** |
| **DHAT-CFA** | 61.54 | **37.67** | **30.93** | 5.93 |

### 主实验：ImageNet-1K (WRN28-10, ε=4/255)

| 方法 | Clean↑ | PGD-10↑ | AA↑ | Robust Gap↓ |
|------|--------|---------|-----|-------------|
| AWP | 64.25 | 45.13 | 40.02 | 12.82 |
| UIAT | 62.64 | 45.29 | 40.18 | 14.68 |
| **DHAT** | 65.90 | **46.83** | **41.70** | **9.53** |
| **DHAT-CFA** | 66.26 | **48.27** | **42.45** | 11.64 |

### 跨架构验证 (CIFAR-10, ε=8/255)

| 方法 | ResNet-50 AA↑ | VGG-16 AA↑ | Inception-V3 AA↑ |
|------|--------------|------------|-------------------|
| UIAT | 51.00 | 45.27 | 51.23 |
| **DHAT-CFA** | **52.38** | **47.83** | **52.67** |

### 关键发现

1. **鲁棒性全面领先**：DHAT 在所有数据集、所有攻击方式上均超越现有 SOTA。CIFAR-10 PGD-10 上 DHAT 超 UIAT 1.93%，ImageNet-1K PGD-10 上超 1.54%
2. **鲁棒泛化差距大幅缩小**：Robust Gap 是衡量虚假相关性的直接指标。DHAT 在 CIFAR-10 上将 Gap 从 UIAT 的 7.92% 降至 3.51%，降幅超 55%
3. **即插即用**：DHAT 可与 AWP、CFA 等先进 AT 方法无缝结合，进一步提升性能。DHAT-CFA 在所有实验中表现最佳
4. **跨架构一致**：在 ResNet-50、VGG-16、Inception-V3 上均有提升，验证了方法的通用性
5. **抗不同攻击强度**：在 PGD 和 C&W 不同 ε 下 DHAT 保持稳定优势，性能衰减更平缓

## 亮点与洞察

1. **发现了逆向对抗训练的根本缺陷**：揭示了"高置信不等于正确关注"的关键洞察——逆向对抗攻击通过放大背景特征激活来提高置信度，这是此前工作完全忽视的
2. **因果推理视角**：从因果推断角度分析虚假相关性，将注意力偏移与鲁棒泛化差距建立联系，提供了比单纯提高准确率更深入的理解
3. **Robust Gap 作为虚假相关性指标**：训练集和测试集间的鲁棒性差距直接反映虚假关联程度，DHAT 显著缩小该差距证明方法从根本上减少了虚假相关依赖
4. **设计简洁有效**：DHLR 和 FLOE 各只需一行额外损失计算，无需修改网络结构或攻击流程，即插即用性极强
5. **逻辑闭环**：发现问题（注意力偏移）→ 量化度量（IoU 分析）→ 提出解决方案（去偏+正交）→ 验证缓解效果（Gap 下降），论证完整有力

## 局限性

1. **额外计算开销**：需要生成逆向对抗样本、计算 Grad-CAM、前向传播背景特征，训练成本高于标准 AT
2. **注意力图的硬阈值**：通过固定阈值 $\omega$ 将注意力图二值化分割前景/背景，过于简化，难以处理前景/背景边界模糊的情况
3. **与依赖虚假相关的方法不兼容**：DHAT 不适合与 FSR、SGLR 等利用非鲁棒特征的方法结合，限制了组合灵活性
4. **Grad-CAM 质量依赖**：背景分离质量取决于 Grad-CAM 的准确性，而 Grad-CAM 本身在对抗训练模型上的可靠性存疑
5. **仅在分类任务验证**：未在目标检测、分割等任务上测试对抗鲁棒性提升

## 相关工作

- **对抗攻击**：FGSM → PGD → C&W → AutoAttack，攻击方法持续演进
- **对抗训练**：MART（对齐对抗与自然样本 logit）、TRADES（鲁棒性-准确率权衡）、AWP（权重扰动平滑损失面）、FSR（利用非鲁棒特征的有用信息）、CFA（类级公平对抗训练）、SGLR（自蒸馏 soft-label 校准）
- **逆向对抗训练**：UIAT 和 ACR 通过生成高置信样本引导对抗样本分布，本文发现其隐患并提出修正
- **虚假相关性**：从因果推断视角分析模型对非因果特征的依赖

## 评分

| 维度 | 分数 (1-10) |
|------|-------------|
| 新颖性 | 9 — 首次揭示逆向对抗训练的注意力偏移问题 |
| 理论深度 | 6 — 主要是实验驱动的发现，缺乏形式化理论 |
| 实验充分性 | 9 — 多数据集、多架构、多攻击方式全面验证 |
| 实用价值 | 8 — 即插即用，可与现有 AT 方法组合 |
| 写作质量 | 8 — 问题发现到解决方案的叙事清晰 |
| **总评** | **8** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Fair Representation Learning with Controllable High Confidence Guarantees via Adversarial Inference](../../NeurIPS2025/ai_safety/fair_representation_learning_with_controllable_high_confidence_guarantees_via_ad.md)
- [\[NeurIPS 2025\] Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits](../../NeurIPS2025/ai_safety/understanding_and_improving_adversarial_robustness_of_neural_probabilistic_circu.md)
- [\[NeurIPS 2025\] Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness](../../NeurIPS2025/ai_safety/bridging_symmetry_and_robustness_on_the_role_of_equivariance_in_enhancing_advers.md)
- [\[NeurIPS 2025\] Boosting Adversarial Transferability with Spatial Adversarial Alignment](../../NeurIPS2025/ai_safety/boosting_adversarial_transferability_with_spatial_adversarial_alignment.md)
- [\[ICCV 2025\] Backdooring Self-Supervised Contrastive Learning by Noisy Alignment](backdooring_self-supervised_contrastive_learning_by_noisy_alignment.md)

</div>

<!-- RELATED:END -->
