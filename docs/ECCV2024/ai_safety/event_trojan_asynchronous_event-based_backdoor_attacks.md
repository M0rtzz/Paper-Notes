---
title: >-
  [论文解读] Event Trojan: Asynchronous Event-Based Backdoor Attacks
description: >-
  [ECCV2024][AI安全][backdoor attack] 提出 Event Trojan 框架，首次研究直接在异步事件数据流中注入后门触发器（immutable trigger 和 mutable trigger），揭示了事件相机视觉任务面临的后门攻击安全风险。
tags:
  - ECCV2024
  - AI安全
  - backdoor attack
  - event camera
  - event data
  - trojan trigger
  - AI safety
---

# Event Trojan: Asynchronous Event-Based Backdoor Attacks

**会议**: ECCV2024  
**arXiv**: [2407.06838](https://arxiv.org/abs/2407.06838)  
**代码**: 待确认  
**领域**: ai_safety  
**关键词**: backdoor attack, event camera, event data, trojan trigger, AI safety

## 一句话总结

提出 Event Trojan 框架，首次研究直接在异步事件数据流中注入后门触发器（immutable trigger 和 mutable trigger），揭示了事件相机视觉任务面临的后门攻击安全风险。

## 背景与动机

事件相机（event camera）因高动态范围和高时间分辨率，在自动驾驶、目标跟踪、手势识别等任务中应用日益广泛。事件数据通常需通过表示模块（如 EST）转换为类图像表示后送入深度网络。

现有后门攻击方法（如 BadNets、FIBA）只能在图像级表示上注入触发器，但事件表示模块与下游模型紧密耦合，攻击者通常无法访问中间表示。而且从未被修改的原始事件流重新构建表示时，图像级后门会失效。因此，直接毒化异步事件数据流本身具有更高的实际威胁价值。

事件数据的异步、多维、对人类难以感知等特性使得隐蔽注入恶意事件成为可能，而此前几乎没有相关安全研究。

## 核心问题

如何设计一种后门攻击方法，能够直接在异步事件数据流（而非图像表示）中注入高隐蔽性、高攻击成功率的恶意触发器？

具体挑战包括：

1. 事件数据由异步离散事件组成（每个事件含 x, y 坐标、时间戳 t、极性 p），与传统图像结构完全不同
2. 注入的触发器需要在经过任意事件表示方法转换后仍然有效
3. 需要同时保证攻击有效性和隐蔽性

## 方法详解

### 威胁模型

- **攻击者能力**：无法控制训练过程（模型结构、损失函数等），但可访问部分训练数据；推理时仅能访问原始事件流，无法操控推理过程或获知表示方法
- **攻击目标**：创建含隐蔽后门的事件视觉模型，当注入特定模式时输出攻击者预设的标签

### Immutable Trigger（不可变触发器）

在预定义的空间位置和时间戳处放置恶意事件，所有中毒样本共享相同的空间坐标和时间参数：

- 空间坐标 (x, y) 从预设区域内采样
- 时间戳统一设为 $\alpha = 10^{-2}$
- 极性统一设为 $\beta = 1.0$
- 触发器大小默认 $10 \times 10$，事件数量 $m = 100$

注入方式为简单将触发器事件拼接到原始事件流中，并修改标签为攻击者目标类。

**优点**：实现简单，对多数分类器有效  
**缺点**：固定设置忽略了不同事件数据的分布差异，在某些模型（VGG、ViT、DeiT）上攻击成功率较低

### Mutable Trigger（可变触发器）

设计自适应学习机制，通过 trigger injector $T_\xi(\cdot)$ 为每个事件流生成自适应的时间戳：

1. **空间一致性**：所有中毒样本的触发器保持相同空间坐标（保证表示中形状一致）
2. **时间自适应**：从原始事件流中随机采样 $m$ 个时间戳作为 MLP 的输入，生成自适应的恶意时间戳

**触发器生成器**：5 层 MLP，每层 64 通道

**触发器优化损失**：

$$\mathcal{L}_T = \lambda_1 \frac{T_\xi(\mathbf{t}) \cdot \mathbf{t}}{\|T_\xi(\mathbf{t})\| \times \|\mathbf{t}\|} + \lambda_2 \psi(T_\xi(\mathbf{t}), \mathbf{t})$$

- **余弦相似度项**（$\lambda_1 = 1$）：降低恶意时间戳与原始时间戳的余弦相似度，增强攻击能力
- **均值方差约束项**（$\lambda_2 = 2$）：$\psi(\cdot)$ 计算恶意与良性时间戳在期望和方差上的平方差，将生成时间戳的分布拉向原始数据，保证隐蔽性

**联合训练**：分类器 $f_\theta$ 和触发器生成器 $T_\xi$ 同步优化，分类器引导生成器学习最具攻击性的触发器。推理时生成器与分类器解耦。

### 训练细节

- 毒化比例 $\rho = 0.1$
- 优化器：SGD（学习率 $10^{-4}$，动量 0.9）
- 学习率调度：指数衰减（$\gamma = 0.5$）
- 训练 60 epochs，防御微调 20 epochs

## 实验关键数据

### 数据集

| 数据集 | 训练/验证/测试 | 类别数 | 特点 |
|--------|---------------|--------|------|
| N-Caltech101 | 4356/2612/1741 | 101 | 背景噪声多，类别不平衡 |
| N-Cars | 8392/2462/8608 | 2 | 真实场景，数据量大 |

### 主要结果（ResNet-18，N-Caltech101）

| 方法 | CDA ↑ | ASR ↑ |
|------|-------|-------|
| BadNets（表示级） | 57.24 | 0.0 |
| FIBA（表示级） | 82.47 | 43.39 |
| Immutable Trigger | 85.61 | 96.73 |
| **Mutable Trigger** | **86.21** | **99.71** |

### 主要结果（ResNet-18，N-Cars）

| 方法 | CDA ↑ | ASR ↑ |
|------|-------|-------|
| Immutable Trigger | 92.23 | 99.67 |
| **Mutable Trigger** | **92.72** | **100.0** |

### 隐蔽性比较

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| BadNets | 39.77 | 0.996 | 0.005 |
| FIBA | 27.77 | 0.459 | 0.074 |
| Immutable T. | **75.06** | **1.000** | **0.000** |
| Mutable T. | 65.45 | 1.000 | 0.000 |

### 防御实验（Neural Polarizer）

防御后 Mutable Trigger 仍保持 CDA=83.03、ASR=64.11，而 BadNets 降至 ASR=1.03、FIBA 降至 ASR=0.0，说明直接毒化事件流的攻击更难防御。

### 触发器尺寸消融

$10 \times 10$ 为性能与隐蔽性的最佳平衡点。$1 \times 10$ 和 $5 \times 5$ 的 ASR 不足 10%，$20 \times 20$ 和 $30 \times 30$ 虽然 ASR 更高但隐蔽性下降。

### 跨表示方法泛化

在 EST、Event Frame、Time Surface、Voxel Grid、Tencode 五种表示方法下均有效，ASR 最低 86.35%（Time Surface），最高 99.77%（Voxel Grid）。

## 亮点

1. **首创性**：首次研究直接在异步事件数据流上执行后门攻击，填补了事件视觉安全研究的空白
2. **双触发器设计**：immutable trigger 简单高效，mutable trigger 兼顾自适应性和隐蔽性
3. **精巧的损失函数**：余弦相似度项增强攻击性，均值方差约束保证隐蔽性，两者形成互补
4. **表示无关性**：攻击在事件流层面完成，对下游使用的表示方法无依赖
5. **强鲁棒性**：在 SOTA 防御方法 Neural Polarizer 下仍保持 64% ASR，因恶意特征与良性特征在事件流中紧密交织

## 局限与展望

1. **任务范围有限**：仅在分类任务上验证，未涉及语义分割、目标检测等更通用的事件任务
2. **数据集规模小**：N-Caltech101 仅 4356 训练样本，在 Inception-V3 等模型上 mutable trigger 表现不佳
3. **immutable trigger 泛化不足**：在含大量噪声和不平衡分布的数据集上，对 VGG、ViT、DeiT 攻击成功率低
4. **防御研究不足**：仅测试了 Neural Polarizer 一种防御方法，未考虑更多样的防御策略
5. **可扩展至多维自适应**：当前 mutable trigger 仅自适应时间戳，可探索同时自适应空间坐标和极性

## 与相关工作的对比

| 维度 | 图像后门（BadNets/FIBA） | Event Trojan |
|------|------------------------|--------------|
| 攻击层面 | 图像表示级 | 事件数据流级 |
| 实际威胁性 | 低（需访问表示模块） | 高（数据传输即可注入） |
| 隐蔽性 | PSNR 27-39 | PSNR 65-75 |
| 防御难度 | 易被 Neural Polarizer 防御 | 较难被防御 |
| 适用性 | 依赖特定表示方法 | 跨表示方法通用 |

与点云后门攻击（PointBA）相比，事件数据的时间维度为攻击提供了额外的自由度，但也带来了更复杂的设计挑战。

## 启发与关联

1. **安全意识**：事件数据由于人类难以直接感知，其安全问题可能比传统图像更严峻，公共数据集的安全审查需要新的工具和方法
2. **防御方向**：需要开发事件数据专用的后门检测方法，例如分析事件流的时空分布异常
3. **自适应攻击范式**：mutable trigger 的"联合训练分类器和触发器生成器"思路可推广到其他非图像模态的后门攻击研究
4. **多模态安全**：随着事件相机与传统相机融合使用，跨模态后门攻击的研究值得关注

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次针对异步事件数据流的后门攻击，问题定义新颖
- 实验充分度: ⭐⭐⭐⭐ — 22 个分类器、多表示方法、防御和消融实验全面，但仅两个数据集和分类任务
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，威胁模型、方法、实验层层递进
- 价值: ⭐⭐⭐⭐ — 填补事件视觉安全研究空白，对社区有重要警示价值

<!-- RELATED:START -->

## 相关论文

- [SkyMask: Attack-Agnostic Robust Federated Learning with Fine-Grained Learnable Masks](skymask_attack-agnostic_robust_federated_learning_with_fine-grained_learnable_ma.md)
- [Preventing Catastrophic Overfitting in Fast Adversarial Training: A Bi-level Optimization Perspective](preventing_catastrophic_overfitting_in_fast_adversarial_training_a_bi-level_opti.md)
- [Resilience of Entropy Model in Distributed Neural Networks](resilience_of_entropy_model_in_distributed_neural_networks.md)
- [Towards Multi-modal Transformers in Federated Learning](towards_multi-modal_transformers_in_federated_learning.md)
- [Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning](fisher_calibration_for_backdoor-robust_heterogeneous_federated_learning.md)

<!-- RELATED:END -->
