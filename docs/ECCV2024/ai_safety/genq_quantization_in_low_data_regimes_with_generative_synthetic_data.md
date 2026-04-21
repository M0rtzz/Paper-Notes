---
title: >-
  [论文解读] Event Trojan: Asynchronous Event-based Backdoor Attacks
description: >-
  [ECCV2024][AI安全][backdoor attack] 提出 Event Trojan 框架，首次针对异步事件数据流设计后门攻击方法，包含不可变触发器和可变触发器两种模式，直接在事件流层面注入恶意事件实现隐蔽高效的后门攻击。
tags:
  - ECCV2024
  - AI安全
  - backdoor attack
  - event camera
  - event data
  - immutable trigger
  - mutable trigger
  - AI safety
---

# Event Trojan: Asynchronous Event-based Backdoor Attacks

**会议**: ECCV2024  
**arXiv**: [2407.06838](https://arxiv.org/abs/2407.06838)  
**代码**: 待确认  
**领域**: ai_safety  
**关键词**: backdoor attack, event camera, event data, immutable trigger, mutable trigger, AI safety

## 一句话总结

提出 Event Trojan 框架，首次针对异步事件数据流设计后门攻击方法，包含不可变触发器和可变触发器两种模式，直接在事件流层面注入恶意事件实现隐蔽高效的后门攻击。

## 背景与动机

事件相机（event camera）因其高动态范围、低延迟、高时间分辨率等优势，在自动驾驶、目标跟踪、监控、手势识别等视觉任务中被广泛使用。异步事件数据通常需要转换为图像化表示（如 Event Spike Tensor, EST）后才能输入深度学习模型。

现有的图像后门攻击方法（如 BadNets、FIBA）只能在事件表示层面注入触发器，但实际场景中攻击者通常无法获取表示模块的中间输出。此外，如果从未被篡改的原始事件流重建表示，基于表示层的后门会失效。因此，直接在原始事件数据流中注入触发器具有更高的实际威胁价值。

事件数据由大量异步事件点组成，人眼难以直接感知，这为高隐蔽性后门攻击提供了天然条件。然而，此前几乎没有研究探讨事件数据的后门攻击风险，相关任务面临潜在安全威胁。

## 核心问题

1. 如何在异步事件数据流中直接注入后门触发器，绕过对事件表示模块的依赖？
2. 如何设计触发器使其兼具高攻击成功率（ASR）和高隐蔽性？
3. 如何让触发器自适应不同事件数据的分布特征，提升泛化能力？

## 方法详解

### 威胁模型

- **攻击者能力**：无法控制模型训练细节（结构、损失函数等），但可访问部分训练数据；推理阶段仅能访问原始事件数据，无法操控推理过程或获取表示方法信息
- **攻击者目标**：创建含隐蔽后门的事件模型，注入特定触发模式时输出攻击者预设标签，同时在正常数据上保持正确预测

### 事件数据基础

每个事件 $e_k = (x_k, y_k, t_k, p_k)$ 包含空间坐标 $(x, y)$、时间戳 $t$ 和极性 $p$。当像素亮度变化超过阈值 $\sigma$ 时触发事件。事件流通过 EST 等方法转换为网格表示 $V_{\pm}(x_w, y_h, t_n)$ 后输入下游模型。

### 不可变触发器（Immutable Trigger）

核心思路是在事件流的特定空间位置和时间戳处放置恶意事件：

- 空间坐标 $(x, y)$ 从预定义区域采样
- 时间戳固定为 $\alpha$（默认 $10^{-2}$）
- 极性固定为 $\beta$（默认 $1.0$）
- 注入后在不同事件流的图像化表示中呈现一致的模式

优点是实现简单，能有效攻击大部分分类器；缺点是固定设置导致泛化能力有限，在背景噪声多或数据分布不均匀的场景下表现下降。

### 可变触发器（Mutable Trigger）

为克服不可变触发器的局限性，提出自适应学习机制：

1. **时间戳自适应**：恶意事件在不同事件流中保持相同的空间坐标（保证触发器形状一致），但采用自适应时间戳，使得触发器在图像化表示中呈现不同的像素值
2. **触发器生成器** $T_\xi(\cdot)$：基于 5 层 MLP（每层 64 通道）构建，输入从原始事件随机采样的 $m$ 个时间戳，输出恶意时间戳
3. **联合训练**：分类器与触发器生成器联合优化，分类器指导生成器学习最优触发模式

### 触发器优化损失函数

$$\mathcal{L}_T = \lambda_1 \frac{T_\xi(\mathbf{t}) \cdot \mathbf{t}}{\|T_\xi(\mathbf{t})\| \times \|\mathbf{t}\|} + \lambda_2 \psi(T_\xi(\mathbf{t}), \mathbf{t})$$

- **余弦相似度项**（$\lambda_1=1$）：降低恶意时间戳与原始时间戳的余弦相似度，增大触发器与干净数据的差异性，提升攻击能力
- **期望-方差约束项** $\psi(\cdot)$（$\lambda_2=2$）：约束恶意时间戳的期望和方差接近原始数据，防止生成的事件偏离正常数据分布太远，保持在干净数据上的分类性能

两项协同作用：余弦项负责攻击有效性，$\psi$ 项负责隐蔽性。

### 训练流程

按 Algorithm 1，每次迭代中：
1. 采样小批量事件数据
2. 通过可变触发器生成器生成毒化事件流
3. 同时优化分类器参数 $\theta$、表示模块参数 $\omega$、触发器生成器参数 $\xi$

## 实验关键数据

### 数据集

- **N-Caltech101**：Caltech101 的事件版本，101 类，4356/2612/1741（训练/验证/测试），数据量在类间差异大
- **N-Cars**：事件版汽车检测数据集，2 类，8392/2462/8608

### 主要结果（ResNet-18 上 N-Caltech101）

| 方法 | CDA↑ | ASR↑ |
|------|------|------|
| 表示触发器 BadNets | 57.24 | 0.0 |
| 表示触发器 FIBA | 82.47 | 43.39 |
| 不可变触发器 | 85.61 | 96.73 |
| **可变触发器** | **86.21** | **99.71** |

### 主要结果（N-Cars 多模型）

可变触发器在 N-Cars 上对 ResNet-18、VGG-16、Swin-S、ViT-S 均达到接近或等于 100% 的 ASR，同时 CDA 保持最高（92.72、92.93、94.76、87.17）。

### 防御实验（Neural Polarizer）

| 方法 | CDA↑ | ASR↑ |
|------|------|------|
| BadNets | 60.00 | 1.03 |
| FIBA | 15.63 | 0.0 |
| 不可变触发器 | 66.84 | 22.01 |
| **可变触发器** | **83.03** | **64.11** |

可变触发器在防御后仍保持最高的攻击成功率，因为触发器直接注入事件数据本身，良性特征与毒化特征紧密交织，难以被极化分离。

### 消融实验

- 去掉余弦相似度项（w/o cos.）：ASR 降至 11.93%，攻击能力大幅下降
- 去掉 $\psi$ 约束项：ASR 为 100% 但 CDA 降至 85.67%，对干净数据引入混淆
- 完整损失函数：CDA 86.21%、ASR 99.71%，最优平衡

## 亮点

1. **开创性工作**：首次系统研究异步事件数据的后门攻击，揭示了事件视觉任务的安全隐患
2. **两级触发器设计**：从简单固定到自适应学习，不可变触发器作为 baseline，可变触发器实现全面提升
3. **损失函数设计精巧**：余弦项与期望-方差项协同，在攻击有效性和隐蔽性之间取得良好平衡
4. **实验规模大**：22 个分类器（CNN+Transformer）× 2 数据集的全面评估
5. **对防御方法的鲁棒性**：即使面对 Neural Polarizer 等防御仍保持较高攻击成功率

## 局限与展望

1. **数据规模依赖**：在小规模、高噪声数据集（N-Caltech101）上部分模型攻击成功率不理想，如 ViT-S 仅 87.73%，Inception-V3 表现更差
2. **防御方法有限**：仅测试了 Neural Polarizer 一种防御，未测试 Spectral Signatures、Fine-Pruning、STRIP 等经典防御
3. **仅考虑分类任务**：未扩展到检测、分割等更复杂下游任务
4. **触发器空间维度固定**：可变触发器仅在时间戳维度自适应，空间坐标仍固定，可进一步探索全维度自适应
5. **缺少对真实部署场景的验证**：所有实验在公开数据集上进行，未考虑实际事件相机场景

## 与相关工作的对比

| 维度 | 图像后门攻击（BadNets/FIBA） | Event Trojan |
|------|------------------------------|--------------|
| 攻击层面 | 图像/表示层 | 原始事件数据流 |
| 攻击可行性 | 需获取表示模块输出 | 仅需访问原始事件流 |
| 隐蔽性 | 表示层可见 | 事件流人眼难感知 |
| 鲁棒性 | 重建表示即失效 | 触发器嵌入数据本身 |
| 防御抗性 | 易被极化分离 | 特征紧密交织，难分离 |

与 3D 点云后门攻击（PointBA）类似，本文利用了数据本身的异步稀疏特性来隐藏触发器，但事件数据还额外包含时间维度信息可供利用。

## 启发与关联

- 揭示了事件视觉系统在安全关键场景（自动驾驶、监控）中的脆弱性，对 AI Safety 研究有重要警示意义
- 可变触发器的自适应学习思路可推广到其他非标准数据模态（如点云、雷达信号）的对抗攻击研究
- 触发器优化损失中"有效性 vs 隐蔽性"的权衡设计模式具有通用参考价值
- 为事件数据的防御方法研究指明方向：需设计专门针对数据流层面的清洗和检测机制

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次系统性地研究事件数据后门攻击，选题新颖
- 实验充分度: ⭐⭐⭐⭐ — 22 个模型评测全面，但防御实验偏少
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ — 对事件视觉安全有重要警示，开辟新研究方向

<!-- RELATED:START -->

## 相关论文

- [SkyMask: Attack-Agnostic Robust Federated Learning with Fine-Grained Learnable Masks](skymask_attack-agnostic_robust_federated_learning_with_fine-grained_learnable_ma.md)
- [Towards Multi-modal Transformers in Federated Learning](towards_multi-modal_transformers_in_federated_learning.md)
- [Resilience of Entropy Model in Distributed Neural Networks](resilience_of_entropy_model_in_distributed_neural_networks.md)
- [Preventing Catastrophic Overfitting in Fast Adversarial Training: A Bi-level Optimization Perspective](preventing_catastrophic_overfitting_in_fast_adversarial_training_a_bi-level_opti.md)
- [Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning](fisher_calibration_for_backdoor-robust_heterogeneous_federated_learning.md)

<!-- RELATED:END -->
