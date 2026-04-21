---
title: >-
  [论文解读] PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection
description: >-
  [CVPR 2025][AI安全][后门检测] 提出 PSBD 方法，发现被植入后门的模型在推理时开启 dropout 后，干净数据的预测会偏移向目标类别而后门数据预测保持稳定（Prediction Shift 现象），基于此设计 Prediction Shift Uncertainty (PSU) 指标实现 SOTA 后门训练数据检测。
tags:
  - CVPR 2025
  - AI安全
  - 后门检测
  - 预测偏移
  - Dropout不确定性
  - 神经元偏置
  - 对抗鲁棒性
---

# PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection

**会议**: CVPR 2025  
**arXiv**: [2406.05826](https://arxiv.org/abs/2406.05826)  
**代码**: [GitHub](https://github.com/WL-619/PSBD)  
**领域**: AI安全 (AI Safety / Backdoor Detection)  
**关键词**: 后门检测, 预测偏移, Dropout不确定性, 神经元偏置, 对抗鲁棒性

## 一句话总结

提出 PSBD 方法，发现被植入后门的模型在推理时开启 dropout 后，干净数据的预测会偏移向目标类别而后门数据预测保持稳定（Prediction Shift 现象），基于此设计 Prediction Shift Uncertainty (PSU) 指标实现 SOTA 后门训练数据检测。

## 研究背景与动机

深度神经网络容易受到后门攻击——攻击者在训练数据中插入含有特定触发器的恶意样本，使模型在遇到触发器时做出攻击者指定的错误预测，而在正常输入上表现正常。后门攻击的隐蔽性使其在安全关键领域（自动驾驶、医疗等）构成严重威胁。

现有防御策略主要集中在三个方向：模型重构（去除后门影响）、模型检测（判断模型是否被植入后门）、和毒化抑制。但在最源头的**后门训练数据检测**方向，现有方法普遍面临两个问题：要么 TPR（真正率）低——漏检后门样本，要么 FPR（假正率）高——误删干净样本。现有方法如 Spectral Signatures、STRIP、Scale-up 主要在**数据层面**操作（改变输入、分析表征），未充分利用模型本身的内在属性。

本文提供了一个全新视角——**模型预测不确定性**。作者发现了一个引人注目的 Prediction Shift (PS) 现象：在推理阶段开启 dropout 后，被投毒模型对干净数据的预测会从正确标签偏移到目标类别，而后门数据的预测保持稳定。这种现象源于"神经元偏置"效应——训练过程中某些神经元路径变得偏向特定类别。基于此洞察，PSBD 通过计算 PSU 实现了简单高效的后门数据检测。

## 方法详解

### 整体框架

PSBD 的工作流程：（1）在可疑训练集上用标准监督学习训练模型（可选数据增强）；（2）通过自适应策略选择合适的 dropout rate $p$；（3）对训练数据和少量无标签干净验证数据计算 PSU 值；（4）基于阈值 $T$（验证集 PSU 的第 25 百分位）将 PSU 低于阈值的样本判定为后门样本。

### 关键设计

1. **Prediction Shift (PS) 现象发现**:
    - 功能：揭示干净数据与后门数据在开启 dropout 后的行为差异
    - 核心思路：定义预测偏移函数 $\phi_{PS}(\mathbf{x}) = \mathbb{I}(\mathcal{Y}(\mathbf{x};\boldsymbol{\theta}) \neq \mathcal{Y}(\mathbf{x};\boldsymbol{\theta}'))$，以及偏移率 $\sigma(\mathcal{D}) = \frac{1}{k|\mathcal{D}|}\sum_{\mathbf{x} \in \mathcal{D}} \phi_{PS}(\mathbf{x})$。实验发现：在合适的 dropout rate $p$ 下，干净数据的 $\sigma$ 达到约 0.8 且几乎全部偏移到攻击目标类别，而后门数据的 $\sigma$ 接近 0。这种差异在 BadNets、WaNet 等多种攻击下一致存在
    - 设计动机：标准 MC-Dropout 不确定性（标准差）在高级攻击（如 WaNet）下失效，PS 现象提供了更鲁棒的区分信号

2. **神经元偏置效应（Neuron Bias Effect）**:
    - 功能：解释 PS 现象的机制
    - 核心思路：后门训练使网络中某些路径偏向目标类别。无 dropout 时，干净数据有足够特征做正确预测；开启 dropout 后，关键区分特征被丢弃，模型依赖训练形成的神经元偏置，将干净数据分到目标类别。后门数据的触发器特征更稳定和显著，即使部分特征被 dropout 丢弃仍能正确分类到目标类别。作者通过可视化最后一层 512 个 feature map 验证：开启 dropout 后干净图和后门图的特征变得几乎完全相同
    - 设计动机：为 PS 现象提供理论解释，增强方法的可信度和可解释性

3. **Prediction Shift Uncertainty (PSU) 检测方法**:
    - 功能：量化预测偏移强度用于后门检测
    - 核心思路：PSU 计算无 dropout 时最高置信类别 $c$ 的置信度与 $k$ 次 dropout 推理中该类别平均置信度的差值：$\phi_{PSU}(\mathbf{x}) = P_c(\mathbf{x};\boldsymbol{\theta}) - \frac{1}{k}\sum_{i=1}^{k}P_c(\mathbf{x};p,\boldsymbol{\theta}_i')$。干净数据 PSU 值高（预测偏移大），后门数据 PSU 值低（预测稳定）。阈值 $T$ 设为验证集 PSU 的第 25 百分位；dropout rate $p$ 通过自适应策略选择——找到验证集 $\sigma$ 接近 0.8 且训练集/验证集偏移率差异最大的 $p$ 值
    - 设计动机：不仅考虑标签变化（PS），还考虑置信度变化，捕获更细粒度的信号（某些干净样本标签不变但置信度显著下降）

### 损失函数 / 训练策略

- **标准交叉熵损失**: 在可疑训练集上正常训练
- **Dropout 位置**: 在 ResNet 每个残差连接后、激活函数前添加 dropout 层
- **推理次数**: $k=3$ 次前向推理
- **数据增强**: 当模型泛化能力不足时使用（如 Tiny ImageNet、Adaptive-Blend），可增强神经元偏置
- **模型选择**: 使用训练后期模型（增强数据拟合和神经元偏置路径）

## 实验关键数据

### 主实验

CIFAR-10 数据集（10% 投毒率，TPR↑ / FPR↓）：

| 攻击方法 | PSBD (Ours) | SS | STRIP | SCAN | SCP | CD-L |
|---------|-------------|-----|-------|------|-----|------|
| BadNets | **1.000/0.104** | 0.389/0.512 | 1.000/0.113 | 1.000/0.009 | 1.000/0.205 | 0.998/0.158 |
| WaNet | **1.000/0.116** | 0.456/0.505 | 0.050/0.101 | 0.891/0.034 | 0.869/0.251 | 0.863/0.144 |
| Adaptive-Blend | **0.982/0.184** | 0.608/0.145 | 0.014/0.069 | 0.000/0.023 | 0.721/0.257 | 0.432/0.167 |
| **Average** | **0.994/0.136** | 0.439/0.456 | 0.689/0.107 | 0.832/0.013 | 0.899/0.244 | 0.855/0.157 |

### 消融实验

| 配置 | 说明 |
|------|------|
| MC-Dropout 标准差 | 在 WaNet 等攻击下失效，后门/干净不确定性接近 |
| PS（仅标签变化） | 有效但粒度不够，部分干净样本标签不变但置信度变化大 |
| PSU（标签+置信度） | 最细粒度，覆盖率最高 |
| 无数据增强 (Tiny ImageNet) | 模型泛化不足时检测效果下降 |
| 有数据增强 (Tiny ImageNet) | 增强神经元偏置，显著提升检测效果 |

### 关键发现

- PSBD 在 7 种攻击 × 3 个数据集上平均 TPR 最高，尤其在高级攻击（WaNet、Adaptive-Blend）上优势巨大——STRIP 在 WaNet 上 TPR 仅 0.050，SCAN 在 Adaptive-Blend 上 TPR 为 0
- 仅需 5% 训练集大小的无标签干净验证数据
- PS 偏移方向几乎全部指向攻击目标类别（class 0），这一规律性令人惊讶
- 数据增强可以增强神经元偏置效应，有助于检测

## 亮点与洞察

- **Prediction Shift 现象的发现极具启发性**：dropout 使干净数据"坠入"后门目标类别的引力场，这揭示了后门攻击在权重空间中留下的深刻痕迹，为理解后门机制提供了新视角
- **方法极其简洁实用**：仅需在推理时开启 dropout 做 3 次前向传播计算 PSU，不需要额外训练任何辅助模型或优化触发器模板，时间开销极小
- **对高级攻击的鲁棒性突出**：在 STRIP 和 SCAN 完全失效的 WaNet 和 Adaptive-Blend 上仍保持 >0.98 的 TPR

## 局限与展望

- FPR 偏高（平均约 13-20%），可能误删部分干净训练数据
- dropout rate $p$ 的自适应选择依赖启发式阈值（$\sigma$ 接近 0.8），在某些场景下可能不准确
- 在 Tiny ImageNet 等复杂数据集上需要配合数据增强才能获得好效果
- 未验证在更大规模模型（如 ViT-Large）和更多样化攻击下的表现

## 相关工作与启发

- **vs Spectral Signatures (SS)**: SS 利用特征统计区分干净/后门，但 TPR 普遍低（平均仅 0.44）；PSBD 利用模型级不确定性，TPR 达 0.99
- **vs STRIP**: STRIP 通过混合样本分析预测熵，在 WaNet 上完全失效（TPR=0.050）；PSBD 对所有攻击类型保持高 TPR
- **vs SCAN**: SCAN 在 CIFAR-10 的 FPR 极低（0.013），但在 Adaptive-Blend 上 TPR 为 0，且计算开销大（GTSRB 超时）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ PS 现象发现和神经元偏置假设极具原创性
- 实验充分度: ⭐⭐⭐⭐ 7 种攻击 × 3 数据集 × 6 种 baseline，10 次重复实验
- 写作质量: ⭐⭐⭐⭐ 从 pilot study 到发现再到方法的叙事逻辑流畅
- 价值: ⭐⭐⭐⭐⭐ 简洁高效的方法在高级攻击检测上取得突破性进展

<!-- RELATED:START -->

## 相关论文

- [Disparate Conditional Prediction in Multiclass Classifiers](../../ICML2025/ai_safety/disparate_conditional_prediction_in_multiclass_classifiers.md)
- [Rethinking VLMs for Image Forgery Detection and Localization](rethinking_vlms_for_image_forgery_detection_and_localization.md)
- [Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs](../../ICML2025/ai_safety/is_your_model_fairly_certain_uncertainty-aware_fairness_evaluation_for_llms.md)
- [INACTIVE: Invisible Backdoor Attack against Self-supervised Learning](invisible_backdoor_attack_against_self-supervised_learning.md)
- [Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)

<!-- RELATED:END -->
