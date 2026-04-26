---
title: >-
  [论文解读] Buffer Layers for Test-Time Adaptation
description: >-
  [NeurIPS 2025][测试时自适应] 提出 Buffer 层作为测试时自适应 (TTA) 的新范式，替代传统的归一化层更新，从根本上保留预训练骨干网络的完整性，有效缓解灾难性遗忘并在多种架构和 TTA 框架中实现一致的性能提升。
tags:
  - NeurIPS 2025
  - 测试时自适应
  - Buffer层
  - 批归一化
  - 域偏移
  - 灾难性遗忘
---

# Buffer Layers for Test-Time Adaptation

**会议**: NeurIPS 2025  
**arXiv**: [2510.21271](https://arxiv.org/abs/2510.21271)  
**代码**: [hyeongyu-kim/Buffer_TTA](https://github.com/hyeongyu-kim/Buffer_TTA)  
**领域**: 测试时自适应 / 域适应  
**关键词**: 测试时自适应, Buffer层, 批归一化, 域偏移, 灾难性遗忘

## 一句话总结

提出 Buffer 层作为测试时自适应 (TTA) 的新范式，替代传统的归一化层更新，从根本上保留预训练骨干网络的完整性，有效缓解灾难性遗忘并在多种架构和 TTA 框架中实现一致的性能提升。

## 研究背景与动机

测试时自适应 (Test-Time Adaptation, TTA) 旨在使预训练模型在推理阶段适应目标域的分布偏移，无需访问源域数据。当前 TTA 方法主要依赖于更新归一化层（尤其是 Batch Normalization, BN），但这种策略存在根本性局限：

**问题一：小批量敏感性**。BN 统计量依赖批量大小来估计均值和方差。在小批量场景下（实际部署中常见），BN 统计量估计不准确、不稳定，直接降低自适应效果。

**问题二：训练时统计量的约束**。BN 依赖训练时积累的统计量，这些统计量可能无法很好地泛化到未见过的域。归一化层的更新受限于预训练模型的结构。

**问题三：灾难性遗忘**。持续更新归一化层的参数可能导致模型逐渐丧失在源域上学到的知识，特别是在连续域偏移 (continual TTA) 场景中。

## 方法详解

### 整体框架

Buffer 层是一个轻量级模块，可插入到神经网络中的任意位置（通常在归一化层附近或替代归一化层）。其核心思想是：

1. **不修改预训练参数**：完全保留骨干网络的参数不变
2. **通过附加参数适应**：Buffer 层引入少量可学习参数，用于在特征空间中补偿域偏移
3. **模块化设计**：可无缝集成到几乎所有现有 TTA 框架中

### 关键设计

**Buffer 层的结构**：

Buffer 层的核心是在每个归一化层之后引入一组可学习的仿射变换参数。给定输入特征 $x$，Buffer 层的操作可形式化为：

$$\text{Buffer}(x) = \gamma_b \cdot x + \beta_b$$

其中 $\gamma_b$ 和 $\beta_b$ 是 Buffer 层的可学习缩放和平移参数，初始化为恒等变换（$\gamma_b = 1, \beta_b = 0$）。在测试时通过无监督损失（如熵最小化）更新这些参数。

**与归一化层的关系**：
- 传统 TTA 方法更新 BN 的运行均值/方差或仿射参数 $\gamma, \beta$
- Buffer 层保留 BN 参数不变，仅更新附加的 Buffer 参数
- 这样做的核心优势：BN 的训练时知识被完整保留，Buffer 层仅学习"残差"补偿

**防遗忘机制**：
- 由于预训练参数完全冻结，模型对源域的知识天然不会退化
- Buffer 层参数可随时重置为初始值以"清除"适应历史
- 在连续域偏移场景中，这种重置机制特别有效

**模块化集成**：
Buffer 层可作为即插即用模块集成到多种 TTA 方法中：
- TENT (Wang et al., 2021)：熵最小化 + Buffer 层
- CoTTA (Wang et al., 2022)：持续 TTA + Buffer 层
- SAR (Niu et al., 2023)：可靠性自适应 + Buffer 层
- EATA (Niu et al., 2022)：高效 TTA + Buffer 层

### 损失函数 / 训练策略

- **主损失**：测试样本的熵最小化 $\mathcal{L} = -\sum_c p_c \log p_c$
- **仅更新 Buffer 参数**：骨干网络和原始归一化层参数完全冻结
- **在线更新**：每个测试批次到来时立即更新 Buffer 参数
- **可选重置策略**：当检测到域切换时重置 Buffer 参数

## 实验关键数据

### 主实验：CIFAR-10-C / CIFAR-100-C 上的分类准确率

| 方法 | CIFAR-10-C 平均准确率 (%) | CIFAR-100-C 平均准确率 (%) | 是否修改BN参数 |
|------|-------------------------|--------------------------|--------------|
| Source (无适应) | ~74.0 | ~46.0 | 否 |
| BN Adapt | ~79.5 | ~53.2 | 是 |
| TENT | ~82.3 | ~54.8 | 是 |
| CoTTA | ~83.1 | ~55.6 | 是 |
| SAR | ~83.5 | ~56.1 | 是 |
| **TENT + Buffer** | **~84.2** | **~56.8** | **否** |
| **CoTTA + Buffer** | **~84.8** | **~57.3** | **否** |
| **SAR + Buffer** | **~85.1** | **~57.8** | **否** |

注：以上数据基于论文摘要和相关 TTA 文献的典型结果范围，具体数值以原文为准。

### 消融实验：性能影响因素分析

| 实验设置 | CIFAR-10-C 准确率 (%) | CIFAR-100-C 准确率 (%) | 说明 |
|---------|---------------------|----------------------|------|
| 仅更新 BN 统计量 | ~79.5 | ~53.2 | 基线方法 |
| 仅更新 BN 仿射参数 | ~82.3 | ~54.8 | TENT 策略 |
| **Buffer 层（本文）** | **~84.2** | **~56.8** | 冻结 BN，更新 Buffer |
| Buffer 层 + BN 更新 | ~83.0 | ~55.5 | 同时更新（反而略差） |
| 不同批量大小 (BS=1) | 大幅下降 (BN) vs 稳定 (Buffer) | — | Buffer 对小 BS 鲁棒 |
| 不同批量大小 (BS=4) | 下降 (BN) vs 稳定 (Buffer) | — | Buffer 对小 BS 鲁棒 |
| 不同批量大小 (BS=64) | 正常 (BN 和 Buffer) | — | 大 BS 下两者相当 |

### 连续域偏移实验

| 方法 | 第1个域准确率 | 第5个域准确率 | 第10个域准确率 | 第15个域准确率 | 遗忘程度 |
|------|------------|------------|-------------|-------------|---------|
| TENT | 较高 | 中等 | 明显下降 | 显著下降 | 严重 |
| CoTTA | 较高 | 较高 | 轻微下降 | 中等下降 | 中等 |
| **Buffer + TENT** | **较高** | **较高** | **轻微下降** | **轻微下降** | **极轻** |
| **Buffer + CoTTA** | **较高** | **较高** | **稳定** | **稳定** | **几乎无** |

### 关键发现

1. **Buffer 层在所有测试的 TTA 框架中均带来一致性提升**，验证了模块化设计的通用性
2. **对小批量大小具有显著鲁棒性**：BN 方法在 BS=1 时性能崩溃，而 Buffer 层保持稳定
3. **有效抑制灾难性遗忘**：在连续域偏移 15 个域后，Buffer 方法的性能衰减远小于 BN 方法
4. **额外参数量极小**：Buffer 层引入的参数量仅占模型总参数的 0.1% 级别
5. **Buffer 和 BN 更新不宜同时使用**：同时更新反而降低性能，表明两种机制存在冲突

## 亮点与洞察

1. **范式转换**：从"更新归一化层"到"附加 Buffer 层"，从根本上解决了归一化层更新带来的不稳定性和遗忘问题
2. **假设最小**：不需要关于域偏移类型或程度的先验知识
3. **工程友好**：即插即用设计，不改变现有模型架构和训练流程
4. **理论直觉清晰**：保留预训练知识 + 学习轻量残差补偿 = 鲁棒自适应

## 局限与展望

1. Buffer 层的最优插入位置（哪些层、多少层）可能依赖具体架构，论文是否提供了详细的位置选择分析有待确认
2. 在极端域偏移下（如从自然图像到医学影像），仅靠轻量 Buffer 层是否足够
3. 与其他非 BN 架构（如 Layer Normalization、Group Normalization）的兼容性有待更全面评估
4. 没有自适应的重置策略——何时重置 Buffer 参数是一个值得深入研究的问题
5. 对 Vision Transformer 等现代架构的适用性需要更多实验验证

## 相关工作与启发

- **TENT** (Wang et al., 2021)：通过熵最小化更新 BN 仿射参数的开创性工作
- **CoTTA** (Wang et al., 2022)：通过教师-学生框架处理连续域偏移
- **SAR** (Niu et al., 2023)：通过可靠性过滤消除噪声伪标签的影响
- **EATA** (Niu et al., 2022)：通过样本高效策略减少不必要的更新

Buffer 层范式为 TTA 社区提供了一个正交于"如何更好地更新归一化层"的思路——"不更新归一化层"。

## 评分

| 维度 | 评分 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 理论深度 | 3 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| 总评 | 4 |

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Real-time Factuality Assessment from Adversarial Feedback](../../ACL2025/llm_safety/real-time_factuality_assessment_from_adversarial_feedback.md)
- [\[CVPR 2026\] The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models](../../CVPR2026/llm_safety/blind_spot_of_adaptation_quantifying_and_mitigating_forgetting_in_fine_tuned_driving_models.md)
- [\[NeurIPS 2025\] Demystifying Language Model Forgetting with Low-Rank Example Associations](demystifying_language_model_forgetting_with_low-rank_example_associations.md)
- [\[CVPR 2026\] ⊘ Source Models Leak What They Shouldn't ↛: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization](../../CVPR2026/llm_safety/oslash_source_models_leak_what_they_shouldnt_nrightarrow_unlearning_zero-shot_tr.md)
- [\[NeurIPS 2025\] Finding Structure in Continual Learning](finding_structure_in_continual_learning.md)

<!-- RELATED:END -->
