---
title: >-
  [论文解读] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning
description: >-
  [ICCV 2025][AI安全][联邦学习] 提出 FedPoisonMIA，一种基于角度偏差最大化的联邦学习投毒成员推理攻击，同时提出 Angular Trimmed-mean (ATM) 防御机制，通过角度距离过滤恶意梯度。
tags:
  - ICCV 2025
  - AI安全
  - 联邦学习
  - 成员推理攻击
  - 投毒攻击
  - 拜占庭鲁棒聚合
  - 隐私保护
---

# Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning

**会议**: ICCV 2025  
**arXiv**: [2507.00423](https://arxiv.org/abs/2507.00423)  
**代码**: 无  
**领域**: AI Safety / Federated Learning Privacy  
**关键词**: 联邦学习, 成员推理攻击, 投毒攻击, 拜占庭鲁棒聚合, 隐私保护

## 一句话总结

提出 FedPoisonMIA，一种基于角度偏差最大化的联邦学习投毒成员推理攻击，同时提出 Angular Trimmed-mean (ATM) 防御机制，通过角度距离过滤恶意梯度。

## 研究背景与动机

**领域现状**：联邦学习（FL）允许多客户端在不共享原始数据的情况下协作训练全局模型，被视为隐私保护的关键范式。然而 FL 的分布式特性使其易受投毒攻击，恶意客户端可发送精心构造的更新来影响模型。

**现有痛点**：大多数联邦学习投毒攻击聚焦于破坏模型完整性（如降低准确率），对隐私攻击关注不足。投毒成员推理攻击（PMIA）通过操纵梯度来推断其他良性客户端数据集中是否包含特定样本，但现有 PMIA 方法在面对拜占庭鲁棒机制时效果有限。

**核心矛盾**：PMIA 比传统投毒攻击更难设计——需要在不显著影响全局模型性能（不然容易被检测）的情况下窃取隐私信息。直接使用错误标签的梯度会产生过大的角度偏差，容易被鲁棒聚合规则过滤。

**本文切入角度**：攻击端——精心构造恶意梯度使其角度偏差尽量大（增强攻击效果），但确保不超过良性梯度之间的最大角度差异（避免被检测）；防御端——利用角度偏差作为判别标准过滤恶意更新。

**核心 idea**：攻击是一个约束优化问题：最大化恶意梯度与良性梯度的角度偏差，同时约束该偏差不超过良性梯度间的最大角度差。利用贪心算法选择 mask 样本来掩盖攻击梯度，并优化缩放系数 α。

## 方法详解

### 整体框架

**攻击框架 FedPoisonMIA**：攻击者有攻击样本集 $D_{attack}$ 和掩码样本集 $D_{mask}$，构造最终恶意梯度 $g_{malicious} = \alpha \cdot g_{attack} + g_{mask}$，其中 $g_{attack}$ 从错误标签样本计算，$g_{mask}$ 从精选正确标签样本计算。

**防御框架 ATM**：服务端计算每个梯度与所有其他梯度的平均角度，去掉平均角度最大的 2b 个梯度后取均值。

### 关键设计

1. **攻击梯度生成（Step I）**：将 $D_{attack}$ 中样本的真实标签替换为随机错误标签，计算梯度 $g_{attack}$。通过错误标签引发目标样本的损失上升，如果良性客户端有相同样本（正确标签），损失会被拉回，从而通过损失差异推断成员身份。

2. **贪心掩码样本选择（Step II）**：从 $D_{mask}$ 中选择子集 $\hat{D}_{mask}$，优化目标为：
    $\arg\max_{\hat{D}_{mask}} \max_{i \in \mathcal{B}} \angle(g_{malicious}, g_{benign}^i)$
   约束：$\max_{i \in \mathcal{B}} \angle(g_{malicious}, g_{benign}^i) \leq \max_{i,j \in \mathcal{B}} \angle(g_{benign}^i, g_{benign}^j)$
   
   由于穷举所有子集是 NP-hard 的，采用贪心策略逐一添加使目标函数最大化的样本，直达目标数量 $\lfloor \gamma |D_{mask}| \rfloor$。

3. **缩放系数优化（Step III）**：固定 $\hat{D}_{mask}$ 后优化 α，在保持角度偏差约束的前提下最大化偏差。

4. **ATM 防御机制**：

    - 计算所有梯度两两间角度 $\theta_{i,j}$
    - 对每个梯度算平均角度 $\bar{\theta}_k = \frac{1}{|G|-1} \sum \theta_{k,l}$
    - 按平均角度升序排列，去除最大的 2b 个梯度
    - 对剩余梯度取均值作为聚合结果

### 损失函数 / 训练策略

ATM 聚合公式：$\bar{g} = \frac{1}{|G| - 2b} \sum_{g \in G'} g$，其中 G' 为去除异常梯度后的集合。

收敛保证（Theorem 1）：ATM 的 L2 偏差有上界 $\frac{2(n-m)(b+1)\sigma^2}{(n-b-m)^2}$，确保聚合结果趋近真实均值。

## 实验关键数据

### 主实验：攻击准确率（C=0.8, Full-knowledge）

| 数据集 | 攻击方法 | FedAvg | Median | Trimmed-mean | ATM |
|--------|---------|--------|--------|-------------|-----|
| Texas100 IID | Passive | 0.643 | 0.583 | 0.630 | 0.600 |
| Texas100 IID | GA | 0.826 | 0.820 | 0.810 | 0.766 |
| Texas100 IID | AGREvader | 0.804 | 0.761 | 0.756 | 0.741 |
| Texas100 IID | **FedPoisonMIA** | **0.897** | **0.913** | **0.880** | **0.803** |
| CIFAR-10 IID | **FedPoisonMIA** | **0.913** | **0.753** | **0.857** | **0.713** |

FedPoisonMIA 在所有数据集、所有防御下均达到最高攻击准确率。ATM 整体上将攻击准确率压至最低。

### 消融实验：贪心选择 vs 随机选择（Texas100）

| 选择方式 | FedAvg | Median | Trimmed-mean | ATM |
|---------|--------|--------|-------------|-----|
| 随机选择 | 0.771 | 0.744 | 0.751 | 0.751 |
| **贪心选择** | **0.884** | **0.807** | **0.884** | **0.880** |

贪心掩码选择比随机选择提升超过 20% 攻击准确率，验证了精选 mask 样本的关键作用。

### 关键发现

- **攻击强度**：FedPoisonMIA 在 Texas100 Non-IID 下相比最佳基线提升 8.0-16.0% 攻击准确率，在 Fang/Multi-Krum/DeepSight 等更复杂防御下也始终最优。
- **部分知识场景下也有效**：仅用恶意客户端梯度（不知良性梯度），攻击准确率最大仅降 8.0%。
- **防御有效性**：ATM 在多数场景下实现最低攻击准确率，且不增加客户端计算开销。ATM 排序维度是客户端数量（十量级），而 Median/Trimmed-mean 排序维度是参数维度（十万量级），因此 ATM 计算开销显著更低。
- **IID vs Non-IID**：IID 设置下攻击准确率更高，因为更新一致性使攻击更容易"借势"。

## 亮点与洞察

- **攻防一体**：从攻击设计出发（最大化角度偏差）自然启发了防御策略（基于角度偏差过滤），逻辑链完整。
- **实用威胁模型**：攻击者不需要知道服务端使用何种聚合规则，且在部分知识场景下仍有效。
- **ATM 计算效率**优秀：排序在客户端数量维度而非参数维度，与 Median/Trimmed-mean 形成鲜明对比。

## 局限与展望

- ATM 将攻击准确率降至 0.7-0.8 范围，但未能完全消除隐私泄露风险，仍存在改进空间。
- 攻击需要攻击者拥有目标样本或相似样本，在实际中获取这些样本的难度未充分讨论。
- 自适应攻击（知道 ATM 的存在）仍能达到一定效果，但准确率低于 FedPoisonMIA。
- 异步 FL 设置下攻击效果下降但仍显著，需要更强防御手段。

## 相关工作与启发

- AGREvader [Zhang 2023] 也用 mask 梯度掩盖攻击，但以欧氏距离为约束；FedPoisonMIA 改用角度距离，效果更好。
- 传统拜占庭鲁棒方法（Median, Trimmed-mean, Multi-Krum）主要防数据/模型完整性攻击，对成员推理攻击效果有限——这是一个被忽视的安全维度。
- 差分隐私（DP）虽然也能降低攻击效果，但同时会显著损害模型准确率。

## 评分

- 新颖性：⭐⭐⭐⭐ — 角度偏差最大化的攻击思路新颖，攻防一体设计
- 技术深度：⭐⭐⭐⭐ — 严谨的优化问题建模、收敛性分析
- 实验充分度：⭐⭐⭐⭐⭐ — 4 数据集、8+ 防御机制、IID/Non-IID、同步/异步、多指标
- 实用性：⭐⭐⭐⭐ — 揭示了联邦学习中真实存在的隐私威胁

<!-- RELATED:START -->

## 相关论文

- [FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)
- [Active Membership Inference Test (aMINT): Enhancing Model Auditability with Multi-Task Learning](active_membership_inference_test_amint_enhancing_model_audit.md)
- [Hide and Find: A Distributed Adversarial Attack on Federated Graph Learning](../../ICLR2026/ai_safety/hide_and_find_a_distributed_adversarial_attack_on_federated_graph_learning.md)
- [A Framework for Double-Blind Federated Adaptation of Foundation Models](a_framework_for_double-blind_federated_adaptation_of_foundation_models.md)
- [MARS: A Malignity-Aware Backdoor Defense in Federated Learning](../../NeurIPS2025/ai_safety/mars_a_malignity-aware_backdoor_defense_in_federated_learning.md)

<!-- RELATED:END -->
