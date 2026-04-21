---
title: >-
  [论文解读] Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off
description: >-
  [ICML 2025][AI安全][联邦学习] 提出 FedCEO 框架，通过在服务器端对堆叠的客户端模型参数进行张量低秩近端优化，利用不同客户端间的语义互补性恢复 DP 噪声破坏的语义信息，将效用-隐私权衡界改进了 $O(\sqrt{d})$ 量级。
tags:
  - ICML 2025
  - AI安全
  - 联邦学习
  - 差分隐私
  - 效用-隐私权衡
  - 张量低秩
  - 语义互补
---

# Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off

**会议**: ICML 2025  
**arXiv**: [2402.07002](https://arxiv.org/abs/2402.07002)  
**代码**: https://github.com/6lyc/FedCEO_Collaborate-with-Each-Other  
**领域**: AI安全  
**关键词**: 联邦学习, 差分隐私, 效用-隐私权衡, 张量低秩, 语义互补

## 一句话总结
提出 FedCEO 框架，通过在服务器端对堆叠的客户端模型参数进行张量低秩近端优化，利用不同客户端间的语义互补性恢复 DP 噪声破坏的语义信息，将效用-隐私权衡界改进了 $O(\sqrt{d})$ 量级。

## 研究背景与动机

**领域现状**：差分隐私（DP）是联邦学习中保护用户隐私的主流技术标准，通过向上传的模型更新添加随机噪声实现。

**现有痛点**：DP 噪声随机破坏模型的语义完整性，且这种破坏随通信轮数积累——不同客户端被破坏的语义信息各不相同，导致全局语义空间不平滑。

**核心矛盾**：现有改进方法（如正则化、个性化）都基于约束本地更新大小，未利用客户端间的协作关系。

**本文目标**：如何利用客户端间的语义互补性来恢复被 DP 噪声破坏的语义信息？

**切入角度**：将多个客户端的噪声模型参数堆叠成高阶张量，通过截断高频分量实现全局语义空间的平滑化。

**核心 idea**：DP 噪声的随机性意味着不同客户端被破坏的语义部分不同——张量低秩分解能提取客户端间的共享语义并去除个体噪声。

## 方法详解

### 整体框架
FedCEO 在标准 DPFL 基础上，在服务器端增加一步张量低秩近端优化：
1. 各客户端本地训练后添加 DP 噪声，上传噪声模型
2. 服务器将 K 个客户端的模型参数堆叠为三阶张量
3. 执行截断张量奇异值分解（T-tSVD）平滑全局语义空间
4. 将平滑后的参数广播回客户端

### 关键设计

1. **张量低秩近端优化**:

    - 功能：将 K 个客户端的参数矩阵堆叠为三阶张量 $\mathcal{W} \in \mathbb{R}^{m \times d \times K}$，执行低秩近端优化
    - 核心思路：等价于在频谱空间截断高频分量（T-tSVD），保留客户端间共享的低频语义信息，去除各客户端独立的高频噪声
    - 设计动机：DP 噪声在频谱空间主要表现为高频分量，截断可有效去噪

2. **自适应秩控制**:

    - 功能：根据噪声水平（隐私预算 ε）和训练阶段动态调整截断秩
    - 核心思路：噪声越大（ε 越小）截断越激进，训练后期语义趋于收敛时可放松截断
    - 设计动机：固定秩无法适应不同隐私设置和训练阶段的需求

### 损失函数 / 训练策略
- 用户级差分隐私：保护粒度为单个客户端的全部数据
- 梯度裁剪 + 高斯噪声机制
- 服务器端低秩正则化不影响隐私保证（后处理性质）

## 实验关键数据

### 主实验
在 CIFAR-10 上用 MLP 架构、不同隐私预算下的测试准确率：

| 方法 | ε=1 | ε=2 | ε=5 | ε=10 |
|------|-----|-----|-----|------|
| UDP-FedAvg | ~35% | ~42% | ~50% | ~55% |
| CENTAUR | ~40% | ~48% | ~56% | ~60% |
| **FedCEO** | **~48%** | **~55%** | **~62%** | **~66%** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 无低秩处理 | 显著下降 | 退化为标准 DPFL |
| 固定秩 vs 自适应秩 | 自适应更优 | 适应不同训练阶段 |
| 不同客户端数量 K | K 越大效果越好 | 更多互补信息 |

### 关键发现
- 效用-隐私权衡界从先前 SOTA 的 $O(d)$ 改进到 $O(\sqrt{d})$
- DLG 梯度反转攻击下仍保持强隐私保护
- 在 CNN、ResNet 等更复杂架构上同样有效

## 亮点与洞察
- **客户端间语义互补**的视角新颖——DP 噪声的随机性反而成为优势（不同客户端被破坏的部分不同）
- 张量低秩处理是服务器端后处理，不增加隐私预算消耗
- 理论界的 $\sqrt{d}$ 改进在高维场景（如大模型）中非常显著

## 局限与展望
- 张量 SVD 的计算开销随客户端数量和参数维度增长
- 假设客户端间有语义相似性，极端异构数据分布下效果可能减弱
- 仅验证了用户级 DP，样本级 DP 的适用性未讨论

## 相关工作与启发
- **vs CENTAUR/Jain et al.**: 它们独立对每个客户端做 SVD，FedCEO 利用跨客户端张量结构
- **vs PPSGD**: 个性化方法，扩展到复杂模型困难

## 评分
- 新颖性: ⭐⭐⭐⭐ 跨客户端张量低秩去噪的思路新颖
- 实验充分度: ⭐⭐⭐⭐ 多架构、多隐私设置、攻击验证
- 写作质量: ⭐⭐⭐⭐ 可视化直观，理论清晰
- 价值: ⭐⭐⭐⭐ 实用的 DPFL 改进方案

<!-- RELATED:START -->

## 相关论文

- [Mitigating Privacy-Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy](../../NeurIPS2025/ai_safety/mitigating_privacy-utility_trade-off_in_decentralized_federated_learning_via_f-d.md)
- [Towards Trustworthy Federated Learning with Untrusted Participants](towards_trustworthy_federated_learning_with_untrusted_participants.md)
- [Private Model Personalization Revisited](private_model_personalization_revisited.md)
- [Improving the Variance of Differentially Private Randomized Experiments through Clustering](improving_the_variance_of_differentially_private_randomized_experiments_through_.md)
- [Enabling Differentially Private Federated Learning for Speech Recognition: Benchmarks, Adaptive Optimizers and Gradient Clipping](../../NeurIPS2025/ai_safety/enabling_differentially_private_federated_learning_for_speech_recognition_benchm.md)

<!-- RELATED:END -->
