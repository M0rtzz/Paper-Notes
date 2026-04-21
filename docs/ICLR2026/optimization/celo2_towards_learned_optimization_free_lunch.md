---
title: >-
  [论文解读] Celo2: Towards Learned Optimization Free Lunch
description: >-
  [ICLR 2026][优化][学习型优化器] 提出 Celo2——一个仅用 4.5 GPU 小时元训练的学习型优化器，通过归一化 MLP 更新规则和任务增强等简单配方，实现了到 10 亿参数级别模型（GPT-3 XL 1.3B）的稳定泛化（比元训练分布大 6 个数量级），性能超越了此前耗费 4000 TPU-month 的 VeLO 和精心调优的 AdamW 基线。
tags:
  - ICLR 2026
  - 优化
  - 学习型优化器
  - 元学习
  - 元泛化
  - 归一化更新规则
  - AdamW替代
---

# Celo2: Towards Learned Optimization Free Lunch

**会议**: ICLR 2026  
**arXiv**: [2602.19142](https://arxiv.org/abs/2602.19142)  
**代码**: [https://github.com/amoudgl/celo2](https://github.com/amoudgl/celo2)  
**领域**: optimization  
**关键词**: 学习型优化器, 元学习, 元泛化, 归一化更新规则, AdamW替代

## 一句话总结

提出 Celo2——一个仅用 4.5 GPU 小时元训练的学习型优化器，通过归一化 MLP 更新规则和任务增强等简单配方，实现了到 10 亿参数级别模型（GPT-3 XL 1.3B）的稳定泛化（比元训练分布大 6 个数量级），性能超越了此前耗费 4000 TPU-month 的 VeLO 和精心调优的 AdamW 基线。

## 研究背景与动机

基础模型预训练主导了当今计算工作负载，而优化器选择（通常是 Adam 及其变体 AdamW）直接影响训练效率。学习型优化器（Learned Optimizer, LO）通过元学习发现更新规则，理论上可以超越手工设计的优化器。然而，这一方向面临三个核心挑战：

**元泛化（Meta-generalization）困难**: 在小规模任务上元训练的优化器往往无法泛化到大规模任务。VeLO 是此前最强的学习型优化器，尽管投入了 4000 TPU-month（约 10× GPT-3 训练量）的元训练计算，但仍未能泛化到超过 600M 参数的任务。

**元训练成本高昂**: VeLO 的 4000 TPU-month 计算量使得学习型优化器的研究迭代极为缓慢。

**稳定性不足**: 学习型优化器在超出训练分布时容易出现不稳定的训练动态，限制了实际采用。

核心矛盾：如何在极低的元训练成本下获得强大的元泛化能力？

本文给出了一个令人惊讶的答案：通过精心设计一个简单的归一化优化器架构并增强元训练策略，仅需 4.5 GPU 小时就能元训练一个性能优异的通用学习型更新规则，且该规则可以稳定扩展到比元训练分布大 6 个数量级的任务（GPT-3 XL 1.3B）。

核心 idea：不学习步长和调度器（解耦为用户调参），只学习归一化的更新方向——这种解耦使得学习到的规则具有更强的任务不变性和尺度泛化性。

## 方法详解

### 整体框架

Celo2 是一个即插即用的 Optax 优化器变换，可以一行代码替换 AdamW：
- 元训练阶段：在 4 个 8×8 图像分类 MLP 任务上训练小 MLP 更新规则，仅需 4.5 GPU 小时
- 部署阶段：将学习到的更新规则作为 Optax 变换插入标准训练流程，配合学习率调度和权重衰减使用
- 支持现代优化技术：正交化（Newton-Schulz）、1D/2D 参数不同更新规则、解耦权重衰减

### 关键设计

1. **归一化学习更新（Normalized Learned Update）**:
   这是 Celo2 最核心的设计创新。先前的学习型优化器直接使用 MLP 的原始输出作为更新步，但 Celo2 对 MLP 输出进行 **RMS 归一化**：
   $$\Delta\mathbf{p}_t = \frac{\text{MLP}(\mathbf{F})}{\text{RMS}(\text{MLP}(\mathbf{F}))}$$
   
   这一看似简单的改变带来了多重好处：
   - 迫使 MLP 在元训练时学习**任务不变的更新方向**，而非与任务相关的原始步长
   - 产生了与 AdamW 类似的训练动态（权重范数曲线一致，如 Figure 2 所示）
   - 使得学习到的规则在部署到更大规模任务时不会出现梯度爆炸或消失

   作者也对比了其他归一化方案（滚动 RMS、带截断的归一化等），发现简单的逐步 RMS 归一化效果最好（Table 2）。

2. **可调步长解耦（Tunable Step-size Decoupling）**:
   与 VeLO 和 Celo 等前作不同，Celo2 **不学习学习率调度器**，而是将步长调节留给用户。这意味着需要额外调一个超参数（学习率），但换来了到大规模任务的可靠泛化。这个权衡极为关键：Celo（前作）因为学习了调度器反而无法泛化到大规模任务。

3. **简单 MLP 架构**:
   Celo2 使用一个 2 层 MLP（8 个隐藏单元，ReLU 激活），总共不到 200 个参数。每个参数的输入特征包括：
   - 3 个momentum 累积器（$\beta_1, \beta_2, \beta_3 = 0.9, 0.99, 0.999$）
   - 1 个 RMS 梯度累积器（$\beta_4 = 0.95$）
   - Adafactor 的行/列特征
   
   MLP 仅输出方向 $\mathbf{d}$（不输出幅度 $\mathbf{m}$），这是消融实验中的最优选择（Table 1e）。

4. **正交化兼容性**:
   Celo2 与 Muon 优化器的 Newton-Schulz 正交化高度兼容。将正交化应用于学习到的 MLP 更新（而非标准的 momentum）可以进一步提升性能。Figure 4 显示了组合效应：Celo2-base + 正交化 + Adam for 1D 参数，三者叠加带来递进式改善。

5. **任务增强（Task Augmentation）**:
   在元训练期间，随机缩放优化对象网络的参数（$\alpha \sim \text{LogUniform}(0.001, 1000)$），模拟更广泛的最优化景观。这一技术是实现强泛化的关键（消融 Table 1c：去除任务增强后损失从 3.812 升至 4.417）。

### 损失函数 / 训练策略

**元训练设置：**
- 任务：4 个 8×8 图像分类 MLP（MNIST、Fashion-MNIST、CIFAR-10、SVHN）
- 元优化方法：Persistent Evolution Strategies (PES)，避免长展开的梯度偏差
- 内循环步数：$K=50$，展开长度对数均匀采样于 [100, 2000]
- 元目标：展开过程中的平均损失
- 总计算：100K 外循环迭代，8 个并行任务，在 Nvidia L40S GPU 上运行
- 全部约 **4.5 GPU 小时**

**部署设置：**
- 学习率搜索：7 个值，对数均匀分布在 $[10^{-5}, 10^{-3}]$
- 权重衰减：0.0, 0.1, 10.0
- 调度器：余弦衰减 + 线性预热（5%）
- 精度：默认 float32（bfloat16 在 ImageNet 上也稳定）

## 实验关键数据

### 主实验

**语言建模（元训练分布外泛化）:**

| 任务 | 参数量 | 规模比 | Celo2 | AdamW | VeLO |
|------|--------|-------|-------|-------|------|
| LM-30M | 30M | 30,000× | 竞争力 | 基线 | 竞争力 |
| GPT-2 | 124M | 124,000× | 略优 | 基线 | 竞争力 |
| GPT-3 XL | 1.3B | 1,000,000× | **竞争力** | 基线 | **泛化失败** |

这是学习型优化器首次成功泛化到 10 亿参数级别的预训练任务。GPT-3 XL 是元训练分布的 **6 个数量级** 之外。

**ImageNet ViT 分类（长展开泛化，50K 步 = 25× 元训练展开长度）:**

| 指标 | Celo2 | AdamW | VeLO |
|------|-------|-------|------|
| 达到 VeLO 最终损失的步数 | ~50% 步 | 较慢 | 100% |
| 最终验证精度 | ~66% | ~66% | ~66% |
| 训练稳定性 | 高（与 AdamW 一致） | 高 | 非典型动态 |

Celo2 达到 VeLO 的最终损失只需 VeLO 约 50% 的步数。

**强化学习（Atari PPO，高方差梯度下的泛化）:**

| 环境 | Celo2 | AdamW | VeLO |
|------|-------|-------|------|
| Asterix | 与 AdamW 相当 | 基线 | 显著落后/停滞 |
| Freeway | 与 AdamW 相当 | 基线 | 显著落后/停滞 |
| SpaceInvaders | 与 AdamW 相当 | 基线 | 显著落后/停滞 |

VeLO 在所有 RL 任务上出现训练停滞（与 VeLO 原论文 Figure 11 一致），而 Celo2 表现稳定。

### 消融实验

| 配置 | 验证损失 (LM-30M) | 说明 |
|------|-------------------|------|
| 隐藏大小=8（默认） | **3.812** | 最优 |
| 隐藏大小=4 | 4.128 | 过小 |
| 隐藏大小=16 | 3.857 | 过大反而不好 |
| RMS 衰减 $\beta=0.95$（默认） | **3.812** | 最优 |
| $\beta=0.999$ | 3.893 | 经典 Adam 设置 |
| 有任务增强（默认） | **3.812** | 关键组件 |
| 无任务增强 | 4.417 | 严重退化 |
| 归一化（默认） | **3.812** | 关键组件 |
| 无归一化 | 3.961 | 明显退化 |
| 仅输出方向 $\mathbf{d}$（默认） | **3.812** | 最优 |
| 输出 $\mathbf{d}$ 和 $\mathbf{m}$ | 3.900 | 幅度输出反而有害 |

### 关键发现

- **归一化是泛化的关键**: RMS 归一化 MLP 输出使训练动态与 AdamW 一致，是实现跨规模泛化的核心机制
- **任务增强不可或缺**: 去除任务增强后损失从 3.812 升至 4.417（+16%），说明梯度landscape多样性对元泛化至关重要
- **Celo2 与 Muon 竞争力相当**: 在 GPT-2 上 Celo2（3.35588 或 3.36785）与 Muon（3.35636）相差无几（Figure 7），且二者的区别仅在于更新规则——Muon 用 momentum，Celo2 用学习到的 MLP
- **运行时和内存开销**: Celo2-base 与 Adam 有相同的挂钟时间；内存开销约 5×（3 个 momentum + 1 个 RMS + Adafactor 特征，vs Adam 的 3×）；加正交化后挂钟时间为 1.3×

## 亮点与洞察

- **"免费午餐"的惊人发现**: 仅 4.5 GPU 小时的元训练计算就能产出一个实用的通用优化器——对比 VeLO 的 4000 TPU-month，计算效率提升 5-6 个数量级
- **设计哲学的转变**: 从"学习一切"（VeLO 学习更新规则+调度器+步长）到"只学习更新方向"——越少学，泛化越好
- **归一化的威力**: 一个简单的 RMS 归一化就将学习型优化器从"玩具"级别提升到了"实用"级别
- **8×8 图像分类训练出 GPT-3 优化器**: 元训练任务之简单与部署任务之复杂的巨大反差，体现了学习到的更新规则的本质通用性
- **与 Muon 的互补性**: Celo2 将 Muon 的正交化框架从手工 momentum 正交化推广到了学习型更新规则正交化

## 局限与展望

- **需要调学习率**: 与 VeLO 的自调优模式不同，Celo2 需要搜索学习率（7 个候选值），虽然搜索空间不大，但这增加了使用门槛
- **内存开销较高**: 5× 的参数内存开销高于 Adam 的 3×，在内存受限场景下可能是问题
- **元训练任务过于同质**: 仅在 4 个 8×8 图像分类 MLP 上元训练，更多样的元训练任务可能带来更好的泛化
- **尚未在混合精度下充分测试**: 作者承认 float32 是默认精度，bfloat16 仅在 ImageNet 上做了初步测试
- **未学习调度器**: 虽然解耦步长是泛化的关键，但如何安全地将调度器也纳入学习仍是开放问题
- **与更新型的 SOAP、AdaMuon 等优化器的对比不充分**: 仅与 AdamW、VeLO、Muon 对比

## 相关工作与启发

- **VeLO (Metz et al., 2022)**: 此前最强的学习型优化器，计算量 4000 TPU-month，但泛化上限仅 600M 参数
- **Celo (Moudgil et al., 2025)**: Celo2 的前作，在 24 GPU-hour 实现了比 VeLO 更好的计算效率，但性能有所下降
- **Muon (Jordan et al., 2024)**: 手工设计的正交化优化器，在 NanoGPT 速度赛中表现出色，Celo2 与其高度兼容
- **SOAP (Vyas et al., 2024)**: 探索模块范数中的优化，与 Celo2 的方向互补
- **启发**: Celo2 的成功暗示存在一个低维的"通用更新规则空间"——不到 200 个参数的 MLP 就能捕获它。这为理解优化算法的本质提供了新视角

## 评分

- 新颖性: ⭐⭐⭐⭐ — 归一化+步长解耦的设计决策虽然简单，但其效果出人意料，且有充分的消融支持
- 实验充分度: ⭐⭐⭐⭐⭐ — 语言建模(30M→1.3B)、ImageNet ViT、Atari RL，覆盖了多个领域和规模
- 写作质量: ⭐⭐⭐⭐ — 方法描述清晰，消融实验组织得当，附录提供了完整的代码和背景知识
- 价值: ⭐⭐⭐⭐⭐ — 学习型优化器领域的里程碑式工作，首次实现了到十亿参数规模的实用泛化

<!-- RELATED:START -->

## 相关论文

- [Better NTK Conditioning: A Free Lunch from ReLU Nonlinear Activation in Wide Neural Networks](../../NeurIPS2025/optimization/better_ntk_conditioning_a_free_lunch_from_relu_nonlinear_activation_in_wide_neur.md)
- [Problem-Parameter-Free Decentralized Bilevel Optimization](../../NeurIPS2025/optimization/problem-parameter-free_decentralized_bilevel_optimization.md)
- [Covariances for Free: Exploiting Mean Distributions for Training-free Federated Learning](../../NeurIPS2025/optimization/covariances_for_free_exploiting_mean_distributions_for_training-free_federated_l.md)
- [RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization](rrnco_towards_real-world_routing_with_neural_combinatorial_optimization.md)
- [Provable and Practical In-Context Policy Optimization for Self-Improvement](provable_and_practical_in-context_policy_optimization_for_self-improvement.md)

<!-- RELATED:END -->
