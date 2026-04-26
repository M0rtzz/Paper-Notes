---
title: >-
  [论文解读] Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients
description: >-
  [ACL 2025 (Long Paper, pp. 416–429)][联邦学习] 提出 LoRA-A²（Low Rank Adaptation with Alternating freeze and Adaptive rank selection），通过交替冻结 A/B 矩阵解决联邦 LoRA 聚合不一致问题，并结合自适应秩选择机制在大幅压缩上传参数量（最高减少 99.8%）的同时保持鲁棒性，尤其在低秩+高数据异构场景下显著优于现有方法。
tags:
  - ACL 2025 (Long Paper, pp. 416–429)
  - 联邦学习
  - LoRA
  - Aggregation Discordance
  - Alternating Freeze
  - Adaptive Rank Selection
  - Communication Efficiency
---

# Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients

**会议**: ACL 2025  
**arXiv**: [2410.22815](https://arxiv.org/abs/2410.22815)  
**代码**: 未公开  
**领域**: others  
**关键词**: Federated Learning, LoRA, PEFT, Communication Efficiency, Rank Selection  

## 一句话总结

提出 LoRA-A2 框架，通过交替冻结 LoRA 的 A/B 模块与自适应秩选择策略，同时解决联邦学习中 LoRA 聚合不一致和通信开销大的双重难题。

## 研究背景与动机

- **核心问题**：在联邦学习（FL）中直接对 LoRA 的 B 和 A 矩阵分别做加权平均聚合时，由于 ΔW = BA 的双线性结构，聚合后的 ΔW 不等于各客户端本地 ΔW 的加权和，称为"聚合不一致"（aggregation discordance）。
- **已有方案的不足**：FFA-LoRA 永久冻结 A 只训练 B，虽然解决了不一致但严重限制了优化空间（A 始终保持随机初始化值）；FL+LoRA 和 FlexLoRA 在服务器端聚合完整 ΔW 可保持一致性，但通信开销与全秩微调相当。
- **本文动机**：能否设计一种既保留完整优化空间、又能大幅降低通信量的联邦 LoRA 框架？LoRA-A2 给出了肯定的答案。

## 方法详解

### 整体框架

LoRA-A2 在标准联邦学习流程的基础上引入两个机制：每轮通过**交替冻结**让所有客户端共享 A 或 B 之一，从而使聚合不再产生不一致；通过**自适应秩选择**让每个客户端只上传对本地数据最重要的秩对应的参数，实现通信量的进一步压缩。

### 关键设计

1. **交替冻结（Alternating Freeze）**：奇数轮冻结 A 训练 B，偶数轮冻结 B 训练 A。以冻结 A 为例，所有客户端共享同一个 A，聚合 B 后 ΔW = (∑wₖBₖ)A = ∑wₖ(BₖA) = ∑wₖΔWₖ，完美消除不一致。与 FFA-LoRA 不同，A 在偶数轮也得到训练，保留完整优化空间。此外，受 LoRA+ 启发，为 B 和 A 设置不同学习率以强化交替优化效果。

2. **自适应秩选择（Adaptive Rank Selection）**：每个客户端在全局秩 rG 中选择 rᵢ 个最重要的秩。重要性度量标准为贡献度 Sₘ,ᵢ = ‖ΔBₖ[:,i]·A[i,:]‖F（冻结 A 时），衡量每个秩对 ΔW 变化的实际影响。该标准显式考虑了 A 和 B 的交互效应，比单纯的梯度幅度标准更适合 LoRA 结构。选择后通过二值掩码 Mₖ 对更新做 Hadamard 积，只上传非零部分。

3. **跨模块全局秩分配**：秩选择在模型所有目标模块上全局进行（从 rG·N 个秩中选 rᵢ·N 个），自动将秩资源从不重要模块转移到需要更多微调的模块，通信预算越小效果越显著。

### 损失函数

标准任务损失（如交叉熵），本地训练时在每个反向传播步骤前将掩码应用于参数更新。

### 理论保证

**命题 1**：参数空间包含关系为 Ω(FFA-LoRA) ⊊ Ω(FL+LoRA) = Ω(FlexLoRA) ⊂ Ω(LoRA-A2)，即 LoRA-A2 的可达参数空间最大。

## 实验

### 主实验结果

| 方法 | 通信秩 | Natural Instructions | Dolly-15K | 通信压缩比 |
|------|--------|---------------------|-----------|-----------|
| FFA-LoRA | rG=8 | 基线 | 基线 | 1× |
| FL+LoRA | rG=8 | 优于FFA | 优于FFA | 1× |
| FlexLoRA | 动态 | ≈FL+LoRA | ≈FL+LoRA | 部分压缩 |
| **LoRA-A2** | **rᵢ=4** | **最优** | **最优** | **最高99.8%压缩** |

LoRA-A2 在通信秩仅为全局秩一半的条件下，性能全面超越所有基线。

### 消融实验

| 消融维度 | 结论 |
|---------|------|
| 交替冻结 vs 永久冻结 | 交替冻结显著优于 FFA-LoRA 的永久冻结方案 |
| 不同学习率 vs 统一学习率 | 为 A/B 设置不同学习率进一步提升性能 |
| 贡献度标准 vs 梯度幅度标准 | 论文提出的贡献度标准（考虑 A-B 交互）胜出 |
| 全局秩选择 vs 逐模块秩选择 | 全局选择的跨模块重分配效果更好 |

### 关键发现

- 在高数据异质性场景下，自适应秩选择允许不同客户端选择不同重要秩，有效减少客户端间的梯度冲突
- 通信秩可压缩至极低（rᵢ=2）而性能下降有限，说明 LoRA 参数中存在大量冗余
- 交替冻结的额外收敛轮数成本可被通信压缩收益充分覆盖

## 亮点

- 交替冻结设计极其简洁却精准地解决了联邦 LoRA 的核心聚合难题，无需任何额外参数或复杂通信协议
- 自适应秩选择将通信压缩与个性化训练统一起来——选不同的秩既减少了通信量，又让客户端按需微调
- 理论证明了参数空间的严格包含关系，为方法的优越性提供了坚实的理论基础

## 局限性

- 交替冻结使模型每轮只更新一半参数，可能需要更多通信轮数才能收敛
- 秩选择阶段需要额外进行一轮完整的本地训练来估计贡献度，增加了计算开销
- 实验主要在指令微调场景下验证，未探索多模态、代码生成等更多样的下游任务

## 相关工作

- **FFA-LoRA** (Sun et al., 2024)：永久冻结 A，是最直接的比较对象
- **FlexLoRA** (Bai et al., 2024)：允许客户端使用不同秩的 LoRA 并在服务器端聚合完整 ΔW
- **LoRA+** (Hayou et al., 2024)：为 A 和 B 使用不同学习率，LoRA-A2 借鉴了其思想
- **AdaLoRA** (Zhang et al., 2023)：集中式场景下自适应秩分配

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| **总分** | **4.0** |
---
title: >-
  [论文解读] Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients
description: >-
  [ACL 2025 (Long Paper, pp. 416–429)][联邦学习] 提出 LoRA-A²（Low Rank Adaptation with Alternating freeze and Adaptive rank selection），通过交替冻结 A/B 矩阵解决联邦 LoRA 聚合不一致问题，并结合自适应秩选择机制在大幅压缩上传参数量（最高减少 99.8%）的同时保持鲁棒性，尤其在低秩+高数据异构场景下显著优于现有方法。
tags:
  - ACL 2025 (Long Paper, pp. 416–429)
  - 联邦学习
  - LoRA
  - Aggregation Discordance
  - Alternating Freeze
  - Adaptive Rank Selection
  - Communication Efficiency
---

# Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients

**会议**: ACL 2025 (Long Paper, pp. 416–429)  
**arXiv**: [2410.22815](https://arxiv.org/abs/2410.22815)  
**作者**: Jabin Koo, Minwoo Jang, Jungseul Ok (POSTECH, 韩国)
**代码**: 无公开  
**领域**: 联邦学习 / 参数高效微调 / 大语言模型  
**关键词**: federated learning, LoRA, Aggregation Discordance, Alternating Freeze, Adaptive Rank Selection, Communication Efficiency

## 一句话总结
提出 LoRA-A²（Low Rank Adaptation with Alternating freeze and Adaptive rank selection），通过交替冻结 A/B 矩阵解决联邦 LoRA 聚合不一致问题，并结合自适应秩选择机制在大幅压缩上传参数量（最高减少 99.8%）的同时保持鲁棒性，尤其在低秩+高数据异构场景下显著优于现有方法。

## 背景与动机
联邦学习（FL）中微调 LLM 面临巨大的通信开销。LoRA 通过低秩分解 $\Delta W = BA$ 减少可训练参数，但在 FL 中面临一个核心矛盾——**聚合不一致（Aggregation Discordance）**：

服务器对各客户端的 B 和 A 分别加权平均后相乘，不等于对各客户端的 $B_k A_k$ 做加权平均：
$$\frac{1}{K}\sum(B_k + B_j) \cdot \frac{1}{K}\sum(A_k + A_j) \neq \frac{1}{K}\sum B_k A_k$$

现有解决方案 FFA-LoRA 永久冻结 A 只训练 B，虽然消除了不一致，但限制了优化空间（A 始终保持初始值），导致在**低秩 + 高数据异构**条件下性能严重退化。

## 核心问题
如何在联邦 LoRA 框架中同时解决：
1. 聚合不一致问题（保证正确聚合）
2. 保留完整的优化参数空间（训练 A 和 B 两个矩阵）
3. 在低秩、高异构条件下保持鲁棒性
4. 进一步降低通信成本

## 方法详解

### 整体框架
LoRA-A² 包含两个核心组件：**交替冻结**（Alternating Freeze）和**自适应秩选择**（Adaptive Rank Selection）。全局维护一个秩为 $r_G$ 的 LoRA 适配器，每轮交替训练 A 或 B，各客户端根据本地数据自适应选择重要的秩进行训练和上传。

### 关键设计

1. **交替冻结（Alternating Freeze）**

    - 偶数轮冻结 A、训练 B；奇数轮冻结 B、训练 A
    - 当冻结 A 时，所有客户端共享相同的 A，聚合变为：
    $\Delta W = \sum_k w_k B_k \cdot A = \sum_k w_k (B_k A_k) = \sum_k w_k \Delta W_k$
      聚合不一致问题自然消除
    - 与 FFA-LoRA（永久冻结 A）相比，交替冻结使 A 也能被训练，保留了完整的优化空间
    - 借鉴 LoRA+ 的思想，为 A 和 B 设置不同的学习率，进一步增强优化效果

2. **自适应秩选择（Adaptive Rank Selection）**

    - **动机**：关注上传通信（上行带宽通常远慢于下行），允许不同客户端选择不同的秩
    - **贡献度准则**：定义模块 $m$ 中秩 $i$ 的重要性分数：
    $S_{m,i}^{B_k} = \|\Delta B_k[:,i] \cdot A[i,:]\|_F$
      该准则捕获每个秩对模型更新 $\Delta W$ 的贡献，同时考虑了 A 和 B 之间的交互（优于单纯的梯度大小准则）
    - **选择与稀疏化**：从全模型 $r_G \times N$ 个秩中选 top-$(r_i \times N)$ 个秩（$N$ 为目标模块数），生成二值掩码 $M_k$，仅上传 $B_k \odot M_k$（或 $A_k \odot M_k$），实现稀疏通信
    - **两大收益**：(1) 不同客户端可选择不同秩，减少异构数据下的客户端冲突；(2) 将秩资源从不重要的模块重新分配到需要更多微调的模块

3. **理论分析**

    - 证明了参数空间的包含关系：
    $\Omega_{\text{FFA-LoRA}} \subsetneq \Omega_{\text{FL+LoRA}} = \Omega_{\text{FlexLoRA}} \subset \Omega_{\text{LoRA-A}^2}$
    - LoRA-A² 拥有最大的可达参数空间，同时传输的参数量更少

## 实验关键数据

实验在 NLU 任务上评估，使用 Dirichlet 分布 ($\alpha$) 控制数据异构程度，测试不同秩 ($r$) 下的性能。

| 方法 | 聚合方式 | 低秩鲁棒性 | 高异构鲁棒性 | 上传参数量 | 关键特点 |
|------|---------|-----------|------------|----------|---------|
| FL+LoRA | 分别聚合 A, B | ❌ 退化严重 | ❌ 退化严重 | 100% (基线) | 存在聚合不一致 |
| FFA-LoRA | 永久冻结 A | ❌ 低秩退化 | ❌ 异构退化 | ~50% | 优化空间受限 |
| FlexLoRA | 全尺寸矩阵 + SVD | ✅ 较好 | ⚠️ 一般 | 高（需传全矩阵） | 通信开销大 |
| **LoRA-A²** | 交替冻结 + 自适应秩 | ✅ 鲁棒 | ✅ 鲁棒 | **最低 0.2%** | 兼顾鲁棒与高效 |

核心实验发现：
- 在极端条件下（低秩 $r=1$ + 高异构 $\alpha=0.1$），LoRA-A² 仍保持稳定性能，而 FFA-LoRA 和 FL+LoRA 性能显著下降
- 相比全量微调，上传参数量最高减少 **99.8%**，且不损失性能
- 交替冻结本身已带来显著提升，加上自适应秩选择进一步压缩通信同时保持甚至提升性能

## 消融实验要点
- **交替冻结 vs 永久冻结**：交替冻结在各种秩和异构设置下一致优于永久冻结（FFA-LoRA），验证了保留完整优化空间的重要性
- **学习率差异化**：为 A 和 B 设置不同学习率可进一步增强交替优化效果
- **贡献度准则对比**：论文提出的基于 $\|ΔB[:,i] \cdot A[i,:]\|_F$ 的准则优于单纯的梯度大小准则（$\|ΔB[:,i]\|$ 或 $\|ΔA[i,:]\|$），因为它显式建模了 A、B 之间的交互
- **秩选择的效果**：自适应秩选择允许各客户端选择不同的重要秩，有效减少了高异构场景下的客户端冲突

## 亮点
- **设计简洁优雅**：交替冻结是一个极其简单的改动（只需切换每轮冻结哪个矩阵），却同时解决了聚合不一致和优化空间受限两个问题
- **鲁棒性突出**：在极端低秩 + 极端异构的"最难"场景下依然稳定，这是现有方法普遍失败的场景
- **严格的理论支撑**：证明了 LoRA-A² 参数空间严格包含其他方法，提供了方法优势的理论解释
- **自适应秩选择的跨模块重分配**：不仅在同一模块内选秩，而是在全模型所有模块中统一排序选择，使得秩资源可以从不重要的模块流向关键模块
- **通信效率极高**：99.8% 的参数压缩率在联邦学习场景中极具实用价值

## 局限与展望
- 交替冻结导致每轮只优化一半参数（A 或 B），收敛速度可能比同时训练两者更慢，需要更多通信轮数
- 自适应秩选择需要额外跑 1 个 epoch 计算贡献度，增加了本地计算开销
- 论文主要聚焦 NLU 类任务，缺少在生成任务（NLG）和更大模型规模（如 LLaMA-7B/13B）上的验证
- 服务器端聚合后需要将稀疏更新"加到两轮之前的 B（或 A）"上，实现逻辑较复杂，需要维护历史状态
- 未探讨与其他 PEFT 方法（如 Adapter、Prefix Tuning）在 FL 中的组合

## 与相关工作的对比
- **vs FFA-LoRA**（Sun et al., 2024）：永久冻结 A，优化空间受限，低秩/高异构退化；LoRA-A² 交替冻结，参数空间严格包含 FFA-LoRA
- **vs FlexLoRA**（Bai et al., 2024）：聚合全尺寸矩阵后做 SVD 重新分解，通信成本高（需传输 $d_1 \times d_2$ 的全矩阵）；LoRA-A² 仅传稀疏低秩更新，通信更省
- **vs FL+LoRA**（FedAvg + LoRA）：分别聚合 A 和 B，聚合不一致严重，异构敏感；LoRA-A² 彻底消除不一致
- **vs RoLoRA**（Chen et al., 2024）：同样采用交替优化的思路，但 LoRA-A² 额外引入了自适应秩选择以进一步提升效率和异构鲁棒性
- **vs HETLORA**（Cho et al., 2024）：支持异构秩但不解决聚合不一致；LoRA-A² 同时解决两个问题

## 启发与关联
- "交替优化"的思想可以推广到其他联邦 PEFT 场景——任何涉及两组参数乘积的结构都可能受益于交替冻结
- 自适应秩选择中的"跨模块统一排序"思想类似 AdaLoRA 中动态分配秩预算的理念，但在联邦场景下更具意义（不同客户端的重要模块可能不同）
- 该论文引用量已达 19（截至 2026.03），说明联邦 LoRA 是一个快速增长的研究方向
- 99.8% 的参数压缩可以启发边缘设备上的 LLM 部署策略

## 评分
- 新颖性: ⭐⭐⭐⭐ 交替冻结虽然思路简单但切中要害，自适应秩选择的贡献度准则设计有原创性；两者组合形成了有效的框架
- 实验充分度: ⭐⭐⭐⭐ 多种异构程度和秩设置下的对比充分，消融实验覆盖各组件，但缺少 NLG 任务和大规模模型实验
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，理论分析精炼，方法描述系统化；交替冻结+自适应秩选择的组合逻辑流畅
- 对我的价值: ⭐⭐⭐⭐ 联邦 LoRA 聚合问题的清晰分析和简洁解法，对理解 LoRA 在分布式场景下的行为很有参考价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] CoLA: Collaborative Low-Rank Adaptation](cola_collaborative_low-rank_adaptation.md)
- [\[ACL 2025\] Low-Rank Interconnected Adaptation across Layers](low-rank_interconnected_adaptation_across_layers.md)
- [\[ACL 2025\] MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)
- [\[ECCV 2024\] Dropout Mixture Low-Rank Adaptation for Visual Parameters-Efficient Fine-Tuning](../../ECCV2024/others/dropout_mixture_low-rank_adaptation_for_visual_parameters-efficient_fine-tuning.md)
- [\[ACL 2025\] Understanding Cross-Domain Adaptation in Low-Resource Topic Modeling](understanding_cross-domain_adaptation_in_low-resource_topic_modeling.md)

<!-- RELATED:END -->
