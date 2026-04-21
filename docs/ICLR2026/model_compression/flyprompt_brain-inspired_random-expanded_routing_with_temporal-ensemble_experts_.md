---
title: >-
  [论文解读] FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning
description: >-
  [ICLR 2026][模型压缩][持续学习] 受果蝇蘑菇体神经系统启发，提出 FlyPrompt 框架将通用持续学习（GCL）分解为专家路由和专家能力提升两个子问题，通过随机扩展解析路由器（REAR）和时序集成专家（TE2）分别解决，在 CIFAR-100/ImageNet-R/CUB-200 上分别提升 11.23%/12.43%/7.62%。
tags:
  - ICLR 2026
  - 模型压缩
  - 持续学习
  - 参数高效微调
  - 果蝇神经系统
  - 随机扩展路由
  - 时序集成
---

# FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning

**会议**: ICLR 2026  
**arXiv**: [2602.01976](https://arxiv.org/abs/2602.01976)  
**代码**: [GitHub](https://github.com/FlyGCL)  
**领域**: 模型压缩 / LLM效率  
**关键词**: 持续学习, 参数高效微调, 果蝇神经系统, 随机扩展路由, 时序集成

## 一句话总结

受果蝇蘑菇体神经系统启发，提出 FlyPrompt 框架将通用持续学习（GCL）分解为专家路由和专家能力提升两个子问题，通过随机扩展解析路由器（REAR）和时序集成专家（TE2）分别解决，在 CIFAR-100/ImageNet-R/CUB-200 上分别提升 11.23%/12.43%/7.62%。

## 研究背景与动机

**通用持续学习（GCL）** 要求智能系统从非平稳的单次遍历数据流中持续学习，任务之间没有清晰的边界。与传统持续学习不同，GCL 面临更严峻的挑战：（1）需要快速适应；（2）鲁棒的知识保留；（3）有限监督和任务模糊下的高效资源使用。

近年来，基于预训练模型（PTM）的参数高效微调（PET）方法在持续学习中表现出色，如 L2P、DualPrompt、CODA-P 等引入可训练的 prompt 专家来适配 PTM。然而这些方法面临两个根本性挑战：

**专家路由问题**：如何在没有任务标签或迭代训练的条件下，将输入动态路由到合适的专家？现有方法的路由器在 GCL 的模糊任务边界下表现不佳。实验证实即使训练完成后，DualPrompt、MVP 等方法的路由准确率仍然较低。

**专家能力问题**：如何在稀疏且不平衡的监督下确保每个专家的表征能力？即使使用 oracle 路由器（完美选择正确专家），现有方法的最终精度仍然不佳，说明专家自身的表征质量也存在问题。

**生物学启发**：果蝇尽管只有不到 10 万个神经元，却展现出鲁棒的记忆巩固和上下文感知行为。其蘑菇体结构通过稀疏随机投影编码感官输入，投射神经元（PN）随机连接到 Kenyon 细胞（KC）实现约 40 倍维度扩展，不同 KC 子区域在不同时间尺度上表现出可塑性（gamma 短期 / alpha'/beta' 中期 / alpha/beta 长期记忆）。

## 方法详解

### 整体框架

FlyPrompt 将 GCL 分解为两个子问题并分别解决：

1. **REAR（Random Expanded Analytic Router）**：模拟果蝇的稀疏扩展回路，实现快速、无梯度的实例级专家选择
2. **TE2（Task-wise Experts with Temporal Ensemble）**：利用指数移动平均（EMA）在多个时间尺度上捕获知识，模拟蘑菇体的分区巩固机制

### 关键设计

**1. 随机扩展解析路由器（REAR）**

REAR 的核心思想是利用固定随机投影和闭式解析更新来实现专家分配，无需梯度更新。

给定预训练骨干编码器特征 h = f(x)，首先进行随机扩展：phi(x) = sigma(h R)，其中 R 是固定高斯随机矩阵（维度 d x M, M > d），sigma 为 ReLU。这模拟了果蝇中投射神经元到 Kenyon 细胞的约 40 倍稀疏扩展。

在线训练时，为每个任务 t 关联专家 E_t。对每个 batch 累积两个统计量：

- Gram 矩阵：G += Phi_i^T Phi_i（二阶特征相关性）
- 原型矩阵：Q += Phi_i^T C_t（专家级特征和）

路由器矩阵通过岭回归闭式解得到：U_hat^T = (G + lambda I)^(-1) Q

路由器矩阵仅在评估时计算一次。推理时通过 argmax phi(x) U_hat^T 选择专家。

**REAR 理论保证**（Theorem 1）：群体超额风险可分解为近似误差（增大 M 可降低）、估计方差（增大 N 或 lambda 可降低）和正则化偏置三项。通过足够大的随机扩展维度和适当正则化，误路由概率可任意小。

与 RanPAC 等方法的关键区别：REAR 仅用随机投影做专家路由，每个专家的 prompt 和 head 仍然是可训练的；而 RanPAC 直接用岭回归做最终分类。

**2. 时序集成专家（TE2）**

受果蝇 KC 子类型的启发，每个专家 E_t 维护一组 n 个 EMA head，衰减率分别为 {alpha_j}。

训练时只更新在线 head 和 prompt。损失函数使用交叉熵，并加入非参数 logit 掩码 m：对当前 batch 中未出现的类别设为负无穷，抑制未见标签的预测。每次更新后，EMA head 同步更新：

W_t^(j) <- alpha_j * W_t^(j) + (1 - alpha_j) * W

推理时，ensemble 所有 n+1 个 head（在线 + EMA），对每个 head 计算 softmax 后取逐元素最大值： z_hat(x) = max_j softmax(z^(j) + m)

**新任务初始化**：新专家的 prompt 初始化为此前所有已学 prompt 的平均值，在 GCL 有限数据下加速收敛。

**TE2 理论保证**（Theorem 2）：EMA head 的参数误差满足方差-偏置分解。几何 EMA bank 在任何时刻都包含一个接近最优偏差-方差权衡的 head。实践中两个 EMA head（alpha=0.9 和 0.99，对应窗口 10 和 100）即可。

### 损失函数 / 训练策略

- 使用标准交叉熵损失训练在线 head 和 prompt
- 非参数 logit 掩码抑制当前 batch 未见类别的预测，缓解跨任务和任务内的类别不平衡
- REAR 路由器在评估时通过累积统计量的闭式解一次性计算
- 无需 replay buffer 或额外的蒸馏损失
- prompt 使用历史 prompt 均值做暖启动

## 实验关键数据

### 主实验

**表1：GCL 基准性能（Sup-21K 预训练）**

| 方法 | CIFAR-100 A_auc | CIFAR-100 A_last | ImageNet-R A_auc | ImageNet-R A_last | CUB-200 A_auc | CUB-200 A_last |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| L2P | 76.23 | 79.11 | 44.40 | 42.03 | 64.30 | 61.42 |
| DualPrompt | 76.04 | 76.62 | 46.13 | 40.80 | 65.03 | 62.43 |
| CODA-P | 79.13 | 80.91 | 51.87 | 48.09 | 66.01 | 62.90 |
| MVP | 67.74 | 63.22 | 39.50 | 32.63 | 54.69 | 50.07 |
| MISA | 80.35 | 80.75 | 51.52 | 45.08 | 65.40 | 60.20 |
| **FlyPrompt** | **83.24** | **86.76** | **56.58** | **55.27** | **70.64** | **73.40** |

FlyPrompt 全面领先，A_last 提升尤为显著（CIFAR-100: +5.85%, ImageNet-R: +7.18%, CUB-200: +10.50%）。

**表2：跨预训练模型泛化性（iBOT-21K）**

| 方法 | CIFAR-100 A_auc | ImageNet-R A_auc | CUB-200 A_auc |
|------|:---:|:---:|:---:|
| CODA-P | 62.13 | 45.50 | 17.72 |
| MISA | 65.30 | 40.94 | 18.62 |
| **FlyPrompt** | **75.58** | **57.75** | **28.86** |

即使在自监督预训练模型上，FlyPrompt 仍大幅领先。

### 消融实验

消融实验验证了两个核心组件的贡献：

- 移除 REAR（使用其他路由策略）：路由准确率显著下降
- 移除 TE2（使用单一 head）：A_last 明显下降
- 移除 logit 掩码：性能在类别不平衡场景下下降
- EMA head 数量：2 个 EMA head（alpha=0.9, 0.99）即达到最佳

### 关键发现

1. **路由准确率是瓶颈**：现有方法在 GCL 设置下路由准确率远低于理想水平；REAR 通过固定随机投影+闭式解大幅提升路由精度
2. **专家能力同样重要**：即使使用 oracle 路由器，现有方法仍不理想；TE2 通过多时间尺度 EMA heads 有效提升单个专家的鲁棒性
3. **跨 PTM 泛化**：FlyPrompt 在多种预训练模型（Sup-21K、iBOT-21K、DINO-1K、MoCo v3-1K 等）上均有效
4. **前向传播即可路由**：REAR 无需梯度更新，适合 GCL 的在线单次遍历约束
5. **CKA 分析**证实不同专家确实特化在不同特征子空间中

## 亮点与洞察

- 将 GCL 问题清晰分解为专家路由和专家能力两个正交子问题，分析框架比直接端到端设计更有结构
- 果蝇蘑菇体的生物学类比贴切：稀疏随机扩展对应 REAR，多时间尺度可塑性对应 TE2
- REAR 的闭式解在评估时才需计算，训练时仅累积统计量，计算开销可忽略
- 提供了 REAR 和 TE2 的理论保证，不仅仅是经验改进
- 在极端设置下（如 DINO-1K 预训练的 CUB-200）仍能有效工作

## 局限与展望

1. 主要在视觉分类任务上验证，需要扩展到 NLP 和多模态场景
2. 专家数量与任务数量绑定，大量任务时可能导致参数量线性增长
3. Si-Blurry 基准虽是 GCL 标准设置，但与某些实际应用场景可能有差距
4. 随机投影矩阵维度 M 需预先设定，对不同规模问题可能需要调整
5. 当前框架假设任务以 session 形式到达，纯流式场景表现需进一步验证

## 相关工作与启发

FlyPrompt 展示了神经科学启发的 AI 设计（NeuroAI）在持续学习中的潜力。REAR 的随机扩展+闭式路由可推广到 MoE 架构的专家路由设计；TE2 的多时间尺度 EMA 也可用于在线学习、联邦学习等非平稳数据场景。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（生物启发 + 问题分解 + 理论支撑）
- 技术深度: ⭐⭐⭐⭐⭐（完整理论分析 + 闭式解路由 + 多时间尺度集成）
- 实验充分度: ⭐⭐⭐⭐⭐（3 数据集 x 6+ PTMs, 详细消融）
- 实用性: ⭐⭐⭐⭐（无需梯度更新的路由器对部署友好）
- 写作质量: ⭐⭐⭐⭐⭐（问题分析深入，生物类比恰当）

<!-- RELATED:START -->

## 相关论文

- [IDER: IDempotent Experience Replay for Reliable Continual Learning](ider_idempotent_experience_replay_for_reliable_continual_learning.md)
- [Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity](null-space_filtering_for_data-free_continual_model_merging_preserving_stability_.md)
- [LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [TiTok: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA](titok_transfer_token-level_knowledge_via_contrastive_excess_to_transplant_lora.md)
- [ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)

<!-- RELATED:END -->
