---
title: >-
  [论文解读] Trajectory Balance with Asynchrony: Decoupling Exploration and Learning for Fast, Scalable LLM Post-Training
description: >-
  [NeurIPS 2025][模型压缩][强化学习后训练] 提出 TBA（Trajectory Balance with Asynchrony），将 GFlowNet 的轨迹平衡（TB）目标与异步分布式 RL 架构结合，实现 LLM 后训练中探索与学习的解耦，在数学推理、偏好微调和自动红队测试任务上获得 4-50 倍加速且性能不降反升。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 强化学习后训练
  - 异步RL
  - 轨迹平衡
  - 离策略学习
  - GFlowNet
---

# Trajectory Balance with Asynchrony: Decoupling Exploration and Learning for Fast, Scalable LLM Post-Training

**会议**: NeurIPS 2025  
**arXiv**: [2503.18929](https://arxiv.org/abs/2503.18929)  
**代码**: [GitHub](https://github.com/bbartoldson/TBA)  
**领域**: 模型压缩  
**关键词**: 强化学习后训练, 异步RL, 轨迹平衡, 离策略学习, GFlowNet

## 一句话总结

提出 TBA（Trajectory Balance with Asynchrony），将 GFlowNet 的轨迹平衡（TB）目标与异步分布式 RL 架构结合，实现 LLM 后训练中探索与学习的解耦，在数学推理、偏好微调和自动红队测试任务上获得 4-50 倍加速且性能不降反升。

## 研究背景与动机

LLM 后训练中的 RL（如 PPO、RLOO）是提升模型推理和对齐能力的关键步骤，但面临严重的效率瓶颈：

**在策略（on-policy）依赖**：PPO/RLOO 等主流算法要求数据生成和策略更新严格交替进行，导致：数据生成时 GPU 空闲等待，训练时生成器停顿，资源利用率极低。

**异步化的困难**：虽然异步 RL 可以解耦生成和训练实现并行，但现有 RL 目标（如 PPO、DPO）在处理离策略（off-policy）数据时性能会显著下降。Noukhovitch et al. 发现增加离策略程度会损害 win-rate 或加剧策略偏移。

**扩展生成的有限性**：Hou et al. 发现 PPO 中将生成量从 4 扩展到 8-16 带来的收益有限。

**核心洞察**：GFlowNet 的轨迹平衡（TB）目标天然支持离策略学习——训练数据可以来自任意具有完全支撑的分布。这使得异步 RL 中由延迟参数同步产生的离策略数据能被有效利用，而非被视为噪声。

**切入角度**：将 KL 正则化 RL 问题重新表述为概率推断问题，利用 TB 的离策略特性嫁接到异步分布式架构上。

## 方法详解

### 整体框架

TBA 架构包含两类节点：多个 **Searcher 节点**负责并行生成数据（使用 vLLM 加速），1 个 **Trainer 节点**使用 TB 目标从 replay buffer 取数据持续训练。两者完全异步运行，仅每 $k$ 步同步一次参数和 buffer。

### 关键设计

1. **轨迹平衡（TB）目标函数**：将 KL 正则化 RL 的最优策略表示为后验分布 $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \exp(\beta^{-1} r_\phi(y;x))$。使用 VarGrad 变体的 TB 损失：

   $$\mathcal{L}_{\text{TB}}^{\text{VarGrad}}(\mathbf{B};\theta) = \frac{1}{BK} \sum_{i,j} \left( \text{SG}[\log \hat{Z}(\mathbf{x}^{(i)})] + \log \pi_\theta(\mathbf{y}^{(i,j)} | \mathbf{x}^{(i)}) - \log \pi_{\text{ref}}(\mathbf{y}^{(i,j)} | \mathbf{x}^{(i)}) - \frac{1}{\beta} r_\phi(\mathbf{y}^{(i,j)}; \mathbf{x}^{(i)}) \right)^2$$

   其中 $\hat{Z}$ 是基于 $K$ 个样本的批量估计，$\text{SG}$ 表示 stop-gradient。其梯度等价于使用均值基线的 REINFORCE + KL 正则化奖励：$\nabla \mathcal{J}_{\text{TB}} = A^{(i,j)} \nabla \log \pi_\theta$，其中优势函数 $A^{(i,j)} = (r^{(i,j)} - \bar{r}^{(i)}) - \beta(\hat{\text{KL}}^{(i,j)} - \bar{\text{KL}}^{(i)})$。**关键区别**：在离策略数据上这种等价性不成立，TB 能正确处理而 REINFORCE 不能。

2. **异步分布式架构**：Searcher 节点持有延迟的策略副本 $\pi_{\theta'}$，使用 vLLM 生成回复并计算奖励，存入本地 buffer $\mathcal{D}_{\text{local}}$。每 $k$ 步同步时，所有 Searcher 的 $\mathcal{D}_{\text{local}}$ 汇入全局 buffer $\mathcal{D}_{\text{global}}$，同时更新 Searcher 的策略权重。关键设计点：生成和训练完全异步，不存在任何一方等待另一方的瓶颈。

3. **优先级采样策略**：从 $\mathcal{D}_{\text{global}}$ 采样时交替使用两种策略：以概率 $m$ 采样最近同步步添加的数据（近似在策略），以概率 $1-m$ 基于奖励优先级采样（鼓励探索高奖励区域）。$m$ 是关键超参——$m=1$ 为纯近似在策略，$m=0$ 为纯奖励优先。

### 训练策略

- $\beta$（KL 系数）采用线性退火调度：较大值促进稳定性，较小值促进精度提升
- 支持参考策略重置（reference policy resetting），类似 Kimi K1.5 的做法
- 每个 query 生成 $S > K$ 个样本（如 $S=24, K=20$）以增加唯一序列数
- TBA' 变体基于 PRIME-RL 代码库实现，用于更大模型（Qwen 2.5 7B）

## 实验关键数据

### 主实验：数学推理（GSM8K, RhoMath-1B）

| 方法 | GPU | 准确率 | 训练时间 | 加速比 |
|---|---|---|---|---|
| VinePPO | 4xA100 | 52.8% | ~68h | 1x |
| Async DPO | 4xA100 | 53.1% | ~2h | ~34x |
| TBA | 4xA100 | **54.6%** | **~1.4h** | **~50x** |
| PPO | 4xA100 | 41.5% | ~8h | ~8.5x |

### 偏好微调（TL;DR, Pythia-410M）

| 方法 | 离策略步数 | Win Rate ↑ | Perplexity/KL ↓ |
|---|---|---|---|
| Online DPO (on-policy) | 1 | 0.85 | 1.13 |
| Async PPO | ≈15 | 0.76 | 1.10 |
| Proximal RLOO | ≈15 | 0.77 | 1.10 |
| **TBA** | **≈15** | **0.86** | **1.13** |

### 消融实验：离策略程度的影响

| 近策略采样概率 $m$ | Win Rate (PFT) |
|---|---|
| 0.4 | 0.67 |
| 0.5 | **0.82** |
| 0.6 | 0.80 |

### 关键发现

- TBA 在高度异步（≈15步离策略）下性能优于 Online DPO 的在策略设置
- 红队测试中增加 Searcher 数量持续提升攻击成功率和多样性，体现良好的可扩展性
- TBA' 在 Qwen 2.5 7B 上训练 MATH 任务同样优于 Dr. GRPO，尤其在高度离策略设置下
- $\beta$ 退火和参考策略重置对稳定训练至关重要

## 亮点与洞察

- TB 目标的离策略特性与异步 RL 的天然匹配是核心洞察，使"离策略 = 缺点"变为"离策略 = 特性"
- 将 GFlowNet 理论引入 LLM 后训练是跨领域创新
- 50x 加速在实际部署中意义重大——原本需要数天的后训练压缩到数小时
- 优先级采样在奖励稀疏场景（如红队测试）中尤其重要

## 局限与展望

- TB 目标在轨迹级别操作，梯度方差较高，需要每个 query 采样更多回复来缓解
- 目前每个 query 用 20 个样本计算 TB，batch size 较小
- PFT 实验使用 32 位精度且未用 DeepSpeed，与基线条件不完全一致
- 初始 buffer 需要大量初始数据（MR: 500, PFT: 10000 条），冷启动成本高
- 训练末期性能波动，需要更好的方差控制或 early stopping 策略

## 相关工作与启发

- 与 Noukhovitch et al. (2024) 的异步 RL 工作相比，TBA 使用原则性离策略目标而非增加 IS 比率
- 与 Kimi K1.5/K2 的目标函数接近，但 TBA 在参考策略重置策略上有所不同
- GFlowNet 在文本生成中的应用（hu2024amortizing, lee2024learning）为本文奠定基础
- 启发：离策略方法在大规模分布式训练中可能比在策略方法更有优势

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将 GFlowNet TB 目标引入异步 LLM 后训练是高度原创的工作
- 实验充分度: ⭐⭐⭐⭐ 三个任务多个模型规模，消融详细，但部分实验条件与基线不完全匹配
- 写作质量: ⭐⭐⭐⭐ 理论分析清晰，梯度推导完整，架构图直观
- 价值: ⭐⭐⭐⭐⭐ 50x 加速对实际 LLM 训练有巨大意义，开源代码增加可复现性

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Ensemble++: Scalable Exploration via Ensemble](scalable_exploration_via_ensemble.md)
- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)
- [\[NeurIPS 2025\] Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks](spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)
- [\[NeurIPS 2025\] Learning to Better Search with Language Models via Guided Reinforced Self-Training](learning_to_better_search_with_language_models_via_guided_reinforced_self-traini.md)
- [\[NeurIPS 2025\] Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression](learning_grouped_lattice_vector_quantizers_for_low-bit_llm_compression.md)

<!-- RELATED:END -->
