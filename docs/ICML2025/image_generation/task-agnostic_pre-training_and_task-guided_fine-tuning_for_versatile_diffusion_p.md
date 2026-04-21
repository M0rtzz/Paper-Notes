---
title: >-
  [论文解读] Task-Agnostic Pre-training and Task-Guided Fine-tuning for Versatile Diffusion Planner
description: >-
  [ICML2025][图像生成][扩散模型] 提出 SODP 框架：先用大量无奖励标签的次优多任务轨迹预训练扩散规划器，再用基于策略梯度的 RL 微调快速适配下游任务，并引入 BC 正则化防止性能崩溃，在 Meta-World 50 任务上达到 60.56% 成功率（SOTA）。
tags:
  - ICML2025
  - 图像生成
  - 扩散模型
  - 多任务RL
  - 预训练-微调
  - 次优数据
  - 策略梯度
  - 行为克隆正则化
---

# Task-Agnostic Pre-training and Task-Guided Fine-tuning for Versatile Diffusion Planner

**会议**: ICML2025  
**arXiv**: [2409.19949](https://arxiv.org/abs/2409.19949)  
**代码**: 待确认  
**领域**: 扩散模型 / 强化学习 / 多任务规划  
**关键词**: 扩散模型, 多任务RL, 预训练-微调, 次优数据, 策略梯度, 行为克隆正则化

## 一句话总结
提出 SODP 框架：先用大量无奖励标签的次优多任务轨迹预训练扩散规划器，再用基于策略梯度的 RL 微调快速适配下游任务，并引入 BC 正则化防止性能崩溃，在 Meta-World 50 任务上达到 60.56% 成功率（SOTA）。

## 研究背景与动机
- **多任务 RL 的困难**：传统方法假设任务间共享潜在结构，难以捕捉多模态最优行为分布；扩散模型虽然擅长建模复杂分布，但现有方法要么依赖昂贵的专家示范数据，要么需要每个任务的奖励标签
- **核心问题**：能否从大量低质量、无奖励标注的次优轨迹中学习一个通用扩散规划器，使其能快速适应各种下游任务？
- **动机**：类比 LLM 的预训练+RLHF 范式，将"大量次优数据预训练 → 少量任务特定奖励微调"的思路引入多任务 RL 中的扩散规划器

## 方法详解

### 整体框架（两阶段）
SODP = **Sub-Optimal data → Diffusion Planner**，包含预训练和微调两个阶段。

### 阶段一：无引导预训练
在多任务混合数据集 $\mathcal{D} = \cup_{i=1}^{N} \mathcal{D}_i$ 上训练无条件扩散模型，预测基于历史状态的未来动作序列：

$$\max_{\theta} \mathbb{E}_{(\mathbf{s}_t, \mathbf{a}_t) \sim \cup_i \mathcal{D}_i} \left[ \log p_\theta(\mathbf{a}_t^0 | \mathbf{s}_t) \right]$$

其中 $\mathbf{a}_t^0 = (a_t, a_{t+1}, \ldots, a_{t+H-1})$ 为长度 $H$ 的动作序列，$\mathbf{s}_t$ 为长度 $T_o$ 的历史状态。

**预训练损失**（标准去噪损失）：

$$\mathcal{L}_{\text{pre-train}}(\theta) = \mathbb{E}_{k, \epsilon, (\mathbf{s}_t, \mathbf{a}_t^0) \sim D} \left[ \| \epsilon - \epsilon_\theta(\mathbf{a}_t^k, \mathbf{s}_t, k) \|^2 \right]$$

**关键设计**：聚焦共享动作空间（如机器人末端执行器位姿）以促进跨任务泛化，无需奖励标签或任务描述。

### 阶段二：基于奖励的策略梯度微调
将去噪过程建模为 $K$ 步 MDP，采用 PPO 风格的重要性采样进行优化：

**微调目标**：最大化下游任务 $\mathcal{T}$ 的期望累积奖励：
$$J^{\mathcal{T}}(\theta) = \sum_t \mathbb{E}_{p_\theta(\mathbf{a}_t^0 | \mathbf{s}_t)} [r^{\mathcal{T}}(\mathbf{a}_t^0)]$$

**策略梯度估计**（含 PPO 裁剪）：

$$\mathcal{L}_{\text{Imp}}^{\mathcal{T}}(\theta) = \sum_t \mathbb{E}_{p_{\theta_{\text{old}}}} \left[ \sum_{k=1}^{K} -r^{\mathcal{T}}(\mathbf{a}_t^0) \cdot \max\left(\rho_k, \text{clip}(\rho_k, 1+\epsilon, 1-\epsilon)\right) \right]$$

其中 $\rho_k(\theta, \theta_{\text{old}}) = \frac{p_\theta(\mathbf{a}_t^{k-1} | \mathbf{a}_t^k, \mathbf{s}_t)}{p_{\theta_{\text{old}}}(\mathbf{a}_t^{k-1} | \mathbf{a}_t^k, \mathbf{s}_t)}$ 为重要性比率。

### BC 正则化
引入行为克隆正则项防止微调过程中性能崩溃，目标策略 $\mu$ 由近期最优经验近似：

$$\mathcal{L}_{\text{BC}}(\theta) = \mathbb{E}_{k, \mathbf{a}_\mu^k \sim p_\mu} \left[ \| \epsilon(\mathbf{a}_\mu^k, k) - \epsilon_\theta(\mathbf{a}_\mu^k, k) \|^2 \right]$$

**最终微调损失**：$\mathcal{L}_{\text{fine-tuning}}^{\mathcal{T}}(\theta) = \mathcal{L}_{\text{Imp}}^{\mathcal{T}}(\theta) + \lambda \mathcal{L}_{\text{BC}}(\theta)$

## 实验关键数据

### 主实验：Meta-World 50 任务（MT50-rand，次优数据）

| 方法 | 平均成功率 |
|------|-----------|
| RLPD | 10.16% |
| IBRL | 25.29% |
| Cal-QL | 35.09% |
| MTBC | 34.53% |
| MTDQL | 17.33% |
| MTDT | 42.33% |
| MTIQL | 43.28% |
| Prompt-DT | 48.40% |
| MTDIFF-P | 48.67% |
| HarmoDT-F | 57.20% |
| **SODP (本文)** | **60.56%** |

### 高效微调验证（仅 100k 步在线微调）

| 方法 | 平均成功率 |
|------|-----------|
| RLPD | 7.62% |
| Cal-QL | 24.60% |
| HarmoDT-F | 57.37% |
| **SODP** | **59.26%** |

### 正则化消融
- 无正则化：性能快速下降，模型丢失预训练能力
- KL 正则化：模型被困在次优区域附近震荡
- PL（预训练损失）正则化：探索无方向性，可能导向更差区域
- **BC 正则化（本文）**：保留预训练知识的同时有效探索高奖励区域

### 多任务预训练收益
在 MT-10 上预训练后迁移到未见任务（如 drawer-open 34.7%、plate-slide-side 55.3%、handle-pull-side 71.3%），而仅在单任务上预训练则完全失败（0%）。

## 亮点与洞察
1. **范式创新**：将 LLM 领域的"预训练+RLHF"范式成功移植到多任务 RL 扩散规划器，低质量数据也能发挥价值
2. **BC 正则化设计精巧**：用近期最优经验近似目标策略，既防止遗忘又引导探索，优于 KL/PL 正则化
3. **数据效率高**：仅 100k 步微调即可达到 59.26%，性能几乎不降
4. **无需奖励标签预训练**：降低了多任务数据收集的成本门槛

## 局限与展望
1. **仅在仿真环境验证**：Meta-World/Adroit 均为仿真，未在真实机器人上测试
2. **在线微调仍需环境交互**：每个下游任务需要在线 RL 环境，部署到真实场景的 sim-to-real gap 未讨论
3. **分类存疑**：论文实际是 RL 规划而非图像生成，当前文件夹归类为 image_generation 可能不准确
4. **扩展性待验证**：50 个任务的规模有限，更大规模任务集（如数百个任务）的表现未知
5. **去噪步数 K 的计算开销**：PPO 微调需对每个去噪步计算重要性比率，K 较大时计算量显著

## 评分
- 新颖性: ⭐⭐⭐⭐ — 预训练+RL微调范式在扩散规划器上的首次系统探索，BC正则化设计有新意
- 实验充分度: ⭐⭐⭐⭐ — 50任务基准+多种baseline+充分的消融实验，但缺少真实环境验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式推导完整，符号系统规范
- 价值: ⭐⭐⭐⭐ — 为低质量数据驱动的多任务智能体训练提供了可行路径

<!-- RELATED:START -->

## 相关论文

- [LIFT: Latent Implicit Functions for Task- and Data-Agnostic Encoding](../../ICCV2025/image_generation/lift_latent_implicit_functions_for_task-_and_data-agnostic_encoding.md)
- [Tree-Guided Diffusion Planner](../../NeurIPS2025/image_generation/tree-guided_diffusion_planner.md)
- [Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models](zero-shot_adaptation_of_parameter-efficient_fine-tuning_in_diffusion_models.md)
- [Revisiting Diffusion Models: From Generative Pre-training to One-Step Generation](revisiting_diffusion_models_from_generative_pre-training_to_one-step_generation.md)
- [UtilGen: Utility-Centric Generative Data Augmentation with Dual-Level Task Adaptation](../../NeurIPS2025/image_generation/utilgen_utility-centric_generative_data_augmentation_with_dual-level_task_adapta.md)

<!-- RELATED:END -->
