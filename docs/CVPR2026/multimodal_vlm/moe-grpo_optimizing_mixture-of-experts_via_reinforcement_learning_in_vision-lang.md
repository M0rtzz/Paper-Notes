---
title: >-
  [论文解读] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models
description: >-
  [CVPR 2026][多模态][混合专家] 将 MoE 中的专家选择建模为序列决策问题，通过 GRPO 强化学习优化路由策略，引入模态感知路由引导，在 VLM 的图像和视频理解任务上一致超越确定性 top-K 路由及其变体。
tags:
  - CVPR 2026
  - 多模态
  - 混合专家
  - 强化学习
  - 路由策略优化
  - 视觉语言模型
  - GRPO
---

# MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models

**会议**: CVPR 2026  
**arXiv**: [2603.24984](https://arxiv.org/abs/2603.24984)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 混合专家, 强化学习, 路由策略优化, 视觉语言模型, GRPO

## 一句话总结

将 MoE 中的专家选择建模为序列决策问题，通过 GRPO 强化学习优化路由策略，引入模态感知路由引导，在 VLM 的图像和视频理解任务上一致超越确定性 top-K 路由及其变体。

## 研究背景与动机

**领域现状**：Mixture-of-Experts (MoE) 通过稀疏激活子集参数来降低 Transformer 的计算开销，同时保持高模型容量。近期 MoE 已被扩展到视觉语言模型（VLM），实现了高效的多模态理解。标准做法是在每层用确定性 top-K 方式贪心选择专家。

**现有痛点**：确定性 top-K 路由限制了对多样专家组合的探索，容易导致模型过拟合于少数专家子集。V-MoE 等方法通过在 gating score 上加高斯噪声引入随机性，但这种启发式扰动只能部分缓解问题，并没有显式地优化专家选择"策略"。

**核心矛盾**：现有方法要么是确定性选择（缺乏探索），要么是加噪声的随机选择（缺乏方向性），都没有真正学习一个最优的专家路由策略。专家路由本质上是一个序列决策问题，但一直被当作简单的 softmax + top-K 操作。

**本文目标** (1) 如何让 MoE 的路由器学会更优的专家组合？(2) 如何在多模态场景中高效稳定地进行路由策略探索？

**切入角度**：将专家选择显式建模为序列决策问题，借助强化学习（GRPO）通过多组 rollout 探索不同专家组合，并根据奖励反馈优化路由策略。同时观察到不同模态的 token 对专家有不同偏好，可以利用这一先验来约束探索空间。

**核心 idea**：用 GRPO 强化学习替代确定性 top-K 路由，通过 Token-GRPO + Gate-GRPO 联合优化 token 生成和层级专家选择策略，并引入模态感知引导加速收敛。

## 方法详解

### 整体框架

给定输入图像（或视频）和问题 $\boldsymbol{x}$，rollout 模块 $g_\text{old}$ 从 gating network 采样 $G$ 组专家路由策略 $\{\boldsymbol{E}^i\}_{i=1}^G$。每组路由策略 $\boldsymbol{E}^i$ 对应一个跨所有层的专家选择序列，模型在该路由下生成输出 token 序列 $\boldsymbol{y}^i$ 并计算准确率奖励 $R^i$。通过组内相对奖励计算优势值 $\hat{A}^i$，引导策略更新朝高奖励专家组合方向优化。

### 关键设计

1. **Token-GRPO（token 级生成优化）**:

    - 功能：从输出 token 级别优化专家选择策略
    - 核心思路：在标准 GRPO 基础上引入专家路由条件。对每个 rollout 采样的专家策略 $\boldsymbol{E}^i$，模型生成对应的 token 序列 $\boldsymbol{y}^i$，通过 PPO 风格的 clipped ratio 目标函数，基于组内相对奖励优化 token 级生成概率。ratio $r_t^i = \pi_\theta(y_t^i | \boldsymbol{x}, \boldsymbol{y}_{<t}^i; \boldsymbol{E}_{<t}^i) / \pi_\text{old}(y_t^i | \boldsymbol{x}, \boldsymbol{y}_{<t}^i; \boldsymbol{E}_{<t}^i)$
    - 设计动机：Token 级优化直接关联任务奖励，是提升性能的核心驱动力。消融实验表明单独去掉 Token-GRPO 会导致平均准确率从 55.7% 骤降至 50.9%

2. **Gate-GRPO（层级路由策略优化）**:

    - 功能：直接优化每层 gating network 的专家选择策略
    - 核心思路：在每层每个 token 位置计算路由 ratio $\hat{r}_{t,l}^i = g_\theta^l(E_{t,l}^i) / g_\text{old}^l(E_{t,l}^i)$，对所有层和 token 位置取平均，同样用 clipped 目标函数优化。与 Token-GRPO 不同，Gate-GRPO 提供层级的密集监督信号，直接作用于 gating function
    - 设计动机：Token-GRPO 只从输出级别间接影响路由，Gate-GRPO 则对每层路由提供细粒度、密集的监督。二者互补：去掉 Gate-GRPO 平均准确率下降 1.8%

3. **模态感知路由引导（Modality-Aware Router Guidance）**:

    - 功能：约束路由探索空间，避免在不相关专家上浪费探索
    - 核心思路：先统计每个专家被视觉 token 和文本 token 选中的次数 $N_v(e_i)$ 和 $N_t(e_i)$，计算归一化的模态感知分数 $\hat{s}_v(e_i)$ 和 $\hat{s}_t(e_i)$。处理视觉 token 时，按分数排序并将底部 P% 的专家 gating score 设为 $-\infty$ 以屏蔽，然后在剩余专家上进行多项式采样。实验中 P=25%
    - 设计动机：RL 训练需要探索的搜索空间很大（每层 $N$ 选 $K$ 专家，所有层所有 token 的组合），无引导的探索效率低。模态感知引导利用专家的模态偏好先验，减少无效探索，加速收敛。消融显示比无引导的噪声/多项式采样分别高 1.5% 和 0.9%

### 损失函数 / 训练策略

最终目标函数为 $\mathcal{L}_\text{MoE-GRPO} = \mathcal{L}_\text{Token-GRPO} + \mathcal{L}_\text{Gate-GRPO}$。由于 gating network 从头训练无预训练路由策略，不使用 KL 散度正则化（与标准 GRPO 不同）。奖励函数采用准确率奖励（正确=1，错误=0）。训练基于 InternVL3.5-1B 转换的 MoE 架构（N=8 专家，K=2 激活），总参数 2.9B，激活 1.3B。使用 100K 多选视觉指令微调样本，25K 步训练，4 GPU 约一天完成。

## 实验关键数据

### 主实验

| 模型 | 架构 | 激活/总参 | MMBench | MMStar | MLVU | LongVideoBench | Avg. |
|------|------|-----------|---------|--------|------|----------------|------|
| InternVL3.5 + Det-FT | MoE | 1.3B/2.9B | 75.8 | 45.6 | 48.6 | 45.3 | 54.0 |
| InternVL3.5 + Stoch-FT-Noise | MoE | 1.3B/2.9B | 76.3 | 46.1 | 51.1 | 45.3 | 54.3 |
| InternVL3.5 + MoE-GRPO | MoE | 1.3B/2.9B | **77.5** | 45.7 | **53.1** | **46.5** | **56.0** |
| InternVL2.5 (Dense) | Dense | 1B/1B | 70.7 | 50.1 | 57.3 | 47.9 | - |

MoE-GRPO 在 9 个 benchmark 中 7 个最优，平均准确率超过三个基线分别 2.0%、2.3%、1.7%。

### 消融实验

| 配置 | Avg. | 说明 |
|------|------|------|
| Token-GRPO + Gate-GRPO (Full) | 55.7 | 完整模型 |
| 仅 Token-GRPO | 53.9 | 去掉 Gate-GRPO 掉 1.8% |
| 仅 Gate-GRPO | 50.9 | 去掉 Token-GRPO 掉 4.8%，说明 token 级优化是核心 |
| 模态感知引导 | 55.7 | 最优 |
| 模态无关（噪声） | 54.2 | 差 1.5% |
| 模态无关（多项式） | 54.8 | 差 0.9% |

### 关键发现

- Token-GRPO 是性能的核心驱动力，Gate-GRPO 提供互补的层级细粒度监督
- 模态感知引导比无引导方式平均高 0.9-1.5%，且收敛更快、奖励方差更低
- MoE-GRPO 显著提升专家多样性：路由分布熵从 Det-FT 的 1.05 提升到 1.82
- 在跨数据集泛化实验中（CLIP-MoE），MoE-GRPO 比 Det-FT 平均高 3.1%，Det-FT 反而因过拟合而退化
- 跨域泛化中，MoE-GRPO 在所有 OOD 数据集上一致提升，比 CLIP-MoE 平均高 4.1%

## 亮点与洞察

- **RL 优化路由策略的新范式**：首次将 MoE 专家选择建模为序列决策问题并用 RL 优化，思路新颖且有效。这为 MoE 架构的路由优化开辟了新方向，未来可探索更复杂的 RL 算法
- **双层 GRPO 互补设计**：Token-GRPO 和 Gate-GRPO 分别从输出级和层级提供监督，实现了粗粒度目标对齐和细粒度路由优化的结合，这种分层优化思路可迁移到其他分层决策问题
- **模态感知约束探索空间**：利用模态-专家统计先验约束探索空间，在不引入额外复杂度的前提下显著提升训练效率和稳定性。这种用先验知识引导 RL 探索的思路在其他 RL+大模型场景也适用

## 局限与展望

- 目前仅在相对小规模模型（1.3B 激活参数）上验证，更大规模 MoE-VLM（如 DeepSeek-V3 级别）上的效果未知
- rollout 数 G=8 增加了训练计算开销（需生成 8 组输出），如何在大规模训练中保持效率是挑战
- 奖励函数仅用准确率（多选题），对开放式生成任务的适用性有待验证
- 模态感知引导基于静态统计的专家偏好，动态适应能力可能不足

## 相关工作与启发

- **vs V-MoE**：V-MoE 通过加高斯噪声引入探索，但噪声是无方向的启发式扰动，不优化"策略"。MoE-GRPO 显式优化路由策略，效果更好
- **vs Expert Choice / Optimal Transport Routing**：这些方法从负载均衡角度优化路由，MoE-GRPO 从任务奖励角度优化，且与负载均衡 loss 互补（结合后再提升 0.9%）
- **vs 标准 GRPO（DeepSeek-R1）**：标准 GRPO 只在 token 级探索，MoE-GRPO 将动作空间扩展到层级专家选择，提供更细粒度的控制

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将 RL 应用于 MoE 路由策略优化，formulation 清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 图像+视频 9 个 benchmark，跨数据集泛化，域泛化，多个消融，路由分析
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表丰富，方法推导完整
- 价值: ⭐⭐⭐⭐ 为 MoE 路由优化提供新范式，但实际部署中 rollout 开销可能限制应用

<!-- RELATED:START -->

## 相关论文

- [Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token-aware_adaptive_error_reconstruction_with_mixture_of_experts_.md)
- [On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models](on_tokens_dilemma_dynamic_moe_with_drift-aware_token_assignment_for_continual_le.md)
- [MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](modes_moe_dynamic_expert_skipping.md)
- [TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition](trivia_self-supervised_fine-tuning_of_vision-language_models_for_table_recogniti.md)
- [EMO-R3: Reflective Reinforcement Learning for Emotional Reasoning in Multimodal Large Language Models](emo-r3_reflective_reinforcement_learning_for_emotional_reasoning_in_multimodal_l.md)

<!-- RELATED:END -->
