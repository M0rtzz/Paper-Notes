---
title: >-
  [论文解读] Spotlight on Token Perception for Multimodal Reinforcement Learning
description: >-
  [ICLR 2026][RLVR] 提出 VPPO（Visually-Perceptive Policy Optimization），通过量化每个 token 的视觉依赖度，在轨迹级和 token 级两个层次对学习信号进行精细化调控，显著提升大视觉语言模型的多模态推理能力。
tags:
  - ICLR 2026
  - RLVR
  - 多模态推理
  - token感知
  - 视觉依赖
  - 策略优化
---

# Spotlight on Token Perception for Multimodal Reinforcement Learning

**会议**: ICLR 2026  
**arXiv**: [2510.09285](https://arxiv.org/abs/2510.09285)  
**代码**: [https://github.com/huaixuheqing/VPPO-RL](https://github.com/huaixuheqing/VPPO-RL)  
**领域**: 多模态强化学习 / 视觉语言模型  
**关键词**: RLVR, 多模态推理, token感知, 视觉依赖, 策略优化

## 一句话总结

提出 VPPO（Visually-Perceptive Policy Optimization），通过量化每个 token 的视觉依赖度，在轨迹级和 token 级两个层次对学习信号进行精细化调控，显著提升大视觉语言模型的多模态推理能力。

## 研究背景与动机

- **RLVR 在多模态中的局限**：现有 RLVR（如 GRPO、DAPO）主要为文本推理设计，在多模态场景中忽视了视觉感知的关键作用。它们对所有 token 广播统一的学习信号，无法区分哪些 token 真正依赖视觉信息。
- **感知与推理的耦合**：有效的多模态推理需要准确的视觉感知作为逻辑推理的基础。例如几何题中，模型必须从图像中识别出 OA、OB 是圆的半径，才能得出等腰三角形的结论。
- **核心发现**：
    - **Insight 1**：轨迹中 token 的视觉依赖度呈稀疏分布——仅少数 token 具有高视觉依赖性
    - **Insight 2**：不同推理轨迹在整体视觉依赖度上存在显著异质性——并非所有正确路径都是真正的视觉驱动推理

## 方法详解

### 整体框架

VPPO 在标准 GRPO 基础上引入两个模块：**轨迹级优势塑形（TAS）** 和 **token 级梯度过滤（TGF）**，通过视觉依赖度分数分层调控学习信号。

### 1. 量化 Token 视觉依赖度

定义 token 在时刻 $t$ 的视觉依赖度为原始图像与遮蔽图像条件下输出分布的 KL 散度：

$$\mathcal{S}(s_t, I) := D_{\text{KL}}\left(\pi_\theta(\cdot|s_t, I) \| \pi_\theta(\cdot|s_t, I')\right)$$

其中 $I$ 是原始图像，$I'$ 是非信息性遮蔽版本。高 $\mathcal{S}$ 值表明该 token 的预测高度依赖视觉输入。

### 2. Token 级梯度过滤（TGF, Micro-level）

对每条轨迹 $\tau_i$，选取视觉依赖度最高的 top-$k\%$ token 构建二值梯度掩码：

$$m_{i,t} = \mathbb{I}(t \in \mathcal{K}_i)$$

只对这些关键 token 计算策略梯度，过滤掉通用 token 的噪声，对抗信号稀释。

### 3. 轨迹级优势塑形（TAS, Macro-level）

计算每条轨迹的平均视觉依赖度 $\bar{\mathcal{S}}(\tau_i)$，通过归一化生成塑形因子：

$$\alpha(\tau_i) = \beta_{\min} + (\beta_{\max} - \beta_{\min}) \frac{\bar{\mathcal{S}}(\tau_i) - \min_{\tau_j} \bar{\mathcal{S}}(\tau_j)}{\max_{\tau_j} \bar{\mathcal{S}}(\tau_j) - \min_{\tau_j} \bar{\mathcal{S}}(\tau_j)}$$

塑形后的优势为 $\hat{A}'(\tau_i) = \alpha(\tau_i) \cdot \hat{A}_{\text{GRPO}}(\tau_i)$，放大高视觉参与轨迹的更新，抑制低视觉依赖路径。

### 4. VPPO 目标函数

$$\mathcal{L}^{\text{VPPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} m_{i,t} \cdot \min\left(r_{i,t}(\theta)\hat{A}'_i, \text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}'_i\right)\right]$$

### 理论分析

方差缩减定理：$\text{Var}(\mathbf{g}_{\text{VPPO}}) \approx k \cdot \mathbb{E}[\alpha(\tau)^2] \cdot \text{Var}(\mathbf{g}_{\text{GRPO}})$，其中 $k \in (0,1)$ 是稀疏率，$\alpha(\tau)$ 被缩放到 1 附近的窄带，因此方差显著降低。

## 实验结果

### 主实验：8 个多模态推理基准 (avg@8 acc %)

| 模型 | MathVerse | DynaMath | MMK12 | Geo3k | MathVision | We-Math | LogicVista | MMMU-Pro | Avg. |
|------|-----------|----------|-------|-------|------------|---------|------------|----------|------|
| Qwen2.5-VL-7B | 39.0 | 55.7 | 42.5 | 37.1 | 18.4 | 46.4 | 42.4 | 25.1 | 38.3 |
| + GRPO | 66.5 | 65.8 | 72.3 | 40.2 | 30.7 | 68.1 | 45.6 | 35.2 | 53.1 |
| + DAPO | 68.3 | 66.6 | 82.1 | 41.5 | 30.5 | 68.0 | 46.8 | 35.9 | 55.0 |
| **+ VPPO** | **71.6** | **68.1** | **82.8** | **46.5** | **33.3** | **71.5** | **47.9** | **37.9** | **57.5** |

### 32B 规模扩展

| 模型 | Avg. |
|------|------|
| Qwen2.5-VL-32B + GRPO | 62.6 |
| Qwen2.5-VL-32B + DAPO | 63.5 |
| **Qwen2.5-VL-32B + VPPO** | **64.6** |

### 关键发现

- 7B 模型上 VPPO 相比 baseline 平均提升 **19.2%**，超越所有开源 RL 方法
- 32B 模型上带来 **7.6%** 平均提升
- 训练更稳定、收敛更快

## 消融实验

| 设置 | Avg. Acc |
|------|----------|
| VPPO (完整) | 57.5 |
| 仅 TAS（轨迹级优势塑形） | 55.8 |
| 仅 TGF（token 级梯度过滤） | 56.2 |
| 无 TAS + 无 TGF (DAPO baseline) | 55.0 |

- TAS 和 TGF 各自独立有效，组合后效果最优
- 梯度过滤比例 $k=0.4$ 为最佳选择

## 亮点与洞察

1. **首次从 token 感知角度分析多模态 RLVR**：揭示了视觉依赖的稀疏分布和轨迹异质性两个关键 insight
2. **双层次信号调控**：轨迹级 + token 级的层次化设计优雅且有效
3. **即插即用**：可无缝集成到 GRPO、DAPO 等主流算法中
4. **理论支撑**：证明了方差缩减效果

## 局限性

- 视觉依赖度计算需要额外的遮蔽图像前向传播，增加了计算开销
- 仅在 Qwen2.5-VL 系列上验证，其他模型架构的泛化性待验证
- 遮蔽策略的选择（如何构造 $I'$）可能影响依赖度估计质量

## 相关工作

- **多模态推理 RL**：GRPO、DAPO、NoisyRollout、VL-Rethinker 等，但均忽视视觉感知
- **奖励设计**：PAPO-D 等感知感知奖励方法，但在算法层面未做改进
- **关键 token 识别**：RLHF 中的分叉点检测、低置信度点探索等，但未针对多模态视觉依赖

## 评分

- **创新性**: ⭐⭐⭐⭐ — 从 token 视觉依赖角度切入多模态 RL，视角新颖
- **技术深度**: ⭐⭐⭐⭐ — 理论分析扎实，方差缩减有证明
- **实验充分性**: ⭐⭐⭐⭐⭐ — 8 个 benchmark，两个规模，消融完备
- **实用价值**: ⭐⭐⭐⭐ — 即插即用，效果显著

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] APPLE: Toward General Active Perception via Reinforcement Learning](apple_toward_general_active_perception_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Real-World Reinforcement Learning of Active Perception Behaviors](../../NeurIPS2025/reinforcement_learning/real-world_reinforcement_learning_of_active_perception_behaviors.md)
- [\[ICLR 2026\] Learning to Generate Unit Test via Adversarial Reinforcement Learning](learning_to_generate_unit_test_via_adversarial_reinforcement_learning.md)
- [\[ICLR 2026\] Metis-SPECS: Decoupling Multimodal Learning via Self-distilled Preference-based Cold Start](metis-specs_decoupling_multimodal_learning_via_self-distilled_preference-based_c.md)
- [\[CVPR 2026\] Linking Perception, Confidence and Accuracy in MLLMs](../../CVPR2026/reinforcement_learning/linking_perception_confidence_and_accuracy_in_mllms.md)

<!-- RELATED:END -->
