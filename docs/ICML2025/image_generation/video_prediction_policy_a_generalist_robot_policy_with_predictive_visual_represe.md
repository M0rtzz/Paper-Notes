---
title: >-
  [论文解读] Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations
description: >-
  [ICML 2025 (Spotlight)][图像生成][扩散模型] 利用视频扩散模型（VDM）内部的"预测性视觉表征"（同时编码当前帧和未来帧信息）来隐式学习逆动力学模型，从而以高频闭环方式生成机器人动作，在仿真和真实世界操作任务上大幅超越已有方法。
tags:
  - ICML 2025 (Spotlight)
  - 图像生成
  - 扩散模型
  - Predictive Visual Representation
  - Inverse Dynamics
  - Robot Manipulation
  - Generalist Policy
---

# Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations

**会议**: ICML 2025 (Spotlight)  
**arXiv**: [2412.14803](https://arxiv.org/abs/2412.14803)  
**代码**: [项目主页](https://video-prediction-policy.github.io)  
**领域**: 机器人策略学习 / 视觉表征 / 视频扩散模型  
**关键词**: Video Diffusion Model, Predictive Visual Representation, Inverse Dynamics, Robot Manipulation, Generalist Policy

## 一句话总结

利用视频扩散模型（VDM）内部的"预测性视觉表征"（同时编码当前帧和未来帧信息）来隐式学习逆动力学模型，从而以高频闭环方式生成机器人动作，在仿真和真实世界操作任务上大幅超越已有方法。

## 研究背景与动机

构建能执行多种任务的通用机器人策略是当前研究热点，其中**视觉编码器**是关键组件。现有视觉预训练方法主要包括：

- **单图重建**（如 MAE、VC-1）：仅捕获静态空间信息
- **双图对比学习**（如 R3M、VIP）：仅对比两帧，动态信息有限
- **图文对比学习**（如 CLIP）：侧重语义对齐，缺少物理动态建模

这些方法的共同局限是**无法显式编码未来状态**，而机器人操作本质上需要对未来物理演化的预测。

与此同时，视频扩散模型（VDM）在视频生成上展示了强大的物理世界理解能力。论文提出关键假设：VDM 的中间表征天然具有 $(T, H, W)$ 结构，第 1 帧表示当前状态、后续 $T-1$ 帧表示预测的未来——这种"预测性视觉表征"对机器人任务极为有价值。

此前利用视频预测做控制的工作（如 SuSIE、UniPi）需要完整去噪生成图像，推理慢、控制频率低。GR-1 自回归但每次只输出一帧，且未利用预训练视频基础模型。VPP 直接提取 VDM 内部表征（单步前向），避免多步去噪，实现高频闭环控制。

## 方法详解

### 整体框架

VPP 采用**两阶段训练**：

1. **Stage 1 — 微调 TVP 模型**：将通用视频基础模型 SVD（1.5B 参数）微调为操作领域的文本引导视频预测模型，使用互联网人类操作数据 + 机器人数据 + 自采集数据
2. **Stage 2 — 基于预测表征学习动作**：冻结 TVP 模型作为视觉编码器，通过 Video Former 聚合时空特征，再用 Diffusion Policy 头生成动作序列

核心洞察：下游策略可以在预测表征中**隐式追踪机器人运动轨迹**来学习逆动力学模型。只要视频模型能准确预测未来场景，策略就能通过隐式追踪生成正确动作，从而将视频预测模型的泛化能力迁移到策略上。

### 关键设计一：文本引导视频预测模型（TVP）

基于 Stable Video Diffusion 改造：增加 CLIP 语言特征 $l_{emb}$ 的交叉注意力层实现文本引导，调整输出分辨率至 $16 \times 256 \times 256$，保留 SVD 预训练权重。

训练目标为扩散重建损失：

$$\mathcal{L}_D = \mathbb{E}_{x_0 \sim D, \epsilon, t} \| V_\theta(x_t, l_{emb}, s_0) - x_0 \|^2$$

其中 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_t$ 是加噪视频，$s_0$ 是初始帧。为平衡不同数据集引入采样系数：

$$\mathcal{L}_{video} = \lambda_H \mathcal{L}_{D_H} + \lambda_R \mathcal{L}_{D_R} + \lambda_C \mathcal{L}_{D_C}$$

训练数据：191K Something-Something-v2 人类操作轨迹（采样比 0.30）、约 155K Open X-Embodiment 机器人轨迹、以及下游任务数据。微调约需 2-3 天（8×A100）。

### 关键设计二：单步前向提取预测表征 + Video Former

**单步前向编码**：将当前图像 $s_0$ 与纯噪声拼接输入 TVP 模型，仅做**一次前向传播**，提取上采样层中间特征：

$$L_m = V_\theta(x_{t'}, l_{emb}, s_0)^{(m)}, \quad L_m \in \mathbb{R}^{T \times C_m \times W_m \times H_m}$$

对各层特征插值到统一尺寸并沿通道维拼接得到 $F_p \in \mathbb{R}^{T \times (\sum_m C_m) \times W_p \times H_p}$。可视化表明单步前向虽纹理不精确，但已能反映物体运动和机械臂轨迹。

**Video Former**：初始化可学习 token $Q_{[0:T, 0:L]}$，通过空间-时间注意力聚合多视角多帧特征：

$$Q' = \{ \text{Spat-Attn}(Q[i], (F_p^{static}[i], F_p^{wrist}[i])) \}_{i=0}^{T}, \quad Q'' = \text{FFN}(\text{Temp-Attn}(Q'))$$

### 关键设计三：Diffusion Policy 动作头

$Q''$ 通过交叉注意力注入 Diffusion Transformer 块，用扩散策略生成动作序列：

$$\mathcal{L}_{diff}(\psi; A) = \mathbb{E}_{a_0, \epsilon, k} \| D_\psi(a_k, l_{emb}, Q'') - a_0 \|^2$$

采用 action chunking（10步），整体推理仅需 TVP 单次前向（<160ms），在 RTX 4090 上达到 **7-10 Hz** 控制频率。

## 实验关键数据

### Calvin ABC→D 零样本长程评估

| 方法 | 类型 | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg. Len |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| RT-1 | 直接动作 | 0.533 | 0.222 | 0.094 | 0.038 | 0.013 | 0.90 |
| Diffusion Policy | 直接动作 | 0.402 | 0.123 | 0.026 | 0.008 | 0.00 | 0.56 |
| GR-1 | 未来预测 | 0.854 | 0.712 | 0.596 | 0.497 | 0.401 | 3.06 |
| Vidman | 未来预测 | 0.915 | 0.764 | 0.682 | 0.592 | 0.467 | 3.42 |
| RoboUniview | 3D 方法 | 0.942 | 0.842 | 0.734 | 0.622 | 0.507 | 3.65 |
| **VPP (ours)** | **预测表征** | **0.965** | **0.909** | **0.866** | **0.820** | **0.769** | **4.33** |
| GR-1 (10%数据) | — | 0.672 | 0.371 | 0.198 | 0.108 | 0.069 | 1.41 |
| VPP (10%数据) | — | 0.878 | 0.746 | 0.632 | 0.540 | 0.453 | 3.25 |

VPP 将 SOTA Avg. Len 从 3.65 提升至 **4.33**（+18.6%）。仅用 10% 数据时 VPP (3.25) 仍优于全数据下所有基线。

### 真实世界实验成功率

| 平台/任务类型 | Diffusion Policy | Susie | GR-1 | **VPP** |
|:---|:---:|:---:|:---:|:---:|
| **Franka Panda — 已见任务** | 0.42 | 0.56 | 0.52 | **0.85** |
| **Franka Panda — 未见任务** | 0.25 | 0.46 | 0.38 | **0.73** |
| **灵巧手 — 已见任务** | 0.28 | 0.45 | 0.32 | **0.75** |
| **灵巧手 — 未见任务** | 0.11 | 0.28 | 0.15 | **0.60** |
| **灵巧手 — 工具使用** | 0.05 | 0.23 | 0.15 | **0.68** |

灵巧手上 VPP 在未见任务比最强基线 Susie 提升 **31.6%**，工具使用任务提升尤其显著（0.68 vs 0.23）。

## 关键发现

1. **预测性表征显著优于静态表征**：替换为 Stable-VAE / VC-1 / Voltron 编码器后 Avg. Len 分别仅 2.58 / 1.23 / 1.54（VPP 为 4.33）
2. **单步前向已包含足够信息**：可视化表明单步预测虽纹理不精确，但已能反映物体运动和机械臂轨迹
3. **互联网数据和预训练至关重要**：去掉互联网数据 4.33→3.97；进一步去掉 SVD 预训练降至 1.63
4. **Video Former 不可或缺**：去掉后性能降至 3.86，推理延迟 140ms→450ms
5. **多层特征聚合优于单层**：仅用最终层，Avg. Len 从 4.33 降至 3.60
6. **视频预测质量突出**：Bridge 数据集 FVD = 41.4，远优于 Seer (246.3)
7. **数据效率极高**：10% Calvin 数据即可超越全数据下所有基线

## 亮点与洞察

- **核心创新优雅**：将视频扩散模型从"视频生成器"重新定位为"预测性视觉编码器"，单步前向即可提取包含未来信息的表征，完美平衡表征质量和推理效率
- **泛化机制清晰**：(1) 视频模型通过互联网规模预训练实现对未见场景的合理预测；(2) 低层策略只需隐式追踪机器人运动，无需关注具体物体/背景
- **从"生成"到"表征"的范式转换**：相比 SuSIE/UniPi 做完整去噪再用图像，VPP 直接用中间表征，避免信息损失和高延迟
- **多数据源融合策略合理**：通过采样比例平衡异构数据（人类操作 30%、RT-1 15%、Bridge 15% 等）
- **灵巧手工具使用实验令人印象深刻**：12自由度灵巧手上勺子/锤子/电钻/移液器任务，VPP 平均 68%，基线几乎无法完成

## 局限性

1. **计算开销较大**：TVP 模型 1.5B 参数，单步前向约 140ms，低算力平台部署受限
2. **强依赖视频预训练**：去掉 SVD 预训练后性能骤降至 1.63
3. **未探索更先进视频模型**：仅用 SVD 1.5B，未尝试 CogVideoX 等更强模型
4. **单步前向缺乏理论分析**：为何单步就够用？仅有可视化验证
5. **动作空间对齐仍需少量 demo 数据**：每个新平台需要示教数据

## 相关工作与启发

- **与 GR-1**：GR-1 自回归逐帧生成，VPP 一次性产生多帧预测表征，质量更高且不需训练解码部分
- **与 SuSIE/UniPi**：它们需完整去噪生成目标图像，VPP 直接用中间表征，推理快一个数量级
- **与 Vidman**：Vidman 也用视频扩散表征但未微调视频模型，VPP 的领域微调带来显著提升
- **启发**：生成模型的中间表征可能比最终输出更有价值，可推广到 LLM 中间表征做决策、图像扩散中间特征做检测等

## 评分

⭐⭐⭐⭐⭐

ICML 2025 Spotlight 实至名归。提出了简洁而强大的范式：将视频扩散模型用作预测性视觉编码器，单步前向获得包含未来动态的表征。实验覆盖 2 仿真 + 2 真实世界平台，所有设置下均大幅领先，核心思想具有很强的普适性和启发性。

<!-- RELATED:START -->

## 相关论文

- [Failure Prediction at Runtime for Generative Robot Policies](../../NeurIPS2025/image_generation/failure_prediction_at_runtime_for_generative_robot_policies.md)
- [Theoretical Guarantees on the Best-of-n Alignment Policy](theoretical_guarantees_on_the_best-of-n_alignment_policy.md)
- [Flattening Hierarchies with Policy Bootstrapping](../../NeurIPS2025/image_generation/flattening_hierarchies_with_policy_bootstrapping.md)
- [Discriminative Policy Optimization for Token-Level Reward Models](discriminative_policy_optimization_for_token-level_reward_models.md)
- [Dense Policy: Bidirectional Autoregressive Learning of Actions](../../ICCV2025/image_generation/dense_policy_bidirectional_autoregressive_learning_of_actions.md)

<!-- RELATED:END -->
