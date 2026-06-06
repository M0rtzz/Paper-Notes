---
title: >-
  [论文解读] Mixture of Horizons in Action Chunking
description: >-
  [ICML 2026][机器人][VLA] 本文针对 VLA 模型中"动作块长度（horizon）选择"导致的"长视野规划 vs. 短视野精控"权衡问题，提出 Mixture of Horizons (MoH)：把同一条动作块拆成多个不同长度的子块，用共享的 action transformer 并行预测…
tags:
  - "ICML 2026"
  - "机器人"
  - "VLA"
  - "动作分块"
  - "多尺度 horizon"
  - "门控融合"
  - "动态推理"
---

# Mixture of Horizons in Action Chunking

**会议**: ICML 2026  
**arXiv**: [2511.19433](https://arxiv.org/abs/2511.19433)  
**代码**: 待确认  
**领域**: 机器人 / VLA / Action Chunking  
**关键词**: VLA, 动作分块, 多尺度 horizon, 门控融合, 动态推理

## 一句话总结
本文针对 VLA 模型中"动作块长度（horizon）选择"导致的"长视野规划 vs. 短视野精控"权衡问题，提出 Mixture of Horizons (MoH)：把同一条动作块拆成多个不同长度的子块，用共享的 action transformer 并行预测，再用 2k 参数的线性门控融合，配合负载均衡损失和"跨 horizon 共识"的动态推理，使 $\pi_{0.5}$ 在 LIBERO 上首次达到 99% 平均成功率，并把吞吐量提高到基线的 2.5 倍。

## 研究背景与动机

**领域现状**：现代 Vision-Language-Action (VLA) 模型（如 $\pi_0$、$\pi_{0.5}$、OpenVLA-OFT、StarVLA）几乎清一色采用 Zhao 等人提出的 action chunking 策略——一次性预测未来 $H$ 步动作 $A_t=(a_t,\dots,a_{t+H-1})$，再以 full-attention 的轻量 action transformer 处理这些动作 token，理论基础是平滑执行、减少策略调用次数、利用时间结构信息。"VLM 主干 + chunk 化动作头"已经成为事实上的标配。

**现有痛点**：作者在 LIBERO 上把 $\pi_0$ 的 horizon 分别设为 10/20/30，在 Spatial、Object、Goal、Long 四个任务集上跑了一遍，结果发现一个朴素却普遍被忽视的事实——成功率对 $H$ 极度敏感，且不同任务的最优 $H$ 不一样：Long 任务偏好长 horizon（要规划），Spatial/Object 偏好短 horizon（要精控）。任何固定 $H$ 都注定在某一类任务上吃亏。

**核心矛盾**：长 horizon → 远期可规划但每一步精度被"摊薄"；短 horizon → 精控但缺乏前瞻。这是 chunk-based 表达本身带来的结构性 trade-off，光靠调超参解决不了，而且部署时还没法在线切换 horizon。

**本文目标**：(i) 系统刻画 horizon 对 VLA 的影响；(ii) 用单一模型同时拿到长视野与短视野的好处；(iii) 让推理时的 chunk 长度能根据置信度自适应缩放。

**切入角度**：既然不同 horizon 各擅胜场，那就别选——把多个 horizon 同时放进训练，让模型学会"该长就长、该短就短"。关键是要让这件事几乎零成本：VLA 计算瓶颈在 VLM 主干，action transformer 本身只有 ~300M 参数，多 horizon 的并行 forward 走 tensor parallelism 几乎不增加 wall-clock。

**核心 idea**：把动作块按多个候选长度 $\mathcal{H}=\{h_1,\dots,h_N\}$ 重排成多个子段，用同一个 action transformer 并行预测，再用 2k 参数线性门控按 step 与 horizon 加权融合，副产物——跨 horizon 的预测一致性天然成为执行置信度信号，可驱动动态截断。

## 方法详解

### 整体框架
给定时刻 $t$ 的多视角图像 $V_t$、历史 $h_{<t}$、指令 $T$、本体感知 $s_t$，VLM 主干编码成 context；接着 MoH 把目标动作块 $A_t\in\mathbb{R}^{H\times d_a}$ 拆成 $N$ 个长度递增的截断子块 $A_t^{(h)}=(a_{t,1},\dots,a_{t,h})$，对每个子块 padding 到 $H$ 并配 horizon-specific attention mask（mask 掉 $k>h$ 位置），由**共享的 action transformer 在一次 forward 中并行处理**所有 horizon，得到 horizon-wise 预测 $\hat A_t^{(h)}$；最后一个线性 gating head 输出 logits $g_{t,k,h}$，经掩码 softmax 得到融合权重 $\alpha_{t,k,h}$，组合出最终预测 $\hat a_{t,k}=\sum_{h:k\le h}\alpha_{t,k,h}\hat a_{t,k}^{(h)}$。整套设计对 flow-matching（$\pi_0$/$\pi_{0.5}$/StarVLA）与一步回归（$\pi_{\text{reg}}$）通用，对 backbone 零侵入。

### 关键设计

1. **动作块多 horizon 重排 + 共享 transformer 并行处理**:

    - 功能：把"选一个 horizon"变成"同时用多个 horizon 训练"，让单一策略内同时学会短期精控与长期规划。
    - 核心思路：固定最大 horizon $H$，定义候选集合 $\mathcal{H}=\{h_1,\dots,h_N=H\}$；对每个 $h\in\mathcal{H}$ 截出 $A_t^{(h)}\in\mathbb{R}^{h\times d_a}$，统一 pad 到 $H$ 并配 horizon-specific mask 屏蔽超出位置；所有 horizon 共享同一组 action transformer 权重和同一份 VLM context，靠 batching + 并行 attention 一次算完。损失同时计算两项——融合预测损失 $L_{\text{mix}}$ 和各 horizon 独立预测损失 $L_{\text{ind}}=\sum_h L^{(h)}$，前者保证融合质量、后者保证每个 horizon 分支自身可用。
    - 设计动机：(a) VLM forward 只跑一次、action transformer 又轻，多 horizon 并行 forward 的额外开销可忽略；(b) 共享权重相当于强迫同一个网络学会"既能短又能长"，而不是简单 ensemble 多个独立模型；(c) padding+mask 让多 horizon 走齐序列长度，避免动态 shape 拖慢 GPU。

2. **2k 参数线性门控 + 负载均衡损失**:

    - 功能：在每个时间步 $k$ 上、按"哪些 horizon 在此处更可信"自适应加权融合预测，并防止门控塌缩到少数 horizon。
    - 核心思路：在共享 transformer 顶部加一个**线性层**（仅 ~2k 参数），输出每个 (step, horizon) 的 logits；对每个 $k$ 只保留 $h\ge k$ 的有效 horizon，做 softmax 归一化得到 $\alpha_{t,k,h}=\exp(g_{t,k,h})/\sum_{h':k\le h'}\exp(g_{t,k,h'})$，再加权求和。为防止门控只青睐某几个 horizon，引入 MoE 风格的负载均衡损失：按 horizon 边界把时间轴切成区间 $S_i$，在每个区间 $i$ 上计算各 horizon 的平均使用率 $\bar\alpha_h^{(i)}$，损失 $L_{\text{bal}}=\frac{1}{|\mathcal{I}|}\sum_i \mathrm{CV}^2(\{\bar\alpha_h^{(i)}\}_h)$ 等于使用率的均方变异系数，最小化它逼门控公平分配。总目标 $L=L_{\text{mix}}+\lambda_{\text{ind}}L_{\text{ind}}+\lambda_{\text{bal}}L_{\text{bal}}$，默认 $\lambda_{\text{ind}}=1$、$\lambda_{\text{bal}}=10^{-3}$。
    - 设计动机：遵循 Occam 原则——几乎所有信息都已编码在共享 transformer 的隐状态里，门控只需要做一次轻量加权决策，复杂结构反而过拟合。消融显示去掉 $L_{\text{bal}}$ 仍优于基线（98.5%），但加上后 Long 任务再涨 1.6 个点，证明平衡正则确实能让长 horizon 真正参与而不是被门控冷落。

3. **基于跨 horizon 共识的动态推理**:

    - 功能：把固定截前 $K$ 步执行改成"按置信度动态决定执行多长前缀"，简单运动时多执行、决策点附近多 replan，同时显著提速。
    - 核心思路：在每步 $k$，把每个 horizon-wise 预测 $\hat a_k^{(h)}$ 视为对融合结果 $\hat a$ 的"投票者"，定义加权 $\ell_1$ 分歧 $\bar d_k=\sum_{h\in\mathcal{H}_k}\alpha_{k,h}\cdot\|\hat a-\hat a_k^{(h)}\|$（$\mathcal{H}_k=\{h\ge k\}$）；先取前 $n$ 步分歧的均值乘以缩放因子 $r$ 作为数据自适应阈值 $\textit{thres}=\mathrm{Mean}(\{\bar d_k\}_{k=1}^n)\cdot r$；然后从 $k=n+1$ 起逐步检查，一旦有效 horizon 数小于 $m$ 或 $\bar d_k>\textit{thres}$ 就 break，把执行前缀 $K_{\text{exec}}$ 定在此处，后续动作延迟到下一轮 replan，自然形成"运动平稳时长前缀、决策关键处短前缀"的自截断行为。
    - 设计动机：以往 chunk-based VLA 把执行前缀写死（如 LIBERO 默认 5、RoboTwin 20），既浪费又脆——简单运动可以一次执行更多步省 VLM 调用，关键帧附近又必须频繁 replan 才能稳。MoH 的多 horizon 天然给出"分歧度"信号，不用额外训练就能驱动自适应截断，实测 $\pi_{0.5}$+MoH 在 2.5× 吞吐下仍超基线，是"免费午餐"。

### 损失函数 / 训练策略
- 总目标：$L=L_{\text{mix}}+\lambda_{\text{ind}}L_{\text{ind}}+\lambda_{\text{bal}}L_{\text{bal}}$，$\lambda_{\text{ind}}=1$、$\lambda_{\text{bal}}=10^{-3}$。
- 对 flow-matching 策略，$L_{\text{mix}}$ 与 $L^{(h)}$ 都是速度匹配损失 $\|v_\theta(A_t^{(\tau)},\tau,\cdot)-(A_t-\epsilon)\|_2^2$；对一步回归策略是 $\ell_1$；对分类型是交叉熵。
- 默认 $\mathcal{H}=\{3,6,\dots,30\}$（步长 $d=3$，共 10 个 horizon），在 4 张 A100 上 30k 迭代、batch size 32，单次训练 <10 小时。

## 实验关键数据

### 主实验

LIBERO（4 任务集、500 trial/集、统一执行前 5 步）：

| 基线 | Spatial | Object | Goal | Long | 平均 |
|------|---------|--------|------|------|------|
| $\pi_{\text{reg}}$ (3B, 30k) | 97.8 | 98.2 | 94.6 | 90.2 | 95.2 |
| $\pi_{\text{reg}}$ + MoH | 99.0 (↑1.2) | 98.8 (↑0.6) | 96.4 (↑1.8) | 91.4 (↑1.2) | **96.4 (↑1.2)** |
| $\pi_0$ (3B, 30k) | 97.4 | 98.2 | 95.4 | 84.2 | 93.8 |
| $\pi_0$ + MoH | 97.6 (↑0.2) | 98.8 (↑0.6) | 96.4 (↑1.0) | 87.4 (↑3.2) | **95.1 (↑1.3)** |
| StarVLA (3B, 30k) | 98.0 | 98.2 | 95.8 | 91.4 | 95.9 |
| StarVLA + MoH | 98.4 | 99.6 | 97.6 | 92.4 | **97.0 (↑1.1)** |
| $\pi_{0.5}$ (3B, 30k) | 98.8 | 99.0 | 97.6 | 95.4 | 97.7 |
| $\pi_{0.5}$ + MoH | 98.8 | **100** | 98.8 | 98.4 (↑3.0) | **99.0 (↑1.3)** |

$\pi_{0.5}$+MoH 仅 30k 迭代就以 99% 平均成功率刷新 LIBERO SOTA（此前最好为 Spatial Forcing 7B 的 98.5%），且模型只有 3B。Long 任务上 +3.0 是最大单点提升，恰好印证 MoH 真正缓解了"长 horizon 才能长规划"的短板。RoboCasa 五任务上 GR00T+MoH 平均涨 3.4 个点（28.0→31.4），证明在远未饱和的家庭场景同样有效。RoboTwin 2.0 七任务也观察到 easy/hard 两种模式下 $\pi_0$+MoH 一致最优。

### 消融实验

固定 $H_{\max}=30$，所有变体在 $\pi_{0.5}$ 上跑：

| 配置 | Spatial | Object | Goal | Long | 平均 | 说明 |
|------|---------|--------|------|------|------|------|
| $\pi_{0.5}$ baseline ($\mathcal{H}=\{30\}$) | 98.8 | 99.0 | 97.6 | 95.4 | 97.7 | 单 horizon |
| + MoH, $d=10$（3 个 horizon） | 98.8 | 99.8 | 97.6 | 96.8 | 98.3 | 仅 3 个 horizon 已涨 0.6 |
| + MoH, $d=3$（10 个 horizon） | 98.8 | 100 | 98.8 | 98.4 | **99.0** | 默认配置，最佳 |
| + MoH, $d=1$（30 个 horizon） | 99.0 | 99.4 | 98.4 | 96.2 | 98.3 | 过密反而下降 |
| + MoH 10 个相同 horizon ($H=30$) | 98.6 | 99.4 | 98.6 | 94.8 | 97.9 | 排除"ensemble effect" |
| + 仅时间维 loss reweight，无 MoH | 99.2 | 99.6 | 99.2 | 94.4 | 98.1 | Long 反而掉，未根治 trade-off |
| + MoH，无门控用均值融合 | 98.8 | 99.2 | 98.6 | 96.8 | 98.4 | 最朴素 MoH 已可用 |
| + MoH，无 $L_{\text{bal}}$ | 98.2 | 100 | 99.0 | 96.8 | 98.5 | 平衡损失主要补 Long |

### 关键发现
- **horizon 多样性才是关键，不是"多分支 ensemble"**：10 个相同 $H=30$ 的分支只能把均值从 97.7% 抬到 97.9%，而 10 个不同 horizon 涨到 99.0%，且只在 Long 任务上拉得开差距。
- **3 个 horizon 已经够用，10 个最佳**：从 1 个加到 3 个最大单步收益，加到 10 个达峰，30 个反而下滑——说明 horizon 集合的"密度"存在最优区间，过密会让训练信号互相干扰。
- **loss reweighting 不能替代 MoH**：单纯按 step 加权可以改 Spatial/Object/Goal，但让 Long 进一步退化（95.4→94.4），证实 MoH 的提升不来自隐式 loss 重权。
- **动态推理是免费午餐**：$\pi_{0.5}$+MoH 在动态截断（$r=1.1$）下吞吐 2.5×、平均执行步数随场景在简单运动时拉长、决策点变短，性能仍超固定前缀基线。

## 亮点与洞察
- **把"超参选择"问题变成"模型内部决策"问题**：horizon 一直被当成需要 grid search 的脆弱超参，MoH 干脆把多个 horizon 全塞进训练让门控学习选择，是一种很优雅的去超参化思路，可直接迁移到 diffusion step 数、history length、temporal stride 等其它"必须选一个"的离散尺度。
- **MoE 思想从 expert 维迁到 horizon 维**：本文证明 MoE 的"门控 + 负载均衡"模板换个变量轴照样有效。负载均衡用 $\mathrm{CV}^2$ 而非传统 KL/熵的细节也值得借鉴——它对 horizon 数变化更稳定。
- **跨预测一致性 = 内生置信度**：以往 chunk-based 模型靠固定前缀执行，本文把多 horizon 的预测分歧当成 confidence 信号驱动自截断，整个过程**零额外训练、零额外参数**，是 MoH 设计的"副产品赚到"——这种"用模型内多视角差异做不确定性估计"的思路在分类模型里也有先例（如 deep ensemble），用到序列动作预测里还是新颖的。
- **几乎零开销**：2k 额外参数 + 一次 forward 共享多分支，对 VLA 这种 VLM 主干占大头的架构尤其友好，使该方法成为标准 chunk-based VLA 的"应当默认开启"组件。

## 局限与展望
- **只对 full-attention action transformer 有效**：纯 causal 自回归（如某些 token-level VLA）无法在一次 forward 中得到不同 horizon 的并行预测，需要做架构调整。
- **horizon 集合仍需手工挑**：虽然消融给出"stride=3、$H_{\max}=30$"的经验最优，但跨平台/跨任务最优值未必稳定，理想做法是让 $\mathcal{H}$ 也可学。
- **评估集中在桌面级 manipulation**：LIBERO/RoboTwin/RoboCasa 都偏短到中等 horizon，论文未涉及真正长时距任务（如多分钟整理房间），MoH 在更极端长 horizon 下是否仍能拉开差距尚未验证。
- **门控可解释性有限**：作者在附录给出 horizon 使用率统计，但具体什么场景偏好哪个 horizon、是否可通过指令显式控制 horizon 偏好，留待后续。

## 相关工作与启发
- **vs. ACT (Zhao 2023)**：ACT 首次提出 chunk-based prediction，固定 $H$；本文指出固定 $H$ 是性能瓶颈，给出多 horizon 解。
- **vs. CogACT (Li 2024)**：CogACT 用相似度加权融合**同一 horizon 内**重叠帧；MoH 融合的是**不同 horizon**的预测，正交且互补。
- **vs. $\pi$ 系列 / OpenVLA-OFT**：这些工作专注 backbone（flow-matching、PaliGemma、OFT 微调），MoH 不动 backbone，作为通用 chunk 模块插上即用，可与它们叠加增益。
- **vs. Switch Transformer / MoE**：思想血缘明显——门控 + 负载均衡损失直接借鉴；区别在 expert 替换为 horizon，意义从"扩容"变成"消除超参 trade-off"。
- **vs. 动态 action chunking / replan 文献**：以往动态 replan 靠值函数或 RL 信号；MoH 用同一模型的多 horizon 一致性免费得到置信度，无需额外训练。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Real-Time Robot Execution with Masked Action Chunking](../../ICLR2026/robotics/real-time_robot_execution_with_masked_action_chunking.md)
- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](../../CVPR2026/robotics/adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [\[ICML 2026\] Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models](neural_implicit_action_fields_from_discrete_waypoints_to_continuous_functions_fo.md)
- [\[ICML 2026\] LangForce: Bayesian Decomposition of Vision-Language-Action Models via Latent Action Queries](langforce_bayesian_decomposition_of_vision_language_action_models_via_latent_act.md)
- [\[ICML 2026\] Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies](discrete_diffusion_vla_bringing_discrete_diffusion_to_action_decoding_in_vision-.md)

</div>

<!-- RELATED:END -->
