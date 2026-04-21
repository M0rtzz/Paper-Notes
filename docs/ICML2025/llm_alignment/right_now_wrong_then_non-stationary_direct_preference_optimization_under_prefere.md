---
title: >-
  [论文解读] Right Now, Wrong Then: Non-Stationary Direct Preference Optimization under Preference Drift
description: >-
  [ICML 2025][LLM对齐][偏好漂移] 提出 NS-DPO，通过 Dynamic Bradley-Terry 模型引入单一指数衰减参数 γ 对训练数据进行时序加权，使 DPO 在偏好随时间漂移的场景下仍能鲁棒对齐，同时在平稳场景下不损失性能。
tags:
  - ICML 2025
  - LLM对齐
  - 偏好漂移
  - 非平稳优化
  - DPO
  - Bradley-Terry模型
  - 偏好对齐
---

# Right Now, Wrong Then: Non-Stationary Direct Preference Optimization under Preference Drift

**会议**: ICML 2025  
**arXiv**: [2407.18676](https://arxiv.org/abs/2407.18676)  
**代码**: 有（论文中提及提供代码，具体链接见论文脚注）  
**领域**: LLM对齐/RLHF  
**关键词**: 偏好漂移, 非平稳优化, DPO, Bradley-Terry模型, 偏好对齐

## 一句话总结

提出 NS-DPO，通过 Dynamic Bradley-Terry 模型引入单一指数衰减参数 γ 对训练数据进行时序加权，使 DPO 在偏好随时间漂移的场景下仍能鲁棒对齐，同时在平稳场景下不损失性能。

## 研究背景与动机

现有偏好优化算法（DPO、IPO 等）假设人类偏好是**平稳的**（stationary），即不随时间变化。但现实中偏好会因多种原因发生漂移：

- **新信息出现**：社会事件、科学发现改变公众观点
- **社会文化趋势**：道德标准、安全要求随时代演进
- **数据收集时间跨度增大**：长期收集的偏好数据集中必然包含变化的偏好

当偏好漂移发生时，平稳算法将其视为**标签噪声**，对过时数据和新鲜数据同等对待，导致模型严重错位。已有研究表明数据质量是微调性能的关键因素，因此偏好漂移会极大影响 LLM 对齐效果。

**核心问题**：如何在仅知道总偏好漂移上界（而不知具体哪些偏好变了）的情况下，设计计算高效的偏好优化算法？

## 方法详解

### 整体框架

NS-DPO 的核心思路非常优雅：在标准 DPO 损失函数中引入一个**指数时序衰减权重** $\gamma^{T-t_i-1}$，使得最近的数据点权重最大，越老的数据权重越小。整个修改仅增加一个超参数 γ。

框架流程：

1. 数据集中每个偏好对附带时间戳 $t_i$
2. 在 DPO 损失中对每个数据点乘以指数衰减权重
3. 通过理论分析确定 γ 的最优设置
4. 正常训练 LLM 策略

### 关键设计

1. **Dynamic Bradley-Terry 模型**：将标准 BT 模型扩展为时变版本 $p(a_i \succ a_i'|x_i, t_i) = \sigma(r(x_i, a_i, t_i) - r(x_i, a_i', t_i))$，奖励函数 $r(x,a,t)$ 显式依赖时间步 $t$。这是首次在直接偏好优化框架中使用动态 BT 模型，捕捉了偏好的时变本质。

2. **指数加权损失函数**：NS-DPO 的损失定义为 $\mathcal{L}^{NS}(\theta_T) = \sum_{(x_i,a_i,a_i',t_i) \in \mathcal{D}} -\gamma^{T-t_i-1} \log\sigma(\tau h_{\pi_{\theta_T}}(x_i, a_i, a_i'))$，其中 $\gamma \in (0,1)$ 控制旧数据的折扣速率。当 $\gamma \to 1$ 时退化为标准 DPO。**关键洞察**：由于不知道哪些数据点的偏好发生了变化，NS-DPO 对所有旧数据统一降权，这是在不确定性下的稳健策略。

3. **变差预算（Variation Budget）假设**：不假设偏好如何漂移，仅假设最优策略参数 $\theta_t^*$ 的总漂移上界为 $B_T$，即 $\sum_{t=1}^{T-1}\|\theta_{t+1}^* - \theta_t^*\|_2 \leq B_T$。这是一个温和假设，允许任意时刻发生剧烈变化，只要总变化受限。

### 损失函数 / 训练策略

**NS-DPO 目标函数推导**：

与标准 DPO 类似，先定义时间步 $T$ 的 RLHF 目标 $\mathcal{J}_T(\pi)$，推导出隐式奖励 $r(x,a,T) = \tau\log\frac{\pi_T^*(a|x)}{\pi_{ref}(a|x)} + \tau\log Z_T^*(x)$，再代入动态 BT 模型的指数加权负对数似然，得到最终的 NS-DPO 损失。

**正则化版本（理论分析用）**：$\mathcal{L}_{reg}^{NS}(\theta) = \frac{1}{n}\mathcal{L}^{NS}(\theta) + \frac{\lambda c_{\sigma,\tau}\tau^2}{2}\|\theta\|^2$

**最优 γ 的选择**：理论分析表明当 $\gamma = 1 - (B_T/T)^{3/4}$ 时可获得最优 regret bound。实际中，Llama-2-7b 使用 $\gamma=0.95$；Llama-3.2-1b 使用 $\gamma=0.85$；TV-HH 数据集上根据变化点 $t_{cp}$ 自适应调整 $\gamma = 1 - \frac{1}{(100-t_{cp})\log(100)}$。

**梯度分析**：NS-DPO 仅调整梯度的**缩放项**（通过时序折扣进一步降低远离 $T$ 的数据点的梯度贡献），**不改变梯度方向**。具体地，梯度中的 $\gamma^{T-t_i-1}\sigma(-h_\theta)$ 缩放项使旧数据的梯度贡献指数衰减。

**理论保证**：

- **估计误差** = 学习误差 + 跟踪误差：学习项 $O(\sqrt{d/n})$ 与标准 DPO 相同，跟踪项 $O(\sqrt{T/(1-\gamma)^3} \cdot B_T)$ 刻画非平稳代价
- **Regret bound**：当 $\gamma = 1-(B_T/T)^{3/4}$ 时，$R_T^{off} = \tilde{O}(dB_T^{3/4}n^{-1/4})$
- **平稳恢复**：当 $B_T \to 0$ 时，恢复标准 $O(n^{-1/2})$ 复杂度，说明 NS-DPO 严格推广了 DPO 的理论

## 实验关键数据

### 主实验

实验在多个非平稳偏好数据集上验证 NS-DPO 的有效性：

| 数据集 (ρ_diff, t_cp) | 模型 | 指标 | NS-DPO | DPO | SW-DPO |
|----------------------|------|------|--------|-----|--------|
| UltraFeedback-LM (0.7, 21) | Llama-3.2-1b-it | LCWR | **8.93** | 7.29 | 6.09 |
| UltraFeedback-LM (0.7, 51) | Llama-3.2-1b-it | LCWR | **8.38** | 7.85 | 4.93 |
| UltraFeedback-LM (0.7, 81) | Llama-3.2-1b-it | LCWR | **7.85** | 7.17 | 4.63 |
| UltraFeedback-LM (1.0, 21) | Llama-3.2-1b-it | LCWR | **9.00** | 8.23 | 6.71 |
| UltraFeedback-LM (1.0, 51) | Llama-3.2-1b-it | LCWR | **7.41** | 6.99 | 5.59 |
| UltraFeedback-LM (1.0, 81) | Llama-3.2-1b-it | LCWR | **7.36** | 6.49 | 4.83 |
| UltraFeedback-LM (平稳) | Llama-3.2-1b-it | LCWR | **9.12** | 8.81 | 8.81 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| γ=0.5~0.9（合成实验） | Reward accuracy >80% | γ 在大范围内表现稳定 |
| γ>0.97 | 性能退化接近 DPO | γ 过大使折扣失效，退化为平稳 DPO |
| SW-DPO (w=33) | 最终精度≈NS-DPO | 但收敛速度显著慢于 NS-DPO |
| SW-DPO (w=50) | 性能下降 | 窗口过大引入过时数据 |
| tDPO（prompt加时间信息） | ≈DPO | ICL 方式无法有效处理偏好漂移 |
| 平稳数据集 (tcp=0) | NS-DPO ≈ DPO | 无偏好漂移时不损失性能 |
| 渐变偏好漂移（TV-HH） | NS-DPO > DPO +10% | 渐变漂移场景同样显著优于基线 |
| 2C NSGO (US→Germany) | NS-DPO >60%, DPO ~55% | 文化偏好渐变中差距约10% |

### 关键发现

1. **变化点越晚，NS-DPO 优势越大**：当偏好变化发生在较晚的时间步（tcp=81）时，NS-DPO 与 DPO 的差距最大。因为此时大部分数据带有旧偏好，DPO 被错误标签"淹没"
2. **ρ_diff 越大，NS-DPO 越有价值**：当更多数据点的偏好发生翻转（ρ_diff=0.9），非平稳算法的收益最大
3. **平稳场景无副作用**：在无漂移数据上 NS-DPO 性能匹配甚至轻微超过 DPO（LCWR 9.12 vs 8.81）
4. **Win rate 实验**：Llama-3.2-1b-it 在 TV-HH 上，后期变化点 NS-DPO 的 win rate 稳定 >0.5 战胜 DPO
5. **NS-DPO 对 γ 鲁棒**：合成实验中 γ∈[0.5, 0.97] 均保持 >80% 准确率，而 SW-DPO 对窗口大小更敏感

## 亮点与洞察

- **极简改动，高度实用**：仅在 DPO 损失中加一个 $\gamma^{T-t_i-1}$ 权重，实现难度极低，任何使用 DPO 的代码库都可快速集成
- **理论与实践双重保证**：不仅有 log-linear 设定下的 regret bound，还在多个 LLM 规模实验中验证效果
- **默认可用策略**：由于平稳场景无性能损失，NS-DPO 可作为 DPO 的"默认安全替代"，当不确定数据是否含漂移时直接使用
- **非平稳数据集构造方法**：论文提出的多种构造非平稳偏好数据集的方法（突变型/渐变型、多奖励模型切换、跨文化偏好插值）本身就是有价值的实验基础设施

## 局限与展望

1. **需要时间戳信息**：NS-DPO 依赖数据点的时间标记，而现有大多数偏好数据集不提供此信息，限制了直接应用
2. **γ 的选择仍需先验**：虽然理论给出最优 γ 需要 $B_T$，但实际中 $B_T$ 未知，γ 需要调参
3. **仅考虑离线场景**：论文聚焦离线设定，未扩展到 online RLHF 或迭代对齐场景（虽然提到可扩展）
4. **可能放大新数据偏见**：NS-DPO 重视新数据，如果新数据本身有偏见，会被放大
5. **理论分析限于 log-linear 策略**：实际 LLM 是深度神经网络，理论保证的适用性有局限

## 相关工作与启发

- **DPO 系列**：Rafailov et al. 2024 (DPO), Azar et al. 2024 (IPO), SimPO, ORPO 等都未考虑时变偏好
- **非平稳 bandit**：Russac et al. 2019 的指数加权策略是本文方法的直接灵感来源
- **Carroll et al. 2024**：讨论了偏好漂移下 LLM 对齐的多种目标设定，但未给出实际算法
- **启发**：该思路可扩展到多目标对齐（不同维度偏好以不同速率漂移）、在线持续对齐（模型部署后持续接收新偏好反馈）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 问题提出有洞察，但技术手段（指数衰减加权）属于经典策略的迁移
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4个数据集、2种LLM、突变/渐变/文化漂移、合成验证理论，覆盖全面
- **写作质量**: ⭐⭐⭐⭐⭐ — 理论推导严谨，实验说明清晰，问题动机论述compelling
- **实用价值**: ⭐⭐⭐⭐ — 改动极简且无副作用，但实际数据缺乏时间戳是落地障碍

<!-- RELATED:START -->

## 相关论文

- [TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization](tgdpo_harnessing_token-level_reward_guidance_for_enhancing_direct_preference_opt.md)
- [D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples](d-fusion_direct_preference_optimization_for_aligning_diffusion_models_with_visua.md)
- [Rethinking Direct Preference Optimization in Diffusion Models](../../NeurIPS2025/llm_alignment/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [ADHMR: Aligning Diffusion-based Human Mesh Recovery via Direct Preference Optimization](adhmr_aligning_diffusion-based_human_mesh_recovery_via_direct_preference_optimiz.md)
- [On Extending Direct Preference Optimization to Accommodate Ties](../../NeurIPS2025/llm_alignment/on_extending_direct_preference_optimization_to_accommodate_ties.md)

<!-- RELATED:END -->
