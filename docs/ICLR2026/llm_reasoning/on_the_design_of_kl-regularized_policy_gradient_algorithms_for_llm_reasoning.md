---
title: >-
  [论文解读] On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning
description: >-
  [ICLR 2026][LLM推理][KL正则化] 提出 Regularized Policy Gradient (RPG) 框架，系统推导并分析了基于 Forward/Reverse KL 散度（归一化和非归一化形式）的策略梯度方法，发现 GRPO 的 KL 项存在理论不一致性，并在数学推理任务上取得优于 GRPO、REINFORCE++、DAPO 的结果。
tags:
  - ICLR 2026
  - LLM推理
  - KL正则化
  - 策略梯度
  - GRPO
  - REINFORCE
---

# On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning

**会议**: ICLR 2026  
**arXiv**: [2505.17508](https://arxiv.org/abs/2505.17508)  
**代码**: [https://github.com/complex-reasoning/RPG](https://github.com/complex-reasoning/RPG)  
**领域**: LLM Reasoning / Reinforcement Learning  
**关键词**: KL正则化, 策略梯度, LLM推理, GRPO, REINFORCE

## 一句话总结

提出 Regularized Policy Gradient (RPG) 框架，系统推导并分析了基于 Forward/Reverse KL 散度（归一化和非归一化形式）的策略梯度方法，发现 GRPO 的 KL 项存在理论不一致性，并在数学推理任务上取得优于 GRPO、REINFORCE++、DAPO 的结果。

## 研究背景与动机

策略梯度方法（如 PPO、GRPO）已广泛用于 LLM 的 RLHF 和推理能力增强。KL 散度正则化是稳定策略优化的关键技术，能防止策略偏离参考策略过远，避免灾难性遗忘和过度自信输出。

然而，现有方法在 KL 散度的具体实现上存在显著差异：
- **KL 方向选择**：Forward KL（零强制/zero-forcing）vs. Reverse KL（模式寻找/mode-seeking），两者具有不同的优化性质
- **归一化处理**：标准归一化 KL vs. 非归一化 KL（UKL），后者通过 $k_3$ 估计器与 GRPO 中的实现形式相关
- **估计器类型**：完全可微分形式 vs. REINFORCE 风格（使用 stop-gradient 算子）
- **离策略估计**：在 off-policy 设置中，重要性权重的处理方式影响梯度的正确性

作者指出 GRPO 的 KL 惩罚项在 off-policy 估计中缺少重要性权重，导致梯度无法精确对应目标函数的梯度。REINFORCE++ 的 KL 处理也存在非标准性——其 KL 项基于旧策略和 SFT 策略，而非当前正在优化的策略。

## 方法详解

### 整体框架

RPG 是一个迭代训练框架：每次迭代中，参考模型 $\pi_{\text{old}}$ 更新为上一轮的策略 $\pi_{\theta^{(t)}}$，提供动态自适应的正则化目标。框架的核心是构建 KL 正则化目标函数 $J(\theta) = \mathbb{E}_{\pi_\theta}[R] - \beta \cdot \text{KL}$，并推导对应的代理损失函数用于梯度下降。

### 关键设计

1. **Forward KL 正则化 (FKL)**：

    - 目标函数：$J_{\text{FKL}}(\theta) = \mathbb{E}_{\pi_\theta}[R(x)] - \beta \text{KL}(\pi_{\text{old}} \| \pi_\theta)$
    - 梯度：$\nabla_\theta J = \mathbb{E}_{x \sim \pi_{\text{old}}}[(w(x)R(x) + \beta) \nabla_\theta \log \pi_\theta(x)]$
    - 代理损失：$\mathcal{L}_{\text{FKL}} = \mathbb{E}[-w(x)R(x) - \beta \log \pi_\theta(x)]$
    - 当 $R=0$ 时退化为 MLE，与 SFT 训练目标一致
    - **设计动机**：Forward KL 鼓励 $\pi_\theta$ 覆盖 $\pi_{\text{old}}$ 的支撑集（zero-forcing），避免遗漏高概率区域

2. **Reverse KL 正则化 (RKL)**：

    - 目标函数：$J_{\text{RKL}}(\theta) = \mathbb{E}_{\pi_\theta}[R(x)] - \beta \text{KL}(\pi_\theta \| \pi_{\text{old}})$
    - 代理损失：$\mathcal{L}_{\text{RKL}} = \mathbb{E}[w(x)(-R(x) + \beta \log w(x))]$
    - **设计动机**：Reverse KL 鼓励 $\pi_\theta$ 集中在 $\pi_{\text{old}}$ 高概率区域（mode-seeking），适合聚焦已知好策略

3. **非归一化 Forward KL (UFKL)**：

    - 引入非归一化 KL 散度，包含质量修正项（mass correction）
    - 代理损失：$\mathcal{L}_{\text{UFKL}} = Z_{\text{old}} \mathbb{E}[-w(x)R(x) + \beta(w(x) - \log w(x) - 1)]$
    - 正则化项 $w(x) - \log w(x) - 1$ 正是 $k_3$ 估计器的形式
    - **设计动机**：处理分布可能非归一化的场景，并建立与 GRPO 使用的 $k_3$ 估计器之间的联系

4. **非归一化 Reverse KL (URKL)**：

    - 证明 $k_3(\pi_{\text{old}}/\pi_\theta)$ 的期望等价于 $\text{UKL}(\pi_\theta \| \pi_{\text{old}})$
    - 代理损失：$\mathcal{L}_{\text{URKL}} = Z_{\text{old}} \mathbb{E}[-w(x)R(x) + \beta(w(x)\log w(x) - w(x))]$
    - **设计动机**：梯度中有效奖励缩放因子更简洁（$R(x) - \beta \log w(x)$），且与 $k_3$ 估计器等价

5. **REINFORCE 风格变体**：

    - 对所有四种 KL 形式提供 REINFORCE 风格的代理损失，使用 $\text{SG}(\cdot)$（stop-gradient）算子
    - 一般形式：$\mathcal{L}^{\text{REINFORCE}} = -\mathbb{E}[\text{SG}(\text{Weight}(x, \theta)) \log \pi_\theta(x)]$
    - 提供了实现灵活性，可适配不同框架需求

6. **GRPO 理论不一致性分析**：

    - GRPO 使用 $k_3$ 估计器作为 KL 惩罚，但在 off-policy 设置下直接减去该项，缺少重要性权重 $w_{i,t}$
    - 这导致 GRPO 目标函数的梯度无法精确对应 $J_{\text{Clip}} - \beta \text{UKL}(\pi_\theta \| \pi_{\text{ref}})$ 的梯度
    - RPG 框架通过显式引入重要性权重修正了这一问题

### 损失函数 / 训练策略

- **Dual-Clip 目标**：采用 PPO 的 Dual-Clip 变体稳定训练，对正负优势值分别处理
- **基线减法**：使用批次平均奖励作为基线减少梯度方差
- **动态采样 + 组过滤**：借鉴 DAPO 的策略，对困难 prompt 过采样，过滤近完美或近零准确率的样本
- **过长惩罚**：在奖励中对过度冗长的输出进行惩罚
- **内存效率**：$\pi_{\text{old}}$ 的 log 概率可预计算存储，训练时 GPU 上只需保留一个模型 $\pi_\theta$

## 实验关键数据

### 主实验

基于 Qwen2.5-7B-Instruct，使用 DAPO-Math-17k（英文部分 13.9k 样本），训练 400 步。

| 方法 | AMC23 (Best) | AIME24 (Best) | AIME25 (Best) |
|------|-------------|---------------|---------------|
| GRPO | 0.7250 | 0.1406 | 0.0948 |
| REINFORCE++ | 0.7664 | 0.1177 | 0.0740 |
| REINFORCE++-Baseline | 0.8711 | 0.1510 | 0.0969 |
| DAPO | 0.8734 | 0.1240 | 0.1063 |
| **RPG-FKL** | **0.8836** | 0.1490 | 0.1083 |
| **RPG-RKL** | 0.8672 | 0.1469 | **0.1240** |
| **RPG-UFKL** | 0.8703 | 0.1427 | 0.1177 |
| RPG-REINFORCE-FKL | 0.8727 | **0.1667** | 0.0875 |
| RPG-REINFORCE-URKL | 0.8531 | 0.1500 | 0.0938 |

### 消融实验

| 配置 | 关键发现 |
|------|---------|
| Clip (0.1,0.1) vs (0.2,0.28) | RPG-REINFORCE 在 Qwen-Math-7B 上使用大 clip 参数会崩溃，小 clip 更稳定 |
| AdamW vs Schedule-Free AdamW | Schedule-Free 改善高方差算法（GRPO、REINFORCE++）的稳定性 |
| 不同 KL 类型对比 | FKL 在 AMC23 最优，RKL 在 AIME25 最优，各有优势 |

### 关键发现

- RPG 变体在训练稳定性上显著优于 GRPO（后者呈现更大波动性）
- Forward KL 与 MLE 的联系表明其在稳定训练中的作用类似于 SFT 的正则化效果
- 完全可微分版本和 REINFORCE 风格版本性能互补，前者在 AMC23 更优，后者在 AIME24 更优
- 已充分预训练的模型（Qwen-Math-7B）需要更紧的 clip 参数来鼓励利用而非探索
- Schedule-Free 优化器通过内部参数平均提供更稳定的训练动态

## 亮点与洞察

- **系统性框架**：将 KL 正则化策略梯度统一到一个框架下，覆盖 Forward/Reverse × 归一化/非归一化 × 可微分/REINFORCE 风格 = 8 种变体，便于研究者理解和选择
- **GRPO 纠错**：指出 GRPO KL 项缺少重要性权重的理论缺陷，提供修正方案，对理解和改进现有 RLHF 方法具有重要价值
- **内存效率**：RPG 训练时只需一个模型在 GPU 上，比同时需要当前策略和参考策略的 GRPO/REINFORCE++ 更高效
- **$k_3$ 估计器的理论根基**：证明了 $k_3$ 估计器等价于非归一化 KL 散度，为其广泛使用提供了理论支撑

## 局限与展望

- 仅在数学推理任务上验证，未涉及代码生成、通用指令跟随等场景
- 仅使用 7B 模型实验，更大规模模型上的表现尚未验证
- 不同 KL 变体的最优选择依赖具体任务，缺乏明确的选择准则
- Dual-Clip 改变了原始 KL 正则化目标，引入了近似误差
- 训练超参数（如 $\beta$、clip 范围）的敏感性分析有限，不同模型可能需要不同配置

## 相关工作与启发

- **GRPO** (Shao et al., 2024)：使用组相对优势估计和 $k_3$ KL 惩罚，但 KL 项缺少重要性权重
- **REINFORCE++** (Hu, 2025)：引入 token 级 KL 惩罚和归一化，但 KL 项基于 $\pi_{\text{old}}^{\text{RL}}$ 而非 $\pi_\theta$
- **DAPO** (Yu et al., 2025)：提出 Clip-Higher、过采样、token 级损失等稳定训练技术
- **Dr. GRPO** (Liu et al., 2025)：发现并修正 GRPO 优势估计器中的偏差
- **VAPO** (Yuan et al., 2025)：长度自适应优势估计
- **DPO/SimPO**：直接偏好优化方向的代表工作，与策略梯度方法互补

**启发**：该工作提示我们在设计 RL 算法时需要仔细审视 KL 正则化的数学正确性，特别是在 off-policy 设置下。框架化的思维方式有助于发现和修正既有方法的理论缺陷。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 框架本身是系统性整理而非全新方法，但 GRPO 纠错和理论统一有价值
- 实验充分度: ⭐⭐⭐⭐⭐ — 8 种变体 × 2 种优化器 × 2 种模型，消融充分
- 写作质量: ⭐⭐⭐⭐ — 数学推导严谨清晰，但因变体众多显得冗长
- 价值: ⭐⭐⭐⭐ — 为 LLM RL 社区提供了重要的理论参考和实用方法

<!-- RELATED:START -->

## 相关论文

- [DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)
- [DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)
- [Nudging the Boundaries of LLM Reasoning](nudging_the_boundaries_of_llm_reasoning.md)
- [RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format](rain-merging_a_gradient-free_method_to_enhance_instruction_following_in_large_re.md)
- [RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following Through Model Merging](rain-merging_a_gradient-free_method_to_enhance_instruction_following_through_mod.md)

<!-- RELATED:END -->
