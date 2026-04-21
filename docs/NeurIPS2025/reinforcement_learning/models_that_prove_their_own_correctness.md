---
title: >-
  [论文解读] Models That Prove Their Own Correctness
description: >-
  [NeurIPS 2025][自证模型] 本文提出 Self-Proving Models 框架，让模型通过交互式证明系统向验证算法证明其输出的正确性，并设计了 Transcript Learning (TL) 和 Reinforcement Learning from Verifier Feedback (RLVF) 两种学习方法，在 GCD 计算任务上实验验证 Annotated TL 可达 96% 的 Verifiability。
tags:
  - NeurIPS 2025
  - 自证模型
  - 交互式证明
  - 验证器反馈强化学习
  - 形式验证
  - LLM可靠性
---

# Models That Prove Their Own Correctness

**会议**: NeurIPS 2025  
**arXiv**: [2405.15722](https://arxiv.org/abs/2405.15722)  
**代码**: [https://github.com/orrp/self-proving-models](https://github.com/orrp/self-proving-models)  
**领域**: 强化学习  
**关键词**: 自证模型, 交互式证明, 验证器反馈强化学习, 形式验证, LLM可靠性

## 一句话总结
本文提出 Self-Proving Models 框架，让模型通过交互式证明系统向验证算法证明其输出的正确性，并设计了 Transcript Learning (TL) 和 Reinforcement Learning from Verifier Feedback (RLVF) 两种学习方法，在 GCD 计算任务上实验验证 Annotated TL 可达 96% 的 Verifiability。

## 研究背景与动机
模型的准确率通常以输入分布上的平均性能来衡量，这意味着对于任何**特定输入**，我们无法保证模型输出的正确性。一个学生用 LLM 求解代数问题，即使 LLM 在某 benchmark 上达到 90% 准确率，他也无法确定眼前这道题的答案是否正确。这是机器学习中"平均保证 vs 最坏情况保证"之间的根本矛盾。

要求模型给出"自然语言解释"不够可靠——模型可能用看似合理的推理说服用户接受错误答案；即使答案正确，模型也可能无法生成令人信服的证明。真正需要的是**形式化证明**，让验证算法（而非人类判断）来确认正确性。

核心 idea 源自交互式证明系统（Interactive Proofs）理论：让模型扮演"证明者"角色，向一个手动定义的高效验证算法证明其每个输出的正确性。验证器的**完备性**保证正确输出总能被证明，**可靠性**保证错误输出不可能被接受（与证明者的计算能力无关）。这样就把 ML 的平均保证提升为了逐输入的最坏情况保证。

## 方法详解

### 整体框架
Self-Proving Model $P_\theta$ 的工作流程：给定输入 $x$，模型先生成输出 $y \sim P_\theta(x)$，然后与验证器 $V$ 进行 $R$ 轮交互。每轮 $V$ 发送查询 $q_i$，$P_\theta$ 回复答案 $a_i$。最终 $V$ 决定接受或拒绝。训练目标是最大化 Verifiability：

$$\text{ver}_{V,\mu}(\theta) := \Pr_{x \sim \mu, y \sim P_\theta(x)} [\langle V, P_\theta \rangle(x,y) \text{ accepts}] \geq \beta$$

关键思想：验证器 $V$ 是**手动定义**的（非学习得到），因此其可靠性有形式保证。Verifiability 蕴含 Correctness：如果 $P_\theta$ 是 $\beta$-Verifiable 且 $V$ 的 soundness error 为 $s$，则 $P_\theta$ 是 $(\beta - s)$-correct。

### 关键设计
1. **Transcript Learning (TL)**:

    - 功能：基于接受转录本（accepting transcripts）的监督学习
    - 核心思路：将模型不仅训练为 $x \mapsto y^*$（正确输出），而是 $x \mapsto y^* \pi^*$（正确输出 + 接受转录本），使模型自回归地学习生成能被验证器接受的完整交互序列
    - 梯度估计：TL 估计的是下界函数 $A(\theta) \leq \text{ver}_V(\theta)$ 的梯度，其中 $A(\theta) := \Pr[\pi = \pi^*]$ 是当前模型生成的转录本与诚实转录本的一致概率
    - 更新规则：$\theta_{i+1} := \theta_i + \lambda \cdot \prod_{s} \alpha_s(\theta_i) \cdot \sum_s \vec{d}_s(\theta_i)$，其中 $\alpha_s$ 是 token 级别的选择概率，$\vec{d}_s$ 是对应的 log-probability 梯度
    - 设计动机：类似于 chain-of-thought 的 process supervision，提供中间步骤的监督信号加速学习

2. **Reinforcement Learning from Verifier Feedback (RLVF)**:

    - 功能：无需接受转录本，仅依赖验证器的接受/拒绝反馈
    - 核心思路：模型自行生成转录本 $\pi \sim P_\theta$，由验证器判定接受与否，仅在被接受时更新参数
    - 梯度关键公式：$\nabla_\theta \text{ver}(\theta) = \mathbb{E}[\text{Acc}_V(\cdot) \cdot \sum_s \vec{d}_s(\theta)]$，其中 $\text{Acc}_V$ 是 0-1 指示变量
    - 特点：属于 on-policy 算法，更新步骤比 TL 更简单（不需要追踪 $\alpha_s$），但需要初始 Verifiability > 0 才能采到接受样本
    - 设计动机：类比 RLHF，但用算法验证器替代人类反馈，且验证器可被高效模拟、无需额外标注成本

3. **Annotated Transcript Learning (ATL)**:

    - 功能：在 TL 基础上添加"注释"——证明过程的中间计算步骤
    - 核心思路：将证明 $\pi$ 扩展为 $\tilde{\pi} = A(x, \pi)$，包含前 $T$ 步的中间推导过程，推理时通过 extractor $E$ 去掉注释只发送实际证明
    - 设计动机：注释相当于 chain-of-thought，显著降低学习难度。实验中 ATL 的 Verifiability 从 TL 的 60.3% 跃升至 96%

### 损失函数 / 训练策略
- TL：最大化接受转录本的 log-likelihood，实质是梯度上升优化 $A(\theta)$
- RLVF：REINFORCE 风格的策略梯度，奖励为验证器的二值决策
- 推荐组合方案：先用 TL 获得 $\delta > 0$ 的基础 Self-Proving 模型，再用 RLVF 放大 Verifiability
- 收敛保证：在凸性和 Lipschitz 条件下，TL 在 $O(C^2 B_\text{Norm}^2 B_\text{Lip}^2 / \varepsilon^2)$ 样本后输出 $(1-\varepsilon)$-Self-Proving 模型

## 实验关键数据

### 主实验

| 学习方法 | Correctness | Verifiability | 说明 |
|----------|-------------|---------------|------|
| GPT (baseline) | 99.8% | — | 能计算 GCD 但不能证明 |
| GPT+TL | 98.8% | 60.3% | 100K iterations |
| GPT+TL+RLVF | 98.9% | 78.3% | 4M iterations RLVF |
| GPT+ATL | 98.6% | **96.0%** | 100K iterations, 注释显著提升 |

### 消融实验

| 配置（注释步数 T） | Verifiability | 说明 |
|-------------------|---------------|------|
| T=0 (无注释, TL) | ~60% | 基线 |
| T=1 | ~82% | 注释开始生效 |
| T=3 | ~91% | 持续提升 |
| T=5 | ~96% | 接近饱和 |
| T=7 | ~96% | 进一步增加注释收益递减 |

### 关键发现
- 模型在 Correctness 接近 100% 的情况下，Verifiability 需要专门的训练方法才能提高——"会算"不等于"会证明"
- RLVF 的后续实现（如 RLVR，加入 KL 正则化）已在全尺度 LLM 上获得广泛成功，验证了本文理论框架的实用价值
- 注释（Chain-of-Thought）对 Verifiability 有决定性影响，且模型能泛化到超出注释长度的情况

## 亮点与洞察
- **理论与实践的完美桥接**：将交互式证明系统这一理论计算机科学经典概念应用于 ML 模型可信度问题，定义了 Verifiability 概念并给出了可操作的训练算法。RLVF 已经成为 LLM 后训练的核心技术。
- **从平均保证到逐输入保证的突破**：Soundness 对所有输入成立（worst-case），这让用户可以逐个输入地信任模型——通过运行验证器，而非依赖 benchmark 统计数据。

## 局限与展望
- 实验仅在小规模 GCD 任务上验证（6.3M 参数 GPT），尚未在大规模 LLM 和复杂推理任务上全面测试
- RLVF 的收敛理论尚未完成（依赖于策略梯度收敛的开放问题）
- 需要预定义一个高效的验证器——对于很多实际任务，设计这样的验证器本身就很困难
- Soundness error $s$ 只对 $\text{ver} > s$ 时才有意义，极低 Verifiability 场景下可靠性保证有限
- TL 对 transcript generator 质量敏感，低质量的转录本会导致学习效率大幅下降
- 自然语言推理任务中如何形式化定义验证器和证明是开放问题

## 相关工作与启发
- **vs Prover-Verifier Games (PVGs)**: PVGs 使用学习的验证器，本文使用手动定义的验证器——后者有形式化的 soundness 保证，不受对抗攻击影响
- **vs RLHF**: RLVF 可视为 RLHF 的"理想化版本"——用确定性的数学验证器替代噪声大的人类偏好，奖励信号完美无歧义
- **vs Chain-of-Thought**: TL 可以被看作由验证器诱导的 CoT 训练，但与普通 CoT 不同，这里的中间步骤有形式化的正确性含义
- **vs IP-PAC**: IP-PAC 让学习器证明"模型在分布上大致正确"，本文让模型在逐个输入的基础上证明正确性，保证更强

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 开创性地将交互式证明引入 ML 训练框架，RLVF 已被大规模采用
- 实验充分度: ⭐⭐⭐ 实验仅在 GCD 这一个小任务上验证，规模有限
- 写作质量: ⭐⭐⭐⭐⭐ 数学严谨，概念清晰，理论发展层次分明
- 价值: ⭐⭐⭐⭐⭐ RLVF 已被广泛采用为 LLM 后训练的标准方法（如 RLVR），影响深远

<!-- RELATED:START -->

## 相关论文

- [Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds](foundation_models_as_world_models_a_foundational_study_in_text-based_gridworlds.md)
- [Checklists Are Better Than Reward Models For Aligning Language Models](checklists_are_better_than_reward_models_for_aligning_langua.md)
- [Training Language Models to Reason Efficiently](training_language_models_to_reason_efficiently.md)
- [Self-Improving Embodied Foundation Models](self-improving_embodied_foundation_models.md)
- [Complexity Scaling Laws for Neural Models using Combinatorial Optimization](complexity_scaling_laws_for_neural_models_using_combinatorial_optimization.md)

<!-- RELATED:END -->
