---
title: >-
  [论文解读] Understanding the Role of Training Data in Test-Time Scaling
description: >-
  [ICLR2026][LLM推理][测试时缩放] 从理论上分析训练数据属性如何影响 test-time scaling 的效果，证明 CoT 推理等价于伪牛顿法迭代，提出基于特征协方差最小特征值的任务难度度量，揭示"更多思考不一定更好"的 overthinking 现象机制，并给出多任务训练中最优任务选择策略——训练集应多样、相关且困难。
tags:
  - ICLR2026
  - LLM推理
  - 测试时缩放
  - chain-of-thought
  - 上下文学习
  - task hardness
  - 过度思考
  - training data selection
---

# Understanding the Role of Training Data in Test-Time Scaling

**会议**: ICLR2026  
**arXiv**: [2510.03605](https://arxiv.org/abs/2510.03605)  
**代码**: 无  
**领域**: llm_reasoning  
**关键词**: 测试时缩放, chain-of-thought, 上下文学习, task hardness, 过度思考, training data selection

## 一句话总结
从理论上分析训练数据属性如何影响 test-time scaling 的效果，证明 CoT 推理等价于伪牛顿法迭代，提出基于特征协方差最小特征值的任务难度度量，揭示"更多思考不一定更好"的 overthinking 现象机制，并给出多任务训练中最优任务选择策略——训练集应多样、相关且困难。

## 研究背景与动机

**领域现状**：Test-time scaling（如 OpenAI o1、DeepSeek R1）通过在推理时分配更多计算资源生成更长的 CoT 来提升推理能力，已在数学竞赛、编程等任务上取得显著成功。

**核心问题**：尽管实践效果显著，训练数据在什么条件下支持 test-time scaling 仍不清楚——具体地：
   - 增加 test-time 计算是否**总是**能提升下游推理表现？
   - 增加 test-time 计算能否降低训练时的计算需求？
   - 什么是"困难"训练样本？它们为何对 test-time scaling 有益？

**现有工作不足**：先前关于训练数据多样性和难度的研究大多是经验性的，缺乏严格的理论框架解释 test-time scaling 的机制。

**本文切入角度**：在 linear regression 的 in-context learning 框架下，从理论和实验两个维度回答上述三个问题。

## 方法详解

### 理论框架：ICL 权重预测 + 线性自注意力

**任务设定**：每个 prompt $P_\tau = (x_{\tau,1}, y_{\tau,1}, \ldots, x_{\tau,n}, y_{\tau,n})$，其中 $y_{\tau,i} = \langle w_\tau, x_{\tau,i} \rangle$，$x_{\tau,i} \sim \mathcal{N}(0, \Lambda)$，$w_\tau \sim \mathcal{N}(0, I_d)$。模型需从 prompt 中估计权重向量 $w_\tau$。

**模型**：单层线性自注意力（LSA），输入嵌入矩阵包含数据和权重估计位：

$$E_\tau = \begin{bmatrix} X_\tau & 0 \\ y_\tau & 0 \\ 0_{d \times n} & \hat{w}_0 \\ 0_{1 \times n} & 1 \end{bmatrix}$$

**训练损失**：最小化权重预测的均方误差：

$$L(\theta) = \frac{1}{2} \mathbb{E}\left[\left\| f_{\text{LSA}}(E_\tau;\theta)_{[:,-1]} - (0_d, 0, w_\tau, 1) \right\|^2\right]$$

**Theorem 3.1（梯度下降收敛到全局最优）**：在合适的初始化下，梯度下降以常数步长收敛到全局最优解 $V_* = -\Gamma^{-1}/c$，其中：

$$\Gamma := \left(1 + \frac{1}{n}\right)\Lambda + \frac{1}{n}\text{tr}(\Lambda) I_d$$

### Test-time CoT 等价于伪牛顿法

**Proposition 3.2**：在测试时执行 $k$ 步 CoT，模型的权重估计递推更新为：

$$w_{i+1} = w_i - \frac{1}{m} \Gamma^{-1} X_{\text{test}} X_{\text{test}}^\top (w_i - w_{\text{test}})$$

这等价于对二次损失 $\ell(w) = \frac{1}{2m}\|y_{\text{test}} - X_{\text{test}}^\top w\|^2$ 的**伪牛顿法**——用 $\Gamma^{-1}$ 近似 Hessian 的逆 $\Lambda^{-1}$。经 $k$ 步后：

$$w_{k+1} = \left(I - \left(I - \frac{1}{m}\Gamma^{-1} X_{\text{test}} X_{\text{test}}^\top\right)^k\right) w_{\text{test}}$$

### 任务难度度量

**Theorem 3.3** 给出直接 ICL（无 CoT）的估计误差上界：

$$\mathbb{E}\|\hat{w} - w_{\text{test}}\|^2 \leq \frac{d}{n^2}\left(1 + \frac{\text{tr}(\Lambda)}{\lambda_{\min}(\Lambda)}\right)^2 + \frac{d}{m}\left(1 + \frac{\text{tr}(\Lambda)}{\lambda_{\min}(\Lambda)}\right)$$

由此定义**任务难度**：

$$\text{Hard}(\Lambda) := \frac{\text{tr}(\Lambda)}{\lambda_{\min}(\Lambda)}$$

**直觉解读**：$\Lambda$ 的每个特征向量代表一种"技能"，特征值表示该技能的强度。容易任务依赖少数几种均衡的技能（特征值接近），困难任务依赖多种技能且分布长尾（存在很小的特征值）。

### Test-time Scaling Law

**Corollary 3.5**：$k$ 步 CoT 后的估计误差：

$$\mathbb{E}\|w_{k+1} - w_{\text{test}}\|^2 \leq d \left(1 + \frac{n}{1 + \text{Hard}(\Lambda)}\right)^{-2k} (1 + o(1))$$

**关键推论**：
- 固定目标误差 $\varepsilon$——增大 $k$（更多 test-time 计算）可以减少训练 prompt 长度 $n$
- 越困难的任务需要越长的 CoT
- 计算复杂度为 $O(kd^2)$

### Overthinking 机制

**Remark 4.1**：test-time thinking 的效果由 $\text{tr}((I - \Gamma^{-1/2}\Sigma\Gamma^{-1/2})^{2k})$ 控制。当目标任务的某些技能方向（$\Sigma$ 的特征向量）在训练数据中**未被充分覆盖**时（$\Gamma$ 在该方向很弱），该项随 $k$ 增大而增大——更多思考反而**损害**性能。

### 最优任务选择

**Proposition 4.3**：在多任务训练中，最优任务选择概率 $\{\pi_\ell\}$ 应满足：至少 50% 的采样概率分配给"困难"任务（$\sigma_{\min}(\Lambda_\ell)$ 小的任务）。

最优选择可转化为可高效求解的二次优化问题：

$$\min_{\{\pi_\ell\}} \left\| I - \Sigma^{-1} \sum_{\ell} \Lambda_\ell \pi_\ell \right\|_F^2 \quad \text{s.t.} \sum \pi_\ell = 1, \pi_\ell \geq 0$$

核心结论：训练集应同时具备**多样性**（覆盖目标任务所有方向）、**相关性**（与目标任务特征分布对齐）和**难度**（包含长尾技能分布的困难样本）。

## 实验关键数据

### LSA 模型验证

| 设定 | 结论 |
|------|------|
| 训练 prompt 长度 $n=10,20,30$ | 增大 $k$ 可弥补较短的训练上下文，$n=10$ 在 $k=20$ 时达到 $n=30$ 直接预测的误差水平 |
| 训练协方差倾斜 ($\lambda_i \propto 1/i$) | 训练/测试分布不匹配时，$k$ 增大后测试误差先降后升——overthinking 出现 |
| overthinking 时大 $n$ 反而更差 | 与非 overthinking 情况相反——更长训练上下文在倾斜分布下"学得更偏" |

### GPT-2（9.5M 参数）验证

| 实验 | 结果 |
|------|------|
| 训练 $n=20,30,40$，变化 $k$ | 与 LSA 趋势一致：更长 CoT 允许用更短训练上下文达到同等性能 |
| 倾斜协方差 + 全等测试 | GPT-2 同样出现 overthinking：大约 $k>10$ 后误差上升 |

### 任务选择实验

| 任务类型 | $(\alpha, B)$ | 平均选择概率 |
|----------|--------------|-------------|
| Easy-Short | (0.2, 20) | 最低 |
| Hard-Short | (0.8, 20) | 中等 |
| Easy-Long | (0.2, 100) | 中等偏低 |
| **Hard-Long** | **(0.8, 100)** | **最高** |

### 真实推理基准（Qwen 2.5-7B）

| 模型 | CoT 长度 [0, 1k) | CoT 长度 [1k, 2k] |
|------|------------------|--------------------|
| Qwen-Base | 30.39% | 27.2% |
| Qwen-GCD（训练对齐） | **75%**（+44.6） | **38.4%**（+11.2） |
| Qwen-Poly（训练不对齐） | 29%（-1.4） | 20.83%（**-6.4**） |

训练数据对齐时更多 thinking 有帮助，不对齐时更多 thinking 有害——完美验证了理论预测。

## 亮点与洞察
- **CoT = 伪牛顿法**：将 test-time CoT 与优化算法建立了精确的数学对应，提供了理解推理过程的新视角
- **Overthinking 的理论解释**：首次从理论上解释了为什么更多推理有时会损害性能——训练数据未覆盖的技能方向在迭代中被放大
- **任务难度的特征谱定义**：$\text{Hard}(\Lambda) = \text{tr}(\Lambda)/\lambda_{\min}(\Lambda)$ 是一个简洁而有洞察力的度量
- **训练-测试计算的可替代性**：严格证明了 test-time compute 可以补偿训练时 context length 的不足——为实践中的资源分配提供了理论指导

## 局限与展望
- **理论局限于线性模型**：主要分析限于 linear regression + LSA，对非线性任务和深层 Transformer 的推广需进一步工作
- **GPT-2 实验仍在合成数据上**：真实推理基准实验（Qwen）只涉及两个特定任务（GCD 和多项式根），覆盖面有限
- **任务难度定义依赖协方差谱**：实际 NLP 任务中"技能"和"特征分布"难以直接测量，理论到实践的 gap 明显
- **未考虑 RL 训练场景**：当前分析基于 SFT/ICL，o1/R1 类模型的 RL 训练范式下理论是否成立未知

## 相关工作与启发
- **vs Snell et al. (2024) / Muennighoff et al. (2025)**：先前工作是经验性地展示 test-time scaling 效果，本文提供了理论基础
- **vs Su et al. (2025) (overthinking)**：先前经验观察到 LLM 在简单问题上 overthink，本文给出了数学解释
- **vs Huang et al. (2025a)**：先前分析限于各向同性特征（$\Lambda = I$）且在训练时用 CoT，本文扩展到一般协方差并分析纯 test-time CoT
- **启发**：在实际系统设计中，应根据训练数据覆盖度动态调整 test-time 计算量——对训练数据充分覆盖的任务加大 compute，对覆盖不足的任务适度限制

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次从理论上系统地分析训练数据与 test-time scaling 的关系，overthinking 和任务选择的理论都是新贡献
- 实验充分度: ⭐⭐⭐⭐ LSA/GPT-2 合成实验充分验证理论，Qwen 真实基准实验为亮点，但真实任务覆盖面可更广
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨清晰，直觉解释到位，从单任务到多任务层层递进
- 价值: ⭐⭐⭐⭐⭐ 对理解和改进 test-time scaling 有重要指导意义，任务选择策略可直接用于 RL reasoning 训练

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory](../../ACL2025/llm_reasoning/rethinking_the_role_of_prompting_strategies_in_llm_test-time_scaling_a_perspecti.md)
- [\[ICLR 2026\] Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs](plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)
- [\[CVPR 2026\] Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models](../../CVPR2026/llm_reasoning/understanding_the_role_of_hallucination_in_reinforcement_post-training_of_multim.md)
- [\[ICLR 2026\] ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](atts_asynchronous_test-time_scaling_via_conformal_prediction.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](../../ACL2026/llm_reasoning/parallel_test-time_scaling_for_latent_reasoning_models.md)

<!-- RELATED:END -->
