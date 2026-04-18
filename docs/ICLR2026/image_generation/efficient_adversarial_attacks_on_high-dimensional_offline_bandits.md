---
title: >-
  [论文解读] Efficient Adversarial Attacks on High-dimensional Offline Bandits
description: >-
  [ICLR 2026][图像生成][离线多臂老虎机] 揭示了离线多臂老虎机（MAB）评估框架的安全漏洞：攻击者只需对公开的奖励模型权重进行极小的不可感知扰动，就能完全劫持 bandit 的决策行为，且所需扰动范数随输入维度增加而降低（$\widetilde{\mathcal{O}}(d^{-1/2})$），使基于图像的生成模型评估特别脆弱。
tags:
  - ICLR 2026
  - 图像生成
  - 离线多臂老虎机
  - 对抗攻击
  - 奖励模型
  - 高维数据
  - 生成模型评估
---

# Efficient Adversarial Attacks on High-dimensional Offline Bandits

**会议**: ICLR 2026  
**arXiv**: [2602.01658](https://arxiv.org/abs/2602.01658)  
**代码**: [GitHub](https://github.com/hadi-hosseini/adversarial-attacks-offline-bandits) (有)  
**领域**: 对抗攻击 / 多臂老虎机  
**关键词**: 离线多臂老虎机, 对抗攻击, 奖励模型, 高维数据, 生成模型评估

## 一句话总结

揭示了离线多臂老虎机（MAB）评估框架的安全漏洞：攻击者只需对公开的奖励模型权重进行极小的不可感知扰动，就能完全劫持 bandit 的决策行为，且所需扰动范数随输入维度增加而降低（$\widetilde{\mathcal{O}}(d^{-1/2})$），使基于图像的生成模型评估特别脆弱。

## 研究背景与动机

多臂老虎机（MAB）算法近年来被广泛用于评估生成模型（扩散模型、LLM 等），通过 UCB 等策略高效识别最优模型，替代昂贵的穷举对比。这些评估通常依赖部署在 Hugging Face 等平台上的**公开权重奖励模型**（如 CLIP、BLIP、美学评分器）。

**核心安全隐患**：离线 bandit 评估（用固定 logged 数据替代在线评估）引入了两个被忽视的风险：

**容易被对抗操纵**：攻击者可以偏倚模型选择过程

**容易对奖励模型过拟合**：同一数据集上反复调整评估指标

**研究空白**：以往对抗攻击研究聚焦于训练过程中篡改奖励值，从未考虑过**在训练前扰动奖励模型本身**这一更现实的威胁模型。

**直觉解释**：高维空间中估计均值本身就不稳定（维度诅咒），而 bandit 本质上在反复估计各 arm 的高维均值。当每 arm 的观测数 $T/K \ll d$ 时，估计天然不可靠。此时只需微小地操纵奖励函数（利用其高维参数空间的自由度），就能轻松误导 bandit。

## 方法详解

### 整体框架

攻击者 $\mathscr{A}$ 可以访问离线 logged 数据集 $\mathcal{D}_1, \ldots, \mathcal{D}_K$，在 bandit 训练前对奖励模型 $r(\cdot)$ 施加微小扰动 $\boldsymbol{\delta}$，目标是使 bandit 无法识别最优 arm。

### 关键设计

1. **三种攻击策略**：

    - **Full Trajectory Attack**：强制 bandit 遵循完全预设的目标轨迹 $\widetilde{A}_t$，需 $(T-K)(K-1)$ 个约束
    - **Trajectory-Free Attack**：仅阻止最优 arm $i^*$ 被选中，不指定具体选哪个，约束数降为 $T-K$
    - **Online Score-Aware (OSA) Attack**：在线逐步添加约束——仅当最优 arm 即将被选中时才加约束，实际约束数仅为 $\mathcal{O}(\log T)$，大幅降低计算成本
   
   对于线性奖励模型 $r(\mathbf{X}) = \mathbf{w}^\top \mathbf{X}$，所有攻击都可归结为凸二次规划（QP）：
    $\boldsymbol{\delta}^* = \arg\min_{\boldsymbol{\delta}} \|\boldsymbol{\delta}\|_2^2 \quad \text{s.t.} \quad \boldsymbol{\delta}^\top \mathbf{T}_{i,t} > R_{i,t}, \forall (i,t) \in \mathcal{I}$

2. **从线性到神经网络的推广**：

    - 利用 Neural Tangent Kernel (NTK) 理论，对于足够宽的随机初始化网络：
    $\text{NN}_{\boldsymbol{\theta}+\boldsymbol{\delta}}(\mathbf{X}) \approx \text{NN}_{\boldsymbol{\theta}}(\mathbf{X}) + \nabla_{\boldsymbol{\theta}}\text{NN}_{\boldsymbol{\theta}}(\mathbf{X})^\top \boldsymbol{\delta}$
    - 即过参数化网络在参数空间近似线性，攻击退化为线性情形
    - 隐藏层宽度超过 750 时，攻击成功率达到 100%

3. **理论保证**：

    - **可行性定理（Theorem 3.3）**：当 $d > (T-K)(K-1)$ 且数据分布非退化时，攻击以概率 1 可行
    - **攻击范数定理（Theorem 3.4）**：在 $d \geq KT$ 的高维情形下，全轨迹攻击所需扰动范数满足 $\|\boldsymbol{\delta}^*\|_2 \leq \mathcal{O}\left(\sqrt{\frac{T^3 \log T \cdot \log d}{Kd}}\right)$，即扰动随维度增大而缩小

### 损失函数 / 训练策略

攻击优化目标为凸二次规划，求解复杂度为 $\widetilde{\mathcal{O}}(|\mathcal{I}|^3 + d|\mathcal{I}|^2)$。OSA 方法通过减少约束数 $|\mathcal{I}|$ 至 $\mathcal{O}(\log T)$ 实现高效攻击。

防御机制：在运行 bandit 前随机打乱部分 logged 数据，打乱 $T/2$ 数据即可显著降低攻击成功率。

## 实验关键数据

### 主实验

在合成数据和真实数据上验证攻击效果：

| 实验设置 | 攻击方法 | ASR | 备注 |
|---------|---------|-----|------|
| 线性模型，合成数据 | Full Trajectory | 100% | 高计算成本 |
| 线性模型，合成数据 | Trajectory-Free | 100% | 中等成本 |
| 线性模型，合成数据 | OSA | 100% | **成本降低数量级** |
| 非线性模型，合成数据 | OSA | 100% | 宽度>750 |
| 真实模型（HF 美学评分器） | OSA | ~80% 概率 ASR∈[90-100%] | 5 个生成模型 |
| 真实模型（HF Image Reward） | OSA | ~80% 概率 ASR∈[90-100%] | 30 个随机 prompt |

| 其他 Bandit 算法 | UCB | ETC | ε-greedy |
|---|---|---|---|
| ASR | 100% | 100% | 100% |

### 消融实验

| 实验 | K | T | d | ASR | 扰动范数趋势 |
|------|---|---|---|-----|-------------|
| 维度增大 | 3 | 100 | 100→5000 | 100% | ℓ₂ 和 ℓ∞ **持续下降** |
| 随机噪声对比 | 3 | 100 | 1000 | ~25% | 无论范数大小 |
| 防御（打乱T/2） | 3 | 100 | 100 | 显著降低 | - |
| 攻击 Fast-Slow 鲁棒算法 | 3,5 | 1000 | 1000 | 100% | ~0.3 |
| 攻击 ε-contamination | 3,5 | 100-1000 | 1000 | 100% | ~0.2 |

### 关键发现

1. **随机扰动无效**：即使范数很大，随机噪声 ASR 始终约 25%（等于随机选择），证明需要定向优化
2. **维度越高越脆弱**：ℓ₂ 和 ℓ∞ 范数都随维度增加显著下降
3. **OSA 方法极其高效**：约束数从 $\mathcal{O}(TK)$ 降至 $\mathcal{O}(\log T)$，同时保持 100% ASR
4. **鲁棒 bandit 算法同样被攻破**：Fast-Slow 和 ε-contamination 都无法防御

## 亮点与洞察

- 提出了一个全新的、高度现实的威胁模型：攻击奖励模型权重而非训练数据
- 理论与实验完美契合：高维效应的理论预测被实验精确验证
- 对实践有重要警示：在 Hugging Face 上公开奖励模型权重，意味着评估可被操纵
- OSA heuristic 以 $\mathcal{O}(\log T)$ 约束实现近最优攻击，工程价值高
- 防御方案（打乱数据）虽简单但方向正确，提示了未来研究的可能路径

## 局限与展望

- 防御机制仍不完善，完全防御仍是开放问题
- NTK 线性近似对实际的深层窄网络可能不够准确
- 真实实验仅冻结大部分权重、只扰动小部分，完全扰动的可行性未探讨
- 未考虑多个奖励模型集成的防御策略
- 对 contextual bandits 或更复杂的 bandit 变体的攻击未涉及

## 相关工作与启发

- **Jun et al. (2018)**：在线训练时篡改奖励值，本文转向训练前攻击权重
- **Garcelon et al. (2020)**：线性 contextual bandits 的对抗攻击
- **NTK 理论（Jacot et al. 2018）**：为非线性攻击提供线性近似的理论基础
- 启发：任何依赖公开权重评估器的自动评测系统都可能存在类似漏洞

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 全新的威胁模型和攻击范式，高维效应的理论发现令人惊讶
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据，多种 bandit 算法，但真实场景验证可更丰富
- 写作质量: ⭐⭐⭐⭐ 理论部分严谨，但符号较重，非 bandit 领域读者需要较多背景知识
- 价值: ⭐⭐⭐⭐ 对 AI 安全和模型评估领域有重要警示意义，但直接应用场景较窄
