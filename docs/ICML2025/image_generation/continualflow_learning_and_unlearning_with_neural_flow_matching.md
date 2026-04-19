---
title: >-
  [论文解读] ContinualFlow: Learning and Unlearning with Neural Flow Matching
description: >-
  [ICML 2025][图像生成][Flow Matching] 提出 ContinualFlow，一种基于 Flow Matching 的生成模型定向遗忘框架，通过能量函数重加权软性减去数据分布中不需要的区域，无需重新训练或直接访问待遗忘样本即可实现高效遗忘。
tags:
  - ICML 2025
  - 图像生成
  - Flow Matching
  - 机器遗忘
  - 能量函数
  - 最优传输
  - 生成模型
---

# ContinualFlow: Learning and Unlearning with Neural Flow Matching

**会议**: ICML 2025  
**arXiv**: [2506.18747](https://arxiv.org/abs/2506.18747)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: Flow Matching, 机器遗忘, 能量函数, 最优传输, 生成模型

## 一句话总结

提出 ContinualFlow，一种基于 Flow Matching 的生成模型定向遗忘框架，通过能量函数重加权软性减去数据分布中不需要的区域，无需重新训练或直接访问待遗忘样本即可实现高效遗忘。

## 研究背景与动机

机器遗忘（Machine Unlearning）已成为生成模型中的核心问题，涉及法律合规（如 GDPR）、伦理和模型部署等多个层面。与判别模型不同，生成模型学习的是从潜在分布到数据分布的复杂映射，内部表征高度纠缠，很难精确隔离特定输入数据的影响。

现有方法主要分为两类：

**输出抑制**（Output Suppression）：在推理时通过解码引导或过滤器阻止生成敏感内容，但不修改模型内部知识，容易被对抗性 prompt 绕过

**模型修补**（Model Patching）：通过微调或定向更新修改模型参数，虽更持久，但可能损害无关能力，存在遗忘有效性和泛化保持之间的权衡

这两类方法都有明显局限：输出抑制不触及底层知识，模型修补缺乏理论保证且可能引发"附带损害"。本文另辟蹊径，利用 Flow Matching 的几何（轨迹）视角，将遗忘问题重新定义为**分布传输问题**，通过能量函数引导生成轨迹远离不需要的区域。

## 方法详解

### 整体框架

ContinualFlow 的核心思想是将生成模型的遗忘视为对数据分布的"软质量减法"（Soft Mass Subtraction）。框架分为两个场景：

**场景一：完全访问遗忘集**  
当直接可以访问遗忘集 $\mathcal{D}_{\text{forget}}$ 时，利用 OT-FM（Optimal Transport Flow Matching）直接建模从原始模型 $G_\theta$ 生成的样本到保留集 $\mathcal{D}_{\text{retain}}$ 的传输。这消除了对预定义先验的依赖，简化训练。

**场景二：无法访问遗忘集（主要贡献）**  
在更实际的场景中，无法直接获得 $\mathcal{D}_{\text{forget}}$，而是依赖代理函数（如分类器或评分模型）来检测不需要的内容。将分类器输出视为未归一化的能量函数 $F(x) \propto -\log q_f(x)$，实现无需样本的生成轨迹更新。

### 关键设计

#### 1. 能量重加权的软质量减法

设 $q_0(x)$ 为已知的可采样源分布，$q_f(x)$ 为未知的遗忘分布。通过能量函数 $F(x)$ 对 $q_0(x)$ 进行密度调制：

$$\tilde{R}(x) \propto q_0(x) \cdot \sigma(-\lambda F(x))$$

其中 $\sigma(z) = \frac{1}{1+e^{-z}}$ 是 sigmoid 函数，$\lambda > 0$ 控制抑制灵敏度。归一化后的目标分布：

$$\tilde{q}_1(x) = \frac{1}{Z} \tilde{R}(x), \quad Z = \int q_0(x) \sigma(-\lambda F(x)) dx$$

**设计动机**：sigmoid 函数提供平滑、可微的密度重加权，避免了硬阈值或样本排除的不连续性。$\lambda$ 控制抑制强度——$\lambda$ 越大，高能量（与遗忘分布相关）区域的权重越低，遗忘越彻底。

#### 2. 能量重加权 Flow Matching 目标（ERFM）

在标准 CFM 损失基础上引入能量权重，定义 ERFM 损失：

$$\mathcal{L}_{\text{ERFM}}(\theta) = \mathbb{E}_{x_0, x_1 \sim q_0, t \sim \mathcal{U}[0,1], x \sim p_t(x|x_0,x_1)} \left[ \sigma(-\lambda F(x_1)) \cdot \|v_\theta(t,x) - u_t(x|x_0,x_1)\|^2 \right]$$

**核心定理（Theorem 4.1）**：ERFM 损失的梯度等价于标准 CFM 向软质量减法目标 $\tilde{q}_1$ 训练的梯度（差一个正常数）：

$$\nabla_\theta \mathcal{L}_{\text{ERFM}}(\theta) = C \cdot \nabla_\theta \mathcal{L}_{\text{CFM}}^{q_0 \to \tilde{q}_1}(\theta), \quad C > 0$$

证明关键在于：$\frac{\tilde{q}_1(x_1)}{q_0(x_1)} \propto \sigma(-\lambda F(x_1))$，即能量权重恰好等于从 $q_0$ 到 $\tilde{q}_1$ 的重要性采样权重。

#### 3. 分类器作为能量函数代理

论文进一步证明（Proposition B.1），贝叶斯最优二分类器 $C(x)$ 可以自然地转化为能量函数：

$$F(x) = -\log\left(\frac{C(x)}{1-C(x)}\right)$$

此时 $\sigma(-\lambda F(x)) = \frac{(1-C(x))^\lambda}{(1-C(x))^\lambda + C(x)^\lambda}$，分类器越确信样本属于遗忘类，该权重越低。

#### 4. 能量函数的可逆性

ContinualFlow 的独特性质在于能量函数的**可组合与可逆**特性：通过反转能量函数的符号，可以恢复被遗忘的内容而无需直接访问其样本。例如在 MNIST 实验中，先用 $F(x)$ 抑制奇数数字，再用 $-F(x)$ 可以恢复奇数数字生成。

### 损失函数 / 训练策略

**训练算法（Algorithm 1）**：
1. 从 $q_0$ 采样 $\{x_0^{(j)}\}$ 和 $\{x_1^{(j)}\}$
2. 均匀采样时间 $t^{(j)} \sim \mathcal{U}(0,1)$
3. 计算插值 $x_t^{(j)} = (1-t^{(j)})x_0^{(j)} + t^{(j)}x_1^{(j)}$
4. 计算能量权重 $w^{(j)} = \sigma(-\lambda F(x_1^{(j)}))$
5. 计算归一化加权损失：$\mathcal{L} = \frac{\sum_j w^{(j)} \|v_\theta(x_t^{(j)}, t^{(j)}) - (x_1^{(j)} - x_0^{(j)})\|^2}{\sum_j w^{(j)}}$

训练策略上，通过 mini-batch OT 近似最优传输计划，实现高效训练。推理时使用 10 步 flow 积分生成样本。

## 实验关键数据

### 主实验

论文在 2D 合成数据（Circles、Moons、Gaussians、Checkerboard）、MNIST 和 CIFAR-10 上进行评估。

| 数据集 | 方法 | MMD ↓ | Accuracy ↑ | Forget Rate ↓ | Leakage ↓ | 训练时间(s) |
|--------|------|-------|-----------|---------------|-----------|-------------|
| MNIST | Retrain (GT) | 0.0004 | 0.9861 | 0.0050 | 0.0108 | 300.00 |
| MNIST | Fine-tuning | 0.0039 | 0.9551 | 0.0143 | 0.0214 | 92.86 |
| MNIST | **CFlow (本文)** | **0.0020** | **0.9673** | **0.0005** | **0.0015** | **158.74** |
| CIFAR-10 | Retrain (GT) | 0.0056 | 0.8920 | 0.1127 | 0.1546 | 802.37 |
| CIFAR-10 | Fine-tuning | 0.0077 | 0.9005 | 0.2157 | 0.2401 | 252.89 |
| CIFAR-10 | **CFlow (本文)** | **0.0064** | **0.8847** | **0.1704** | **0.1748** | **427.15** |

在 MNIST 上，ContinualFlow 的 Forget Rate (0.0005) 和 Leakage (0.0015) 远优于 Fine-tuning (0.0143/0.0214)，甚至优于 Retrain (0.0050/0.0108)。

### 消融实验

| 配置 ($\lambda$ 值) | 效果 | 说明 |
|---------------------|------|------|
| $\lambda = 0.5$ | 轻微抑制 | 遗忘集仍有较多残留生成 |
| $\lambda = 2$ | 中等抑制 | 遗忘类明显减少但未完全消除 |
| $\lambda = 5$ | 强力抑制 | 遗忘类接近消失 |
| $\lambda = 1000$ | 近乎完全抑制 | 几乎只生成保留集内容 |
| 反转能量 ($-F$) | 恢复遗忘内容 | 验证能量函数可逆性 |

### 关键发现

1. **遗忘效果超越重训练**：在 MNIST 上 CFlow 的 Forget Rate/Leakage 甚至低于从头重训练的基线，说明能量引导能更精准地定向抑制
2. **2D 实验直观验证**：在 Checkerboard 数据集上，CFlow 的 MMD (0.0063) 反而优于 Retrain (0.0136)，表明 OT-FM 可能学到更好的保留分布近似
3. **能量可逆性**：通过反转能量函数可恢复被遗忘的类别，对隐私可控的生成建模有重要意义
4. **分类器可作为能量代理**：理论证明贝叶斯最优分类器的 logit 输出可直接作为能量函数，使得方法易于实际部署

## 亮点与洞察

1. **理论扎实**：ERFM 损失等价性定理（Theorem 4.1）提供了将能量重加权与 Flow Matching 统一的严格理论基础，而非 heuristic 设计
2. **无需遗忘样本**：最大亮点在于不需要访问待遗忘数据本身，只需一个能量代理（如分类器），极大提升了实用性
3. **可组合与可逆**：能量函数的模块化设计支持组合（多个遗忘目标叠加）和可逆（恢复被遗忘内容），为持续遗忘提供灵活机制
4. **轨迹级控制**：不同于输出抑制或模型编辑，本方法在训练阶段直接调制生成轨迹，提供更根本的分布层面遗忘

## 局限与展望

1. **能量函数质量依赖**：遗忘效果强烈依赖能量函数与真实遗忘分布的对齐程度；如果分类器/评分模型不够准确，遗忘可能不完整或过度
2. **仅验证了简单场景**：实验局限于 2D 合成数据、MNIST 和 CIFAR-10 的潜在空间，未在大规模文本到图像模型（如 Stable Diffusion）上验证
3. **语义级遗忘待探索**：当前能量函数主要基于类别级或二元代理，扩展到语义级、几何级表述是未来方向
4. **CIFAR-10 效果有限**：在 CIFAR-10 上 Forget Rate (0.1704) 明显高于 MNIST (0.0005)，高维复杂数据的遗忘仍存在挑战
5. **持续多轮遗忘**：虽然名为 ContinualFlow，但论文未深入验证多轮迭代遗忘场景下的性能退化问题

## 相关工作与启发

- **Flow Matching (Lipman et al., 2023; Tong et al., 2023)**：本文的技术基础，利用 CFM 和 OT-CFM 的框架
- **Selective Amnesia (Heng & Soh, 2023)**：利用持续学习工具（如 EWC）进行遗忘，但需要访问遗忘数据
- **Erasing Concepts (Gandikota et al., 2023)**：扩散模型概念擦除的代表工作，属于输出抑制/模型编辑类
- **启发**：将遗忘问题形式化为分布传输的思路非常优雅，可以启发将类似框架应用到其他生成范式（如 diffusion、VAE）中

## 评分

- 新颖性: ⭐⭐⭐⭐ - 将 Flow Matching 与能量函数结合实现遗忘，视角新颖，但核心是重要性加权的巧妙应用
- 实验充分度: ⭐⭐⭐ - 2D + MNIST + CIFAR-10 验证了方法可行性，但缺乏大规模实验和更多基线对比
- 写作质量: ⭐⭐⭐⭐ - 理论推导清晰，可视化直观，整体结构规范
- 价值: ⭐⭐⭐⭐ - 建立了 Flow Matching 遗忘的理论基础，对后续工作有启发意义

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [Flow Matching Neural Processes](../../NeurIPS2025/image_generation/flow_matching_neural_processes.md)
- [Improving Flow Matching by Aligning Flow Divergence](improving_flow_matching_by_aligning_flow_divergence.md)
- [Gaussian Mixture Flow Matching Models](gaussian_mixture_flow_matching_models.md)
- [Curly Flow Matching for Learning Non-gradient Field Dynamics](../../NeurIPS2025/image_generation/curly_flow_matching_for_learning_non-gradient_field_dynamics.md)
- [Elucidating Flow Matching ODE Dynamics via Data Geometry and Denoisers](elucidating_flow_matching_ode_dynamics_with_respect_to_data_geometries_and_denoi.md)

<!-- RELATED:END -->
