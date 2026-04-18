---
title: >-
  [论文解读] A Certified Unlearning Approach without Access to Source Data
description: >-
  [ICML 2025][AI安全][机器遗忘] 提出首个无需访问原始训练数据的认证遗忘框架，利用代理数据集（surrogate dataset）近似原始数据统计特性，通过基于源分布与代理分布之间统计距离的噪声缩放机制，实现可证明的数据删除保证。
tags:
  - ICML 2025
  - AI安全
  - 机器遗忘
  - 认证遗忘
  - 代理数据集
  - 差分隐私
  - 噪声校准
---

# A Certified Unlearning Approach without Access to Source Data

**会议**: ICML 2025  
**arXiv**: [2506.06486](https://arxiv.org/abs/2506.06486)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 机器遗忘, 认证遗忘, 代理数据集, 差分隐私, 噪声校准

## 一句话总结

提出首个无需访问原始训练数据的认证遗忘框架，利用代理数据集（surrogate dataset）近似原始数据统计特性，通过基于源分布与代理分布之间统计距离的噪声缩放机制，实现可证明的数据删除保证。

## 研究背景与动机

随着 GDPR、CCPA、CPPA 等数据隐私法规的推行，从已训练模型中删除特定数据点的需求日益迫切。**认证遗忘（Certified Unlearning）** 通过严格的概率保证，确保遗忘后的模型与从头重训模型在统计上不可区分。

现有认证遗忘方法面临一个核心假设问题：**它们通常要求访问完整的原始训练数据**。然而在实际场景中，这一假设往往不成立：

- **隐私约束**：模型可能由第三方训练，原始数据不可共享
- **资源限制**：旧数据可能因存储成本已被删除
- **法规壁垒**：数据留存政策可能禁止保存原始数据

因此，一个关键的开放问题是：**当遗忘机制完全无法访问原始训练样本时，如何仅依赖一个近似原始数据分布的代理数据集来实现认证遗忘？**

尽管 zero-shot unlearning 方面已有一些工作（如零样本遗忘、JiT 遗忘、对抗样本生成等），但这些方法均**缺乏形式化的认证遗忘保证**。本文首次填补了这一理论空白。

## 方法详解

### 整体框架

本文方法建立在**二阶 Newton 更新**的认证遗忘框架上，核心创新在于用代理数据集 $\mathcal{D}_s$（采样自分布 $\nu$）替代原始数据集 $\mathcal{D}$（采样自分布 $\rho$），并通过三个关键步骤实现认证遗忘：

1. **Hessian 估计**：用代理数据集的 Hessian 近似原始数据集的 Hessian
2. **模型更新**：基于估计的 Hessian 进行单步 Newton 更新
3. **噪声校准**：根据源分布与代理分布之间的统计距离注入高斯噪声

**符号定义**：
- $\mathcal{D}$：原始数据集（$n_1$ 个样本，分布 $\rho$）
- $\mathcal{D}_s$：代理数据集（$n_2$ 个样本，分布 $\nu$）
- $\mathcal{D}_u$：待遗忘数据集（$m$ 个样本）
- $\mathcal{D}_r = \mathcal{D} \setminus \mathcal{D}_u$：保留数据集
- $\boldsymbol{w}^*$：在 $\mathcal{D}$ 上训练的模型
- $\boldsymbol{w}_r^*$：在 $\mathcal{D}_r$ 上重训的模型
- $\hat{\boldsymbol{w}}_r$：本文遗忘机制产出的模型

### 关键设计

#### 步骤一：Hessian 估计

传统方法中 Newton 更新需要保留数据 $\mathcal{D}_r$ 的 Hessian $\mathbf{H}_{\mathcal{D}_r}$。由于 $\mathcal{D}_r$ 不可用，本文用代理数据集的 Hessian $\mathbf{H}_{\mathcal{D}_s}$ 进行近似：

$$\hat{\mathbf{H}}_{\mathcal{D}_r} = \frac{n \cdot \mathbf{H}_{\mathcal{D}_s} - m \cdot \mathbf{H}_{\mathcal{D}_u}}{n - m}$$

这里利用了：完整数据 Hessian 可以由保留集 Hessian 和遗忘集 Hessian 线性组合得到，而代理数据集的 Hessian 近似了完整数据集的 Hessian。

#### 步骤二：模型更新

利用训练到收敛时 $\nabla \mathcal{L}(\mathcal{D}, \boldsymbol{w}^*) = 0$ 的性质，保留集上的梯度可以表示为：

$$\nabla \mathcal{L}(\mathcal{D}_r, \boldsymbol{w}^*) = \frac{-m \cdot \nabla \mathcal{L}(\mathcal{D}_u, \boldsymbol{w}^*)}{n - m}$$

将估计的 Hessian 和梯度代入 Newton 更新公式：

$$\hat{\boldsymbol{w}}_r = \boldsymbol{w}^* + \frac{m}{n - m} \hat{\mathbf{H}}_{\mathcal{D}_r}^{-1} \nabla \mathcal{L}(\mathcal{D}_u, \boldsymbol{w}^*)$$

注意：这一步只需要遗忘集 $\mathcal{D}_u$ 和代理数据集 $\mathcal{D}_s$，完全不需要原始保留数据。

#### 步骤三：噪声校准与认证保证

为实现 $(\epsilon, \delta)$-认证遗忘，向更新后的模型注入高斯噪声：

$$\hat{\boldsymbol{w}}_r' = \hat{\boldsymbol{w}}_r + \boldsymbol{n}, \quad \boldsymbol{n} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$$

其中噪声标准差为：

$$\sigma = \frac{\Delta}{\epsilon} \sqrt{2 \ln(1.25/\delta)}$$

**核心理论结果（Theorem 4.2）**：在损失函数满足 $L$-Lipschitz、$\alpha$-强凸、$\beta$-光滑、$\gamma$-Hessian Lipschitz 假设下，重训模型与遗忘模型的差的上界为：

$$\|\boldsymbol{w}_r^* - \hat{\boldsymbol{w}}_r\|_2 \leq \Delta = \frac{2\gamma L m^2}{\alpha^3 n_1^2} + \frac{\|\nabla \mathcal{L}(\mathcal{D}_u, \boldsymbol{w}^*)\|_2 \cdot m(n_1 - n_2)\beta + 2mn_2\beta \cdot \text{TV}(\rho \| \nu)}{(n_1\alpha - m\beta)(n_2\alpha - m\beta)}$$

关键观察：上界 $\Delta$ 由两项组成——第一项是标准 Newton 近似误差，第二项取决于**源分布与代理分布之间的全变差距离 $\text{TV}(\rho \| \nu)$**。当两个分布一致时（$\text{TV} = 0$），上界退化为已有方法的结果。

#### 统计距离的实用估计

在实际中 $\text{TV}(\rho \| \nu)$ 不可直接计算。本文提出一套启发式方法：

1. **TV → KL 转换**：利用 Bretagnolle-Huber 不等式 $\text{TV}(\nu \| \rho) \leq \sqrt{1 - e^{-\text{KL}(\nu \| \rho)}}$
2. **KL 散度分解**（Proposition 4.5）：将 $\text{KL}(\nu \| \rho)$ 分解为条件分布散度和边际分布散度两部分
3. **条件分布散度**：用原始模型 $\boldsymbol{w}^*$ 和在代理数据集上训练的模型 $\tilde{\boldsymbol{w}}^*$ 的输出概率来近似
4. **边际分布散度**：利用分类器隐含的能量模型（Energy-Based Model），通过 SGLD 采样从 $\rho(\boldsymbol{x})$ 生成样本，再用 Donsker-Varadhan 变分表示估计 KL 散度

### 损失函数 / 训练策略

**损失函数假设（Assumption 4.1）**：
- $L$-Lipschitz：参数空间中损失变化有界
- $\alpha$-强凸：保证唯一全局最优解
- $\beta$-光滑：梯度 Lipschitz 连续
- $\gamma$-Hessian Lipschitz：二阶导数 Lipschitz 连续

**实验超参数设置**：$\alpha = 1 + \lambda$, $L = 1$, $\beta = 1$, $\gamma = 1$，$L_2$ 正则化常数 $\lambda = 0.01$，隐私参数 $\epsilon = 5e3$, $\delta = 1$。SGLD 采样步长 0.02，生成 1000 个样本，每个样本迭代 4000 次。KL 估计网络为三层线性网络，500 个 epoch，学习率 0.0001，Adam 优化器。

## 实验关键数据

### 主实验

**合成实验**：原始数据集 15000 个样本，50 维标准高斯分布；代理数据集通过调节协方差矩阵非对角元素 $\zeta \in [0.01, 0.1]$ 控制 KL 散度。

| 方法 | $\zeta$ | Train Acc | Test Acc | Forget Acc | MIA | RT |
|------|---------|-----------|----------|------------|-----|-----|
| Retrain | – | 77.0% | 72.0% | 73.6% | 47.63% | 10 |
| Unlearn (+) | – | 77.1% | 72.6% | 74.1% | 47.63% | 10 |
| Unlearn (-) | 0.02 | 77.2% | 72.2% | 74.8% | 48.89% | 7 |
| Unlearn (-) | 0.06 | 77.3% | 72.2% | 74.3% | 48.37% | 10 |
| Unlearn (-) | 0.10 | 77.3% | 72.7% | 74.1% | 48.30% | 10 |

**真实数据实验（StanfordDogs, $\xi=100$）**：

| 方法 | Train Acc | Test Acc | Retain Acc | Forget Acc | MIA | RT |
|------|-----------|----------|------------|------------|-----|-----|
| Retrain | 82.9% | 76.0% | 84.2% | 71.1% | 52.5% | 19 |
| Unlearn (+) | 83.8% | 75.7% | 85.1% | 72.2% | 52.0% | 21 |
| Unlearn (-) | 83.7% | 75.6% | 85.0% | 72.0% | 51.9% | 16 |

### 消融实验

**不同遗忘比例（StanfordDogs, $\xi=100$）**：

| 遗忘比例 | 方法 | Train Acc | Forget Acc | MIA | RT |
|---------|------|-----------|------------|-----|-----|
| 0.01 | Unlearn (-) | 87.1% | 74.1% | 53.1% | 10 |
| 0.1 | Unlearn (-) | 83.7% | 72.0% | 51.9% | 16 |
| 0.2 | Unlearn (-) | 85.0% | 72.6% | 52.0% | 40 |

**不同模型架构（CIFAR-10, $\xi=100$）**：

| 架构 | 方法 | Train Acc | Test Acc | MIA | RT |
|------|------|-----------|----------|-----|-----|
| Linear (L) | Unlearn (-) | 78.1% | 77.2% | 48.83% | 32 |
| Conv+Linear (C+L) | Unlearn (-) | 80.5% | 78.1% | 50.71% | 46 |
| 2Conv+Linear (2C+L) | Unlearn (-) | 82.9% | 80.5% | 50.05% | 22 |

### 关键发现

1. **Unlearn (-) 与 Unlearn (+) 性能接近**：即使无法访问原始数据，本文方法在所有指标上均与有数据访问的方法表现相当
2. **MIA 接近 50%**：成员推理攻击准确率接近随机猜测，说明遗忘彻底
3. **噪声缩放的必要性**：Forget Score 实验表明，当使用代理数据集时，必须使用本文提出的噪声缩放（而非直接使用有数据方法的噪声），才能实现有效的认证遗忘
4. **KL 近似有效**：启发式 KL 估计值与精确值高度吻合，验证了实用性
5. **跨架构泛化**：从线性模型到多层卷积网络，方法均保持有效

## 亮点与洞察

1. **首个无源数据认证遗忘理论保证**：填补了 source-free unlearning 领域缺乏形式化认证保证的重要空白，将认证遗忘的适用范围大幅拓展
2. **优雅的理论框架**：上界 $\Delta$ 清晰地刻画了代理数据质量（通过 $\text{TV}(\rho \| \nu)$）对遗忘保证的影响——分布越接近，所需噪声越小，模型效用越高
3. **完整的实践方案**：从理论保证到 KL 散度的无源估计（能量模型 + SGLD + Donsker-Varadhan），提供了一套从理论到实践的完整路径
4. **MNIST-USPS 跨域实验**：展示了真正跨数据集（不同来源）作为代理数据集的可行性，具有很强的现实意义

## 局限与展望

1. **强凸性假设**：损失函数需要 $\alpha$-强凸等严格假设，限制了对深度非凸模型的直接适用性（混合线性网络虽能部分缓解，但仍是近似）
2. **KL 估计的理论 gap**：启发式 KL 估计虽然实验中表现良好，但理论上可能导致比精确值更弱的隐私保证
3. **SGLD 采样质量**：能量模型采样的质量直接影响 KL 估计精度，在高维复杂场景下可能退化
4. **大规模模型适用性**：实验主要在线性模型和浅层 CNN 上验证，对 Transformer 等现代大模型的适用性待验证
5. **单步 Newton 更新的局限**：对于大批量遗忘请求，单步近似可能不够精确

## 相关工作与启发

- **Guo et al. (2019)**：开创性的认证数据删除工作，使用影响函数和 Newton 更新
- **Sekhari et al. (2021)**：建立了 $(\epsilon, \delta)$-认证遗忘的标准定义和理论框架
- **Zhang et al. (2024)**：将认证遗忘扩展到深度神经网络
- **Ahmed et al. (CVPR 2025)**：source-free 遗忘的实证方法，但无认证保证
- **Chundawat et al. (2023)**：零样本遗忘，使用噪声最大化和门控知识蒸馏
- **Grathwohl et al. (2019)**：能量模型视角启发了本文的边际分布采样方法

**对后续研究的启发**：代理数据集的思路可推广到联邦遗忘、隐私合规审计等场景；理论框架可尝试放松到非凸损失或更一般的模型类。

## 评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ | 首次为无源数据认证遗忘提供形式化理论保证 |
| 理论深度 | ⭐⭐⭐⭐⭐ | 完整的定理证明体系，从上界到认证保证 |
| 实验充分性 | ⭐⭐⭐⭐ | 合成+真实数据集，多种消融，但模型规模偏小 |
| 实用性 | ⭐⭐⭐⭐ | KL 估计方案使方法可落地，但强凸假设限制适用范围 |
| 写作质量 | ⭐⭐⭐⭐ | 逻辑清晰，理论到实践过渡自然 |
| 综合推荐 | ⭐⭐⭐⭐☆ | 理论贡献突出，是认证遗忘领域的重要进展 |
