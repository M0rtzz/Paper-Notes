---
title: >-
  [论文解读] Come Together, But Not Right Now: A Progressive Strategy to Boost Low-Rank Adaptation
description: >-
  [模型压缩] > 提出 CoTo（Come Together），一种渐进式训练策略：在微调早期随机关闭 LoRA adapter，激活概率从 0 线性增长至 1，促使梯度在各层间均匀分布；理论上保证了 dropout 稳定性与线性模式连通性，实验表明可同时提升单任务泛化、多任务合并、剪枝鲁棒性并降低训练开销。
tags:
  - 模型压缩
---

# Come Together, But Not Right Now: A Progressive Strategy to Boost Low-Rank Adaptation

| 信息 | 内容 |
|------|------|
| **会议** | ICML 2025 |
| **arXiv** | [2506.05713](https://arxiv.org/abs/2506.05713) |
| **代码** | [zwebzone/coto](https://github.com/zwebzone/coto) |
| **领域** | LoRA/模型微调 |
| **关键词** | LoRA, 参数高效微调, 渐进式训练, Adapter Dropout, 线性模式连通性, 模型合并, 剪枝 |

## 一句话总结

> 提出 CoTo（Come Together），一种渐进式训练策略：在微调早期随机关闭 LoRA adapter，激活概率从 0 线性增长至 1，促使梯度在各层间均匀分布；理论上保证了 dropout 稳定性与线性模式连通性，实验表明可同时提升单任务泛化、多任务合并、剪枝鲁棒性并降低训练开销。

---

## 研究背景与动机

### 现有问题

LoRA 作为当前最主流的参数高效微调（PEFT）方法，通过将权重增量分解为低秩矩阵 $\Delta W = \alpha BA$ 来减少可训练参数量。然而 vanilla LoRA 存在两个关键缺陷：

**"懒惰训练"（Lazy Training）现象**：标准梯度优化的 lazy dynamics 使 adapter 收敛到初始化附近的次优极小值，限制了模型的泛化能力。

**层间梯度不平衡**：高层 adapter 获取绝大部分梯度信号并主导任务表现，低层 adapter 则严重利用不足。这不仅影响单任务泛化，还损害下游的 adapter 合并（merging）和剪枝（pruning）操作。

### 已有方案的不足

- **元素级/列级 Dropout**（如 LoRA 矩阵内部的 dropout）：没有考虑 adapter 在层间的顺序计算特性，无法纠正高层 adapter 获取不成比例梯度更新的问题。
- **自适应秩方案**（AdaLoRA、ALoRA、LoRA-drop）：自动调整每层 rank，但未显式解决层间优化不平衡。
- **初始化/优化改进**（PiSSA、LoRA-GA、rsLoRA、LoRA+、LoRA-Pro）：提升收敛速度或最终性能，但同样不直接解决层间不平衡。

### 核心动机

能否设计一种**无需改变架构**的训练策略，从优化过程本身入手，让所有层的 adapter 得到均衡利用，并同时提升泛化、可合并性和可剪枝性？

---

## 方法详解

### 整体框架

CoTo 的核心思想极其简洁：在微调过程中**渐进地提高每个 adapter 的激活概率**。

- **训练前 75%**：每个 adapter 以时变概率 $p(t)$ 被随机激活，$p(t)$ 从 0 线性增长至 1。
- **训练后 25%**：所有 adapter 全部激活（$p(t)=1$），退化为标准 LoRA 微调。

这种"先稀疏后完整"的课程式（curriculum-like）调度鼓励模型在损失景观中进行更广泛的探索。

### 激活概率调度

对于总训练步数 $T$，在第 $t$ 步，每个 adapter $i$ 的激活指示变量为：

$$\delta_i \sim \text{Bernoulli}(p(t))$$

其中激活概率的调度函数为：

$$p(t) = \begin{cases} \frac{4t}{3T} & t < \frac{3T}{4} \\ 1 & t \geq \frac{3T}{4} \end{cases}$$

模型输出相应调整为：

$$\hat{y} = f\left(x_0; \left\{W_i + \delta_i \mathbf{1} \odot \Delta W_i\right\}_{i=1}^{L}\right)$$

训练目标为最小化期望损失：

$$\min_{\{\Delta W_i\}} \mathbb{E}_{\boldsymbol{\delta}}\left[\ell(\hat{y}, y)\right]$$

### 关键设计 1：渐进式优化视角

将 CoTo 视为训练一个**部分 LoRA 子网络的加权集成**。定义恰好有 $j$ 个 adapter 激活时的期望模型预测为 $\tilde{y}_j$，则在步骤 $t$ 时恰有 $j$ 个 adapter 激活的概率为二项分布权重：

$$w_j(p(t)) = \binom{L}{j} p(t)^j (1-p(t))^{L-j}$$

**定理 3.1**：当损失函数 $\ell(\cdot, y)$ 为凸函数时，CoTo 的期望目标是各子网络损失加权和的上界：

$$\min_{\{\Delta W_i\}} \mathbb{E}_{\boldsymbol{\delta}}[\ell(\hat{y}, y)] \geq \min_{\{\Delta W_i\}} \sum_{j=1}^{L} w_j(p) \ell(\tilde{y}_j, y)$$

这保证了 CoTo 在优化时隐式地让模型在**任意 adapter 子集被关闭**的情况下都表现良好，从而促进 dropout 稳定性和线性模式连通性（LMC）。

### 关键设计 2：合作博弈视角

将每个 adapter 视为合作博弈中的"玩家"，使用 **Shapley 值**量化每个 adapter 的边际贡献。对 adapter 子集 $\mathcal{R}$ 定义价值函数：

$$v(\mathcal{R}) = \mathbb{E}_x\left[\ell\left(f\left(x; \{W_i + \delta_i \mathbf{1} \odot \Delta W_i\}_{i=1}^L\right), y\right)\right]$$

通过多线性扩展（Multilinear Extension）近似 Shapley 值：

$$\varphi_i(v) = \int_0^1 c_i(p) dp, \quad c_i(p) = \mathbb{E}[v(\mathcal{R}_i \cup \{i\}) - v(\mathcal{R}_i)]$$

实验表明 vanilla LoRA 的 Shapley 值高度集中在高层（69% 集中在最高 4/12 层），而 CoTo 使各层贡献更均衡（偏差 ±8%，早停版本仅 ±3%）。

### 计算节省

当 $\delta_i = 0$ 时，adapter $i$ 被完全跳过，不需要执行 $A_i$ 和 $B_i$ 的矩阵乘法。因此 CoTo 在训练早期减少了前向和反向计算。

---

## 实验关键数据

### 主实验 1：视觉基准（11 个图像分类任务，ViT-B/16，rank=2）

| 方法 | 平均准确率 |
|------|-----------|
| LoRA | 82.95% |
| LoRA-CoTo | **83.48%** (+0.53) |
| DoRA | 83.45% |
| DoRA-CoTo | **83.93%** (+0.48) |
| HiRA | 83.98% |
| HiRA-CoTo | **84.34%** (+0.36) |

CoTo 在所有 LoRA 变体上均取得一致提升。

### 主实验 2：常识推理（8 任务，LLaMA-3-8B，rank=32）

| 方法 | 平均准确率 |
|------|-----------|
| LoRA | 80.79% |
| LoRA-CoTo | **85.02%** (+4.23) |
| DoRA | 85.20% |
| DoRA-CoTo | **85.49%** (+0.29) |
| HiRA | 86.72% |
| HiRA-CoTo | **87.00%** (+0.28) |

在 LLaMA-2-7B 上 LoRA-CoTo 相比 LoRA 提升 3.02%（77.61→80.63）。

### 主实验 3：数学推理（GSM8K，LLaMA-2-7B，rank=8）

| 方法 | 不用 CoTo | 用 CoTo | 提升 |
|------|----------|---------|------|
| LoRA | 42.08 | 55.85 | **+13.77** |
| DoRA | 53.07 | 56.56 | +3.49 |
| HiRA | 54.51 | 56.68 | +2.17 |
| rsLoRA | 45.62 | 56.99 | +11.37 |
| LoRA-Pro | 54.23 | 57.16 | +2.93 |

CoTo 对 vanilla LoRA 的提升尤为惊人（+13.77），说明基础方法受 lazy training 影响最严重。

### 多任务合并（LLaMA-2-7B，9 个语言理解任务）

| 合并策略 | 不用 CoTo | 用 CoTo | 提升 |
|---------|----------|---------|------|
| Fusion | 47.17% | 58.53% | **+11.36** |
| Ensemble | 56.84% | 56.66% | -0.18 |
| LoRA-LEGO | 62.21% | 67.19% | **+4.98** |

线性模式连通性实验：在 $\lambda=0.5$ 插值处，LoRA-CoTo 保持 79% 准确率，而 vanilla LoRA 降至 39%。

### 剪枝鲁棒性

- **结构化剪枝**：在所有剪枝模式（交替层、低层、中层、高层）下，CoTo 均显著优于 LoRA。
- **非结构化剪枝**：50% 稀疏度下，LoRA-CoTo 比 LoRA 高出 10% 准确率。

### 消融实验

| 消融项 | 关键发现 |
|-------|---------|
| 第一阶段比例 | 75% 为最佳平衡点 |
| Dropout 策略 | 均匀 dropout（CoTo）和从高层开始 dropout（CoTo-H）均优于从低层开始（CoTo-L） |
| Rank 敏感性 | rank=8/32/128 均一致提升 |
| 学习率敏感性 | 5e-5/1e-4/2e-4 均有提升 |
| 插入模块 | Attention/Projection/Gating 均有效 |

### 训练时间（单卡 A6000，数学推理任务）

| 方法 | 不用 CoTo | 用 CoTo | 加速比 |
|------|----------|---------|--------|
| LoRA | 7h38min | 7h05min | 7.20% |
| DoRA | 19h00min | 14h30min | **23.69%** |
| HiRA | 11h39min | 8h50min | **24.21%** |

adapter 越大的变体（DoRA、HiRA）节省越多。

---

## 亮点与洞察

1. **极致简洁**：CoTo 不改变任何架构，仅用一个线性增长的激活概率调度即可带来全方位提升，是典型的"一行代码改进"。
2. **理论与实践双重支撑**：渐进优化视角（凸损失上界）和合作博弈视角（Shapley 值均衡化）从不同角度解释了方法有效性。
3. **"一石四鸟"**：同时改善单任务泛化、多任务合并、剪枝鲁棒性、训练效率，四个维度全部获益。
4. **通用兼容性**：对 LoRA、DoRA、HiRA、PiSSA、rsLoRA、LoRA+、LoRA-Pro 等多种变体均有效，是一种正交的训练策略。
5. **线性模式连通性提升极大**：在 $\lambda=0.5$ 处从 39% 提升到 79%，这对实际部署中的模型合并意义重大。
6. **计算免费午餐**：由于早期跳过部分 adapter，训练反而更快，DoRA 加速 24%。

---

## 局限性

1. **超参选择**：虽然 75%/25% 的分割比例实验表现最优，但不同任务和模型规模下是否需要调整尚未充分探讨。
2. **线性调度的最优性**：$p(t)$ 的线性增长是否最优？余弦、阶梯等其他调度方式未被系统比较。
3. **超大模型验证**：实验最大在 LLaMA-2-13B 上进行，对 70B+ 规模模型的效果未知。
4. **联合优化空间**：论文指出 CoTo 与自适应秩方案（AdaLoRA）、量化（QLoRA）等的联合使用是未来方向，但当前未做验证。
5. **理论假设**：定理 3.1 依赖损失函数的凸性假设，实际深度网络的损失函数通常是非凸的。

---

## 相关工作

- **PEFT 方法**：Adapter (Houlsby et al., 2019)、Prompt-Tuning (Lester et al., 2021)、Prefix-Tuning (Li & Liang, 2021)
- **LoRA 变体**：DoRA (Liu et al., 2024a)、HiRA (Huang et al., 2025)、LoRA-FA (Zhang et al., 2023a)、FourierFT (Gao et al., 2024b)、AdaLoRA (Zhang et al., 2023b)
- **LoRA 初始化与优化**：PiSSA (Meng et al., 2024)、LoRA-GA (Wang et al., 2024b)、rsLoRA (Kalajdzievski, 2023)、LoRA+ (Hayou et al., 2024)、LoRA-Pro (Wang et al., 2024c)
- **模型合并**：LoraHub (Huang et al., 2023)、LoRA-LEGO (Zhao et al., 2024b)、ZipLoRA (Shah et al., 2025)
- **随机正则化**：Dropout (Srivastava et al., 2014)、Stochastic Depth (Huang et al., 2016)、LayerDrop (Fan et al., 2020)

---

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐ |
| 理论深度 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| **总分** | **⭐⭐⭐⭐** |

方法本身极其简单（新颖性一般），但理论分析扎实，实验覆盖面极广（视觉/语言/扩散模型），且实用价值极高——几乎零成本集成到任何 LoRA 工作流中。

<!-- RELATED:START -->

## 相关论文

- [Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [DataDecide: How to Predict Best Pretraining Data with Small Experiments](datadecide_how_to_predict_best_pretraining_data_with_small_experiments.md)
- [Who Taught You That? Tracing Teachers in Model Distillation](../../ACL2025/model_compression/who_taught_you_that_tracing_teachers_in_model_distillation.md)
- [Variance-Based Pruning for Accelerating and Compressing Trained Networks](../../ICCV2025/model_compression/variance-based_pruning_for_accelerating_and_compressing_trained_networks.md)

<!-- RELATED:END -->
