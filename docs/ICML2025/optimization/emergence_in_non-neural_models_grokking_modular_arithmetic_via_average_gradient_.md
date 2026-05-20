---
title: >-
  [论文解读] Emergence in Non-Neural Models: Grokking Modular Arithmetic via Average Gradient Outer Product
description: >-
  [ICML2025][优化/理论][grokking] 本文证明 grokking（延迟泛化）现象并非神经网络或梯度下降特有，而是源于**任务相关特征的逐步学习**——利用非神经网络的 Recursive Feature Machines (RFM) 在核机器上复现了模算术的 grokking…
tags:
  - "ICML2025"
  - "优化/理论"
  - "grokking"
  - "模算术"
  - "特征学习"
  - "AGOP"
  - "循环矩阵"
  - "核方法"
  - "Recursive Feature Machines"
---

# Emergence in Non-Neural Models: Grokking Modular Arithmetic via Average Gradient Outer Product

**会议**: ICML2025  
**arXiv**: [2407.20199](https://arxiv.org/abs/2407.20199)  
**代码**: [nmallinar/rfm-grokking](https://github.com/nmallinar/rfm-grokking)  
**领域**: 优化 / Grokking / 特征学习  
**关键词**: grokking, 模算术, 特征学习, AGOP, 循环矩阵, 核方法, Recursive Feature Machines

## 一句话总结

本文证明 grokking（延迟泛化）现象并非神经网络或梯度下降特有，而是源于**任务相关特征的逐步学习**——利用非神经网络的 Recursive Feature Machines (RFM) 在核机器上复现了模算术的 grokking，揭示分块循环（block-circulant）特征矩阵是泛化的核心。

## 研究背景与动机

Grokking 是指模型在训练精度早已达到 100% 后，测试精度才突然从接近零跃升至完美的"涌现"现象，被视为深度学习中"技能涌现"的典型案例。现有研究普遍将其归因于：

- 神经网络的特殊结构
- SGD 等梯度优化器的隐式正则化
- 训练损失的逐步下降

然而，这些解释是否充分？作者提出三个关键质疑：

1. Grokking 是否只能在神经网络中出现？
2. 是否必须依赖梯度优化？
3. 传统指标（训练/测试损失）能否预测涌现时间点？

本文通过在**核机器 + RFM** 这一完全不同的学习框架中复现 grokking，回答了以上三个问题均为"否"。

## 方法详解

### 核心框架：Recursive Feature Machines (RFM)

RFM 是一种迭代算法，通过 **Average Gradient Outer Product (AGOP)** 为无特征学习能力的核机器赋予特征学习功能。每次迭代包含两步：

**Step 1 — 训练预测器**：求解核岭回归

$$f^{(t)}(x) = k(x, X; M_t) \alpha, \quad \alpha = k(X, X; M_t)^{-1} y$$

**Step 2 — AGOP 更新特征矩阵**：

$$M_{t+1} = [G(f^{(t)})]^{s}, \quad s = \frac{1}{2}$$

其中 AGOP 定义为训练数据上雅可比矩阵的平均外积：

$$G(f) = \frac{1}{n} \sum_{j=1}^{n} \frac{\partial f(x^{(j)})}{\partial x} \frac{\partial f(x^{(j)})}{\partial x}^\top \in \mathbb{R}^{d \times d}$$

### 数据编码

输入 $(a, b) \in \mathbb{Z}_p \times \mathbb{Z}_p$ 采用 one-hot 编码拼接为 $\mathbf{e}_a \oplus \mathbf{e}_b \in \mathbb{R}^{2p}$，标签为 $\mathbf{e}_{f^*(a,b)} \in \mathbb{R}^p$。

### 关键发现：分块循环特征矩阵

RFM 最终学到的 $2p \times 2p$ 特征矩阵 $M^*$ 具有分块循环结构：

$$M^* = \begin{pmatrix} A & C^\top \\ C & A \end{pmatrix}$$

其中 $C \in \mathbb{R}^{p \times p}$ 为循环矩阵（每行是前一行的循环移位），$A = c_1 I + c_2 \mathbf{1}\mathbf{1}^\top$。

对于乘法/除法任务，需通过**离散对数**重排行列后才能观察到循环结构（因为离散对数将乘法转化为加法）。

### 两个隐藏进展度量

1. **循环偏差 (Circulant Deviation)**：衡量特征子矩阵与循环矩阵的距离

$$\mathcal{D}(A) = \frac{1}{\|A\|_F^2} \sum_{j=0}^{p-1} \mathrm{Var}(\mathcal{S}(A)[j])$$

循环矩阵对角线元素恒定，因此 $\mathcal{D} = 0$ 代表完美循环结构。

2. **AGOP 对齐 (AGOP Alignment)**：当前迭代特征与最终特征的余弦相似度

$$\rho(M_t, M^*) = \frac{\langle \tilde{M}_t, \tilde{M}^* \rangle}{\|\tilde{M}_t\| \|\tilde{M}^*\|}$$

### 与傅里叶乘法算法 (FMA) 的理论联系

作者证明，配备循环特征的核机器实质上实现了 FMA：

1. 对每个数字向量做离散傅里叶变换：$\hat{x}_{[1]} = Fx_{[1]}, \hat{x}_{[2]} = Fx_{[2]}$
2. 逐元素相乘：$\hat{x}_{[1]} \odot \hat{x}_{[2]}$
3. 投影回类别空间：$y_{\mathrm{add}}(x; \ell) = \sqrt{p} \cdot \langle F\mathbf{e}_a \odot F\mathbf{e}_b, F\mathbf{e}_\ell \rangle_{\mathbb{C}}$

这与先前研究中神经网络被认为学到的泛化方案一致，表明 RFM 和神经网络发现了**相同的算法**来解决模算术。

## 实验关键数据

实验使用模数 $p = 61$，训练比例 50%，核函数为二次核和高斯核。

| 实验设置 | 训练损失 | 涌现迭代 | 最终测试精度 |
|---------|---------|---------|------------|
| RFM (二次核) 加法 | 恒为 0 | ~12 次迭代 | 100% |
| RFM (二次核) 减法 | 恒为 0 | ~12 次迭代 | 100% |
| RFM (二次核) 乘法 | 恒为 0 | ~15 次迭代 | 100% |
| RFM (二次核) 除法 | 恒为 0 | ~15 次迭代 | 100% |
| 神经网络 (二次激活) 加/减 | 逐步下降 | ~25 epoch | 100% |
| 神经网络 (二次激活) 乘/除 | 逐步下降 | ~35 epoch | 100% |

**关键对比**：

- **随机循环特征 vs. RFM 学习特征**：用随机循环矩阵变换输入后，标准核机器直接泛化，甚至优于 RFM 经多次迭代学到的特征
- **随机循环特征加速神经网络**：训练比例 17.5% 时，变换后的网络数百 epoch 达到 100%，未变换网络 3000 epoch 仍不收敛
- **NFM 与 AGOP 的皮尔逊相关**：加法 0.955、减法 0.942、乘法 0.924、除法 0.929，高度一致

**隐藏进展度量 vs. 标准度量**：在测试损失和测试精度完全不变的前 8-10 次迭代中，循环偏差和 AGOP 对齐已呈现近乎线性的稳步改善。

## 亮点与洞察

1. **解耦贡献**：将特征学习与预测器训练彻底分离，证明涌现纯粹源于特征学习能力，与网络架构和优化器无关
2. **统一视角**：RFM（核机器）和神经网络学到的特征结构高度一致（均为分块循环），暗示存在任务固有的"正确特征"
3. **隐藏进展度量**：提出循环偏差和 AGOP 对齐两个指标，在标准度量（损失/精度）完全静默时即可检测到模型朝泛化方向的稳步进展
4. **正则化解释**：通过 NFA（Neural Feature Ansatz），将权重衰减的作用解释为对 AGOP 迹的正则化，AGOP 正则化可替代权重衰减实现 grokking
5. **实用启示**：随机循环特征即可使核/神经网络快速泛化，说明泛化不需要精确学到的特征，只需正确的**结构类型**

## 局限与展望

1. **任务受限**：仅验证了模算术（加减乘除），是否能推广到更复杂的代数结构或自然语言中的"技能涌现"尚未探讨
2. **AGOP 对齐是后验指标**：需要已知最终收敛模型的特征矩阵，无法作为训练过程中的实时预测工具
3. **理论证明条件强**：FMA 定理要求在全部数据上训练且使用特定核函数，部分训练数据场景的理论保证缺失
4. **多技能场景初步**：虽在附录中展示了两个任务的不同 grokking 速率，但未深入分析技能间的交互与竞争
5. **计算开销**：RFM 每步需计算 $d \times d$ AGOP 矩阵并求逆，对大规模问题的可扩展性存疑

## 相关工作与启发

- **Power et al. (2022)**：首次发现并命名 grokking 现象
- **Nanda et al. (2023)**：通过机械可解释性发现神经网络用傅里叶乘法算法解决模加法
- **Radhakrishnan et al. (2024)**：提出 RFM 和 Neural Feature Ansatz，本文核心框架来源
- **Schaeffer et al. (2024)**：质疑涌现可能是度量选择造成的"幻象"，本文提供了更深层的反驳——即使连续度量（测试损失）也呈现sharp transition

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次在非神经网络模型中复现 grokking，视角独特
- 实验充分度: ⭐⭐⭐⭐ — 四种运算 × 两类模型 × 多种度量，消融充分
- 写作质量: ⭐⭐⭐⭐⭐ — 逻辑清晰，图示精美，理论与实验交织得当
- 价值: ⭐⭐⭐⭐ — 为理解涌现/grokking 提供了重要的去耦合证据

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Grokking at the Edge of Linear Separability](grokking_at_the_edge_of_linear_separability.md)
- [\[NeurIPS 2025\] Natural Gradient VI: Guarantees for Non-Conjugate Models](../../NeurIPS2025/optimization/natural_gradient_vi_guarantees_for_non-conjugate_models.md)
- [\[ICML 2025\] Provable In-Context Vector Arithmetic via Retrieving Task Concepts](provable_in-context_vector_arithmetic_via_retrieving_task_concepts.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](../../NeurIPS2025/optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[ICLR 2026\] Optimizer Choice Matters for the Emergence of Neural Collapse](../../ICLR2026/optimization/optimizer_choice_matters_for_the_emergence_of_neural_collapse.md)

</div>

<!-- RELATED:END -->
