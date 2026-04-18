---
title: >-
  [论文解读] Reparameterized LLM Training via Orthogonal Equivalence Transformation
description: >-
  [NeurIPS 2025][LLM/NLP][reparameterized training] 提出 POET 训练框架，通过将权重矩阵重参数化为"两个可学习正交矩阵 × 固定随机权重"的形式来保持谱性质不变，实现更稳定的训练和更好的泛化，且比 AdamW 更节省参数。
tags:
  - NeurIPS 2025
  - LLM/NLP
  - reparameterized training
  - orthogonal transformation
  - spectrum preservation
  - LLM pretraining
  - efficient training
---

# Reparameterized LLM Training via Orthogonal Equivalence Transformation

**会议**: NeurIPS 2025  
**arXiv**: [2506.08001](https://arxiv.org/abs/2506.08001)  
**代码**: [spherelab.ai/poet](https://spherelab.ai/poet)  
**领域**: llm_nlp  
**关键词**: reparameterized training, orthogonal transformation, spectrum preservation, LLM pretraining, efficient training

## 一句话总结

提出 POET 训练框架，通过将权重矩阵重参数化为"两个可学习正交矩阵 × 固定随机权重"的形式来保持谱性质不变，实现更稳定的训练和更好的泛化，且比 AdamW 更节省参数。

## 研究背景与动机

- LLM 预训练通常使用 AdamW 直接优化权重矩阵，但存在三个核心问题：
    - 计算密集且随模型规模增长扩展性差
    - 需要精细的超参数调优以确保稳定收敛
    - 即使训练损失完美最小化，泛化性能仍可能次优
- 权重矩阵的谱性质（奇异值）与泛化能力密切相关——更小的谱范数通常对应更强的泛化
- 现有谱控制方法的不足：
    - **谱控制无效**：只约束最大奇异值，无法有效正则化整个奇异值谱
    - **计算开销大**：谱范数正则化和谱归一化都需要计算最大奇异值（即使用 power iteration）
- 核心思想：正交变换不改变奇异值 → 用正交矩阵变换固定权重 = 自动保持谱性质

## 方法详解

### 整体框架

POET 将权重矩阵 $\mathbf{W} \in \mathbb{R}^{m \times n}$ 重参数化为：

$$\mathbf{W}_{RP} = \mathbf{R} \mathbf{W}_0 \mathbf{P}$$

其中：
- $\mathbf{W}_0$：随机初始化后**固定不变**的权重矩阵
- $\mathbf{R} \in \mathbb{R}^{m \times m}$：可学习的左正交矩阵（变换列空间/左奇异向量）
- $\mathbf{P} \in \mathbb{R}^{n \times n}$：可学习的右正交矩阵（变换行空间/右奇异向量）

前向传播为 $\mathbf{y} = (\mathbf{R}\mathbf{W}_0\mathbf{P})^\top \mathbf{x}$，训练后可将 $\mathbf{R}, \mathbf{P}$ 合并入权重，推理速度不变。

**谱保持性质**：由于 $\mathbf{W}_0 = \mathbf{U}\mathbf{\Sigma}_0\mathbf{V}^\top$，重参数化后 $\mathbf{W}_{RP} = \mathbf{RU}\mathbf{\Sigma}_0\mathbf{V}^\top\mathbf{P}$，奇异值 $\mathbf{\Sigma}_0$ 完全不变。

### 关键设计

**1. 随机原始子矩阵优化（SPO）**

直接优化 $m \times m$ 正交矩阵计算代价巨大。SPO 将其分解为多个"原始正交矩阵"的乘积：

- **Full stochastic SPO**：随机采样 $b$ 个索引的子集 $\mathbf{S}$，构造 $b \times b$ 小正交矩阵嵌入到单位矩阵中：
  $$\mathbf{R} = \prod_{i=1}^c \left(\mathbf{I}_m + \mathbf{D}(\mathbf{S}^i)(\tilde{\mathbf{G}}_i - \mathbf{I}_b)\mathbf{D}(\mathbf{S}^i)^\top\right)$$

- **Block stochastic SPO**：构造块对角正交矩阵 + 随机置换，确保所有维度都被变换：
  $$\mathbf{R} = \prod_{i=1}^c \left(\mathbf{\Psi}_i^\top \cdot \text{Diag}(\tilde{\mathbf{G}}_i^1, \ldots, \tilde{\mathbf{G}}_i^{\lceil m/b \rceil}) \cdot \mathbf{\Psi}_i\right)$$

**2. Cayley-Neumann 参数化（CNP）**

用截断 Neumann 级数近似 Cayley 参数化中的矩阵逆：

$$\mathbf{R} = (\mathbf{I}+\mathbf{Q})(\mathbf{I}-\mathbf{Q})^{-1} \approx (\mathbf{I}+\mathbf{Q})\left(\mathbf{I} + \sum_{i=1}^k \mathbf{Q}^i\right)$$

其中 $\mathbf{Q}$ 为反对称矩阵。$k=3$ 即可取得性能-速度的好权衡。

**3. Merge-then-Reinitialize 技巧**

每固定步数将学到的正交矩阵合并到权重中（$\mathbf{W} \leftarrow \mathbf{RWP}$），然后将正交矩阵重置为单位矩阵。这：
- 极大减少 GPU 显存使用（每次只存一个原始矩阵）
- 防止 Neumann 近似误差累积
- 允许重新采样索引集/置换

**4. 初始化方案**

提出两种新初始化：
- **Normalized Gaussian**：对标准高斯采样的神经元做归一化（实验最优）
- **Uniform Spectrum**：对标准初始化做 SVD 后将所有奇异值设为 1

### 训练策略

完整训练算法：
1. 用 Normalized Gaussian 初始化权重 $\mathbf{W} \leftarrow \mathbf{W}_0$
2. 随机采样索引/置换，初始化小正交矩阵为单位矩阵
3. 构造 $\mathbf{R}, \mathbf{P}$（通过 SPO + CNP）
4. 内循环训练：前向传播用 $\mathbf{RWP}$，反向传播更新小正交参数
5. 合并后重初始化，回到第 2 步

## 实验关键数据

### 主实验：LLaMA 预训练

| 模型 (tokens) | AdamW | GaLore | LoRA (r=64) | POET-BS b=256 | POET-FS b=1/2 |
|--------------|-------|--------|-------------|--------------|--------------|
| 60M (30B) | 26.68 (25.3M) | 29.81 (25.3M) | 39.70 (4.85M) | 25.29 (9.66M) | 25.37 (8.54M) |
| 130M (40B) | 20.82 (84.9M) | 22.35 (84.9M) | 32.07 (11.2M) | 19.88 (22.3M) | 19.94 (28.6M) |
| 350M (40B) | 16.78 (302M) | 17.99 (302M) | 25.19 (30.3M) | 16.27 (60.3M) | 15.95 (102M) |
| 1.3B (50B) | 14.73 (1.21B) | 18.33 (1.21B) | 20.55 (59.4M) | **14.56** (118M) | **13.70** (407M) |

POET-FS (b=1/2) 在 1.3B 模型上以约 1/3 可训练参数超越 AdamW（13.70 vs 14.73）。在 3B 模型上同样保持优势（16.90 vs 19.61）。

### 下游微调（GLUE）

| 微调方式 | CoLA | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | STS-B |
|---------|------|------|------|------|-----|-----|-------|-------|
| Full FT + AdamW | 0.361 | 0.658 | 0.696 | 0.818 | 0.829 | 0.534 | 0.914 | 0.880 |
| Full FT + POET | 0.523 | 0.818 | 0.824 | 0.885 | 0.902 | 0.661 | 0.920 | 0.873 |
| POET FT + POET | **0.505** | **0.821** | **0.826** | **0.892** | **0.902** | **0.682** | **0.931** | **0.887** |

POET 预训练的模型在所有任务和微调方式上一致优于 AdamW 预训练的基线。

### 消融实验

| 消融项 | 结论 |
|--------|------|
| 初始化方案 | Normalized Gaussian 最优（25.37），Uniform Spectrum 反而最差（27.29） |
| 合并频率 $T_m$ | 200 和 400 最优，过小（5）或过大（1600）性能下降 |
| Neumann 项数 $k$ | $k=0$ 训练发散；$k=3$ 达到性能-效率最优平衡 |
| $\mathbf{R}:\mathbf{P}$ 参数分配 | 50:50 均匀分配最优 |
| FS vs BS | Block stochastic 更参数高效（更好覆盖权重参数） |

### 关键发现

- POET 训练过程有三个独特阶段：(1) 锥壳搜索阶段，(2) 锥壳上稳定学习阶段，(3) 末期微调阶段
- POET 即使 AdamW 训练了近 3 倍 tokens 仍然保持优势（非平凡的泛化提升）
- POET 保持高 SVD 熵（多样谱分布），优于 AdamW 甚至 Muon
- POET 在各层保持低 hyperspherical energy（神经元均匀分布）
- 性能与参数预算高度相关，呈现类似 scaling law 的特性

## 亮点与洞察

1. **原理优雅**：从谱保持的数学理论出发推导出 POET，Theorem 1 证明保持谱性质的线性变换必须是正交等价变换
2. **泛化理论支撑**：通过 spectrally-normalized margin bound 提供泛化保证
3. **效率突破**：以 1/3~1/10 的可训练参数超越 AdamW 全参训练
4. **推理零开销**：训练后将正交矩阵合并入权重，推理结构完全不变
5. **是 OFT 的自然推广**：从能量保持训练推广到谱保持训练，增加了右正交矩阵 $\mathbf{P}$ 带来的灵活性
6. **三阶段学习动态**：vector probing 分析揭示了正交矩阵学习的有趣模式

## 局限与展望

- POET 早期训练收敛比 AdamW 慢（Phase II 特征），总训练时间可能更长
- Merge-then-reinitialize 频率 $T_m$ 是需要调优的超参数
- SPO 块大小 $b$ 也需要选择，不同模型规模的最优设置不同
- CNP 的 $k=0$ 导致训练发散，对正交性维护有强依赖
- 未充分探索与更先进优化器（如 Muon、SOAP）的结合效果
- Uniform spectrum 初始化表现差的原因需要进一步理论分析

## 相关工作与启发

- **对 LoRA 的替代**：在相似参数预算下 POET 显著优于 LoRA，说明谱保持 > 低秩假设
- **对 GaLore 的超越**：GaLore 依赖低秩梯度近似，POET 通过正交变换避免了信息损失
- **与 Muon 的互补**：Muon 也促进谱多样性，POET 可以与其结合使用
- **Random Neural Network 连接**：POET 训练后的权重在统计上与随机初始化不可区分（高斯等距不变性）
- **对预训练范式的影响**：POET 表明"学习变换"而非"直接学习权重"是一条有前景的路径

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 从谱保持理论推导出重参数化训练方法，SPO + CNP 设计精巧
- 实验充分度: ⭐⭐⭐⭐ 从 60M 到 3B 规模验证，下游任务评估完整，消融详尽
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导完整，实验分析深入，三阶段学习动态分析有趣
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 预训练方法有重要启示，可能改变大模型训练范式
