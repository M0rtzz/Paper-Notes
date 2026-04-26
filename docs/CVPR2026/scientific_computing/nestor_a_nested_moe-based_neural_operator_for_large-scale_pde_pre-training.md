---
title: >-
  [论文解读] NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training
description: >-
  [CVPR 2026][科学计算][神经算子] 提出嵌套式 MoE 神经算子 NESTOR，通过 image-level MoE 捕获不同 PDE 类型的全局特征 + token-level Sub-MoE 捕获物理场内局部相关性，在 12 个 PDE 数据集上实现大规模预训练并有效迁移到下游任务。
tags:
  - CVPR 2026
  - 科学计算
  - 神经算子
  - 混合专家(MoE)
  - 偏微分方程
  - 大规模预训练
  - 傅里叶注意力
---

# NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training

**会议**: CVPR 2026  
**arXiv**: [2602.22059](https://arxiv.org/abs/2602.22059)  
**代码**: [有](https://github.com/Event-AHU/OpenFusion)  
**领域**: 科学计算  
**关键词**: 神经算子, 混合专家(MoE), PDE求解, 大规模预训练, 傅里叶注意力  

## 一句话总结

提出嵌套式 MoE 神经算子 NESTOR，通过 image-level MoE 捕获不同 PDE 类型的全局特征 + token-level Sub-MoE 捕获物理场内局部相关性，在 12 个 PDE 数据集上实现大规模预训练并有效迁移到下游任务。

## 研究背景与动机

偏微分方程（PDE）广泛应用于物理、流体力学等领域。传统数值方法（FEM、FDM）计算成本高，神经算子（FNO、DeepONet 等）通过学习函数空间映射实现快速推理，但面临两个核心挑战：

**训练数据稀缺**：PDE 训练数据通常需要昂贵的实验或数值模拟获取

**单一架构局限**：现有大规模 PDE 预训练（如 DPOT、MPP）采用单一网络架构，难以同时处理：
   - **PDE 间的宏观差异**：不同方程的动力学机制、边界条件、变量维度差异巨大
   - **PDE 内的微观异质**：同一方程物理场内存在复杂的时空局部相关性

核心洞察：PDE 系统的多样性和复杂性需要不同专家网络针对不同输入进行专门化处理，而非"一个网络处理所有"。MoE 的路由机制天然适合这一需求，但单层 MoE 仅能区分方程类型，无法捕获同一方程内部的区域异质性。

## 方法详解

### 整体框架

NESTOR 采用自回归预测框架：输入最近 $T$ 帧 PDE 状态 $u_{t-T+1:t}$，预测下一帧 $u_{t+1}$。整体流程：

1. **Patch 嵌入 + 时空编码**：将输入划分为 patch 并映射到隐空间
2. **嵌套 MoE 模块**（核心）：image-level MoE 选择全局专家 → 每个专家内部包含 token-level Sub-MoE
3. **输出头**：预测下一帧 PDE 状态

### 关键设计

#### 1. 时空编码（Spatio-Temporal Encoding）

**功能**：将多帧 PDE 输入编码为统一的隐表示。

**核心思路**：输入 $x \in \mathbb{R}^{B \times C \times H \times W}$ 先划分为不重叠的 patch $X_p \in \mathbb{R}^{B \times N \times C \times P_H \times P_W}$，经线性映射和位置编码后得到 $X \in \mathbb{R}^{B \times N \times D}$。然后重排为 $X \in \mathbb{R}^{B \times X \times Y \times T \times C}$，通过可学习权重矩阵在时间维度压缩：

$$Y = \sum_{t=1}^{T} W_t X_t, \quad Y \in \mathbb{R}^{B \times X \times Y \times C_{\text{out}}}$$

**设计动机**：不同 PDE 的输入帧数可能不同，时间维度压缩保证统一维度输入后续模块。

#### 2. Image-level MoE（全局专家选择）

**功能**：基于输入样本的全局特征，动态选择最适合当前 PDE 类型的专家网络。

**核心思路**：采用 Top-$k$ 路由策略。对输入特征做全局平均池化得到 $\bar{x}_b \in \mathbb{R}^C$，经线性层生成专家分数，softmax 归一化后选取概率最高的 $k$ 个专家：

$$s_b = \bar{x}_b W^\top + b, \quad p_b = \text{softmax}(s_b)$$

$$w_{b,i} = \frac{p_{b,i}}{\sum_{j \in \mathcal{I}_b} p_{b,j}}, \quad i \in \mathcal{I}_b$$

架构配置：6 个非共享专家 + 1 个共享专家，门控网络每次激活 2 个非共享专家。

**专家设计**：
- **共享专家**：AFNO（自适应傅里叶神经算子），在频域捕获全局低频空间特征。对输入做 FFT → 频域复数卷积 → IFFT
- **非共享专家**：Flash Attention，捕获细粒度时空特征。Q/K/V 注意力后接 Sub-MoE

**设计动机**：实验验证表明不同专家对不同 PDE 类型有显著偏好（例如 Expert 0+1 偏好 NS 方程，Expert 2+3 偏好浅水波方程），说明 image-level 路由有效实现了功能分工。

#### 3. Token-level Sub-MoE（局部专家选择）

**功能**：在每个 image-level 专家内部，对每个 token（空间位置）选择最适合的局部专家。

**核心思路**：替代 Flash Attention 中的 FFN 层。同样采用 Top-$k$ 路由，但路由粒度为 token 级别而非 image 级别。每个专家是标准 MLP：

$$\text{ExpertMLP}(x) = W_2 \sigma(W_1 x + b_1) + b_2$$

其中 $W_1 \in \mathbb{R}^{C \times (rC)}$，$W_2 \in \mathbb{R}^{(rC) \times C}$，$r$ 为 MLP ratio，激活函数为 GELU。

配置同样为 6 非共享 + 1 共享专家，Top-2 激活。

**设计动机**：可视化分析表明，不同 token-level 专家在空间上呈现区域特异性的激活模式，验证了其捕获物理场内局部相关性的能力。

#### 4. 嵌套架构的"宏观分类-微观分区"机制

**功能**：image-level MoE 和 token-level Sub-MoE 形成层级协作。

**核心思路**：
- **宏观层**：image-level MoE 根据 PDE 类型自适应选择专家组合（如 NS 方程激活 Expert 0+1，SWE 激活 Expert 2+3）
- **微观层**：token-level Sub-MoE 在选定的专家内部，进一步识别物理场的空间区域特征

这种嵌套设计使模型总参数量达 83M，但激活参数仅 13M（激活率 16.67%），实现了大容量与低计算成本的平衡。

### 损失函数 / 训练策略

总损失由三部分组成：

$$\mathcal{L} = \mathcal{L}_2 + \alpha \mathcal{L}_{\text{aux}_1} + \beta \mathcal{L}_{\text{aux}_2}$$

- **主任务损失** $\mathcal{L}_2$：L2 相对误差（L2RE），$\mathcal{L}_2 = \frac{\|\hat{y}_i^{(c)} - y_i^{(c)}\|_2}{\|y_i^{(c)}\|_2}$
- **Image-level 负载均衡损失** $\mathcal{L}_{\text{aux}_1}$：防止专家分配不均
- **Token-level 负载均衡损失** $\mathcal{L}_{\text{aux}_2}$：同上

负载均衡损失统一定义为 $\mathcal{L}_{\text{aux}} = E \sum_{i=1}^{E} p_i \cdot f_i$，其中 $p_i$ 为平均路由概率，$f_i$ 为实际 token 分配比例。

训练策略：向输入帧注入小尺度噪声增强鲁棒性（沿用 DPOT 的去噪预训练策略）。

## 实验关键数据

### 主实验

12 个 PDE 数据集上的预训练和微调结果（L2RE↓）：

| 模型 | 激活参数 | FNO-ν 1e-5 | FNO-ν 1e-4 | FNO-ν 1e-3 | PDEBench Avg(1) | PDEBench Avg(0.1) | DR | SWE | CFDBench |
|---|---|---|---|---|---|---|---|---|---|
| FNO | 0.5M | 0.116 | 0.092 | 0.016 | 0.130 | 0.153 | 0.032 | 0.009 | 0.027 |
| DPOT-T (预训练) | 7M | 0.098 | 0.061 | 0.010 | 0.029 | 0.018 | 0.032 | 0.006 | 0.010 |
| **Ours (预训练)** | 13M | 0.120 | 0.095 | **0.009** | **0.027** | **0.016** | 0.031 | **0.005** | 0.011 |
| DPOT-FT500 | 7M | 0.052 | 0.037 | 0.006 | 0.015 | 0.016 | 0.015 | 0.002 | 0.004 |
| **Ours-FT500** | 13M | **0.051** | **0.022** | **0.004** | **0.011** | **0.010** | **0.012** | 0.003 | **0.004** |

微调 500 epochs 后，在 12 个任务中 9 个达到 SOTA，全局最优 10/12。

### 消融实验

PDEBench 六个子任务上的消融（FT-500，Avg L2RE↓）：

| 方法 | Avg L2RE | 性能下降 |
|---|---|---|
| **完整模型** | **0.0173** | - |
| w/o Sub-MoE | 0.0197 | +0.0024 |
| w/o 负载均衡损失 | 0.0178 | +0.0005 |
| FlashAttn + AFNO 直接相加 | 0.0196 | +0.0023 |

### 关键发现

1. **Sub-MoE 贡献最大**：移除后误差增加 0.0024，验证了 token 级别细粒度专家选择的重要性
2. **MoE 融合优于简单相加**：将 AFNO 和 FlashAttn 的 MoE 融合替换为直接相加，误差增加 0.0023，证明路由选择机制优于固定融合
3. **专家数不是越多越好**：6 个非共享专家在 FT-500 下取得最佳平均性能，12 个专家反而因优化困难而性能下降
4. **预训练数据量有正向影响**：12 个数据集预训练的 Avg L2RE（0.0208）优于 3 个数据集（0.0234）
5. **下游任务迁移效果显著**：在 512×512 高分辨率湍流任务上，微调后精度提升 47.3%
6. **激活效率**：总参数 83M 中仅激活 13M（16.67%），远低于 MoE-POT-T 的 56.67%

## 亮点与洞察

1. **嵌套 MoE 设计有清晰的物理对应**：image-level → PDE 类型分工，token-level → 物理场空间区域分工，"宏观分类-微观分区"具有良好的可解释性
2. **异构专家设计**：共享专家用 AFNO（频域全局特征），非共享专家用 Flash Attention（空间局部特征），两类互补而非冗余
3. **可解释性分析充分**：Table 5 的专家激活频率统计和 Figure 5 的 token 级别空间热图，清晰展示了 MoE 的功能分化
4. **大容量低成本**：83M 总参数但仅 16.67% 激活率，为 PDE 神经算子的高效扩展提供了思路

## 局限与展望

1. **预训练阶段部分数据集不占优**：FNO-ν 1e-5 和 1e-4 上预训练性能不如 DPOT，说明嵌套 MoE 在数据有限时可能过拟合到特定专家
2. **NS-cond 和 PDE Arena-NS 表现较弱**：这两个数据集上 Ours-FT500 与 DPOT-FT500 基本持平甚至略差
3. 仅实验了 2D PDE，3D PDE 的可扩展性未验证
4. 专家数量（6）和激活数（2）为手动设定，缺少自适应机制
5. 负载均衡损失贡献有限（仅 0.0005），可探索更有效的专家平衡策略
6. 可结合物理约束损失（如 PDE 残差损失）进一步提升物理一致性

## 评分

⭐⭐⭐⭐ 4/5

将嵌套 MoE 引入 PDE 神经算子是有意义的创新，"宏观分类-微观分区"的设计直觉清晰、实验验证充分。在 12 个基准中 10 个 SOTA 的结果令人信服。扣分在于：创新主要是 MoE 架构的工程组合（AFNO + Flash Attention + 双层路由），各组件均为已有技术；此外 83M 参数仅激活 13M 虽然计算高效，但总内存开销仍然较大。

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers](../../ICLR2026/scientific_computing/one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd.md)
- [\[AAAI 2026\] PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations](../../AAAI2026/scientific_computing/physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)
- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](../../ICLR2026/scientific_computing/drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[NeurIPS 2025\] Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections](../../NeurIPS2025/scientific_computing/enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)
- [\[ICLR 2026\] Astral: Training Physics-Informed Neural Networks with Error Majorants](../../ICLR2026/scientific_computing/astral_training_physics-informed_neural_networks_with_error_majorants.md)

<!-- RELATED:END -->
