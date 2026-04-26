---
title: >-
  [论文解读] LayerIF: Estimating Layer Quality for Large Language Models using Influence Functions
description: >-
  [NeurIPS 2025][模型压缩][影响函数] LayerIF 提出用影响函数（Influence Functions）逐层量化 LLM 的训练质量，通过聚合各层的正向影响分数得到数据驱动的层重要性估计，并将其应用于 LoRA-MoE 专家分配和层级稀疏剪枝两个下游任务，在 Mistral-7B 和 Gemma-7B 上分别获得 1.61% 和 0.90% 的准确率提升。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 影响函数
  - 层质量估计
  - LoRA-MoE
  - 模型剪枝
  - 层稀疏分配
---

# LayerIF: Estimating Layer Quality for Large Language Models using Influence Functions

**会议**: NeurIPS 2025  
**arXiv**: [2505.23811](https://arxiv.org/abs/2505.23811)  
**代码**: 无  
**领域**: model_compression  
**关键词**: 影响函数, 层质量估计, LoRA-MoE, 模型剪枝, 层稀疏分配

## 一句话总结
LayerIF 提出用影响函数（Influence Functions）逐层量化 LLM 的训练质量，通过聚合各层的正向影响分数得到数据驱动的层重要性估计，并将其应用于 LoRA-MoE 专家分配和层级稀疏剪枝两个下游任务，在 Mistral-7B 和 Gemma-7B 上分别获得 1.61% 和 0.90% 的准确率提升。

## 研究背景与动机

**领域现状**：LLM 各层的训练质量差异显著，已有研究发现中间层提供更强的下游特征，最深层可以被剪枝而损失最小。层质量评估对模型压缩（剪枝）和高效微调（LoRA-MoE专家分配）等下游任务至关重要。

**现有痛点**：现有层质量估计方法主要依赖模型侧启发式：AlphaPruning 用权重矩阵的重尾谱密度 (PL_Alpha_Hill)，OWL 用异常激活分布，MoLA 用固定模式（如 2-4-6-8）。这些方法完全忽略了训练数据的影响。

**核心矛盾**：同一模型在不同任务/数据集上的层级专精化模式不同，仅基于权重统计的方法无法捕获这种任务特异性。

**本文目标** 提出一个数据+模型双驱动的层质量估计框架，能为同一 LLM 在不同任务上产生不同的层重要性评分。

**切入角度**：影响函数量化训练样本对模型损失的敏感度；训练良好的层对数据扰动不敏感（IF值小），训练不足的层更敏感（IF值大）——这可以作为层质量的代理指标。

**核心 idea**：用逐层影响函数分数作为层训练质量的数据驱动代理，指导 MoE 专家分配和稀疏剪枝，实现任务自适应的层级资源分配。

## 方法详解

### 整体框架
输入：预训练 LLM、训练数据 $D^{train}$、验证数据 $D^{\mathcal{V}}$。对每个 Transformer 块计算层级影响函数分数 → 聚合为标量层质量估计 → 通过映射函数转换为下游任务的层级资源分配（专家数或稀疏率）。

### 关键设计

1. **层级影响函数计算（LayerIF Core）**:

    - 功能：量化每层对训练-验证交互的贡献
    - 核心公式：$I^{(l)}(z_i) = -\sum_{j=1}^m \nabla_{\theta^{(l)}} \ell(z_j^{\mathcal{V}}, \theta)^\top (H^{(l)}(\theta))^{-1} \nabla_{\theta^{(l)}} \ell(z_i, \theta)$
    - 聚合：仅保留正向影响样本（对模型有益的），逐层求和 $S^{(l)} = \sum_{i=1}^n \mathbb{I}[I^{(l)}(z_i) > 0] \cdot I^{(l)}(z_i)$
    - 关键洞察：$S^{(l)}$ 高的层对训练数据更敏感，说明训练尚不充分；$S^{(l)}$ 低的层更稳定，训练更成熟
    - 使用 DataInf 方法近似 Hessian 逆矩阵，比 Gauss-Newton Hessian 快约 160×

2. **LoRA-MoE 专家分配**:

    - 功能：将固定数量的专家按层质量非均匀分配
    - 映射步骤：(a) 符号翻转 $\tilde{v}_i = -v_i$ 使高质量层得分更高；(b) 幂变换 $\hat{v}_i = \tilde{v}_i^\beta$ 控制分配不均匀程度；(c) 归一化到目标总专家数 $T$，保证每层至少1个专家
    - 最终分配：$s_i = \lfloor f_i \rfloor + 1$，余量按小数部分分配给 top-$r$ 层
    - 路由策略：Top-K 路由 + softmax 门控 + load balancing loss
    - 每层输出：$h^{i,t} = W_0^{i,t}x + \sum_{j=1}^K S_j^{i,t}(x) A_j^{i,t} B_j^{i,t} x$

3. **层级稀疏剪枝分配**:

    - 功能：为不同层分配不同的剪枝比率（训练不足的层多剪，训练充分的层少剪）
    - 预处理：取绝对值 → min-max 归一化到 [0,1] → Savitzky-Golay 滤波平滑噪声
    - 映射函数：$\phi(\tilde{s})_i = \eta \left(\frac{\tilde{s}_i - \tilde{s}_{\min}}{\tilde{s}_{\max} - \tilde{s}_{\min}}(e_2 - e_1) + e_1\right)$
    - 约束：全局稀疏率 $\sum_{i=1}^L \phi(\tilde{s})_i \cdot d_i = S \cdot \sum_{i=1}^L d_i$
    - 适用于 Magnitude、Wanda、SparseGPT 三种主流剪枝方法

### 设计特点
- **模型无关**：适用于任何 Transformer 架构
- **任务特异性**：同一模型对不同数据集产生不同的层级分配（通过热力图可视化验证）
- **Per-block 而非 Per-matrix**：消融实验表明在整个 Transformer 块粒度计算比按 Attention/MLP 分别计算效果更好（+2.14%准确率）

## 实验关键数据

### LoRA-MoE 专家分配（Mistral-7B，160个专家）

| 方法 | MRPC | CoLA | TextSciQ | CommonQ | OpenBookQ | Avg |
|------|------|------|----------|---------|-----------|-----|
| AlphaLora | 75.30 | 82.07 | 78.52 | 82.06 | 84.60 | 80.51 |
| MoLA-5555 | 82.02 | 82.26 | 64.65 | 80.91 | 64.00 | 74.77 |
| MoLA-8642 | 80.70 | 84.56 | 76.39 | 81.00 | 82.20 | 80.97 |
| **LayerIF (Top 25%)** | **82.63** | **83.80** | **79.59** | 80.75 | 84.60 | **82.27** |

LayerIF 最佳配置平均准确率 82.27%，比 MoLA-8642 (+1.30%) 和 AlphaLora (+1.76%) 都有显著提升。

### 层稀疏剪枝（Mistral-7B，50%稀疏率）

| 方法 | Magnitude | Wanda | SparseGPT | Avg |
|------|-----------|-------|-----------|-----|
| Uniform | 56.41 | 58.71 | 60.45 | 58.52 |
| AlphaPruning | 56.74 | 58.48 | 60.07 | 58.43 |
| OWL | 56.16 | 58.75 | 59.99 | 58.30 |
| **LayerIF** | **56.89** | **58.94** | **60.61** | **58.81** |

LayerIF 在所有三种剪枝方法上都取得最优，平均提升 0.90%。

### 消融：反转分配策略（70%稀疏率，Mistral-7B）

| 方法 | Magnitude | Wanda | SparseGPT | Avg |
|------|-----------|-------|-----------|-----|
| LayerIF (正常) | 33.44 | 32.50 | 41.14 | 35.69 |
| 反转分配 | 33.10 | 32.02 | 38.45 | 34.52 |

反转后性能明显下降，验证了"低 IF 敏感度 = 训练成熟 = 应多保留"的假设。

### 关键发现
- LayerIF 的层级分配在不同数据集间差异显著（热力图可视化），验证了任务特异性
- 在 80、160、224 三种专家总数设置下均保持领先，鲁棒性好
- Per-block 优于 Per-matrix 计算（+2.14%）
- Hessian-free 方法（TracIn）与 Hessian-based 方法（DataInf）的 Spearman 相关性低（平均约0.15），说明 Hessian 信息对层质量估计至关重要
- Top 25% 正向影响样本是最优聚合策略，因为 IF 值呈重尾分布

## 亮点与洞察
- **数据+模型双视角**：首次将影响函数用于层质量估计，弥补了纯模型启发式方法忽略数据影响的缺陷。这种思路可以推广到任何需要量化模型子组件重要性的场景
- **任务自适应分配**：同一模型不同任务产生不同层级分配，比固定模式（如 MoLA 的 2-4-6-8）更合理。热力图对比非常直观
- **通用框架设计**：LayerIF 与具体 IF 方法解耦（DataInf 可替换为未来更高效的 IF 方法），与剪枝方法也解耦
- **Savitzky-Golay 平滑的实用价值**：IF 分数在相邻层间因 Hessian 近似噪声而波动大，简单平滑就能显著提升稳定性

## 局限与展望
- IF 计算开销较大（需要 Hessian 近似），虽然文中强调相对预训练时间可忽略，但具体 wall-clock 时间仍值得关注
- 仅在 7B 规模模型上验证，更大模型（70B+）的可扩展性存疑
- 超参数 $\beta$（幂变换指数）需要调优，论文未给出自动选择方法
- Layer-wise 粒度可能过粗，更细粒度（如 per-head 或 per-neuron）可能有更好的效果
- 仅验证了 NLP 任务，视觉/多模态模型待探索

## 相关工作与启发
- **vs AlphaPruning**: 基于重尾谱密度 (PL_Alpha_Hill) 估计层质量，纯模型侧；LayerIF 加入数据视角，在剪枝任务上持续领先 0.90%
- **vs OWL**: 基于异常激活分布做非均匀稀疏分配；LayerIF 在所有稀疏率下均领先
- **vs MoLA/AlphaLora**: MoLA 用固定模式分配专家，AlphaLora 用重尾理论自适应但不考虑数据；LayerIF 的数据驱动分配在不同数据集上自动调整
- **vs Shapley Value 方法**: Shapley Value 也能估计层贡献，但计算成本更高（指数级采样）；LayerIF 通过 IF 的梯度近似更高效

## 评分
- 新颖性: ⭐⭐⭐⭐ 影响函数用于层质量估计是有意义的新方向，但 IF 方法本身是已有工具
- 实验充分度: ⭐⭐⭐⭐ 两个下游任务、两个模型、多种剪枝方法、消融+对比齐全，但仅7B规模
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，pipeline 图直观，但部分公式较冗长
- 价值: ⭐⭐⭐⭐ 为 MoE 专家分配和剪枝提供了新的数据驱动视角，实用性明确

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)
- [\[NeurIPS 2025\] Correlation Dimension of Auto-Regressive Large Language Models](correlation_dimension_of_auto-regressive_large_language_models.md)
- [\[NeurIPS 2025\] Restoring Pruned Large Language Models via Lost Component Compensation](restoring_pruned_large_language_models_via_lost_component_compensation.md)
- [\[NeurIPS 2025\] The Structure of Relation Decoding Linear Operators in Large Language Models](the_structure_of_relation_decoding_linear_operators_in_large_language_models.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)

<!-- RELATED:END -->
