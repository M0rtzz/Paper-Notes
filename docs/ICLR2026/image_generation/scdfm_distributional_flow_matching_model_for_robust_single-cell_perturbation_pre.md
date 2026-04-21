---
title: >-
  [论文解读] scDFM: Distributional Flow Matching for Robust Single-Cell Perturbation Prediction
description: >-
  [ICLR 2026][图像生成][单细胞扰动预测] 提出 scDFM，基于条件流匹配（CFM）的生成式框架，通过 MMD 正则化保证分布级保真度，并设计 PAD-Transformer 骨干处理噪声稀疏的单细胞数据，在组合扰动预测上比最强基线 CellFlow 的 MSE 降低 19.6%。
tags:
  - ICLR 2026
  - 图像生成
  - 单细胞扰动预测
  - 条件流匹配
  - MMD正则化
  - 差分注意力
  - 基因共表达图
---

# scDFM: Distributional Flow Matching for Robust Single-Cell Perturbation Prediction

**会议**: ICLR 2026  
**arXiv**: [2602.07103](https://arxiv.org/abs/2602.07103)  
**代码**: [GitHub](https://github.com/AI4Science-WestlakeU/scDFM)  
**领域**: 医学图像（计算生物学）  
**关键词**: 单细胞扰动预测, 条件流匹配, MMD正则化, 差分注意力, 基因共表达图

## 一句话总结
提出 scDFM，基于条件流匹配（CFM）的生成式框架，通过 MMD 正则化保证分布级保真度，并设计 PAD-Transformer 骨干处理噪声稀疏的单细胞数据，在组合扰动预测上比最强基线 CellFlow 的 MSE 降低 19.6%。

## 研究背景与动机
- 预测细胞在基因/药物扰动后的转录组响应是系统生物学和药物发现的核心挑战
- 由于 RNA 测序的破坏性本质，无法观察同一细胞扰动前后的状态（未配对数据）
- 现有方法（CPA、GEARS 等）主要关注均值表达谱，忽略了更高阶的分布统计量（方差、偏度、亚群比例变化）
- 单细胞数据稀疏、零膨胀、噪声严重，基因间存在复杂调控网络但大多数模型将基因视为独立特征
- **核心动机**：需要一个能建模完整分布变化、同时鲁棒处理噪声和稀疏性的生成框架

## 方法详解

### 整体框架
scDFM 建立在条件流匹配（CFM）之上，学习一个时间依赖的速度场 $v_\theta(x_t | t, c_x, c_p)$，将噪声源分布变换为扰动后的基因表达分布。训练结合 CFM 损失和多核 MMD 正则器，骨干网络为 PAD-Transformer。

### 关键设计

1. **条件流匹配 (CFM)**:

    - 在高维基因表达空间中直接应用 FM 框架（首次尝试）
    - 源分布 $x_0$ 为噪声基因表达，目标分布 $x_1$ 为扰动后表达
    - 线性插值路径：$\pi_t(x_0, x_1) = (1-t)x_0 + tx_1$
    - 训练目标：$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}[\|v_\theta(x_t | t, c_x, c_p) - v(x_t | x_0, x_1, t, c_x, c_p)\|_2^2]$
    - **动机**：FM 直接学习条件变换，适合从噪声中间态到真实扰动态的映射

2. **多核 MMD 正则化**:

    - CFM 仅保证局部动态一致性，不保证终端分布对齐
    - 引入 MMD 直接比较生成分布 $\hat{X}_1$ 与真实扰动分布 $X_1$
    - 混合高斯 RBF 核：$k_{\text{mix}}(x, x') = \frac{1}{L}\sum_{\ell=1}^L \exp(-\frac{\|x-x'\|^2}{2\sigma_\ell^2})$
    - 一步预测端点：$\hat{x}_1 = x_t + (1-t) \cdot v_\theta(x_t | t, c_x, c_p)$
    - 最终目标：$\mathcal{L} = \mathcal{L}_{\text{CFM}} + \lambda \mathcal{L}_{\text{MMD}}$
    - **动机**：弥补 CFM 在全局分布对齐上的不足，确保群体水平保真度

3. **PAD-Transformer (Perturbation-Aware Differential Transformer)**:

    - **基因共表达图注意力掩码**：基于 Pearson 相关系数 $w_{ij} = |\text{Cov}(x_i, x_j) / (\sigma(x_i)\sigma(x_j))|$ 构建 KNN 图，约束注意力仅在生物学相关基因间计算
    - **差分注意力模块**：$\alpha_{\text{diff}} = A_1 - \lambda A_2$，抑制噪声基因的无关注意力
    - **每层扰动注入**：将扰动嵌入 $e_p$ 在每一层通过 MLP adapter 注入
    - 三步精炼：扰动注入 → 自差分注意力 → 跨差分注意力（用控制表示 $h_c$ 指导扰动态精炼）
    - **动机**：标准 Transformer 容易过度关注噪声 token；差分注意力可区分控制态和扰动态信号

### 损失函数 / 训练策略
- 总损失：$\mathcal{L} = \mathcal{L}_{\text{CFM}} + \lambda \mathcal{L}_{\text{MMD}}$，$\lambda > 0$ 平衡轨迹一致性和终端分布保真度
- MMD 带宽通过 median heuristic 自适应选择
- 时间步 $t$ 使用正弦余弦嵌入 + MLP，提供 adaLN-Zero 调制

## 实验关键数据

### 主实验（Norman Additive Split）

| 模型 | MSE ↓ | MAE ↓ | DE-Spearman ↑ | DS ↑ | Pearson $\hat{\Delta}_{20}$ ↑ |
|------|-------|-------|---------------|------|------|
| scDFM (Ours) | **0.00315** | **0.02155** | **0.5705** | **0.9737** | **0.9260** |
| CellFlow | 0.00392 | 0.02207 | 0.5503 | 0.9321 | 0.8988 |
| GEARS | 0.01387 | 0.06624 | 0.5624 | 0.8601 | 0.2032 |
| scGPT | 0.01349 | 0.03796 | 1.07e-5 | 0.5404 | 0.2414 |
| CPA | 0.03435 | 0.07894 | 0.0713 | 0.6021 | 0.2254 |

### 消融实验

| 配置 | 关键指标变化 | 说明 |
|------|---------|------|
| w/o MMD | MSE 上升, DS 下降 | MMD 对分布级保真度至关重要 |
| w/o 基因共表达图 | DE-Spearman 下降 | 生物先验引导注意力有效 |
| w/o 差分注意力 | 噪声敏感度增加 | 差分注意机制抑制噪声 |
| 标准 Transformer 替代 PAD | 全面下降 | PAD-Transformer 各组件互补 |

### 关键发现
- scDFM 比 CellFlow 的 MSE 降低 19.6%（0.00315 vs 0.00392），同时判别得分（DS）达到 0.9737
- 在 Holdout 设置（未见过的扰动）下同样表现优异，验证了泛化能力
- scGPT 等预训练模型在 DE-Spearman 上几乎为零，表明基础模型难以捕捉扰动特异性效应
- Additive 基线本身就有竞争力（与 Ahlmann-Eltze 一致），说明组合扰动常具有近似加性效应

## 亮点与洞察
- 首次在高维基因表达空间中直接应用条件流匹配，相比在 PCA 空间操作的 CellFlow 更直接
- MMD 正则化巧妙弥补了 CFM 只保证局部一致性的缺陷，实现局部（轨迹）+ 全局（分布）双保真
- 基因共表达图作为生物学先验注入注意力掩码，有效过滤噪声、保留调控结构
- 差分注意力机制对噪声生物数据特别适用——只有部分基因响应扰动，其余应被抑制

## 局限与展望
- 仅在 Norman（基因扰动）和 ComboSciPlex（药物扰动）两个数据集验证
- 需要预先计算基因共表达图，增加了数据准备的计算开销
- 分布级评估指标（DS）虽有用但对实际生物意义的反映不够直接
- 未与最新的 diffusion-based 方法（scDiffusion）进行详细对比

## 相关工作与启发
- CellFlow (Klein et al. 2025)：在 PCA 空间做流匹配，scDFM 在原始表达空间操作
- GEARS (Roohani et al. 2024)：引入基因本体等生物先验，scDFM 使用共表达图
- Diff Transformer (Ye et al. 2025)：差分注意力原始提出，scDFM 将其适配到扰动预测

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ CFM + MMD + PAD-Transformer 的组合很创新且设计合理
- 实验充分度: ⭐⭐⭐⭐ 多设置评估、多指标覆盖、有消融
- 写作质量: ⭐⭐⭐⭐ 技术描述清晰，动机充分
- 价值: ⭐⭐⭐⭐⭐ 对计算生物学有重要价值，代码开源

<!-- RELATED:START -->

## 相关论文

- [RenderFlow: Single-Step Neural Rendering via Flow Matching](../../CVPR2026/image_generation/renderflow_single-step_neural_rendering_via_flow_matching.md)
- [Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)
- [SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning](flow-based_single-step_completion_for_efficient_and_expressive_policy_learning.md)
- [FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching](flowcast_advancing_precipitation_nowcasting_with_conditional_flow_matching.md)
- [Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)

<!-- RELATED:END -->
