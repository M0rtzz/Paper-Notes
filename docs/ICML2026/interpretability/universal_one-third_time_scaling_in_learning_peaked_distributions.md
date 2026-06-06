---
title: >-
  [论文解读] 学习尖峰分布中的通用 1/3 时间缩放
description: >-
  [ICML 2026][可解释性][神经缩放律] 通过分析 softmax 与交叉熵在学习峰值概率分布时的数学性质，论文揭示了 LLM 训练损失呈现通用 1/3 幂律衰减的根本原因——这是一个与数据结构无关的架构层面的优化瓶颈。
tags:
  - "ICML 2026"
  - "可解释性"
  - "神经缩放律"
  - "时间缩放指数"
  - "幂律收敛"
  - "softmax-交叉熵"
  - "峰值分布"
---

# 学习尖峰分布中的通用 1/3 时间缩放

**会议**: ICML 2026  
**arXiv**: [2602.03685](https://arxiv.org/abs/2602.03685)  
**代码**: https://github.com/liuyz0/TimeScaling  
**领域**: 可解释性 / 模型缩放规律 / 优化动力学  
**关键词**: 神经缩放律, 时间缩放指数, 幂律收敛, softmax-交叉熵, 峰值分布

## 一句话总结
通过分析 softmax 与交叉熵在学习峰值概率分布时的数学性质，论文揭示了 LLM 训练损失呈现通用 1/3 幂律衰减的根本原因——这是一个与数据结构无关的架构层面的优化瓶颈。

## 研究背景与动机

**领域现状**：神经缩放律（如 Chinchilla 定律）通过实验观察到 LLM 训练损失随时间/数据量按幂律衰减，但这一现象的根本机制仍未明确。现有理论多数归因于数据中的幂律结构或特征频率分布。

**现有痛点**：为什么指数接近 0.28 而非其他值？为什么这个指数跨越不同模型尺寸保持一致？仅从数据分布难以解释这种"通用性"。此外现有 MSE 损失的理论分析无法捕捉 softmax 和交叉熵的特殊非线性。

**核心矛盾**：一方面 LLM 必须输出"尖峰"的下一个词分布（低熵）以准确预测；另一方面 softmax 在低温（logit 方差大）时产生幂律衰减的损失和梯度——这两者的结合形成基本的优化瓶颈。

**本文目标**：找出 LLM 训练缓慢的架构根源而非数据根源，推导通用幂律指数。

**切入角度**：从统计物理学中的"通用性"概念出发，最小化模型（单层 softmax+交叉熵），通过低温展开精确求解，而后验证在实际 LLM 上的适用性。

**核心 idea**：softmax 和交叉熵损失在学习峰值分布时必然导致 $L \sim \tau^{-1/3}$ 的幂律衰减，与具体数据结构无关。

## 方法详解

### 整体框架
"最小模型→理论分析→LLM 验证"的三层递进架构。首先构造教师-学生模型，证明 softmax+交叉熵生成幂律；其次用低温展开精确求导幂律指数；最后在 Pythia/OLMo 真实模型上验证预测。

### 关键设计

1. **对齐学生假说（Aligned Student Ansatz）**:

    - 功能：通过观察权重演化轨迹，假设学生权重始终与教师权重方向一致，只有范数变化。
    - 核心思路：由于零初始化和小学习率，学生权重在早期可快速对齐方向，后续专注于范数增长。这将复杂的非线性动力学简化为单变量 $\beta$（逆温度）的演化：$\frac{d\beta}{d\tau} = -\frac{c_{\text{eff}}}{n}\frac{dL(\beta)}{d\beta}$。
    - 设计动机：验证显示即使初始错位，学生也会逐步对齐，最终进入 $\beta \sim \tau^{1/3}$ 体制。

2. **低温展开与自由能分析**:

    - 功能：在峰值分布下（$\beta \gg c_0 = \sqrt{2\ln n}$），用泰勒级数展开自由能 $F(\beta)$ 和内能 $U(\beta)$。
    - 核心思路：$F(\beta) = -c_0 - c_1\beta^{-1} - c_2\beta^{-2} + \cdots$，导出 $L \approx c_2\beta^{-1}$ 和 $-\frac{dL}{d\beta} \approx c_2\beta^{-2}$，代入梯度流获得 $\beta \sim \tau^{1/3}$，因而 $L \sim \tau^{-1/3}$。
    - 设计动机：这是通用性的根源——展开系数与能量分布形式无关，只需低温条件，因此指数 1/3 对所有峰值分布都成立。

3. **动态时间 $\tau$ 的精确映射**:

    - 功能：将 Adam 等自适应优化器的复杂学习率调度统一为单一动态时间 $\tau = \int_0^t \eta_{t'} dt'$。
    - 核心思路：不同学习率的训练曲线在 $(L, \tau)$ 平面上重合，说明 $\tau$ 而非步数 $t$ 是基本变量。
    - 设计动机：解释为何 Chinchilla 缩放律（用数据量）的指数 0.28 略小于 1/3，根本原因是学习率调度与数据量非线性关系。

## 实验关键数据

### 主实验：玩具模型验证

| 逆温度范围 | 损失衰减类型 | 拟合指数 | 说明 |
|-----------|-----------|--------|------|
| 高温（小 $\beta^*$）| 指数衰减 | N/A | 非幂律 |
| 中温（$c_0 < \beta < \beta^*$） | 幂律 | $-1/3$ | 理论预测区间 |
| 低温（$\beta \approx \beta^*$） | 饱和 | N/A | 学生接近收敛 |

### 消融与因素分析

| 配置 | 观察现象 | 结论 |
|------|--------|------|
| 固定学习率，扫描 $\beta^*$ | 高 $\beta^*$ 时幂律更明显 | 峰值分布触发 1/3 缩放 |
| 权重衰减+低学习率 | 损失仍为 $\tau^{-1/3}$ 但 $\beta$ 不增长 | 参数旋转也能产生幂律 |
| 不同初始化比例 | 最终都对齐进入 1/3 体制 | 对齐学生假说的鲁棒性 |

### 关键发现
- LLM 验证：在 Pythia 上拟合 $L = \frac{c_\tau}{\tau^{\alpha_\tau}} + L_{\backslash\tau}$，得 $\alpha_\tau \approx 0.33 \pm 0.02$（理论值 1/3）。
- 不同模型尺寸的曲线在 $\tau$ 坐标下重合，表明通用性。
- Logit 标准差增长指数为 0.38≈1/3，验证了低温进入阶段。

## 亮点与洞察
- **架构为根源**：首次严格证明 softmax+交叉熵本身（而非数据）导致幂律——优化层面的通用物理规律。
- **低温展开的威力**：通过泰勒展开和极值分布理论从复杂非线性系统中解析提取幂律指数。
- **动态 vs 静态视角的统一**：证明 Chinchilla 的静态缩放律实际反映了动态时间缩放（$\tau^{-1/3}$）。

## 局限与展望
- 理论局限——假设梯度流动力学，未考虑有限批量噪声和大学习率的影响；对齐学生假说在参数大幅偏离时失效。
- 实验局限——仅在 Pythia/OLMo 验证，样本量有限；logit 并非严格 i.i.d. Gaussian。
- 改进方向：设计对参数旋转敏感的优化器以打破 1/3 瓶颈；探索降低 logit 熵的架构（如词表分层）。

## 相关工作与启发
- **vs 数据结构理论（Bordelon et al. 2024）**：本文证明即使无幂律数据，softmax+交叉熵也产生幂律。
- **vs MSE 损失理论**：MSE 不产生幂律，突出了 softmax 的特殊角色。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐  首次揭示 softmax-交叉熵与幂律的因果关系，颠覆"幂律源于数据"的共识。
- 实验充分度: ⭐⭐⭐⭐  玩具模型完整，LLM 验证缺少更多规模。
- 写作质量: ⭐⭐⭐⭐⭐  逻辑链条清晰，物理直觉到位。
- 价值: ⭐⭐⭐⭐⭐  为 LLM 缩放律提供可行的改进方向。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](block-em_preventing_emergent_misalignment_via_latent_blocking.md)
- [\[ICML 2026\] Verified SHAP: 神经网络精确 Shapley 值的可证明界](verified_shap_provable_bounds_for_exact_shapley_values_of_neural_networks.md)
- [\[ICML 2026\] Courtroom Analogy: New Perspective on Uncertainty-Aware Classification](courtroom_analogy_new_perspective_on_uncertainty-aware_classification.md)
- [\[ICML 2026\] Interpretable Self-Supervised Learning via Representer Landmarks and Nyström Approximation](interpretable_self-supervised_learning_via_representer_landmarks_and_nyström_app.md)
- [\[ICML 2026\] Finding the Correct Visual Evidence Without Forgetting: Mitigating Hallucination in LVLMs via Inter-Layer Visual Attention Discrepancy](finding_the_correct_visual_evidence_without_forgetting_mitigating_hallucination_.md)

</div>

<!-- RELATED:END -->
