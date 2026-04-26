---
title: >-
  [论文解读] GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding
description: >-
  [NeurIPS 2025][时空因果推断] 提出 GST-UNet，将 U-Net 时空编码器与迭代 G-computation 相结合，从**单条时空观测轨迹**中估计位置特异性的条件平均潜在结果 (CAPO)，可同时处理干扰（interference）、空间混杂、时间延续和时变混杂，并在加州山火烟雾对呼吸系统住院率的因果分析中验证了实用价值。
tags:
  - NeurIPS 2025
  - 时空因果推断
  - G-computation
  - UNet
  - 时变混杂
  - 潜在结果
---

# GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding

**会议**: NeurIPS 2025  
**arXiv**: [2502.05295](https://arxiv.org/abs/2502.05295)  
**代码**: [moprescu/GSTUNet](https://github.com/moprescu/GSTUNet)  
**领域**: causal_inference  
**关键词**: 时空因果推断, G-computation, UNet, 时变混杂, 潜在结果

## 一句话总结
提出 GST-UNet，将 U-Net 时空编码器与迭代 G-computation 相结合，从**单条时空观测轨迹**中估计位置特异性的条件平均潜在结果 (CAPO)，可同时处理干扰（interference）、空间混杂、时间延续和时变混杂，并在加州山火烟雾对呼吸系统住院率的因果分析中验证了实用价值。

## 研究背景与动机
**领域现状**: 时空观测数据中的因果效应估计在公共卫生、环境科学、政策评估中至关重要，但随机实验通常不可行，需要从观测数据中做因果推断。

**现有痛点**:
   - **经典方法**（DID, Synthetic Control, MSM）依赖平行趋势、无干扰等强假设，无法处理空间溢出效应
   - **深度学习预测模型**（CNN/RNN/Transformer）虽能捕获复杂时空模式，但不做因果调整，无法估计反事实
   - **纵向因果推断方法**（MSM, G-computation, 基于 RNN/Transformer 的扩展）假设独立时间序列（如不同患者），无法建模跨单元交互
   - **现有时空因果模型**（如 Tec et al.）仅处理静态暴露，不处理时变混杂和干扰

**核心矛盾**: 时空数据中同时存在 ①空间干扰（邻域的干预影响本地结果）②空间混杂 ③时间延续效应 ④时变混杂（协变量受过去干预影响又影响未来干预），且往往只有**单条时空轨迹**可用

**本文目标**: 在单条时空轨迹下，同时处理上述四个挑战，给出有理论保证的因果效应估计

**切入角度**: 将经典的迭代 G-computation（递归回归消除时变混杂）与 U-Net 时空表征学习结合，用 representation-based time invariance 假设实现从单轨迹中池化训练样本

**核心 idea**: 学一个时不变的历史嵌入 ϕ(H₁:t, At)，使得条件于该嵌入后转移分布与时间无关，从而将单轨迹切分为可交换的前缀片段，再用迭代 G-computation 递归地消除时变混杂

## 方法详解

### 整体框架
1. **数据结构**: NX×NY 空间网格上每个位置 s 在每个时刻 t 有 (Xs,t, As,t, Ys,t)，加上静态特征 Vs。目标是估计给定历史 H₁:t 和干预序列 a_{t:t+τ-1} 下的 CAPO: E[Y_{t+τ}[a] | H₁:t]
2. **前缀构造**: 从长度 T 的单条轨迹中，对每个 t ∈ {1,...,T-τ} 提取前缀 P_t^τ = (X_{1:t+τ}, A_{1:t+τ}, Y_{1:t+τ}, V)，获得 T-τ 个重叠但非独立的训练片段
3. **识别 → 估计 → 推断**: 先通过 Assumption 1,2 建立识别结果 (Theorem 1)，再用迭代 G-computation 进行估计 (Theorem 2)，最后用 GST-UNet 架构实例化

### 关键设计
1. **Representation-Based Time Invariance (Assumption 2)**: 存在嵌入函数 ϕ: H×A → Z，使得条件于 z=ϕ(H₁:t, At) 后，(X_{t+1}, Y_{t+1}) 的分布不依赖于 t。这比经典的平稳性假设更弱——不要求边际分布时不变，仅要求在嵌入后的转移机制时不变。这使得来自不同时间点的前缀可以被视为条件可交换的，从而池化用于回归训练。

2. **迭代 G-Computation (Theorem 1)**: 对 τ≥2 的干预序列，直接条件于历史和干预会引入时变混杂偏差。通过递归定义：

    - Q_τ(H, A) = E[Y_{t+τ} | ϕ(H_{1:t+τ-1}, A_{t+τ-1})]（直接回归观测结果）
    - Q_{k}(H, A) = E[Q_{k+1}(H^a, a_{t+k}) | ϕ(H_{1:t+k-1}, A_{t+k-1})]（递归回归伪结果）
    - 最终 Q₁(h₁:t, at) = E[Y_{t+τ}[a] | H₁:t]
   关键是从 Q_τ 逆向到 Q₁，每一步用前一步的预测作为伪结果（pseudo-outcomes），在替换了干预值的历史上进行前向预测。

3. **U-Net 时空编码器**: 

    - **空间模块**: U-Net 编码器-解码器结构，带 skip connections，逐步下采样再上采样空间网格
    - **时序模块**: 在编码器中集成 ConvLSTM 层，维护跨时间步的隐藏状态，同时通过卷积聚合空间信息；之后拼接静态协变量 V 作为额外通道
    - **注意力门控**: 在解码器中使用 Attention Gates 选择性高亮相关空间区域，精炼 skip connections
    - 输出 d_h × NX × NY 的特征图，捕获干扰、空间混杂和静态特征

4. **Neural Causal Module (G-heads)**: 在 U-Net 特征图上附加 τ 个 G-computation head，每个 Qk 是小型卷积模块或前馈网络。Q_τ 对比真实观测结果监督，Q_{k<τ} 对比前一步生成的伪结果监督（detached forward pass 防止梯度回传）。

### 损失函数 / 训练策略
**联合损失**:
$$\mathcal{L}(\theta; e) = \frac{1}{\tau} \sum_{k=1}^{\tau} \alpha_k^{(e)} \mathcal{L}_k(\theta)$$
其中 $\mathcal{L}_k$ 是每个 G-head 的 MSE 损失，$\alpha_k^{(e)}$ 是 epoch 相关的权重。

**课程训练 (Curriculum Training)**:
- 阶段 p(e) = min{τ, ⌈e/ec⌉}，ec 为课程周期超参
- 初始阶段 (e ≤ ec): 仅训练 Q_τ（有真实监督），α_τ = 1，其余为 0
- 逐步引入: 每过 ec 个 epoch 多激活一个 G-head（从 Q_{τ-1} 向前）
- 最终所有 head 等权 1/τ
- **动机**: 防止早期 Q_τ 不准确时，其伪结果误导前面的 G-head 过拟合噪声

## 实验关键数据

### 合成实验：RMSE（τ=5, 不同混杂强度 β₁）

| 方法 | β₁=0.0 | β₁=0.5 | β₁=1.0 | β₁=1.5 | β₁=2.0 |
|------|--------|--------|--------|--------|--------|
| UNet+ (无因果调整) | **0.28** | 0.36 | 0.54 | 0.71 | 0.81 |
| STCINet | 0.29 | 0.38 | 0.62 | 0.80 | 0.90 |
| IPWUNet | 0.60 | 0.58 | 0.58 | 0.59 | 0.59 |
| GST-UNet w/o Attention | 0.50 | 0.46 | 0.51 | 0.45 | 0.47 |
| GST-UNet w/o Curriculum | 0.69 | 0.64 | 0.63 | 0.61 | 0.61 |
| **GST-UNet** | 0.33 | **0.35** | **0.40** | **0.44** | **0.40** |

### 合成实验：RMSE（τ=10, 不同混杂强度 β₁）

| 方法 | β₁=0.0 | β₁=0.5 | β₁=1.0 | β₁=1.5 | β₁=2.0 |
|------|--------|--------|--------|--------|--------|
| UNet+ | **0.28** | 0.61 | 1.18 | 1.45 | 1.71 |
| STCINet | 0.31 | 0.68 | 1.25 | 1.47 | 1.60 |
| IPWUNet | 0.78 | 0.80 | 0.96 | 1.19 | 1.08 |
| **GST-UNet** | 0.38 | **0.55** | **0.68** | **0.73** | **0.85** |

### 消融实验

| 组件 | 效果 |
|------|------|
| 去掉 Curriculum | τ=5 时 RMSE 增加 50%-70%，尤其在低混杂时劣化严重 |
| 去掉 Attention | RMSE 增加 10%-30%，在局部动态主导时影响较小 |
| 去掉 G-computation (即 UNet+) | 无混杂时最优，但混杂增强后 RMSE 急剧恶化 (0.28→1.71) |
| IPW 替代 G-computation | 因无法校正空间干扰，即使无混杂也有较大偏差 |

### 真实数据：2018 加州 Camp Fire 山火对呼吸系统住院的因果效应

| 方法 | 估计超额住院人次 (10天) | 95% Bootstrap CI | 特点 |
|------|------------------------|-------------------|------|
| **GST-UNet** | ~4,650 (465/天) | [1888, 6535] | 稳定且符合先验知识 |
| UNet+ | ~3,981 | [-899, 5202] | 置信区间包含负值 |
| STCINet | ~88 | [-3077, 3281] | 高度不稳定，接近零 |
| IPWUNet | ~20,500 | - | 不合理的高估（稀有事件支撑不足） |

参考: Letellier et al. 报告 259 例/天超额，但其统计窗口更长（Nov 8-Dec 5）且强度更低。

### 关键发现
- 当无时变混杂时 (β₁=0)，简单的 UNet+ 预测模型即足够，G-computation 反而引入噪声
- 混杂越强，UNet+/STCINet 退化越严重，GST-UNet 的优势越明显（τ=10, β₁=2.0 时 RMSE 降低 21-47%）
- 课程训练对长时域 (τ=10) 尤其关键
- 在真实山火数据中，GST-UNet 是唯一给出合理且稳定估计的方法

## 亮点与洞察
- **理论与实践统一**: 少见地将识别定理 (Theorem 1) + 一致性定理 (Theorem 2) + 神经网络实现完整打通
- **Representation-Based Time Invariance**: 比经典平稳性假设更弱，允许非平稳过程只要存在时不变的嵌入
- **课程训练策略**: 简单但关键——解决了共享编码器 + 递归伪结果的训练不稳定问题
- **单轨迹因果推断**: 在最数据稀缺的设定下（只有一条时空轨迹）仍能工作

## 局限与展望
- **Assumption 2 的验证**: Representation-Based Time Invariance 难以在实践中直接验证，且学到的 ϕ 是否真正满足此假设缺乏诊断工具
- **计算开销**: τ 个 G-head + 共享编码器的联合训练随 τ 增大计算量显著增加
- **空间分辨率**: 实验中使用 40×44 和 64×64 网格，更高分辨率时 U-Net 的扩展性待验证
- **二值干预**: 当前框架聚焦于二值 treatment，连续 treatment 需要额外适配
- **因果假设**: Sequential Unconfoundedness 在观测数据中不可测试，未测量的混杂仍可能导致偏差

## 相关工作与启发
- **vs UNet+ (Tec et al. 2022)**: 共享 U-Net 架构但仅处理静态暴露+空间混杂，不处理时变混杂和干扰
- **vs STCINet (Ali et al. 2024)**: 估计直接/间接效应但不建模时变混杂，本文的 G-computation 更原则性
- **vs 纵向因果方法 (Bica et al., Melnychuk et al.)**: 假设独立时间序列，无法处理空间溢出；GST-UNet 通过卷积编码器自然捕获空间依赖
- **vs IPW 方法**: 在稀有事件 / 空间干扰下估计不稳定，GST-UNet 的回归方法更鲁棒
- **启发**: 迭代 G-computation 与深度学习的结合范式可推广到其他结构化因果推断问题（如图因果推断）

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个将 G-computation 与时空深度学习端到端结合的框架，理论贡献扎实
- 实验充分度: ⭐⭐⭐⭐ 合成实验控制变量充分，真实数据案例有实际意义；但仅一个真实数据集稍显不足
- 写作质量: ⭐⭐⭐⭐ 理论-实现的衔接清晰，符号系统完整但公式密集
- 价值: ⭐⭐⭐⭐ 对环境健康、政策评估等领域有直接应用价值，框架通用性强

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[ICML 2025\] Causal Abstraction Inference under Lossy Representations](../../ICML2025/causal_inference/causal_abstraction_inference_under_lossy_representations.md)
- [\[ACL 2025\] IRIS: An Iterative and Integrated Framework for Verifiable Causal Discovery](../../ACL2025/causal_inference/iris_an_iterative_and_integrated_framework.md)
- [\[ICML 2025\] Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains](../../ICML2025/causal_inference/learning_time-aware_causal_representation_for_model_generalization_in_evolving_d.md)
- [\[CVPR 2025\] Image Quality Assessment: Investigating Causal Perceptual Effects with Abductive Counterfactual Inference](../../CVPR2025/causal_inference/image_quality_assessment_investigating_causal_perceptual_effects_with_abductive_.md)

<!-- RELATED:END -->
