---
title: >-
  [论文解读] Optimizer Choice Matters for the Emergence of Neural Collapse
description: >-
  [ICLR 2026][Neural Collapse] 通过 3,900+ 次训练实验和理论分析，揭示了优化器选择（特别是权重衰减的耦合方式）对 Neural Collapse 现象涌现起关键决定性作用——AdamW（解耦权重衰减）无法产生 Neural Collapse，而 SGD 和 Adam（耦合权重衰减）可以。
tags:
  - ICLR 2026
  - Neural Collapse
  - 优化器选择
  - 权重衰减耦合
  - AdamW vs Adam
  - 隐式偏置
---

# Optimizer Choice Matters for the Emergence of Neural Collapse

**会议**: ICLR 2026  
**arXiv**: [2602.16642](https://arxiv.org/abs/2602.16642)  
**代码**: 无  
**领域**: 优化理论 / 深度学习理论  
**关键词**: Neural Collapse, 优化器选择, 权重衰减耦合, AdamW vs Adam, 隐式偏置

## 一句话总结

通过 3,900+ 次训练实验和理论分析，揭示了优化器选择（特别是权重衰减的耦合方式）对 Neural Collapse 现象涌现起关键决定性作用——AdamW（解耦权重衰减）无法产生 Neural Collapse，而 SGD 和 Adam（耦合权重衰减）可以。

## 研究背景与动机

Neural Collapse (NC) 是 Papyan et al. (2020) 发现的深度网络训练末期现象：最后一层特征向量和分类器权重会自组织成高度对称的几何结构。NC 包含四个性质：
- **NC1**：类内变异性消失（特征坍缩到类均值）
- **NC2**：类中心收敛到 Simplex ETF（等角紧框架）
- **NC3**：分类器权重与类均值对齐（Self-Duality）
- **NC4**：分类简化为最近类中心分类器

已有的理论分析大多忽略了优化器的角色，暗示 NC 对所有优化方法是普遍的。本文挑战这一假设，证明优化器的选择——特别是权重衰减的耦合方式——对 NC 的涌现至关重要。一个关键发现是：**Adam 能产生 NC，但算法上非常相似的 AdamW 却不能**。

## 方法详解

### 整体框架

本文在理论和实验两个层面研究优化器对 NC 的影响：
1. 引入新的诊断指标 NC0
2. 对 SGD 和 SignGD（Adam/AdamW 的特例）进行理论分析
3. 大规模实验（3,900+ 训练运行）验证理论

### 关键设计

1. **NC0 诊断指标**：

    - 定义：$\alpha_t = \frac{1}{K}\|W_t^\top \mathbf{1}\|_2^2$，即最后一层权重矩阵行和的平方范数
    - 核心性质：NC0 收敛到零是 NC2 和 NC3 成立的**必要条件**（Proposition 2.1）
    - 优势：比原始 NC 指标更易追踪和理论分析；若 NC0 发散，可以确定性地断言 NC 不可能发生

2. **权重衰减耦合 vs 解耦**：

    - **耦合权重衰减**（如 SGD/Adam）：$V_{t+1} = \beta V_t + \nabla L_{CE} + \lambda W_t$，衰减项在梯度内部
    - **解耦权重衰减**（如 SGDW/AdamW）：$W_{t+1} = (1-\eta\lambda)W_t - \eta V_{t+1}$，衰减项直接作用于参数
    - 对于 vanilla SGD 两者等价，但对自适应优化器（如 Adam）两者**不等价**

3. **理论定理**：

    - **Theorem 3.1** (SGD + 解耦 WD)：NC0 以指数速率收敛到零，速率与 $\lambda$ 成正比
    - **Theorem 3.2** (SGD + 耦合 WD)：NC0 以指数速率收敛到零，速率与 $\lambda$ 和 $\beta$ 成正比
    - **Theorem 3.3** (SignGD + 解耦 WD，即 AdamW 特例)：NC0 **单调递增**到正常数 $\frac{(K-2)^2}{\lambda^2}$，**不收敛到零**
    - **Theorem 3.4** (SignGD + 耦合 WD，即 Adam 特例)：在学习率衰减策略下，NC0 可以收敛到零

   理论关键洞察：交叉熵损失梯度的行和 $\nabla L_{CE}(W_t)^\top \mathbf{1}_K = 0$ 为零，这使得 NC0 的动力学仅依赖于权重衰减和动量。

### 损失函数 / 训练策略

- 实验统一使用交叉熵损失 + L2 正则化
- ResNet9 和 VGG9 架构
- MNIST、FashionMNIST、CIFAR10 数据集
- 6 个优化器：Adam, AdamW, SGD, SGDW, Signum, SignumW
- 3 个学习率 × 6 个动量值 × 6 个权重衰减值 = 108 种超参数组合/优化器
- 200 epochs，batch size 128，学习率在 1/3 和 2/3 处衰减 10 倍

## 实验关键数据

### 主实验

ResNet9 在 FashionMNIST 上的最终 NC 指标（越低越好）：

| 优化器 | NC0↓ | NC1↓ | NC2↓ | NC3↓ |
|--------|------|------|------|------|
| SGD | 2.14e-04 (<-99.5%) | 0.05 (-99.3%) | 0.29 (-63.0%) | 0.35 (-75.1%) |
| Adam | 0.34 (-80.6%) | 0.04 (-99.5%) | 0.29 (-63.9%) | 0.29 (-79.5%) |
| AdamW | **5.33 (>100%)** | 0.20 (-97.2%) | 0.54 (-32.4%) | 0.78 (-45.2%) |
| SGDW | 0.55 (-68.9%) | 0.26 (-96.3%) | 0.46 (-42.4%) | 0.80 (-43.5%) |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Adam vs AdamW 插值 | 随耦合 WD 增加，NC0/NC2/NC3 平滑改善 | 验证准确率基本不变 |
| 动量对 NC 的加速 | 相同训练损失下，mom=0.9 比 0.7 的 NC 指标明显更低 | 动量的 NC 加速效应超越了对训练损失的加速 |
| 最优 NC3 的超参数 | SGD NC3=0.13, AdamW NC3=0.49 | SGD 在所有优化器中实现最强 NC |

### 关键发现

1. **耦合权重衰减是自适应优化器产生 NC 的必要条件**：AdamW/SignumW 的 NC 指标始终远高于 Adam/Signum，即使权重衰减高几个数量级也无法改善
2. **动量加速 NC 但不仅仅加速收敛**：两个相同训练损失但不同动量的 SGD 运行会到达几何结构截然不同的解
3. **SGD 的 NC 行为对耦合/解耦不敏感**：SGD 和 SGDW 的 NC 指标差距较小，与理论一致
4. **部分 Neural Collapse**：AdamW 可以在 NC1、NC2 上取得最优值，同时 NC0 发散、NC3 不满足——NC 性质不一定同时出现
5. **NC4 是冗余的**：只要训练准确率接近 100%，NC4 总是满足，与其他 NC 指标不相关

## 亮点与洞察

- **提出了 NC0 这一新的诊断指标**：收敛到零是 NC 的必要条件，比原始指标更易追踪和分析
- **挑战了 NC 的普遍性假设**：证明优化器选择决定性地影响 NC 是否涌现
- **揭示了被忽视的细微差异**：Adam 和 AdamW 之间看似微小的权重衰减耦合方式差异，导致截然不同的表征几何
- **NC 不一定意味着更好的泛化**：所有优化器都能达到相似的验证准确率，但 NC 强度差异显著——这限制了用 NC 来理解泛化
- **规模宏大的实验**：3,900+ 训练运行，系统性地控制变量

## 局限与展望

1. **理论分析限于简化设定**：Theorem 3.3/3.4 基于 UFM（无约束特征模型）中的 SignGD，未完全捕捉深度网络和自适应优化器的复杂性
2. **仅分析 NC0**：完整理解 NC1-NC3 在现实优化动力学下的行为仍是开放问题
3. **仅限最后一层**：未分析中间层的 NC 性质（已有工作表明中间层也可能出现 NC）
4. **未覆盖新型优化器**：Lion, MARS, Shampoo, SOAP, Muon 等新优化器的 NC 行为有待探索
5. **需要扩展到更大模型**：ViT 和 DenseNet 等更大架构的实验有限（附录中有初步 ViT 结果）

## 相关工作与启发

- Papyan et al. (2020) 首次发现 NC 现象
- Pan & Cao (2024), Jacot et al. (2024) 研究了权重衰减对 NC 的影响，但未区分耦合/解耦
- Loshchilov & Hutter (2019) 提出 AdamW，但其在 NC 上下文中的影响此前被忽视
- **启发**：优化器不仅影响收敛速度，还决定性地影响学习到的表征的几何结构——优化器选择是一种隐式的归纳偏置

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ （首次揭示优化器依赖的 NC 涌现，提出 NC0 指标）
- 实验充分度: ⭐⭐⭐⭐⭐ （3,900+ 运行，系统性变量控制，多数据集多架构）
- 写作质量: ⭐⭐⭐⭐ （理论和实验结合紧密，结构清晰）
- 价值: ⭐⭐⭐⭐ （对理解深度学习优化和表征几何有重要启示）

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](../../CVPR2026/others/neural_collapse_in_test-time_adaptation.md)
- [\[NeurIPS 2025\] Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry](../../NeurIPS2025/others/reliable_active_learning_from_unreliable_labels_via_neural_collapse_geometry.md)
- [\[AAAI 2026\] The Publication Choice Problem](../../AAAI2026/others/the_publication_choice_problem.md)
- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](../../NeurIPS2025/others/the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [\[ACL 2025\] Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction](../../ACL2025/others/distractor_gen_multiple_choice.md)

<!-- RELATED:END -->
