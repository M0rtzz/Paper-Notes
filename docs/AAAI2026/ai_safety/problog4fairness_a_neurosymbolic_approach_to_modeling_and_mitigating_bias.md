---
title: >-
  [论文解读] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias
description: >-
  [AI安全] 提出 ProbLog4Fairness 框架，利用概率逻辑编程语言 ProbLog 将数据中的偏差机制形式化为可解释的逻辑程序，并通过 DeepProbLog 的远程监督将偏差假设集成到神经网络训练中，实现灵活、原则性的偏差缓解。
tags:
  - AI安全
---

# ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias

- **会议**: AAAI 2026
- **arXiv**: [2511.09768](https://arxiv.org/abs/2511.09768)
- **代码**: 未提供
- **领域**: ai_safety
- **关键词**: 算法公平性, 神经符号AI, 概率逻辑编程, 偏差建模, DeepProbLog

## 一句话总结

提出 ProbLog4Fairness 框架，利用概率逻辑编程语言 ProbLog 将数据中的偏差机制形式化为可解释的逻辑程序，并通过 DeepProbLog 的远程监督将偏差假设集成到神经网络训练中，实现灵活、原则性的偏差缓解。

## 背景与动机

- 算法公平性领域存在大量互不兼容的公平性定义（如统计奇偶性、均等化赔率等），选择哪个约束在规范上存在争议
- 现有方法通常固定一种偏差类型或公平性指标进行优化，缺乏针对具体任务灵活建模不同偏差来源的能力
- 数据生成过程中的多种偏差机制（标签偏差、测量偏差、历史偏差）是模型不公平的重要来源
- 因果模型虽然可以描述偏差机制，但在实践中难以解释和集成到数据管道中
- **核心动机**：与其选择一个固定的公平性约束，不如直接建模并纠正数据中的偏差机制，让模型"自然地"产生更公平的决策

## 方法详解

### 1. ProbLog 偏差建模框架

核心思想是将偏差假设编码为 ProbLog 概率逻辑程序。ProbLog 是 Prolog 的概率扩展，支持概率事实 $p::f$，表示事实 $f$ 以概率 $p$ 为真。

**偏差表示为概率事实**：对于二元变量，偏差转换由四个概率事实完全定义，分别描述：
- $p_1$：敏感群体的负向偏差（将 1 变为 0）
- $p_2$：非敏感群体的负向偏差
- $p_3$：敏感群体的正向偏差（将 0 变为 1）
- $p_4$：非敏感群体的正向偏差

以标签偏差为例，ProbLog 模板为：

$$\tilde{y}(\mathbf{X}) \leftarrow y_h(\mathbf{X}) \wedge \neg \text{label\_neg\_bias}(\mathbf{X})$$
$$\tilde{y}(\mathbf{X}) \leftarrow \neg y_h(\mathbf{X}) \wedge \text{label\_pos\_bias}(\mathbf{X})$$

其中 $y_h(\mathbf{X})$ 是分类器预测的无偏标签，$\tilde{y}(\mathbf{X})$ 是观测到的有偏标签。

### 2. DeepProbLog 集成与远程监督训练

DeepProbLog 允许神经网络预测概率事实的参数，实现在神经网络预测之上的逻辑推理。训练流程：

1. **定义逻辑程序**：包含分类器 $h(\mathbf{X})::y_h(\mathbf{X})$ 和偏差机制的 ProbLog 规则
2. **编译为电路**：逻辑程序被编译为计算查询概率的电路
3. **远程监督**：仅用有偏标签 $\tilde{y}$ 监督，梯度通过逻辑电路反向传播更新网络
4. **推理时剥离**：测试时若有无偏特征，直接用 $h(\mathbf{X})$ 预测；若只有有偏特征，保留偏差转换机制

关键优势是梯度更新考虑了所有与观测有偏数据一致的无偏解释。

### 3. 三种偏差类型的建模

**标签偏差（Label Bias）**：无偏特征可用，但观测标签 $\tilde{Y}$ 是真实标签 $Y$ 的有偏代理。例如贷款审批中审批员对特定族群的歧视。用 Template 1 在训练时建模。

**测量偏差（Measurement Bias）**：观测特征 $\tilde{X}_i$ 是无偏特征 $X_i$ 的噪声代理，标签仅依赖无偏特征。例如用"过去三年工作天数"衡量工作稳定性对产假女性不公。用 Template 2 建模去偏过程：

$$y(\tilde{\mathbf{X}}) \leftarrow \text{debias}(\tilde{\mathbf{X}}, \mathbf{X}) \wedge y_h(\mathbf{X})$$

**历史偏差（Historical Bias）**：特征和标签都受偏差影响，标签基于有偏特征生成。可假设有偏特征到有偏标签的函数与无偏特征到无偏标签的函数相同，在测试时用 Template 2 进行去偏。

### 4. 参数设定

偏差概率参数 $p_i$ 可通过以下方式设定：
- 领域知识直接赋值
- 从同时具有有偏和无偏标签的小规模数据子集中估计
- 基于 Hoeffding 不等式，以 95% 置信度在 10% 误差内估计参数仅需 184 个样本

## 实验

### 合成数据实验

| 实验 | 偏差类型 | 关键发现 |
|------|----------|----------|
| RQ1: 不同偏差类型对比 | 标签/测量/历史 | ProbLog4Fairness 在准确率和统计差异上接近上界基线，显著优于其他缓解方法 |
| RQ2: $A \not\perp Y$ 场景 | 标签偏差 | 能仅移除有问题的偏差，保留合理相关性；其他基线错误地将统计差异强制为零 |
| RQ3: 参数敏感性 | 标签偏差 | 最优准确率在正确参数处取得，且对参数估计误差不敏感 |

### 真实数据实验

| 数据集 | 类型 | 偏差来源 | 敏感变量 | ProbLog4Fairness 结果 |
|--------|------|----------|----------|----------------------|
| Student Alcohol | 表格数据 (856样本) | 标注者偏见（对男生的主观偏差） | 性别 | F1 分数优于所有缓解基线，统计差异接近无偏数据水平 |
| CELEB-A | 图像数据 | 标注不一致（Mouth Slightly Open属性） | Smiling/Blurry/High Cheekbones | 同时纠正三个敏感属性时 F1 和统计差异均显著提升，优于单属性纠正 |

**基线方法**包括：Lower（直接在有偏数据训练）、Upper（在无偏数据训练）、Unawareness（去除敏感变量）、Massaging（数据预处理）、Error Parity（后处理方法）。

## 主要发现

1. ProbLog4Fairness 因其灵活建模能力，在准确率和公平性上同时接近理想上界，是其他固定假设方法无法达到的
2. 当 $A \not\perp Y$（敏感变量与标签非独立）时，本方法能区分有问题的偏差和合理的相关性，而其他方法会过度纠正
3. 参数估计对结果影响较小，少量无偏数据即可有效设定参数
4. 在 CELEB-A 实验中，同时建模多个敏感属性的偏差远优于单个属性的纠正，凸显灵活建模的重要性
5. 简化假设（如"无正向偏差"）在真实数据上可能反而提升性能，因为减少了参数估计误差

## 亮点

- **原则性与灵活性统一**：ProbLog 的声明式编程允许用户根据具体场景灵活添加、修改偏差假设，无需依赖固定的公平性定义
- **可解释性强**：偏差假设以逻辑规则形式表达，领域专家可以直接理解和验证
- **多类型偏差统一框架**：标签、测量、历史三种偏差均可在同一框架下组合建模
- **从表格到图像的泛化**：在 CELEB-A 上成功将 ResNet-50 与 ProbLog 结合，证明对高维非表格数据也有效

## 局限性

- 偏差参数需要预先设定或从小规模无偏数据估计，若参数联合学习则分类器不可辨识
- 目前仅处理二元敏感变量和二元标签，扩展到多类别需要大量概率事实
- 合成实验限于二元/分类特征，连续特征的处理未深入讨论
- DeepProbLog 的编译和推理在复杂程序上计算代价较高（CELEB-A 实验约需 100 小时）
- 未与近年来的因果公平性方法（如反事实公平性）进行直接比较
- 假设偏差机制可被正确建模，但现实中偏差来源可能比预设模板更复杂

## 相关工作

- **公平性约束方法**：预处理（Feldman 2015）、过程中约束（Kamishima 2011）、后处理（Hardt 2016），但依赖固定公平性指标
- **因果公平性**：Kilbertus 2017 用因果图识别歧视路径，Madras 2018 建模决策与混杂变量；本文更灵活但不需要完整因果图
- **神经符号公平性**：Varley 2021 用 SPN 学习依赖关系，Choi 2020 设计满足独立性假设的概率电路，Wagner 2021 用 LTN 主动学习约束
- **PU Learning 与 ProbLog**：Verreet 2024 将 PU Learning 的标注机制表达为 ProbLog 模板，本文将此思路推广到公平性领域

## 评分

⭐⭐⭐⭐ (4/5)

创新性地将概率逻辑编程应用于公平性问题，框架设计优雅、可解释性强。实验全面涵盖合成和真实数据，三种偏差类型验证充分。主要扣分在扩展性（计算开销大、多类别场景受限）和缺乏与近期因果公平方法的对比。

<!-- RELATED:START -->

## 相关论文

- [Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning](learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)
- [Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures](ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)
- [Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)

<!-- RELATED:END -->
