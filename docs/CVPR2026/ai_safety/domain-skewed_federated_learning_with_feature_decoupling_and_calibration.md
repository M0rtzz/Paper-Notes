---
title: >-
  [论文解读] Domain-Skewed Federated Learning with Feature Decoupling and Calibration
description: >-
  [CVPR 2026][AI安全][联邦学习] 提出 F²DC 框架，通过域特征解耦器（DFD）和域特征校正器（DFC）将联邦学习中客户端的局部特征分离为域鲁棒特征和域相关特征，并对域相关特征进行校准以挽救被丢弃的类别信息，配合域感知聚合策略，在三个多域数据集上一致超越 SOTA。
tags:
  - "CVPR 2026"
  - "AI安全"
  - "联邦学习"
  - "域偏移"
  - "特征解耦"
  - "域感知聚合"
  - "表征校准"
---

# Domain-Skewed Federated Learning with Feature Decoupling and Calibration

**会议**: CVPR 2026  
**arXiv**: [2603.14238](https://arxiv.org/abs/2603.14238)  
**代码**: [GitHub](https://github.com/mala-lab/F2DC)  
**领域**: AI安全  
**关键词**: 联邦学习, 域偏移, 特征解耦, 域感知聚合, 表征校准

## 一句话总结

提出 F²DC 框架，通过域特征解耦器（DFD）和域特征校正器（DFC）将联邦学习中客户端的局部特征分离为域鲁棒特征和域相关特征，并对域相关特征进行校准以挽救被丢弃的类别信息，配合域感知聚合策略，在三个多域数据集上一致超越 SOTA。

## 研究背景与动机

**联邦学习中的域偏移问题**：与标签偏移（label skew）不同，域偏移（domain skew）场景中各客户端数据来自不同域（如不同天气下的驾驶数据），类别分布相似但特征分布差异大：$\mathbb{P}_{k_1}(x|y) \neq \mathbb{P}_{k_2}(x|y)$。

**维度坍塌现象**：域偏移导致局部模型的表征坍缩到狭窄低维子空间——特征协方差矩阵的大量奇异值趋近于零，意味着每个客户端仅拟合其专属域的特征而忽略其他子空间。

**"消除式"方法的局限**：FDSE 等方法试图消除域特定偏差，但域相关特征中纠缠了有价值的类别信息（如 sketch 域中笔触构成的物体轮廓），直接消除导致信息丧失——Grad-CAM 显示 FDSE 在 cartoon/sketch 域中遗漏了长颈鹿的鹿角和头部。

**核心 idea**：通过校准而非消除域相关特征，挽救其中纠缠的类别相关线索，从而促进更一致的跨域决策。

## 方法详解

### 整体框架

F²DC 包含两个核心模块和一个聚合策略，嵌入标准 FedAvg 框架：
- **Domain Feature Decoupler (DFD)**：将局部特征解耦为域鲁棒特征 $f^+$ 和域相关特征 $f^-$
- **Domain Feature Corrector (DFC)**：校准 $f^-$ 为修正特征 $f^\star$，捕获额外类别线索
- **Domain-Aware Aggregation (DaA)**：根据各客户端域差异度加权全局聚合

架构上，DFD 和 DFC 在最后一个 backbone 层后插入（以 ResNet-10 为例，在 L4 后），$f^+$ 和 $f^\star$ 相加得到最终特征 $\tilde{f}$ 送入后续层。DFD、DFC 和辅助 MLP $\mathbf{m}$ 仅保留在本地，不参与全局聚合。

### 关键设计

**1. 域特征解耦器 DFD：先把域上下文分出来，而不是急着处理原始特征。**

如果直接拿局部特征去分类，模型会顺着自己专属域的偏差过拟合，根本没机会判断哪些信息是跨域通用的。DFD 的思路是先给特征图的每个单元打一个"跨域鲁棒性"标签，再据此一刀两断。具体做法是用一个两层 CNN（带 BN + ReLU）算出属性映射 $\mathcal{S}_i = \mathcal{A}_D(f_i) \in \mathbb{R}^{C \times H \times W}$，但二值化选择本身不可微，于是用 **Gumbel Concrete 分布** 生成伪二值掩码 $\mathcal{M}_i$——温度 $\sigma \to 0$ 时它趋近真正的硬二值，训练时却保持可微。掩码一出，特征就被切成域鲁棒部分 $f_i^+ = \mathcal{M}_i \odot f_i$ 和域相关部分 $f_i^- = (1 - \mathcal{M}_i) \odot f_i$。

要让这刀切得干净，损失同时管两件事：可分性项最小化 $f^+$ 与 $f^-$ 的余弦相似度，逼两者朝不同方向走；判别性项则借助一个辅助 MLP $\mathbf{m}$ 预测 logits，要求 $f^+$ 命中 ground truth、$f^-$ 反而倒向置信度最高的错误标签（highest-confidence wrong label），把"真正有判别力"的信号都挤进 $f^+$。这正是 DFD 与 FDSE 的分水岭——FDSE 把域相关特征整个消除掉，DFD 是"分而不弃"，把 $f^-$ 留着等下一步校准。

**2. 域特征校正器 DFC：把被丢弃的域相关特征里的类别线索捞回来。**

DFD 切出的 $f^-$ 并不是纯粹的噪声，它里面把域偏差和类别信息缠在一起——比如 sketch 域里笔触勾出的物体轮廓，直接扔掉就白白损失了有价值的判别信号。DFC 用一个与 DFD 同构的两层 CNN $\mathcal{A}_C$ 去学一个残差修正量，把 $f^-$ 重塑成补充特征 $f_i^\star = f_i^- + (1 - \mathcal{M}_i) \odot \mathcal{A}_C(f_i^-)$；残差形式让它只在域相关区域上做加法、不破坏原有结构。训练上挂一个标准交叉熵 $\mathcal{L}_{DFC} = -y_i \cdot \log(\delta(\mathbf{m}(l_i^\star)))$，强制把正确的类别判别信号注入回 $f^\star$。最终 $f^+$ 与 $f^\star$ 相加得到 $\tilde{f}$ 送进后续层，等于"鲁棒主干 + 抢救回来的类别线索"双管齐下。

**3. 域感知聚合 DaA：让全局聚合按域多样性调权，而不是一视同仁。**

朴素 FedAvg 只按样本量加权，完全无视各客户端来自不同域这件事，结果是被某些域的偏差带偏。DaA 先定义一个均匀的全局域分布 $\mathcal{G} = [1/Q, \dots, 1/Q]$（$Q$ 为域数量）作为参照，再算出客户端 $k$ 偏离这个均匀分布的域差异度 $\mathbf{d}_k$，最后给它一个权重 $\mathbf{p}_k = \sigma(\alpha \cdot n_k/N - \beta \cdot \mathbf{d}_k)$——样本越多越加分，域偏得越离谱越扣分，归一化后再聚合。这样既保留了样本量这个传统信号，又把"域代表性"显式纳入，避免少数极端域主导全局模型。

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{CE} + \frac{1}{|L|}\sum_{j=1}^{|L|}(\lambda_1 \cdot \mathcal{L}_{DFD}^{L_j} + \lambda_2 \cdot \mathcal{L}_{DFC}^{L_j})$$

默认 $|L|=1$（仅最后一层），$\lambda_1=0.8, \lambda_2=1.0$，Gumbel 温度 $\sigma=0.1$，分离温度 $\tau=0.06$，聚合参数 $\alpha=1.0, \beta=0.4$。SGD 优化器，lr=0.01，动量 0.9，batch size 64，100 轮通信，每轮 10 个 local epoch。

## 实验关键数据

### 主实验

| 数据集 | 指标 | F²DC | 之前 SOTA (FDSE) | 提升 |
|--------|------|------|------------------|------|
| PACS | AVG Acc ↑ | **76.47** | 73.13 | +3.34 |
| PACS | STD ↓ | **5.83** | 6.83 | -1.00 |
| Office-Caltech | AVG Acc ↑ | **66.82** | 63.18 | +3.64 |
| Office-Caltech | STD ↓ | **3.65** | 4.50 | -0.85 |
| Digits | AVG Acc ↑ | **87.23** | 84.15 | +3.08 |
| Digits | STD ↓ | **13.36** | 16.19 | -2.83 |

在三个数据集上均一致超越全部 9 种对比方法（FedAvg/FedProx/MOON/FPL/FedTGP/FedRCL/FedHEAL/FedSA/FDSE），且跨域公平性（STD 更小）更优。对比方法 MOON 等基于对比的方法在 PACS 上甚至差于 FedAvg，因为强制对齐已受污染的全局表征反而加剧性能退化。

### 消融实验（PACS）

| 配置 | AVG Acc | STD | 说明 |
|------|---------|-----|------|
| FedAvg (baseline) | 66.39 | 11.74 | 无任何模块 |
| + DFD only | 68.43 | 10.15 | 仅解耦 |
| + DFD + DFC | 73.64 | 6.12 | 解耦 + 校正 |
| + DFD + DaA | 75.33 | 6.80 | 解耦 + 域感知聚合 |
| + DFD + DFC + DaA | **76.47** | **5.83** | 完整 F²DC |

### 模块化可插拔性（PACS）

| 基线方法 | + DFD+DFC 后 AVG | 提升 |
|----------|-----------------|------|
| FedAvg | 75.33 | **+8.94** |
| FPL | 75.52 | +4.93 |
| FedHEAL | 75.06 | +1.72 |
| FDSE | 74.79 | +1.66 |

### 关键发现

- **特征分析**：$f^+$ 的 AVG=75.13 远优于 $f^-$ 的 57.87，但校正后 $f^\star$=73.49，证实域相关特征确实包含可挽救类别信息；融合后 $\tilde{f}$=76.47 达最优
- **收敛更快**：F²DC 在 Office-Caltech 和 PACS 上均展现更快的收敛速度
- **开销极小**：无额外通信成本（DFD/DFC 本地保留），训练时间仅增 2%（180.67s vs 176.94s/轮）

## 亮点与洞察

1. **"校准而非消除"**：域偏差中纠缠的类别信息是有价值的，Grad-CAM 可视化直观展示了 F²DC 如何恢复被传统方法忽略的区域（如长颈鹿腰部）
2. **Gumbel Concrete 可微分离**：巧妙解决特征二值分离的不可微问题，使框架端到端可训练
3. **维度坍塌诊断**：奇异值分析定量揭示了域偏移 FL 的核心病因，可推广为通用诊断工具

## 局限与展望

1. 解耦粒度依赖超参数 $\tau$，过于激进的分离反而降低性能
2. 仅在特征层面操作，未考虑参数级别的域偏差解耦
3. 实验仅覆盖 4 域场景（ResNet-10），更多域/更大模型的可扩展性未验证
4. 域感知聚合假设域内类别分布均匀，同时存在域偏移+标签偏移时需要扩展

## 相关工作与启发

- **FDSE (CVPR'25)**：消除式解耦的代表，F²DC 的"校正利用"范式是更优替代
- **FedHEAL (CVPR'24)**：选择性参数更新+公平聚合，但不进行特征层面域偏差处理
- 启发：Gumbel Concrete 技巧在 NAS/剪枝中广泛使用，F²DC 展示了其在 FL 特征选择中的新应用

## 评分

- **新颖性**: ⭐⭐⭐⭐ — "校准而非消除"在域偏移 FL 中较新颖，DFD+DFC 设计合理
- **实验充分度**: ⭐⭐⭐⭐ — 三个数据集、9 种对比方法、完整消融、模块化验证、效率分析、可视化齐全
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰、图表丰富（Grad-CAM/T-SNE/SVD 可视化）
- **价值**: ⭐⭐⭐⭐ — 模块化设计使其易于集成到现有 FL 框架，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)
- [\[CVPR 2025\] A Simple Data Augmentation for Feature Distribution Skewed Federated Learning](../../CVPR2025/ai_safety/a_simple_data_augmentation_for_feature_distribution_skewed_federated_learning.md)
- [\[CVPR 2026\] FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)
- [\[CVPR 2026\] ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning](proxyfl_a_proxy-guided_framework_for_federated_semi-supervised_learning.md)
- [\[CVPR 2026\] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)

</div>

<!-- RELATED:END -->
