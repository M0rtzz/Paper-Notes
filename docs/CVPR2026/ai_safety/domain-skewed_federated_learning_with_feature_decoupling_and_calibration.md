---
title: >-
  [论文解读] Domain-Skewed Federated Learning with Feature Decoupling and Calibration
description: >-
  [CVPR 2026][AI安全][联邦学习] 提出 F²DC 框架，通过域特征解耦器（DFD）和域特征校正器（DFC）将联邦学习中客户端的局部特征分离为域鲁棒特征和域相关特征，并对域相关特征进行校准以挽救被丢弃的类别信息，配合域感知聚合策略，在三个多域数据集上一致超越 SOTA。
tags:
  - CVPR 2026
  - AI安全
  - 联邦学习
  - 域偏移
  - 特征解耦
  - 域感知聚合
  - 表征校准
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

1. **Domain Feature Decoupler (DFD)**

    - **功能**：为特征图中每个单元确定跨域鲁棒性得分，分离为域鲁棒和域相关两部分
    - **为什么**：直接处理原始特征会过拟合域偏差；需先分离出域上下文以便后续校准
    - **怎么做**：使用两层 CNN（含 BN + ReLU）构建属性映射 $\mathcal{S}_i = \mathcal{A}_D(f_i) \in \mathbb{R}^{C \times H \times W}$；通过 **Gumbel Concrete 分布** 生成伪二值掩码 $\mathcal{M}_i$（解决硬离散化的不可微问题），$\sigma \to 0$ 时趋近硬二值；解耦：$f_i^+ = \mathcal{M}_i \odot f_i$，$f_i^- = (1 - \mathcal{M}_i) \odot f_i$
    - **损失**（可分性 + 判别性）：可分性项最小化 $f^+$ 和 $f^-$ 的余弦相似度；判别性项让 $f^+$ 正确分类（ground truth）、$f^-$ 倾向错误分类（highest-confidence wrong label），由辅助 MLP $\mathbf{m}$ 预测 logits
    - **区别**：FDSE 直接消除域特征，DFD 是"分而不弃"，保留域相关特征供后续校准

2. **Domain Feature Corrector (DFC)**

    - **功能**：从域相关特征 $f^-$ 中提取可补充 $f^+$ 的额外类别线索
    - **为什么**：$f^-$ 中纠缠着域偏差和类别信息，直接丢弃损失有价值信号
    - **怎么做**：与 DFD 同架构的两层 CNN $\mathcal{A}_C$，学习残差：$f_i^\star = f_i^- + (1 - \mathcal{M}_i) \odot \mathcal{A}_C(f_i^-)$
    - **损失**：标准交叉熵 $\mathcal{L}_{DFC} = -y_i \cdot \log(\delta(\mathbf{m}(l_i^\star)))$，注入正确判别信号

3. **Domain-Aware Aggregation (DaA)**

    - **功能**：全局聚合中考虑各客户端域差异度
    - **为什么**：朴素 FedAvg 忽视域多样性，等权重聚合导致偏差
    - **怎么做**：定义均匀全局域分布 $\mathcal{G} = [1/Q,...,1/Q]$（Q=域数量），计算客户端 k 的域差异度 $\mathbf{d}_k$；权重 $\mathbf{p}_k = \sigma(\alpha \cdot n_k/N - \beta \cdot \mathbf{d}_k)$ 归一化后聚合

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
