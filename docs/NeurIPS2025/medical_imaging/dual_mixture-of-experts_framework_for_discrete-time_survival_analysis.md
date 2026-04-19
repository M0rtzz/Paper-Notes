---
title: >-
  [论文解读] Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis
description: >-
  [NeurIPS 2025 (TS4H Workshop)][医学图像][生存分析] 提出双混合专家（Dual MoE）框架用于离散时间生存分析，结合特征编码器 MoE（建模患者亚组异质性）与风险网络 MoE（捕获时间动态），在 METABRIC 和 GBSG 乳腺癌数据集上提升 time-dependent C-index 最高 0.04。
tags:
  - NeurIPS 2025 (TS4H Workshop)
  - 医学图像
  - 生存分析
  - 混合专家
  - 离散时间
  - 患者异质性
  - 乳腺癌
---

# Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis

**会议**: NeurIPS 2025 (TS4H Workshop)  
**arXiv**: [2510.26014](https://arxiv.org/abs/2510.26014)  
**代码**: 无  
**领域**: medical_imaging  
**关键词**: 生存分析, 混合专家, 离散时间, 患者异质性, 乳腺癌

## 一句话总结
提出双混合专家（Dual MoE）框架用于离散时间生存分析，结合特征编码器 MoE（建模患者亚组异质性）与风险网络 MoE（捕获时间动态），在 METABRIC 和 GBSG 乳腺癌数据集上提升 time-dependent C-index 最高 0.04。

## 研究背景与动机

**领域现状**：生存分析旨在预测事件发生的时间，广泛用于临床和生物医学研究。传统 Cox 比例风险（CPH）模型假设风险比随时间恒定不变。深度学习方法（DeepHit、ConSurv）用灵活的神经网络替代 CPH 约束，可建模非比例风险。

**现有痛点**：大多数深度生存模型仍使用单一共享特征编码器和单一风险网络。但患者群体具有异质性亚组（如 ER+/ER-、HER2+/HER2-），单一编码器倾向于偏向主导模式；同时，风险不仅因人而异，还随时间变化——同一时间点上不同患者可能展现截然不同的风险轨迹。

**核心矛盾**：单一网络将所有患者和所有时间段绑定到一个共享函数形式，无法充分捕获患者异质性和时间动态的交互。

**本文目标** 如何在特征编码和风险预测两个层面同时建模患者亚组差异和时间变化？

**切入角度**：将 Mixture-of-Experts（MoE）引入生存分析，分别在特征编码和风险预测阶段使用多专家网络，通过软路由实现亚组感知和时间感知的建模。

**核心 idea**：在特征编码和风险预测两个阶段分别引入 MoE，通过患者特征驱动的路由和患者+时间联合驱动的路由，实现细粒度的生存建模。

## 方法详解

### 整体框架
输入为患者协变量 $x_i$、观测时间 $\tau_i$ 和事件指示器 $\delta_i$。模型分两阶段：(1) 特征编码器 MoE——初始编码器提取基础表征后，通过软路由分配到 K 个专家编码器，加权融合得到亚组感知的患者表征；(2) 风险网络 MoE——L 个风险专家根据患者特征+时间嵌入联合路由，输出每个时间段的风险函数。两阶段的 MoE 结合负对数似然损失和负载均衡正则化进行端到端训练。

### 关键设计

1. **特征编码器 MoE**:

    - 功能：通过多专家编码器学习亚组感知的患者表征
    - 核心思路：初始编码器 $g(\cdot)$ 提取基础特征，路由器基于患者特征产生 softmax 路由概率 $\pi_k^{feat}$，最终表征为所有专家输出的加权和：
    $z(x) = \sum_{k=1}^{K} \pi_k^{feat} \cdot f_k(g(x))$
    - 设计动机：不同亚组（如 ER+/- 或 HER2+/-）具有不同的风险特征分布，单一编码器无法同时最优化所有亚组的表征

2. **风险网络 MoE**:

    - 功能：通过多专家风险网络捕获患者-时间交互的细粒度风险模式
    - 核心思路：每个风险专家 $h_l$ 预测所有时间段的风险，路由器同时接收患者特征和可学习时间嵌入 $e_t$，联合决定路由概率 $\pi_{t,l}^{haz}$：
    $\lambda(t|x) = \sum_{l=1}^{L} \pi_{t,l}^{haz} \cdot h_l(z(x), e_t)$
    - 设计动机：这使得专家可以沿时间和患者两个维度同时特化——例如某些专家可能专注于早期风险高的患者亚组，另一些专注于晚期风险的亚组

3. **负载均衡正则化**:

    - 功能：防止路由崩塌到少数专家
    - 核心思路：特征 MoE 均衡损失 $\mathcal{L}_{LB}^{feat} = \alpha(K \sum_k \bar{\pi}_k^{feat^2} - 1)$，风险 MoE 均衡损失 $\mathcal{L}_{LB}^{haz} = \beta(T_{max} \sum_t L \sum_l \bar{\pi}_{t,l}^{haz^2} - 1)$
    - 设计动机：MoE 中专家利用率不均是常见问题，不加约束路由器会收敛到只使用 1-2 个专家

### 损失函数 / 训练策略
- 总损失：$\mathcal{L} = \mathcal{L}_{NLL} + \mathcal{L}_{LB}^{feat} + \mathcal{L}_{LB}^{haz}$
- 负对数似然：$\mathcal{L}_{NLL} = -\sum_{i=1}^{N}[\delta_i \log \hat{p}(\tau_i|x_i) + (1-\delta_i) \log \hat{S}(\tau_i|x_i)]$
- 其中 $\hat{p}(t|x) = \lambda(t|x) S(t-1|x)$, $S(t|x) = \prod_{t' \leq t}(1-\lambda(t'|x))$
- METABRIC 使用 (K=4, L=4) 专家，GBSG 使用 (K=6, L=3) 专家
- 10 个随机种子取平均

## 实验关键数据

### 主实验（C-index）

| 数据集 | 方法 | Dual MoE | C-index | Td-C 10% | Td-C 50% | Td-C 90% |
|--------|------|---------|---------|----------|----------|----------|
| METABRIC | Naïve impl. | ✗ | 0.646 | 0.670 | 0.638 | 0.606 |
| METABRIC | Naïve impl. | ✓ | **0.654** | 0.669 | **0.646** | **0.623** |
| METABRIC | ConSurv | ✗ | 0.657 | 0.656 | 0.649 | 0.617 |
| METABRIC | ConSurv | ✓ | **0.668** | **0.696** | **0.657** | **0.634** |
| GBSG | Naïve impl. | ✗ | 0.662 | 0.744 | 0.669 | 0.652 |
| GBSG | Naïve impl. | ✓ | **0.667** | **0.751** | **0.677** | **0.659** |
| GBSG | ConSurv | ✗ | 0.665 | 0.742 | 0.674 | 0.658 |
| GBSG | ConSurv | ✓ | **0.668** | **0.752** | **0.677** | **0.659** |

### 消融分析（从可视化推断）

| 分析维度 | 关键发现 |
|---------|---------|
| 特征路由可视化 | ER+/ER- 和 HER2+/HER2- 亚组显示出明显不同的专家偏好分布，验证路由器能自动发现亚组结构 |
| 风险路由轨迹 | 不同患者的风险专家分配随时间显著变化，早期和晚期的主导专家不同，说明模型捕获了时间动态 |
| 基线对比 | CoxPH (0.663) 优于 Naïve impl. (0.646)，说明深度模型在小数据上易过拟合，Dual MoE 缓解了这一问题 |

### 关键发现
- Dual MoE 在所有设置下**一致提升**性能，最大提升 Time-dependent C-index +0.04（METABRIC ConSurv 在 10% 时间点）
- 与 ConSurv 结合时效果更佳，说明 Dual MoE 是通用插件模块，可灵活集成到现有深度生存分析流水线
- 特征路由器能自动发现临床上有意义的亚组（ER/HER2 状态差异），无需显式亚组标签
- 风险路由器展示了时间依赖的专家特化——不同时间段由不同专家主导，符合非比例风险的直觉
- 在 GBSG 上 Naïve impl. 不加 MoE 就已接近 CoxPH，说明该数据集上简单深度模型已有竞争力

## 亮点与洞察
- **双层 MoE 的设计理念**：特征 MoE 捕获"谁不同"（患者间异质性），风险 MoE 捕获"何时不同"（时间动态），两者正交互补
- **可解释性强**：路由概率的可视化直接揭示了亚组结构和时间动态，比黑箱模型更有临床可解释性
- **即插即用**：可直接替换现有深度生存模型的编码器和风险头，无需修改损失函数或训练流程
- **时间嵌入作为路由器输入**：将可学习时间嵌入引入风险路由是巧妙的设计，使专家能沿时间轴特化

## 局限与展望
- 仅在两个乳腺癌数据集上验证，需要更多疾病类型和更大规模的数据集
- Workshop 论文，消融实验不够详细——未逐步拆分特征 MoE 和风险 MoE 的独立贡献
- 专家数量 (K, L) 的选择似乎需要针对每个数据集调参，缺乏自适应机制
- 未与其他异质性建模方法对比（如 mixture cure models、random effects models）
- 可扩展到多模态（如影像+临床）的生存分析场景

## 相关工作与启发
- **vs CoxPH**：经典方法假设比例风险，无法处理非比例风险场景，本文通过灵活的 MoE 风险网络克服这一限制
- **vs DeepHit / ConSurv**：使用单一编码器和风险头，本文在两个层面引入 MoE 提升建模能力，且可作为 ConSurv 的插件使用
- **vs 标准 MoE (Shazeer et al.)**：标准 MoE 仅在单一层面使用，本文创新性地在编码和预测两个阶段同时使用，且风险 MoE 的路由额外引入时间嵌入
- 对病理学/影像组学中的生存预测（如全切片图像的预后）有直接借鉴价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 在生存分析中同时引入双层 MoE 是新颖的组合，风险 MoE 的时间条件路由设计巧妙
- 实验充分度: ⭐⭐⭐ Workshop 论文，仅两个数据集，缺乏详细消融和更多基线对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式完整，可视化有效支撑了方法动机
- 价值: ⭐⭐⭐⭐ 提供了即插即用的通用模块，对临床生存预测有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts](../../ICML2025/medical_imaging/i2moe_interpretable_multimodal_interaction-aware_mixture-of-experts.md)
- [Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation](mamba_goes_home_hierarchical_soft_mixture-of-experts_for_3d_medical_image_segmen.md)
- [DFLMoE: Decentralized Federated Learning via Mixture of Experts for Medical Data](../../CVPR2025/medical_imaging/dflmoe_decentralized_federated_learning_via_mixture_of_experts_for_medical_data_.md)
- [MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)
- [SEMC: Structure-Enhanced Mixture-of-Experts Contrastive Learning for Ultrasound Standard Plane Recognition](../../AAAI2026/medical_imaging/semc_structure-enhanced_mixture-of-experts_contrastive_learning_for_ultrasound_s.md)

<!-- RELATED:END -->
