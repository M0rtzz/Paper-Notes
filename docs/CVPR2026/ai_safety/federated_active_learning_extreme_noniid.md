---
title: >-
  [论文解读] Federated Active Learning Under Extreme Non-IID and Global Class Imbalance
description: >-
  [CVPR 2026][AI安全][联邦学习] 系统分析全局类不平衡与客户端异构性对联邦主动学习中 query model 选择的影响，归纳出3个核心 Observation，据此提出 FairFAL——自适应选择 query model + 原型引导伪标签 + 两阶段不确定性-多样性平衡采样的类公平 FAL 框架，在5个基准数据集上一致超越所有基线。
tags:
  - CVPR 2026
  - AI安全
  - 联邦学习
  - 主动学习
  - non-IID
  - 类别不平衡
  - query selection
  - class-fair sampling
  - 原型引导
---

# Federated Active Learning Under Extreme Non-IID and Global Class Imbalance

**会议**: CVPR 2026  
**arXiv**: [2603.10341](https://arxiv.org/abs/2603.10341)  
**代码**: [GitHub](https://github.com/chenchenzong/FairFAL)  
**领域**: 联邦学习 / 主动学习  
**关键词**: 联邦学习, 主动学习, non-IID, 类别不平衡, query selection, class-fair sampling, 原型引导

## 一句话总结

系统分析全局类不平衡与客户端异构性对联邦主动学习中 query model 选择的影响，归纳出3个核心 Observation，据此提出 FairFAL——自适应选择 query model + 原型引导伪标签 + 两阶段不确定性-多样性平衡采样的类公平 FAL 框架，在5个基准数据集上一致超越所有基线。

## 研究背景与动机

- **领域现状**：联邦学习（FL）在不共享原始数据的前提下实现协作训练，主动学习（AL）通过选择性标注降低标注成本。联邦主动学习（FAL）将两者结合——去中心化的客户端在隐私约束下协作识别最有价值的待标注样本。这在医疗影像、自动驾驶等标注昂贵且数据隐私敏感的领域尤为重要。
- **现有痛点**：已有FAL研究在三方面存在盲区：(1) 将客户端异构性仅视为数据划分问题，隐含假设全局类分布大致均衡；(2) 对于FAL中天然存在的两个query模型（全局聚合模型 vs 本地训练模型），缺乏系统性的选择准则；(3) 在全局长尾叠加极端non-IID的条件下，现有采样策略系统性偏向头部类别——LoGo、KAFAL、IFAL 等近期方法虽考虑了 non-IID，但未显式处理全局类不平衡。IFAL 在 CIFAR-100+$\rho=20$ 下甚至低于随机采样（26.82 vs. 27.44），充分说明问题的严重性。
- **核心矛盾**：FAL 中全局模型拥有更好的特征表示（跨客户端聚合），但在不确定性采样中常因过度平滑的预测而丧失判别力；本地模型对客户端特有决策边界更敏感，但在全局高度不平衡时其采样会反映长尾偏斜。两个模型的相对优劣取决于全局不平衡程度 $\rho$ 和客户端异构程度 $\alpha$ 的组合，无法简单固定。
- **本文目标**：在极端 non-IID（$\alpha=0.1$）和全局长尾分布（$\rho=20$）的挑战性设置下，设计一个能自适应选择 query 模型、显式促进类公平采样的 FAL 框架。
- **切入角度**：从系统性经验分析出发——在 CIFAR-10 上系统对比全局/本地模型在不同 $(\alpha, \rho)$ 组合下的采样行为（4种组合 × 2种策略 × 5种子），使用 AULC、Wilcoxon 检验、Hodges-Lehmann 效应量三重统计分析，归纳出3个 Observation，据此设计框架每个组件。
- **核心 idea**：类平衡采样能力（尤其是对少数类的获取）是 FAL 性能的最一致预测因子，比不确定性或多样性本身更重要。

## 方法详解

### 整体框架

FairFAL 在标准 FedAvg 框架上增加三个协同组件，在每轮 FAL 的 query 阶段依次执行：(1) **自适应模型选择**——根据全局不平衡系数 $\bar{\gamma}$ 和本地-全局散度 $d_k$ 决定每个客户端使用全局还是本地模型作为 query selector；(2) **原型引导伪标签**——用全局模型特征计算类原型，对未标注样本分配伪标签，形成类级候选池；(3) **两阶段不确定性-多样性平衡采样**——先在每个类内按不确定性选候选，再在梯度嵌入空间做 $k$-center 多样性筛选。整个流程在每轮 AL query 后进行标准的联邦训练更新。

### 关键设计

1. **[自适应模型选择 (Adaptive Model-Selection)]**:

    - 功能：每个客户端自动决定用全局还是本地模型作为 query selector，无需额外隐私泄露（仅上传标量 $\gamma_k$）
    - 核心思路：估计两个关键量——**全局类不平衡比 $\gamma_k$** 和 **本地-全局分布散度 $d_k$**。对于 $\gamma_k$：在第一轮 AL（标注数据为随机采样，近似 IID）时，对每个客户端构建类平衡子集 $\mathcal{B}^{(k)}$（通过上采样使每类样本数相等），用全局模型预测得到 softmax 先验 $\hat{\pi}_g^{(k)}$，计算 $\gamma_k = \min_c \hat{\pi}_{g,c} / \max_c \hat{\pi}_{g,c} \in (0,1]$，越接近1越平衡。服务器简单平均所有 $\gamma_k$ 得到 $\bar{\gamma}$（仅传标量，隐私安全），仅第一轮计算后固定。对于 $d_k$：用全局和本地模型分别在同一平衡子集上预测，计算归一化对称差异 $d_k = \frac{1}{C}\sum_c \frac{|\hat{\pi}_{g,c} - \hat{\pi}_{\ell,c}^{(k)}|}{\hat{\pi}_{g,c} + \hat{\pi}_{\ell,c}^{(k)}} \in [0,1]$，每轮更新。模型选择分数 $s_k = 1 - \frac{1}{2}(d_k + \bar{\gamma})$：当 $s_k > \delta = 0.75$ 时用全局模型（全局高度不平衡 + 客户端同质），否则用本地模型。
    - 设计动机：由 Observation 1 驱动——不确定性采样中本地模型通常优于全局模型（本地多样性的聚合天然产生全局平衡的查询集），但当全局类不平衡严重且客户端同质时例外（本地查询反映全局长尾偏斜）。消融显示自适应选择 $\mathcal{M}^{(k)}$ 始终优于对立选项 $\tilde{\mathcal{M}}^{(k)}$（59.33 vs. 58.49），且对阈值 $\delta$ 不敏感（0.65-0.85 变化 < 0.5%）。

2. **[原型引导伪标签 (Prototype-Guided Pseudo-Labeling)]**:

    - 功能：对未标注样本分配类感知伪标签，将未标注池划分为类级子集 $\tilde{\mathcal{D}}_{U,c}^{(k)}$，为后续类平衡采样提供基础
    - 核心思路：用全局模型的倒数第二层特征提取器 $\phi^g(\cdot)$ 提取 L2 归一化特征 $\mathbf{z}_i^{(k)}$，对每个类 $c$ 计算原型 $\boldsymbol{\mu}_c^{(k)}$（该类标注样本特征均值），对未标注样本计算与各原型的余弦相似度，分配伪标签 $\hat{y}^{(k)}(x) = \arg\max_c \langle \mathbf{z}^{(k)}(x), \boldsymbol{\mu}_c^{(k)} \rangle$。这绕过了长尾场景下分类器决策边界偏移的问题——在特征空间中做类分配比直接用 logit 更鲁棒。
    - 设计动机：由 Observation 2（类平衡采样与性能高度对齐）和 Observation 3（全局模型的特征质量更优）共同驱动。消融证实：全局原型优于本地原型（59.95 vs. 59.14），全局模型在特征空间提供更干净的类分离。

3. **[两阶段不确定性-多样性平衡采样 (Uncertainty-Diversity Balanced Sampling)]**:

    - 功能：在保证类平衡的前提下兼顾信息性和多样性，避免高不确定性样本的冗余聚集
    - 核心思路：**Stage 1 (类内不确定性筛选)**——对每个类的候选子集用自适应选择的 query model 计算熵，取 top-$\kappa \cdot b_c$ 个最不确定样本组成过完备候选池 $\mathcal{H}_c^{(k)}$（$\kappa = 4$，$b_c$ 为每类均匀预算）。**Stage 2 (梯度嵌入空间 $k$-center)**——用全局模型计算每个候选的梯度嵌入 $\mathbf{g}^{(k)}(x) = \psi(x; \phi^g, f^g)$（分类损失对全局分类器参数的梯度），以已标注样本为锚点，执行贪心 $k$-center 选择 $b_c$ 个样本，最小化最大覆盖半径。
    - 设计动机：直接选最不确定样本会导致严重冗余——高不确定性样本倾向聚集在特征空间的狭窄区域。两阶段设计先保证信息性（排除"安全"样本），再保证多样性（覆盖特征空间）。$\kappa = 4$ 提供足够候选多样性，消融显示 $\kappa = 2/3/4$ 性能差异微小（< 0.3%）。

### 损失函数/训练策略

标准 FedAvg 框架：100 通信轮，每客户端 5 本地 epoch。4 层 CNN 骨干，SGD（momentum=0.9，weight decay=1e-5，batch size=64），学习率 0.01，75 轮后衰减 10x。FAL 进行 9 轮 query，首轮随机标注 5%，后续每轮 query 5%。10 个客户端，Dirichlet 划分（$\alpha = 0.1$ / $\alpha = 100$），全局不平衡比 $\rho = 20$。所有实验 5 个随机种子取均值。

## 实验关键数据

### 主实验

**自然图像数据集（$\alpha = 0.1$, $\rho = 20$，最终轮测试准确率 %）**：

| 方法 | 类型 | FMNIST | CIFAR-10 | CIFAR-100 |
|------|------|--------|----------|-----------|
| Random | 基线 | 85.60 | 55.70 | 27.44 |
| Entropy | 不确定性 | 86.28 | 57.18 | 27.44 |
| BADGE | 混合 | 86.65 | 58.20 | 27.39 |
| Coreset | 多样性 | 86.15 | 57.63 | 27.81 |
| KAFAL | FAL | 87.05 | 60.01 | 27.84 |
| LoGo | FAL | 86.98 | 59.68 | 27.95 |
| IFAL | FAL | 86.80 | 57.51 | 26.82 |
| **FairFAL** | **FAL** | **87.37** | **60.44** | **29.20** |

**医学数据集（$\alpha = 0.1$，自然长尾分布）**：

| 方法 | OctMNIST | DermaMNIST |
|------|----------|------------|
| Random | 68.30 | 72.32 |
| KAFAL | 70.40 | 73.27 |
| LoGo | 70.00 | 73.62 |
| IFAL | 68.40 | 72.97 |
| **FairFAL** | **72.80** | **73.77** |

### 消融实验

**组件逐步添加（CIFAR-10，准确率 %）**：

| 配置 | ($\alpha$=0.1, $\rho$=20) | ($\alpha$=100, $\rho$=20) |
|------|---------------------------|--------------------------|
| 仅模型选择 $\mathcal{M}^{(k)}$ | 59.33 | 63.65 |
| 对立模型 $\tilde{\mathcal{M}}^{(k)}$ | 58.49 | 61.89 |
| + 本地原型 | 59.14 | 63.39 |
| + 全局原型 | 59.95 | 64.02 |
| + 两阶段 $\kappa=2$ | 60.61 | 64.60 |
| + 两阶段 $\kappa=3$ | 60.38 | 64.58 |
| **+ 两阶段 $\kappa=4$ (Final)** | **60.44** | **64.57** |
| + 两阶段 $\kappa=5$ | 60.28 | 64.17 |

**全局/本地 query selector 统计分析（Hodges-Lehmann 效应量，百分点 pp）**：

| ($\alpha$, $\rho$) | Entropy 胜者 | Entropy HL (pp) | Coreset 胜者 | Coreset HL (pp) |
|-----|---------|--------|---------|--------|
| (0.1, 1) | Local | 201 | Global | 28 |
| (100, 1) | Local | 50 | Global | 21 |
| (0.1, 20) | Local | 66 | Global | 50 |
| (100, 20) | **Global** | **106** | **Global** | **92** |

### 关键发现

- **自适应模型选择有效**：$\mathcal{M}^{(k)}$ 始终优于对立选项（高异构 +0.84，近同构 +1.76）
- **全局原型 > 本地原型**（59.95 vs. 59.14）：全局模型在特征空间提供更干净的类分离
- **方法对 $\kappa$ 不敏感**（$\kappa=2/3/4$ 差异 < 0.3%），对 $\delta$ 也不敏感（0.65-0.85 范围 < 0.5%）
- **FL 框架无关**：在 FedProx 和 SCAFFOLD 下也一致最优
- **优势随任务难度增大而扩大**（FMNIST +0.32 → CIFAR-100 +1.25）
- **IFAL 在 CIFAR-100 低于 Random**（26.82 vs. 27.44）——无类平衡机制的方法在复杂长尾场景完全失效
- **Observation 2 的可视化**：全局-本地模型的少数类累计采样比例与最终准确率高度对齐，差距在少数类采样分化后稳定

## 亮点与洞察

- 核心洞察极为清晰且具有普遍性：类平衡采样能力是 FAL 中比不确定性或多样性更一致的性能预测因子
- 三个 Observation 的推导具有教育意义：(1) 不确定性采样中本地模型通常优于全局（除非全局高度不平衡+客户端同质）；(2) 类平衡采样与最终性能高度对齐（因果方向清晰）；(3) 多样性采样中全局模型的特征质量始终更优
- 原型引导伪标签绕过了长尾分类器决策边界偏移——在特征空间做类分配避免 logit 偏移
- 隐私保护设计考虑周全——客户端仅上传标量 $\gamma_k$，所有估计基于本地标注数据
- 统计分析方法（AULC + Wilcoxon 检验 + Hodges-Lehmann 效应量）为 Observation 提供了严格的统计支撑，而非仅靠均值比较

## 局限与展望

- 自适应选择是硬切换（$s_k > \delta$ 全局，否则本地），未尝试软混合——如对两个模型的 query 结果进行加权融合
- 均匀预算分配（每类 $b_c$ 相同）在极端长尾下可能非最优——可考虑按类稀缺程度动态分配更多预算给尾部类
- 仅在图像分类任务上验证（4层CNN），对目标检测、分割等复杂任务和更深骨干（ResNet-50、ViT）的适用性未知
- $\bar{\gamma}$ 仅第一轮估计后固定，不随 query 进程更新——后续轮次中全局分布因主动选择而改变
- 假设第一轮 query 是随机的以保证 IID 近似——实际系统中初始标注可能已有偏斜
- 未与基于生成模型的数据增强方案（解决类不平衡的另一路线）比较

## 相关工作与启发

- **vs LoGo (CVPR'23)**：先本地聚类再全局不确定性打分——两阶段但未显式考虑类平衡；FairFAL 通过原型伪标签显式保证每类采样
- **vs KAFAL (ICCV'23)**：利用全局-本地预测差异识别"知识不一致"样本——但差异大不等于少数类；FairFAL 增加类感知层
- **vs IFAL**：无类平衡机制在 CIFAR-100+$\rho=20$ 下低于 Random，是最直接的反面证据
- 原型学习从 few-shot 和对比学习引入到 FAL 的类感知采样——FL 的全局模型是天然优质特征提取器，即使分类头有偏
- 从 AL 到 FAL 的核心挑战不只是隐私约束，更是"双模型选择 + 全局长尾"的复合难题

## 评分

- 新颖性: ⭐⭐⭐⭐ 从系统性经验分析出发设计框架，每个组件都有清晰的 Observation 对应，方法论值得学习
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集、11个基线、详尽消融（模型选择/原型质量/κ/δ/不确定性度量/FL框架/客户端数）
- 写作质量: ⭐⭐⭐⭐⭐ Observation-driven 结构清晰，统计分析严谨，每个设计选择都有实验支撑
- 价值: ⭐⭐⭐⭐ 对 FAL 社区有直接指导价值，核心洞察（类平衡 > 不确定性/多样性）具有普遍意义

<!-- RELATED:START -->

## 相关论文

- [FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning](fecalfed_privacy-preserving_poultry_disease_detection_via_federated_learning.md)
- [FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)
- [Domain-Skewed Federated Learning with Feature Decoupling and Calibration](domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)
- [FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)
- [ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning](proxyfl_a_proxy-guided_framework_for_federated_semi-supervised_learning.md)

<!-- RELATED:END -->
