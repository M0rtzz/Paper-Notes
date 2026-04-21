---
title: >-
  [论文解读] Federated Active Learning Under Extreme Non-IID and Global Class Imbalance
description: >-
  [CVPR 2026][AI安全][Federated Active Learning] 系统研究了联邦主动学习中查询模型选择问题，发现类别平衡采样是性能关键因素，并提出 FairFAL 框架，通过自适应模型选择、原型引导伪标签和不确定性-多样性平衡采样实现公平高效的联邦主动学习。
tags:
  - CVPR 2026
  - AI安全
  - Federated Active Learning
  - Non-IID
  - Class Imbalance
  - active learning
  - Long-Tailed Distribution
---

# Federated Active Learning Under Extreme Non-IID and Global Class Imbalance

**会议**: CVPR 2026  
**arXiv**: [2603.10341](https://arxiv.org/abs/2603.10341)  
**代码**: [chenchenzong/FairFAL](https://github.com/chenchenzong/FairFAL)  
**领域**: AI安全 / 联邦学习  
**关键词**: Federated Active Learning, Non-IID, Class Imbalance, active learning, Long-Tailed Distribution

## 一句话总结

系统研究了联邦主动学习中查询模型选择问题，发现类别平衡采样是性能关键因素，并提出 FairFAL 框架，通过自适应模型选择、原型引导伪标签和不确定性-多样性平衡采样实现公平高效的联邦主动学习。

## 研究背景与动机

联邦主动学习（FAL）结合了联邦学习的隐私保护和主动学习的标签效率，但在现实场景中面临两个被忽视的严峻挑战：

**全局类别严重不平衡**：真实联邦系统通常呈现长尾全局分布，稀有但关键的类别在各客户端中稀疏出现

**客户端高度异构**：不同客户端的数据分布差异极大（极端 Non-IID）

现有 FAL 方法（如 LoGo、KAFAL、IFAL）虽然开始考虑 Non-IID 场景，但通常将异构性仅视为数据分区问题，隐含假设全局标签分布相对平衡。在长尾全局分布下，现有采集策略难以捕获少数类样本，导致标注预算浪费。

本文提出一个根本性问题：**在 FAL 中，全局模型和本地模型哪个更适合作为查询选择器？这与采样的类别平衡性有何关系？**

## 方法详解

### 整体框架

FairFAL 基于三个实证观察设计了三个核心组件：

- **观察1**：对于不确定性采样，除全局严重不平衡+客户端近似同质的情况外，本地模型通常优于全局模型 → 自适应模型选择
- **观察2**：无论哪个模型，类别平衡采样（尤其是少数类获取）越好，最终性能越优 → 类别感知采样
- **观察3**：对于多样性采样，全局模型在所有设置下都一致优于本地模型 → 全局特征引导多样性

### 关键设计

1. **自适应模型选择（Adaptive Model Selection）**：通过轻量级预测差异估计全局不平衡程度和本地-全局分布偏差，自适应选择查询模型。

   **全局类别不平衡估计**：对每个客户端构建类别平衡子集 $\mathcal{B}^{(k)}$，利用全局模型预测先验 $\hat{\boldsymbol{\pi}}_g^{(k)}$ 估计不平衡比：
   $$\gamma_k = \frac{\min_{c \in \mathcal{C}_k^+} \hat{\pi}_{g,c}}{\max_{c \in \mathcal{C}_k^+} \hat{\pi}_{g,c}} \in (0,1]$$
   各客户端上传标量 $\gamma_k$，服务器平均得到全局系数 $\bar{\gamma}$（仅第一轮计算）。

   **本地-全局分布偏差估计**：
   $$d_k = \frac{1}{C}\sum_{c=1}^{C}\frac{|\hat{\pi}_{g,c} - \hat{\pi}_{\ell,c}^{(k)}|}{\hat{\pi}_{g,c} + \hat{\pi}_{\ell,c}^{(k)}}$$

   **模型选择分数**：$s_k = 1 - \frac{1}{2}(d_k + \bar{\gamma})$，当 $s_k > \delta=0.75$ 时选全局模型，否则选本地模型。直觉：全局严重不平衡（$\bar{\gamma}$ 小）且本地与全局分布接近（$d_k$ 小）时，$s_k$ 大，选全局模型。

2. **原型引导伪标签（Prototype-Guided Pseudo-Labeling）**：使用全局模型特征构建每类原型提供更可靠的类别分配，克服分类器在不平衡数据上的偏差。

   类别原型：$\boldsymbol{\mu}_c^{(k)} = \frac{1}{|\mathcal{D}_{L,c}^{(k)}|}\sum_{y_i=c} \mathbf{z}_i^{(k)}$，其中 $\mathbf{z}_i^{(k)} = \frac{\phi^g(x_i)}{\|\phi^g(x_i)\|_2}$ 为全局模型的归一化特征。

   伪标签通过余弦相似度分配：$\hat{y}^{(k)}(x) = \arg\max_c \langle \mathbf{z}^{(k)}(x), \boldsymbol{\mu}_c^{(k)} \rangle$
   
   基于伪标签将未标注池划分为类别子集 $\widetilde{\mathcal{D}}_{U,c}^{(k)}$，为后续类别感知采样奠定基础。

3. **不确定性-多样性平衡采样（Two-Stage Balanced Sampling）**：
   
   **阶段1-类别候选选择**：对每类以均匀预算 $b_c^{(k)}$ 分配标注配额，选取 $\kappa \cdot b_c^{(k)}$ 个最高不确定性样本组成过完备候选池 $\mathcal{H}_c^{(k)}$（$\kappa=4$）
   
   **阶段2-多样性精炼**：在全局模型的梯度嵌入空间 $\mathbf{g}^{(k)}(x) = \psi(x; \phi^g, f^g)$ 中执行 $k$-center 采样，最小化最大距离：
   $$\mathcal{Q}_c^{(k)} = \arg\min_{\mathcal{Q}'} \max_{x \in \mathcal{H}_c^{(k)}} \min_{a \in \mathcal{A}_c^{(k)} \cup \mathcal{Q}'} d(\mathbf{g}^{(k)}(x), \mathbf{g}^{(k)}(a))$$
   使用贪心 $k$-center 算法获得 2-近似解。

### 损失函数 / 训练策略

- 标准联邦学习训练：FedAvg 框架 + 本地 SGD
- 每轮 FAL 包含完整的联邦训练过程和主动查询
- 每轮查询 5% 训练数据进行标注
- 第一轮随机查询，后续轮次使用 FairFAL 策略

## 实验关键数据

### 主实验

数据集：FMNIST / CIFAR-10 / CIFAR-100，全局不平衡比 $\rho=20$，10 个客户端。

**CIFAR-10, final-round accuracy (α=0.1, ρ=20)**:

| 方法 | 15% | 25% | 35% | 45% |
|------|-----|-----|-----|-----|
| Random | 47.24 | 50.46 | 54.29 | 55.70 |
| KAFAL | 49.99 | 56.34 | 58.41 | 60.01 |
| LoGo | 51.56 | 56.35 | 58.30 | 59.68 |
| IFAL | 47.76 | 52.67 | 55.62 | 57.51 |
| **FairFAL** | **52.12** | **56.90** | **59.62** | **60.44** |

**医学数据集 (α=0.1)**：OctMNIST 上 FairFAL 72.80% vs KAFAL 70.40%；DermaMNIST 上 FairFAL 73.77% vs LoGo 73.62%。

FairFAL 在所有数据集和异构性设置下均一致优于所有基线，且任务难度越大优势越明显。

### 消融实验

| 配置 | (α=0.1, ρ=20) | (α=100, ρ=20) | 说明 |
|------|:---:|:---:|------|
| 自适应模型选择 $\mathcal{M}^{(k)}$ | 59.33 | 63.65 | 选对的查询模型 |
| 替代模型 $\widetilde{\mathcal{M}}^{(k)}$ | 58.49 | 61.89 | 选错的 → 差 0.84~1.76% |
| + 类别采样 (Local 原型) | 59.14 | 63.39 | 本地原型质量较低 |
| + 类别采样 (Global 原型) | 59.95 | 64.02 | 全局原型更准确 (+0.63~0.81%) |
| + 两阶段平衡采样 (κ=2) | 60.61 | 64.60 | κ=2 略优但差异小 |
| + 两阶段平衡采样 (κ=4, Final) | **60.44** | **64.57** | 完整 FairFAL，更灵活的候选池 |

### 关键发现

- **观察的普适性**：类别平衡采样 → 更好性能的规律在所有实验设定中成立
- **自适应选择的必要性**：使用"选错"的模型性能比"选对"的差 0.84~1.76%
- **全局原型优于本地原型**：全局模型的特征表示更具判别性和全局一致性
- **医学数据验证**：在 OctMNIST 和 DermaMNIST（自然长尾分布）上，FairFAL 同样最优（72.80% vs 70.40%）
- **现有方法的崩溃**：在 α=100（近同质客户端）下，IFAL 等缺乏显式类别平衡机制的方法甚至不如随机采样

## 亮点与洞察

1. **系统性实证研究**：首次系统研究了 FAL 中全局/本地查询模型选择问题，提出了三个有价值的观察，基于统计检验（Wilcoxon 检验 + Hodges-Lehmann 估计）而非简单均值比较
2. **观察驱动的方法设计**：每个组件都有明确的实证动机，设计逻辑清晰
3. **隐私保护**：自适应模型选择仅需上传标量 $\gamma_k$，无额外隐私泄露
4. **实用性强**：方法是模块化的，可以组合使用，$\kappa$ 参数对结果不敏感

## 局限与展望

1. **$\delta$ 阈值固定为 0.75**：可能对特定场景不够灵活，可考虑自适应调整
2. **第一轮假设**：假设第一轮随机查询的标注集是 IID 近似，在极端不平衡下可能不成立
3. **仅验证分类任务**：检测、分割等更复杂的任务下的效果未知
4. **客户端规模**：仅测试了 10 个客户端配置
5. **类别数限制**：CIFAR-100 仅 100 类，超大规模标签空间（如 ImageNet-21k）下的表现未验证

## 相关工作与启发

- **BADGE**：不确定性+多样性的经典两阶段采样，FairFAL 在此基础上引入了类别感知机制
- **LoGo**：局部聚类+全局不确定性评分的 FAL 方法，但未考虑全局类别不平衡
- **KAFAL/IFAL**：利用全局-本地预测差异引导采集，但缺乏类别平衡设计在极端不平衡下失效
- **本文启发**：类别平衡是 FAL 的核心，而非仅仅追求不确定性或多样性；全局模型在特征表示层面的优势可用于原型计算

## 评分

- **新颖性**: ⭐⭐⭐⭐ 实证观察有深度，方法设计有清晰的理论逻辑
- **实验充分度**: ⭐⭐⭐⭐ 5个数据集、多种配置、统计检验、消融完整
- **写作质量**: ⭐⭐⭐⭐⭐ 论文结构极好：观察→设计→验证，叙述流畅
- **价值**: ⭐⭐⭐⭐ 填补了 FAL 在极端不平衡+非IID 场景下的空白，对实际部署有指导意义

<!-- RELATED:START -->

## 相关论文

- [FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning](fecalfed_privacy-preserving_poultry_disease_detection_via_federated_learning.md)
- [Domain-Skewed Federated Learning with Feature Decoupling and Calibration](domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)
- [FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)
- [ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning](proxyfl_a_proxy-guided_framework_for_federated_semi-supervised_learning.md)
- [FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)

<!-- RELATED:END -->
