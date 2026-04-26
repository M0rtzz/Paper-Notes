---
title: >-
  [论文解读] UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning
description: >-
  [CVPR 2025][多模态][转导式少样本学习] 提出UNEM，将广义EM（GEM）算法的每次迭代展开为神经网络的一层，通过端到端学习自动优化**类平衡超参数λ**和**温度缩放T**，在11个细粒度数据集上实现vision-language设置下平均77.8%的准确率（vs. EM-Dirichlet的73.6%），vision-only设置下提升最高达10%。
tags:
  - CVPR 2025
  - 多模态
  - 转导式少样本学习
  - EM算法展开
  - 超参数学习
  - CLIP
  - Dirichlet分布
---

# UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning

**会议**: CVPR 2025  
**arXiv**: [2412.16739](https://arxiv.org/abs/2412.16739)  
**代码**: https://github.com/ZhouLong0/UNEM-Transductive (有)  
**领域**: 多模态VLM  
**关键词**: 转导式少样本学习, EM算法展开, 超参数学习, CLIP, Dirichlet分布

## 一句话总结
提出UNEM，将广义EM（GEM）算法的每次迭代展开为神经网络的一层，通过端到端学习自动优化**类平衡超参数λ**和**温度缩放T**，在11个细粒度数据集上实现vision-language设置下平均77.8%的准确率（vs. EM-Dirichlet的73.6%），vision-only设置下提升最高达10%。

## 研究背景与动机
转导式少样本学习（transductive FSL）通过联合推断一批query样本的类别来利用无标注数据的统计信息，显著优于独立推断每个样本的归纳式方法。然而，现有转导式方法引入了关键超参数（特别是控制类平衡程度的λ），其最优值因数据集和预训练模型而异。

**现有痛点**：如Fig.1所示，在EM-Dirichlet算法中，类平衡超参数λ对准确率影响巨大，且其最优值在不同数据集间可能相差数个数量级（如Food101的最优λ约为4，而SUN397约为2000）。目前的做法是在验证集上穷举网格搜索，这不仅**计算不可行**（尤其多个超参数组合时），还可能因搜索范围不够细而**次优**。

**核心矛盾**：超参数对性能影响巨大 vs. 穷举搜索不可行。特别是当同时考虑λ（类平衡）和T（预测soft/hard程度）两个超参数时，二维网格搜索更加棘手。

**切入角度**：引入"学习优化"（learning to optimize）范式——将迭代优化算法展开为神经网络，超参数变为可学习参数，通过反向传播自动优化。

## 方法详解

### 整体框架
UNEM将广义EM算法的L次迭代展开为L层神经网络。每层对应一次EM迭代，包含三个步骤：更新分布参数$\theta_k$、更新类比例$\pi_k$、更新分配向量$u_n$。每层有独立的超参数$(λ^{(\ell)}, T^{(\ell)})$，通过在验证集上最小化交叉熵损失来学习。支持高斯分布（vision-only模型）和Dirichlet分布（CLIP等vision-language模型）。

### 关键设计

1. **广义EM目标函数（Generalized EM Formulation）**
    - 功能：统一现有转导式少样本方法为特殊情况的通用框架
    - 核心思路：优化目标 $\min_{u,\theta} \mathcal{L}(u,\theta) + \lambda\Psi(u) + T\Phi(u)$，其中：
        - $\mathcal{L}$：负对数似然（聚类项），对类平衡有隐式偏好
        - $\Psi$：类分布的Shannon熵，$\Psi(u) = -\sum_k \pi_k \ln \pi_k$，控制类平衡
        - $\Phi$：分配向量的熵屏障，$\Phi(u) = \sum_n \sum_k u_{n,k} \ln u_{n,k}$，控制预测的软硬程度
    - 设计动机：当$T=1, \lambda=|\mathbb{Q}|$时恢复标准EM；当$T=1$时恢复EM-Dirichlet。两个超参数$\lambda$和$T$的显式化使得它们可以被学习
    - 分配更新闭式解：$u_n^{(\ell+1)} = \text{softmax}\left(\frac{1}{T}\left(\ln p(z_n|\theta_k^{(\ell+1)}) + \frac{\lambda}{|\mathbb{Q}|}\ln(\pi_k^{(\ell+1)})\right)_k\right)$

2. **算法展开架构（Unrolling Architecture）**
    - 功能：将迭代算法转化为可端到端训练的神经网络，自动学习超参数
    - 核心思路：L次迭代 → L层网络，每层$\mathcal{L}^{(\ell)}$执行GEM的一次更新，第$\ell$层的超参数$(λ^{(\ell)}, T^{(\ell)})$独立可学习。网络仅有2L个可学习参数（对于L=10仅20个参数）
    - 设计动机：
        - 超参数可以**逐层变化**——迭代早期和后期可能需要不同的类平衡强度和温度
        - 通过梯度下降学习，比网格搜索更高效且可以找到全局更优的配置
        - 参数约束：$\lambda^{(\ell)} = \text{Softplus}(a^{(\ell)})$保证非负；$T^{(\ell)} = 1 + \text{Softplus}(b^{(\ell)})$保证$\geq 1$避免梯度消失
    - 训练损失：标准交叉熵 $L_c(w) = \sum_{n \in \mathbb{Q}} \sum_k y_{n,k} \log(u_{n,k}^{(L)})$

3. **双分布模型支持（Gaussian + Dirichlet）**
    - 功能：使UNEM能同时适用于传统vision-only预训练模型和CLIP等vision-language模型
    - 核心思路：
        - **UNEM-Gaussian**：假设特征$z_n$服从高斯分布$p(z_n|\theta_k) \propto \exp(-\frac{1}{2}\|z_n-\theta_k\|^2)$，参数$\theta_k$即类中心，用于ResNet/WRN等backbone
        - **UNEM-Dirichlet**：假设CLIP的softmax概率输出服从Dirichlet分布$p(z_n|\theta_k) = \frac{1}{\mathcal{B}(\theta_k)}\prod_i z_{n,i}^{\theta_{k,i}-1}$，用于CLIP
    - 设计动机：CLIP输出单纯形上的概率向量，高斯假设不再合适。Dirichlet分布自然建模单纯形上的数据，EM-Dirichlet已被验证有效

## 实验关键数据

### Vision-Only：mini-ImageNet（Tab. 1, WRN28-10 backbone）

| 方法 | 5-shot | 10-shot | 20-shot |
|------|--------|---------|---------|
| PADDLE | 62.6 | 73.0 | 79.2 |
| α-TIM | 71.5 | 75.2 | 78.3 |
| **UNEM-Gaussian** | **71.6** | **79.2** | **83.7** |
- 20-shot下超越PADDLE 4.5个百分点

### Vision-Only：tiered-ImageNet（160类，Tab. 1, WRN28-10）

| 方法 | 5-shot | 10-shot | 20-shot |
|------|--------|---------|---------|
| PADDLE | 43.9 | 59.4 | 69.9 |
| **UNEM-Gaussian** | **54.1** | **66.8** | **74.7** |
- 5-shot下提升10.2个百分点！

### Vision-Language：11个细粒度数据集（Tab. 3, CLIP, 4-shot）

| 方法 | Food101 | Flowers | Cars | SUN397 | ImageNet | 平均 |
|------|---------|---------|------|--------|----------|------|
| Tip-Adapter (归纳) | 76.7 | 83.2 | 63.9 | 66.7 | 62.7 | 68.3 |
| EM-Dirichlet (转导) | 88.7 | 91.3 | 73.5 | 80.9 | 78.4 | 73.6 |
| **UNEM-Dirichlet** | **91.4** | **95.6** | **80.0** | **88.5** | **83.1** | **77.8** |
- 平均提升4.2个百分点，在Cars上提升6.5，SUN397提升7.6

### CUB细粒度鸟类分类（Tab. 2）

| 方法 | 5-shot | 10-shot | 20-shot |
|------|--------|---------|---------|
| PADDLE | 71.2 | 81.8 | 86.8 |
| **UNEM-Gaussian** | **78.5** | **85.3** | **88.6** |

### 关键消融发现
- 逐层可变超参数 vs. 全局固定超参数：逐层可变始终更优，验证了超参数应随迭代变化的假设
- 学到的λ值在不同数据集间差异显著（如ImageNet~300 vs. SUN397~2000），但UNEM可自动适配
- 仅20个可学习参数（L=10层×2个超参数），训练极其轻量

## 亮点与洞察
- **算法展开首次用于少样本学习的超参数优化**：将"学习优化"范式引入FSL领域，解决了长期困扰该领域的超参数搜索难题
- **极致轻量**：整个网络仅20个可学习参数（传统深度学习动辄百万参数），却带来显著性能提升
- **统一框架**：GEM公式将标准EM、EM-Dirichlet、α-TIM等方法统一为特殊情况，理论贡献突出
- **实用性强**：无需针对每个新数据集重新网格搜索超参数，学一次即可泛化——大幅降低实际部署成本
- **逐层超参数变化的洞察**：迭代早期需要更强的类平衡约束，后期需要更hard的分配——这一发现对理解EM优化动态有理论价值

## 局限性
- 展开层数L=10是固定的，对于某些任务可能不够或过多
- 仅验证了高斯和Dirichlet两种分布模型，对其他指数族分布的推广未验证
- 验证集上训练超参数需要一定量的标注数据（虽然远少于网格搜索）
- 仅关注图像分类，未扩展到目标检测、分割等其他视觉任务
- 当query集类别数$K_{eff}$变化时，需要重新训练展开网络

## 相关工作与启发
- EM-Dirichlet → 本文的直接基线，UNEM使其超参数自动化
- 算法展开（LISTA、ADMM-Net）→ 将优化迭代转化为可学习网络层的通用范式，首次用于FSL
- CLIP转导式方法非常匮乏 → 本文填补了CLIP+转导式FSL的方法空白
- 启发：迭代优化算法中的超参数不应是静态的——它们在不同迭代阶段可能需要不同的值，而展开范式自然支持这种灵活性

## 评分
⭐⭐⭐⭐ — 方法思路清晰优雅，将经典的算法展开范式创新性地应用于少样本学习，用极少的可学习参数实现了显著的性能提升。统一的GEM框架理论贡献扎实，实验覆盖全面。在CLIP转导式少样本学习的研究空白中做出重要贡献。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)
- [\[CVPR 2025\] Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning](visual_and_semantic_prompt_collaboration_for_generalized_zero-shot_learning.md)
- [\[CVPR 2025\] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](rethinking_few-shot_adaptation_of_vision-language_models_in_two_stages.md)
- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](../../ICCV2025/multimodal_vlm/causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [\[CVPR 2025\] Single Domain Generalization for Few-Shot Counting via Universal Representation Matching](single_domain_generalization_for_few-shot_counting_via_universal_representation_.md)

<!-- RELATED:END -->
