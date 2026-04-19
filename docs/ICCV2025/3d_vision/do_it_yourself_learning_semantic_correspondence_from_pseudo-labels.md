---
title: >-
  [论文解读] Do It Yourself: Learning Semantic Correspondence from Pseudo-Labels
description: >-
  [ICCV 2025][3D视觉][语义对应] 本文提出 DIY-SC，通过3D感知的伪标签生成策略（链式传播+松弛循环一致性+球面原型过滤）训练轻量适配器来改进基础模型特征的语义对应能力，在 SPair-71k 上实现了超越先前 SOTA 4.5%（PCK@0.1 per-keypoint）的性能，且无需人工关键点标注。
tags:
  - ICCV 2025
  - 3D视觉
  - 语义对应
  - 伪标签
  - 3D感知
  - 基础模型
  - 自训练
---

# Do It Yourself: Learning Semantic Correspondence from Pseudo-Labels

**会议**: ICCV 2025  
**arXiv**: [2506.05312](https://arxiv.org/abs/2506.05312)  
**代码**: [https://genintel.github.io/DIY-SC](https://genintel.github.io/DIY-SC)  
**领域**: 3D视觉 / 语义对应  
**关键词**: 语义对应, 伪标签, 3D感知, 基础模型, 自训练

## 一句话总结

本文提出 DIY-SC，通过3D感知的伪标签生成策略（链式传播+松弛循环一致性+球面原型过滤）训练轻量适配器来改进基础模型特征的语义对应能力，在 SPair-71k 上实现了超越先前 SOTA 4.5%（PCK@0.1 per-keypoint）的性能，且无需人工关键点标注。

## 研究背景与动机

**领域现状**：语义对应（semantic correspondence）是计算机视觉中的经典问题——在不同实例间找到语义对应点。近年来，大型预训练视觉模型（DINO、Stable Diffusion）展示了令人惊喜的零样本语义匹配能力。

**现有痛点**：
   - 基础模型特征在对称物体和重复部件上存在歧义性（如汽车左右轮不可区分）
   - 监督方法（如 TLR）依赖人工关键点标注，标注稀缺且难以扩展到更大数据集
   - 弱监督方法（如 SphMap）通过球面映射解决对称性问题，但球形先验对复杂拓扑物体（如动物）效果差，且需要手动调权重

**核心矛盾**：基础模型编码了丰富的语义知识，但简单的特征拼接或加权平均无法最优利用；而有效利用这些知识需要监督信号，但人工标注成本高且不可扩展。

**本文目标** 不依赖人工关键点标注定义，仅用弱3D监督信号（类别、mask、粗略 pose），通过自生成的伪标签训练适配器来提升语义对应性能。

**切入角度**：零样本匹配在小视角差异时效果好，大视角差时显著下降。利用这一观察，先在小视角差的图像对上生成高质量伪标签，然后通过链式传播获得大视角差的伪标签。

**核心 idea**：用基础模型生成伪标签→通过3D感知链式传播+循环一致性+球面过滤提升伪标签质量→用高质量伪标签训练轻量适配器。

## 方法详解

### 整体框架

DIY-SC 分为两个阶段：（1）伪标签生成与过滤——利用方位角采样、链式传播、松弛循环一致性和球面原型过滤生成高质量匹配对；（2）监督训练——用伪标签训练轻量适配器 $f_p$ 来精炼基础模型特征 $\tilde{\mathcal{F}} = [\mathcal{F}^{DINO}, \mathcal{F}^{SD}]$。

### 关键设计

1. **3D感知的图像对采样与链式传播**

    - 功能：为视角差异大的图像对生成高质量伪标签。
    - 核心思路：观察到零样本匹配在视角差 $< 45°$ 时效果良好（PCK@0.1 为 75.9%），差 $> 90°$ 时急剧下降（54.0%）。因此构建 $K$-元组 $(I_1, ..., I_K)$，每对相邻图像视角差 $< 90°$，通过递归最近邻 $\mathcal{P}^{k+1} = \text{NN}^{k \to k+1}(\mathcal{P}^k)$ 将匹配从"容易对"传播到"困难对"。选 $K=4$ 覆盖完整 $180°$ 视角变化范围。
    - 设计动机：直接在大视角差图像对上做 NN 匹配错误率高；链式传播利用"小步累积"策略，每一步匹配质量有保障。

2. **松弛循环一致性约束**

    - 功能：过滤链式传播中的虚假匹配。
    - 核心思路：传统循环一致性要求 $\text{NN}^{t \to s}(\text{NN}^{s \to t}(p_i^s)) = p_i^s$（严格等式），但零样本匹配难以精确返回原位置。本文放松为 $\|\hat{p}_i^s - p_i^s\|_2 < r_{max}$，允许一个特征patch的偏差，在链的每一段迭代应用。
    - 设计动机：严格循环一致性会拒绝太多有效匹配，松弛版本在保持过滤效果的同时保留更多正确匹配。

3. **规范化球面3D先验过滤**

    - 功能：利用球面映射器 $f_s$ 将 DINO 特征映射到规范化球面 $\mathcal{S}^2$ 上，拒绝映射到球面不同区域的匹配对。
    - 核心思路：对每个匹配对计算球面位置 $\Psi^s = f_s(\mathcal{F}^{DINO}(\mathcal{P}^s))$ 和 $\Psi^t = f_s(\mathcal{F}^{DINO}(\mathcal{P}^t))$，如果球面距离 $\text{sim}(\psi_i^s, \psi_i^t) < \theta_{th}$（$\theta_{th} < 0.15\pi$），则拒绝该匹配。
    - 设计动机：与 SphMap 直接将球面特征融入匹配不同（会破坏原始特征的定位精度），本文仅用球面信息做"删除"——删除不正确的伪标签，不干扰原始零样本匹配的质量。这避免了因球形先验不适合某些物体类别而导致的性能退化。

4. **适配器监督训练**

    - 功能：用伪标签训练4层 bottleneck 适配器（5M参数）精炼基础模型特征。
    - 核心思路：使用两个损失函数——稀疏对比损失 $\mathcal{L}_{sparse} = CL(\mathcal{F}^s(\mathcal{P}^s), \mathcal{F}^t(\mathcal{P}^t))$ 最大化匹配点相似度同时最小化非匹配点相似度；密集损失 $\mathcal{L}_{dense} = \sum \|\hat{p}_i^t - (p_i^t + \epsilon)\|_2$ 通过 WindowSoftArgmax 将梯度也传播到无标签区域。
    - 设计动机：稀疏损失直接优化特征区分度；密集损失确保特征图中无标签区域也得到优化，两者配合实现全局特征改进。

### 损失函数 / 训练策略

- 训练使用 AdamW 优化器，权重衰减 0.001，学习率 $5 \times 10^{-3}$，one-cycle 调度器训练 200K 步。
- 每个类别生成 30K 伪标签图像对，每对最多 50 个随机采样关键点。
- 输入分辨率：SD 用 $960^2$，DINOv2 用 $840^2$，特征图分辨率 $60 \times 60$。

## 实验关键数据

### 主实验：SPair-71k 上的 PCK@0.1 结果（per-keypoint）

| 方法 | 监督类型 | PCK@0.1 (avg) |
|------|---------|---------------|
| SD + DINOv2 零样本 | 无 | 64.0 |
| DistillDIFT (U.S.) | 无监督 | 65.1 |
| SphMap† | 3D弱监督 | 67.8 |
| TLR | 关键点标签 | 69.6 |
| DistillDIFT (W.S.) | 关键点标签 | 70.6 |
| **DIY-SC (Ours)** | **伪标签+3D弱监督** | **74.4** |
| **DIY-SC (IN3D→SPair)** | **伪标签+3D弱监督** | **75.1** |

在对称/重复部件类别上改进最大：bus +15.7%, car +14.0%。

### 消融实验

| 伪标签 | 循环一致 | 松弛CC | 链式传播 | 球面过滤 | PCK@0.1 |
|--------|---------|--------|---------|---------|---------|
| | | | | | 65.0 (零样本) |
| ✓ | | | | | 67.2 |
| ✓ | ✓ | | | | 66.9 |
| ✓ | | ✓ | | | 68.4 |
| ✓ | | ✓ | ✓ | | 70.0 |
| ✓ | | | | ✓ | 72.9 |
| ✓ | | ✓ | ✓ | ✓ | **74.4** |

### 关键发现
- **朴素伪标签已有效**（+2.2%）：仅用 NN 匹配生成伪标签训练适配器就能改进，因为学习组合 SD+DINO 特征优于简单拼接。
- **松弛循环一致性优于严格版本**（68.4 vs 66.9）：严格版本拒绝太多有效匹配。
- **球面过滤贡献最大**（+5.7%/独立使用，+4.4%/在链式之上叠加）：有效解决对称性和重复部件歧义。
- **扩展到更大数据集（ImageNet-3D）进一步改善**：在未见过 SPair-71k 的情况下就超越先前 SOTA，展示了强泛化能力。
- AP-10k 跨数据集评估也超越 SOTA，且不像监督方法那样出现严重过拟合（PCK@0.1: 70.6 vs 监督68.3）。

## 亮点与洞察

- **伪标签+质量控制的通用范式**：先用零样本方法生成伪标签，再通过多重过滤提升质量，最后用于监督训练。这一范式不局限于语义对应，可迁移到其他需要标注的任务。
- **"仅删除不调制"的球面先验使用方式**：不同于 SphMap 将球面特征和原始特征加权混合（会伤害定位精度），本文仅用球面信息删除错误伪标签，巧妙避免了副作用。
- **链式传播思路**：将一个困难问题分解为多个简单子问题，每步保证质量，累积解决全局难题——这种分而治之的策略在很多领域都有应用前景。

## 局限与展望

- 球面先验对复杂拓扑物体仍不够理想（虽然仅用作过滤降低了影响），可考虑更灵活的3D先验。
- 依赖粗粒度方位角标注进行采样，进一步减少监督需求仍有空间。
- 链式传播可能在极长链中累积误差，限制了极端视角变化下的性能。
- 仅在物体级语义对应上验证，未扩展到场景级对应。

## 相关工作与启发

- **vs SphMap**: SphMap 通过加权球面特征与原始特征解决对称性，但对非刚体物体效果差且需调权重；DIY-SC 仅用球面做过滤，不污染原始特征，效果更好且更鲁棒。
- **vs TLR**: TLR 使用数据集特定的关键点标签定义来区分左右，不可迁移到其他数据集；DIY-SC 无需关键点标签即超越其性能。
- **vs DistillDIFT**: DistillDIFT 在3D实例数据上微调特征，但难以泛化到跨实例匹配；DIY-SC 通过跨实例伪标签训练，泛化能力更强。

## 评分

- 新颖性: ⭐⭐⭐⭐ 链式传播+松弛循环一致性+球面过滤的组合方式新颖，但各组件单独来看多为已有技术
- 实验充分度: ⭐⭐⭐⭐⭐ 消融极为全面（逐组件、逐类别、跨数据集），定性分析深入
- 写作质量: ⭐⭐⭐⭐⭐ 动机推导逻辑严密，方法描述清晰，图示直观
- 价值: ⭐⭐⭐⭐⭐ 新 SOTA + 方法可扩展到更大数据集 + 对伪标签方法论有深刻洞察

<!-- RELATED:START -->

## 相关论文

- [CasP: Improving Semi-Dense Feature Matching Pipeline Leveraging Cascaded Correspondence Priors for Guidance](casp_improving_semi-dense_feature_matching_pipeline_leveraging_cascaded_correspo.md)
- [SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images](spatialsplat_efficient_semantic_3d_from_sparse_unposed_images.md)
- [Monocular Semantic Scene Completion via Masked Recurrent Networks](monocular_semantic_scene_completion_via_masked_recurrent_networks.md)
- [Object-Centric Representation Learning for Enhanced 3D Semantic Scene Graph Prediction](../../NeurIPS2025/3d_vision/object-centric_representation_learning_for_enhanced_3d_semantic_scene_graph_pred.md)
- [Cross-View Completion Models are Zero-shot Correspondence Estimators](../../CVPR2025/3d_vision/cross-view_completion_models_are_zero-shot_correspondence_estimators.md)

<!-- RELATED:END -->
