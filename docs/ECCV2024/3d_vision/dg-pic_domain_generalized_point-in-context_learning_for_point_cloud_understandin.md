---
title: >-
  [论文解读] DG-PIC: Domain Generalized Point-In-Context Learning for Point Cloud Understanding
description: >-
  [ECCV 2024][3D视觉][点云理解] 提出 DG-PIC，首个在统一模型中同时处理多领域多任务的点云理解框架，通过双层次源域原型估计和测试时特征平移机制，在不更新模型的情况下提升对未知域的泛化能力。
tags:
  - ECCV 2024
  - 3D视觉
  - 点云理解
  - 领域泛化
  - 上下文学习
  - 多任务学习
  - 测试时自适应
---

# DG-PIC: Domain Generalized Point-In-Context Learning for Point Cloud Understanding

**会议**: ECCV 2024  
**arXiv**: [2407.08801](https://arxiv.org/abs/2407.08801)  
**代码**: https://github.com/Jinec98/DG-PIC (有)  
**领域**: 3D视觉  
**关键词**: 点云理解, 领域泛化, 上下文学习, 多任务学习, 测试时自适应

## 一句话总结

提出 DG-PIC，首个在统一模型中同时处理多领域多任务的点云理解框架，通过双层次源域原型估计和测试时特征平移机制，在不更新模型的情况下提升对未知域的泛化能力。

## 研究背景与动机

**领域现状**：点云理解在自动驾驶、机器人等领域应用广泛，但模型通常在单一数据集上训练和测试。当面对分布不同的新数据时（如从合成数据 ModelNet40 到真实扫描 ScanObjectNN），性能会显著下降。

**现有痛点**：
   - **领域泛化 (DG) 方法**通常只针对单一任务设计（如分类），缺乏多任务处理能力，且忽略了测试数据本身的利用价值
   - **上下文学习 (ICL) 方法**（如 PIC）可以做多任务，但局限于单一数据集，跨域泛化能力差
   - 两类方法都无法同时解决"多域"和"多任务"问题

**核心矛盾**：统一模型需要兼顾任务泛化（多任务）和域泛化（多域），而现有方法只能二选一。

**本文目标** 在一个统一模型中处理多个领域和多个任务的点云理解，并在测试时不更新模型参数即可泛化到未知域。

**切入角度**：将 PIC 的多任务 ICL 与测试时领域泛化结合——预训练阶段用 PIC 学习跨域泛化信息，测试时通过特征平移将目标域拉向源域。

**核心 idea**：双层次源域原型 + 双层次测试时特征平移，无需模型更新即可将未知域测试数据对齐到已知源域。

## 方法详解

### 整体框架

DG-PIC 分为两个阶段：(1) 预训练阶段——基于 Masked Point Modeling (MPM) 框架，在多个源域上训练 PIC 模型，使用跨域 prompt 配对策略；(2) 测试阶段——冻结模型，通过双层次源域原型估计确定测试样本与各源域的距离，再通过双层次特征平移将测试数据对齐到源域，选择最近源域中最相似的样本作为 prompt。

### 关键设计

1. **多域 Prompt 配对 (Multi-domain Prompt Pairing)**:

    - **功能**：在预训练时从不同源域随机选取样本作为 prompt，增强跨域关联。
    - **核心思路**：设 query 来自域 $D_s^i$，prompt 来自域 $D_s^j (j \neq i)$，预测的 masked patch 为：
    $P \sim (D_s^i, D_s^j) = Trans([F_\theta(I_i) \oplus F_\theta(T_i^k) \oplus F_\theta(I_j) \oplus F_\theta(T_j^k)], Mask)$
      训练损失使用 Chamfer Distance：$\text{CD}(P,G) = \frac{1}{|P|}\sum_{x \in P}\min_{y \in G}\|x-y\|^2 + \frac{1}{|G|}\sum_{y \in G}\min_{x \in P}\|y-x\|^2$
    - **设计动机**：跨域配对迫使模型学习域不变的特征表示。

2. **双层次源域原型估计 (Dual-level Source Prototype Estimation)**:

    - **功能**：为每个源域计算全局和局部两个层次的原型，作为测试时特征对齐的锚点。
    - **核心思路**：
        - **局部原型** $Z_{local}^{i,m}$：对域 $D_s^i$ 中所有样本的 patch 级特征取平均：$Z_{local}^{i,m} = \frac{1}{N_{D_s^i}} \sum_{n=1}^{N_{D_s^i}} F_\theta(P_m)$
        - **全局原型** $Z_{global}^i$：对所有 patch 特征做 max pooling 后取平均：$Z_{global}^i = \frac{1}{N_{D_s^i}} \sum_{n=1}^{N_{D_s^i}} max(F_\theta(P_m))$
        - 计算测试样本到每个源域原型的欧氏距离：$\mathcal{E}_{global}^i = \|F_{global} - Z_{global}^i\|$，$\mathcal{E}_{local}^{i,m} = \|F_{local}^m - Z_{local}^{i,m}\|$
    - **设计动机**：全局特征捕获形状上下文，局部特征捕获几何结构细节，双层次才能全面表示源域。

3. **双层次测试时特征平移 (Dual-level Test-time Feature Shifting)**:

    - **功能**：在测试时将目标域特征向源域方向平移，无需更新模型参数。
    - **核心思路**：
        - **宏观语义系数 $\alpha$**：从全局距离导出，控制各源域对特征平移的贡献度：$\alpha = softmax(\mathcal{E}_{global})$
        - **微观位置系数 $\beta^i$**：从局部距离导出，考虑 patch 位置对齐关系：$\beta^i = softmax(\mathcal{E}_{local}^i)$
        - 最终特征平移公式：
    $F'_{local} = \frac{1}{R}\sum_{i=1}^{R}\alpha_i\left(\frac{1}{M}\sum_{m=1}^{M}\beta^{i,m}F_{local}^m\right) + \frac{1}{R}\sum_{i=1}^{R}(1-\alpha_i)\left(\frac{1}{M}\sum_{m=1}^{M}(1-\beta^{i,m})Z_{local}^{i,m}\right)$
    - **设计动机**：$\alpha$ 利用跨域语义相似性调节整体平移强度，$\beta$ 利用同位置 patch 的几何相似性进行精细对齐。同位置的 patch 即使跨域也应具有相似的几何结构（如桌子外围是边缘、内部是平面）。

4. **测试时 Prompt 选择**:

    - **功能**：从最近源域中选择最相似样本作为 prompt。
    - **核心思路**：综合全局和局部距离确定最近源域：$\mathcal{E}^i = \lambda \cdot \mathcal{E}_{global}^i + (1-\lambda) \cdot \frac{1}{M}\sum_{m=1}^{M}\mathcal{E}_{local}^{i,m}$（$\lambda=0.5$），在该域中找特征距离最小的样本作为 prompt。

### 损失函数 / 训练策略

- 预训练使用 Chamfer Distance 损失
- AdamW 优化器，lr=0.001，cosine 学习率调度
- 训练 300 epochs，batch size 128
- 每个点云采样 1024 点，分为 64 个 patch（每个 32 点），mask ratio 0.7
- **测试时完全不更新模型参数**

## 实验关键数据

### 主实验（ScanObjectNN 作为目标域，CD×10⁻³ ↓）

| 方法 | 设定 | 重建 | 去噪 | 配准 |
|------|------|------|------|------|
| **DG-PIC (Ours)** | 测试时 DG | **4.1** | **15.2** | **5.8** |
| PIC | 监督学习 | 72.9 | 80.0 | 12.7 |
| Point-MAE (task-specific) | 监督学习 | 30.4 | 36.0 | 31.2 |
| PCT (multi-task) | 监督学习 | 31.5 | 36.5 | 34.9 |
| PointCutMix (task-specific) | 训练时 DG | 44.8 | 43.5 | 41.3 |

### 消融实验（CD×10⁻³ ↓）

| 模型 | 原型估计 | 特征平移 | 锚定域 | 重建 | 去噪 | 配准 |
|------|---------|---------|--------|------|------|------|
| Model A | 随机 | 平均 | 单域 | 8.4 | 40.5 | 6.7 |
| Model B | 仅全局 | 平均 | 单域 | 7.2 | 38.3 | 6.4 |
| Model C | 仅局部 | 平均 | 单域 | 7.3 | 36.7 | 6.7 |
| Model D | 全局+局部 | 平均 | 单域 | 6.8 | 35.1 | 6.2 |
| Model E | 全局+局部 | 平均 | 全域 | 6.3 | 32.4 | 6.5 |
| Model F | 全局+局部 | 仅宏观 | 全域 | 5.2 | 22.7 | 6.0 |
| Model G | 全局+局部 | 仅微观 | 全域 | 4.9 | 25.6 | 6.2 |
| **Ours** | **全局+局部** | **宏观+微观** | **全域** | **4.1** | **15.2** | **5.8** |

### 关键发现

- DG-PIC 在所有三个任务上大幅超越所有对比方法，重建任务上 CD 仅为 PIC 的 5.6%（4.1 vs 72.9）
- 传统方法（PointNet, DGCNN 等）跨域泛化能力差，CD 值普遍在 30-45 范围
- DG 方法（Pointmixup, PointCutMix）虽引入域泛化，但各任务结果方差小，说明它们只关注域间差异而忽视任务差异
- PIC 能做多任务但跨域失败（CD 72.9-80.0），证明 ICL 不足以解决域差距
- 双层次设计（全局+局部原型、宏观+微观平移）的每个组件都有独立贡献，去噪任务受益最大

## 亮点与洞察

- **开创性设定**：首次提出多域多任务点云理解的设定，并构建了包含 4 个数据集（2 合成 + 2 真实）、7 类物体、3 个任务、30,954 样本的基准测试
- **测试时泛化无需模型更新**：通过特征空间平移而非微调实现域适应，计算开销可控
- **微观位置系数的设计直觉精妙**：同一物体同位置的 patch 跨域应有相似几何结构，这一先验被有效利用
- **统一模型做三个任务**：重建、去噪、配准共享一个网络，通过 prompt 切换任务

## 局限与展望

- benchmark 仅包含 7 个共有类别，规模和多样性有限
- 仅考虑了 xyz 坐标回归型任务（重建/去噪/配准），未涵盖分类、分割等判别型任务
- 源域原型为所有样本的简单平均，可能丢失类内多模态分布信息
- 特征平移公式较为启发式，缺乏理论分析
- 仅在 ScanObjectNN 作为目标域上做了充分实验，其他目标域配置的消融不足

## 相关工作与启发

- **vs PIC**: PIC 是单域多任务 ICL 模型，DG-PIC 在此基础上引入测试时域泛化，性能提升巨大（CD: 72.9→4.1 在重建任务）
- **vs Point-BERT / Point-MAE**: 这些自监督方法虽能学好特征表示，但未考虑跨域泛化，直接迁移效果差
- **vs Pointmixup / PointCutMix**: 训练时 DG 方法通过混合数据增强，但仅做"域不变"而忽略"任务不变"，结果平庸
- **vs DGLSS / SemanticSTF**: 点云 DG 先锋工作，但都是任务特异的，不支持多任务统一模型

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次提出多域多任务点云理解设定，双层次设计有创意
- 实验充分度: ⭐⭐⭐⭐ 消融实验充分，多种 baseline 对比，但目标域过于单一
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述合理，图示直观
- 价值: ⭐⭐⭐⭐ 新设定和基准测试对推动点云泛化研究有较好贡献

<!-- RELATED:START -->

## 相关论文

- [GPSFormer: A Global Perception and Local Structure Fitting-Based Transformer for Point Cloud Understanding](gpsformer_a_global_perception_and_local_structure_fitting-based_transformer_for_.md)
- [Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding](../../CVPR2026/3d_vision/mamba_learns_in_context_structure-aware_domain_generalization_for_multi-task_poi.md)
- [T-MAE: Temporal Masked Autoencoders for Point Cloud Representation Learning](t-mae_temporal_masked_autoencoders_for_point_cloud_representation_learning.md)
- [SegPoint: Segment Any Point Cloud via Large Language Model](segpoint_segment_any_point_cloud_via_large_language_model.md)
- [P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising](p2p-bridge_diffusion_bridges_for_3d_point_cloud_denoising.md)

<!-- RELATED:END -->
