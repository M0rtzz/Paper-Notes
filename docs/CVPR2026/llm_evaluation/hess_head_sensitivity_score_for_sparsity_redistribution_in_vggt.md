---
title: >-
  [论文解读] HeSS: Head Sensitivity Score for Sparsity Redistribution in VGGT
description: >-
  [CVPR 2026][注意力稀疏化] HeSS 提出 Head Sensitivity Score 来量化 VGGT 全局注意力层中每个注意力头对稀疏化的敏感程度，并基于此将注意力预算从不敏感的头重新分配到敏感头，在高稀疏度下显著优于均匀稀疏化方法 SparseVGGT，几乎不增加运行时开销。
tags:
  - CVPR 2026
  - 注意力稀疏化
  - VGGT
  - 头部敏感性
  - Fisher信息矩阵
  - 3D重建加速
---

# HeSS: Head Sensitivity Score for Sparsity Redistribution in VGGT

**会议**: CVPR 2026  
**arXiv**: [2603.25336](https://arxiv.org/abs/2603.25336)  
**代码**: https://github.com/libary753/HeSS  
**领域**: 3D视觉 / 模型压缩  
**关键词**: 注意力稀疏化, VGGT, 头部敏感性, Fisher信息矩阵, 3D重建加速

## 一句话总结

HeSS 提出 Head Sensitivity Score 来量化 VGGT 全局注意力层中每个注意力头对稀疏化的敏感程度，并基于此将注意力预算从不敏感的头重新分配到敏感头，在高稀疏度下显著优于均匀稀疏化方法 SparseVGGT，几乎不增加运行时开销。

## 研究背景与动机

1. **领域现状**：VGGT（Visual Geometry Grounded Transformer）是多视图 3D 重建的强大基础模型，通过交替堆叠的全局注意力（GA）层和帧注意力（FA）层统一了传统 SfM 和 MVS 任务。全局注意力层使所有帧 token 相互交互，对理解整体场景结构至关重要。

2. **现有痛点**：GA 层的自注意力计算量随输入视图数平方增长 $O(S^2)$，在大规模输入时 GPU 显存和计算成本急剧上升，限制了 VGGT 在实时和大规模场景中的应用。

3. **核心矛盾**：现有加速方法（如 SparseVGGT）对所有注意力头施加统一的稀疏模式（相同的 CDF 阈值 $\tau$ 和稀疏比 $\rho$），但实际上不同注意力头对稀疏化的敏感度差异很大——有些头被过度稀疏化后会导致性能急剧下降，而有些头即使大幅稀疏化也几乎不影响结果。

4. **本文目标** (1) 如何量化每个注意力头对稀疏化的敏感程度？(2) 如何根据敏感度差异自适应地分配注意力预算？(3) 如何在保持总计算量不变的前提下，通过预算重分配显著减少高稀疏度下的性能退化？

5. **切入角度**：作者假设性能退化的根本原因是注意力头的敏感度异质性——均匀稀疏化不可避免地过度压缩了关键头。通过 Fisher 信息矩阵近似 Hessian 来度量头部敏感度，将3D重建任务特有的两种误差（相机位姿和点云）结合起来计算敏感度分数。

6. **核心 idea**：用 Fisher 信息矩阵对相机位姿误差和点云误差的 Hessian 近似来量化每个注意力头的稀疏化敏感度，然后按敏感度比例重新分配总注意力预算，对敏感头保留更多注意力、对鲁棒头施加更强稀疏化。

## 方法详解

### 整体框架

HeSS 是一个两阶段管线：(1) 校准阶段：在小型校准集上计算所有 GA 层注意力头的 HeSS 分数并固定；(2) 推理阶段：根据预计算的 HeSS 分数重新分配每个头的注意力预算（block 数量），敏感头分配更多、鲁棒头分配更少。总预算与 SparseVGGT 相同，仅改变头间分配。

### 关键设计

1. **相机位姿误差 $e_{\text{cam}}$**:

    - 功能：作为 HeSS 的第一个误差信号，评估模型对整体场景几何结构的理解
    - 核心思路：先用 Umeyama + ICP 算法对齐预测与真实点云得到变换矩阵 $\mathbf{H}$（对 $\mathbf{H}$ 施加 stop-gradient），然后计算预测相机位置 $\hat{\mathbf{t}}_i$ 变换后与真实位置 $\mathbf{t}_i$ 的 MSE：$e_{\text{cam}} = \frac{1}{2N}\sum_{i=1}^N |\text{sg}(\mathbf{H})\hat{\mathbf{t}}_i - \mathbf{t}_i|_2^2$
    - 设计动机：相机位姿是 3D 视觉的核心——它是所有下游预测的几何支架。对 $\mathbf{H}$ 用 stop-gradient 是为了防止梯度通过辅助对齐步骤

2. **点云误差 $e_{\text{pc}}$**:

    - 功能：补充 $e_{\text{cam}}$ 对细粒度局部几何的评估
    - 核心思路：用置信度阈值 $\epsilon = 0.05$ 筛选内点集 $\mathcal{I}$（与真实点云最近距离 < $\epsilon$ 的预测点），然后计算内点的平均最近点距离：$e_{\text{pc}} = \frac{1}{2|\mathcal{I}|}\sum_{j \in \mathcal{I}} \min_{\mathbf{p} \in P}\|\text{sg}(\mathbf{H})\hat{\mathbf{p}}_j - \mathbf{p}\|_2^2$
    - 设计动机：$e_{\text{cam}}$ 只反映全局几何一致性，对逐像素的精细结构不敏感。点云误差要求模型精确回归每个像素到 3D 空间的位置，能捕捉局部几何细节

3. **HeSS 计算**:

    - 功能：将两种误差的敏感度合并为每个头的统一分数
    - 核心思路：对每个头 $h$，计算误差对 Query 投影权重 $W_Q^h$ 的 Fisher 信息矩阵 $\mathbf{F}_{\text{cam}}^h$ 和 $\mathbf{F}_{\text{pc}}^h$，取其 trace 并在同层所有头之间归一化：$\text{HeSS}_{\text{cam}}(h) = \frac{\text{tr}(\mathbf{F}_{\text{cam}}^h)}{\sum_h \text{tr}(\mathbf{F}_{\text{cam}}^h)}$，最终 $\text{HeSS}(h) = \lambda \cdot \text{HeSS}_{\text{cam}}(h) + (1-\lambda) \cdot \text{HeSS}_{\text{pc}}(h)$，默认 $\lambda = 0.5$
    - 设计动机：FIM 是 Hessian 的可计算近似，用一阶梯度的外积估计二阶信息。选择 $W_Q^h$ 而非 $W_K^h$ 或 $W_V^h$ 是因为实验表明 $W_Q$ 的 Hessian 给出了更可靠的敏感度估计。两种误差互补——$e_{\text{cam}}$ 在低稀疏度更重要，$e_{\text{pc}}$ 在高稀疏度更关键

4. **HeSS 引导的预算重分配**:

    - 功能：根据 HeSS 分数将总注意力预算在头之间重新分配
    - 核心思路：三步流程——(a) 获取总预算 $C_{\text{total}} = \sum_n c_{h_n}$（所有头的基线预算之和）；(b) 按 HeSS 比例计算理想预算 $c_h' = C_{\text{total}} \cdot w_h$，其中 $w_h = \text{HeSS}(h) / \sum_n \text{HeSS}(h_n)$；(c) 迭代 water-filling 算法处理溢出——若某头的理想预算超过其最大容量 $C_{\max}$，将其钳制到 $C_{\max}$ 并将多余预算按 HeSS 比例重新分配给剩余未封顶的头，重复直到无溢出
    - 设计动机：简单按比例分配可能导致某些高敏感头被分配超过其最大可用 block 数量的预算。迭代 capping 确保预算分配在结构上可行

### 训练 / 校准策略

- 校准集使用 CO3Dv2 dev split，每场景采样 20 个视图
- HeSS 一次计算后固定，不随推理数据变化
- 不需要任何训练——纯粹是推理时的预算重分配策略

## 实验关键数据

### 主实验

在 CO3Dv2（相机位姿估计）和 DTU（MVS 重建）上对比：

| 方法 | 稀疏度 | CO3Dv2 AUC@30↑ | DTU Chamfer↓ | 运行时间(s) |
|------|--------|---------------|-------------|-----------|
| VGGT (原始) | 0% | 基准 | 基准 | 10.35 |
| SparseVGGT | 43% | 下降明显 | 下降明显 | 8.42 |
| **HeSS (Ours)** | **43%** | **接近原始** | **接近原始** | **8.37** |
| SparseVGGT | 73% | 严重退化 | 严重退化 | 6.59 |
| **HeSS (Ours)** | **73%** | **显著优于SparseVGGT** | **显著优于SparseVGGT** | **6.58** |

### 消融实验

| 配置 | DTU Chamfer↓ | Acc.↓ | Comp.↓ | 说明 |
|------|-------------|-------|--------|------|
| Ours ($W_Q^h$) | **1.603** | **2.839** | **0.367** | 用 Query 投影的 Hessian |
| $W_K^h$ | 1.917 | 3.450 | 0.384 | 用 Key 投影，性能下降 |
| $W_V^h$ | 1.966 | 3.540 | 0.392 | 用 Value 投影，更差 |
| 无归一化 (Linear) | 1.840 | 3.272 | 0.408 | 不做 sum-normalization |
| 反转 HeSS | 灾难性失败 | — | — | 先剪敏感头验证正确性 |

### 关键发现

- HeSS 分布可视化（图 4）证实了头部敏感度的高度异质性——大多数头在两种误差上都呈现低敏感度，只有少数头极为关键（如 GA19 的 H5）
- 某些头表现出明显的任务偏好——GA13 的 H13 对相机位姿更敏感，GA19 的 H5 对点云更敏感
- $\lambda$ 的消融实验（图 11）表明两种误差互补：仅用 $e_{\text{cam}}$ 在高稀疏度下性能下降，仅用 $e_{\text{pc}}$ 在低稀疏度下下降，组合后在所有稀疏度水平上都保持稳定
- 去掉迭代 capping 导致明显性能退化（图 10），未用完的预算被浪费
- 方法可泛化到 $\pi^3$ 模型，但需要用不同的 $\lambda$（$\lambda = 0$ 最优）

## 亮点与洞察

- **零额外推理开销的性能提升**：HeSS 只改变预算分配方式，总计算量不变（甚至因重分配更高效的调度而略减），但在高稀疏度下性能提升显著。这是一种"免费午餐"式的改进
- **3D 任务特定的敏感度度量**：不同于通用 ViT 剪枝使用分类 loss 的 Hessian，HeSS 针对 3D 重建定义了相机位姿和点云两个互补的误差项，这种领域特定设计使敏感度度量更准确
- **反转 HeSS 的 sanity check**：反转排序后性能灾难性崩溃，这个简单但有力的验证证明了 HeSS 确实正确识别了关键头

## 局限与展望

- 作者承认 HeSS 目前对所有 GA 层统一处理，但不同层对稀疏化的敏感度也不同，且跨层的 FIM 尺度不可直接比较——层级可比的敏感度指标是重要后续方向
- 方法只考虑推理时稀疏化，未涉及训练时适应——如果模型在训练时就对稀疏化具有鲁棒性，可以进一步提升压缩比
- 通过 $\pi^3$ 实验发现不同模型需要不同的 $\lambda$，说明需要某种自动搜索机制
- 校准集的选择和大小可能影响 HeSS 的稳定性，文中未充分分析这一点

## 相关工作与启发

- **vs SparseVGGT**: SparseVGGT 提出了 block-sparse attention 机制但对所有头统一处理，HeSS 在其基础上增加了头级别的预算重分配，总预算相同但分配更智能
- **vs ViT 通用剪枝 (Michel et al.)**: 通用头部剪枝在 50% 以上稀疏度时急剧崩溃，因为它缺乏 3D 几何的归纳偏置；HeSS 保持空间 token 的结构完整性，即使在 75% 稀疏度下仍保持高保真度
- **HeSS 的思路可迁移**：这种"用任务特定误差的 Hessian 来指导注意力头级别的资源分配"思路可以推广到其他多头注意力模型的压缩（如 LLM 的 KV-cache 压缩）

## 评分

- 新颖性: ⭐⭐⭐⭐ 头部敏感度量化和自适应预算分配的思路虽不完全新颖，但在 3D 重建的 VGGT 上的应用设计精巧
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多稀疏度、多消融、泛化实验齐全，sanity check 有说服力
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导详尽，图表直观
- 价值: ⭐⭐⭐⭐ 对 VGGT 实际部署有直接帮助，方法思路可迁移

<!-- RELATED:START -->

## 相关论文

- [Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)
- [Spectral Sensitivity Estimation with an Uncalibrated Diffraction Grating](../../ICCV2025/llm_evaluation/spectral_sensitivity_estimation_with_an_uncalibrated_diffraction_grating.md)
- [Access Denied Inc: The First Benchmark Environment for Sensitivity Awareness](../../ACL2025/llm_evaluation/access_denied_inc_the_first_benchmark_environment_for_sensitivity_awareness.md)
- [Right Answer, Wrong Score: Uncovering the Inconsistencies of LLM Evaluation in Multiple-Choice QA](../../ACL2025/llm_evaluation/right_answer_wrong_score_uncovering_the_inconsistencies_of_llm_evaluation_in_mul.md)
- [Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline](pioneering_perceptual_video_fluency_assessment_a_novel_task_with_benchmark_datas.md)

<!-- RELATED:END -->
