---
title: >-
  [论文解读] Outlier Gradient Analysis: Efficiently Identifying Detrimental Training Samples for Deep Learning Models
description: >-
  [ICML 2025][目标检测][数据中心学习] 提出 Outlier Gradient Analysis (OGA)，将影响函数中识别有害训练样本的问题转化为梯度空间上的异常点检测，绕开了 Hessian 矩阵求逆的高计算开销…
tags:
  - "ICML 2025"
  - "目标检测"
  - "数据中心学习"
  - "影响函数"
  - "梯度空间异常检测"
  - "有害样本识别"
  - "噪声标签校正"
---

# Outlier Gradient Analysis: Efficiently Identifying Detrimental Training Samples for Deep Learning Models

**会议**: ICML 2025  
**arXiv**: [2405.03869](https://arxiv.org/abs/2405.03869)  
**代码**: 无  
**领域**: 目标检测  
**关键词**: 数据中心学习, 影响函数, 梯度空间异常检测, 有害样本识别, 噪声标签校正

## 一句话总结

提出 Outlier Gradient Analysis (OGA)，将影响函数中识别有害训练样本的问题转化为梯度空间上的异常点检测，绕开了 Hessian 矩阵求逆的高计算开销，同时在噪声标签校正、NLP 数据筛选和 LLM 影响力数据识别等任务上取得优于传统影响函数方法的效果。

## 研究背景与动机

数据中心学习（Data-Centric Learning）的核心挑战之一是识别对模型性能有害的训练样本。影响函数（Influence Functions）是该任务中最常用的工具，通过度量无穷小权重扰动对模型参数的影响来估计单个样本的贡献，而无需重新训练模型。

然而，影响函数在深度学习场景中面临两个关键瓶颈：

**凸性假设**：影响函数要求损失函数严格凸，以保证 Hessian 矩阵可逆，而深度模型的损失函数通常是非凸的。

**计算代价**：Hessian 矩阵及其逆的计算代价极高，尤其对于参数量巨大的深度模型和大规模数据集几乎不可行。

尽管已有诸多近似方法（如 LiSSA、DataInf、Kronecker 分解等）试图缓解计算问题，但这些方法本质上仍依赖于 Hessian 的某种形式。本文另辟蹊径——完全绕开 Hessian，直接在梯度空间上做异常检测来识别有害样本。

## 方法详解

### 整体框架

本文的核心思路可以用三步概括：

1. **计算梯度**：对每个训练样本 $z_j = (x_j, y_j)$，计算其损失关于模型参数的梯度 $\nabla_{\hat{\theta}} \ell(z_j; \hat{\theta})$。
2. **梯度空间异常检测**：将所有训练样本的梯度组成梯度集合 $\mathcal{G}$，在该空间上运行异常检测算法 $\mathcal{A}$。
3. **修剪与重训练**：将被检测为异常的样本标记为有害样本并移除，在修剪后的数据集上重新训练模型。

这一过程在 Algorithm 1 中形式化为：

- 输入：训练集 $T$、损失函数 $\ell$、模型参数 $\hat{\theta}$、异常检测算法 $\mathcal{A}$、修剪预算 $k$
- 输出：有害/有益标签集合 $L$、修剪后训练集 $T^*$

### 关键设计

#### 从影响函数到异常检测的桥梁

经典影响函数的公式为：

$$\mathcal{I}(z_j) = -\sum_{z \in T/V} \nabla_{\hat{\theta}} \ell(z; \hat{\theta})^\top \mathbf{H}_{\hat{\theta}}^{-1} \nabla_{\hat{\theta}} \ell(z_j; \hat{\theta})$$

作者观察到这个公式由三个因子组成：(1) 求和中的梯度项、(2) Hessian 逆矩阵、(3) 样本 $z_j$ 的梯度。前两个因子对所有训练样本是共享的，而**第三个因子 $\nabla_{\hat{\theta}} \ell(z_j; \hat{\theta})$ 是唯一依赖于具体样本 $z_j$ 的项**，因此它在决定样本是有益还是有害时起决定性作用。

基于此，作者提出两个关键论断：

- **观察 3.1**：对于收敛的 ERM 模型，绝大多数训练样本对模型有正贡献，有害样本只占极少数——即有害样本是"异常点"。
- **假设 3.2**：存在能在梯度空间中检测有害样本的异常检测算法，其效果等价于通过影响函数评估样本的离散影响。

这一转换的本质在于：影响函数判断有害样本时只需要知道"正/负"（离散影响 $\tilde{\mathcal{I}}$），而不需要精确的连续影响分数。梯度空间中的异常检测恰好可以提供这种二元判别能力。

#### 异常检测算法选择

作者选用了三种异常检测方法：

1. **Isolation Forest (iForest)**：线性时间复杂度，低内存占用，对高维梯度空间友好；通过构建 iTree 集成进行子空间检测，可以有效处理非线性分离的异常点。
2. **L1-norm 阈值法**：计算每个样本梯度的 L1 范数，超过阈值的标记为异常。
3. **L2-norm 阈值法**：类似 L1，但使用 L2 范数。

其中 iForest 是首选方法，因为它在效果和效率之间取得了最佳平衡。

#### LLM 任务的扩展设计

对于 LLM 影响力数据识别任务，需要度量训练样本与测试样本的相似性。作者为每个类别的 prompt 单独训练一个 iForest 估计器（共训练 10 个），每个估计器仅基于该类别训练 prompt 的梯度空间。对于未知测试 prompt，用所有类别的 iForest 估计器分别产生异常分数，从而实现跨集合的影响力估计。

### 损失函数 / 训练策略

本文不修改原始训练损失函数，而是作为一种**后处理数据清洗策略**：

- 先正常训练模型至收敛
- 计算训练样本的梯度
- 在梯度空间运行异常检测，按修剪预算 $k$（默认为训练集的 5%）移除检测到的有害样本
- 在修剪后的数据集上重新训练模型

这种"训练→检测→修剪→重训练"的流程简单且可与其他方法（如损失函数修正）组合使用。

## 实验关键数据

### 主实验

**合成数据集（Two Half Moons）上的异常检测与分类精度**：

| 方法 | 异常检测精度 (%) | 修剪后分类精度 (%) |
|---|---|---|
| Multilayer Perceptron (baseline) | - | 90.0 |
| Exact Hessian | 90.0 | 90.0 |
| LiSSA | 82.0 | 91.0 |
| DataInf | 82.0 | 91.0 |
| Gradient Tracing | 82.0 | 91.0 |
| **Outlier Gradient (iForest)** | **96.0** | **96.0** |
| Outlier Gradient (L1) | 98.0 | 87.0 |
| Outlier Gradient (L2) | 98.0 | 87.0 |

**CIFAR-10N / CIFAR-100N 噪声标签校正（ResNet-34）**：

| 方法 | Aggregate | Random | Worst | Noisy100 |
|---|---|---|---|---|
| Cross Entropy (baseline) | 90.87 | 89.17 | 82.27 | 57.36 |
| LiSSA | 91.49 | 90.05 | 83.38 | 60.48 |
| DataInf | 91.46 | 90.05 | 83.40 | 60.70 |
| Self-LiSSA | 92.07 | 89.58 | 83.01 | 59.48 |
| Outlier Gradient (L1) | 91.86 | **90.66** | **84.20** | 60.32 |
| **Outlier Gradient (L2)** | **92.21** | 90.25 | 82.99 | **61.40** |
| Outlier Gradient (iForest) | 91.36 | 90.20 | 83.72 | 60.99 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|---|---|---|
| 修剪预算 k=2.5% | 略低于5%的精度 | 修剪不够充分 |
| 修剪预算 k=5%（默认） | 最佳平衡 | 在多个数据集上表现一致 |
| 修剪预算 k=12.5% | 精度有所下降 | 过度修剪损伤有益样本 |
| iForest 参数调优 | 对参数不敏感 | 默认参数即可获得稳定表现 |
| 基模型换为 ResNet-18 | 趋势一致 | 方法不依赖特定架构 |
| ImageNet 扩展 | 趋势一致 | 大规模数据集同样有效 |

### 关键发现

1. **合成数据验证**：在非凸 MLP 模型上，传统影响函数的分数混淆了有害与正常样本（影响分数区分度差），而梯度空间中有害样本清晰地被非线性分离，iForest 能有效检测。
2. **计算效率优势**：OGA 的时间复杂度是线性的（对样本数和参数数），远优于需要 Hessian 逆的方法。在 CIFAR-10N 上运行时间比 LiSSA 快数个数量级。
3. **LLM 场景表现完美**：在 Llama-2-13B-chat 上的三个 benchmark 中，OGA 在 Class Detection AUC 和 Recall 上均达到 1.000（完美分数），显著优于 DataInf（AUC 0.999）和 Gradient Tracing（部分 AUC 仅 0.72）。
4. **NLP 数据筛选**：在 RoBERTa 的 GLUE 任务 LoRA 微调中，OGA 在 QNLI、SST2、QQP 上明显优于所有基线方法。

## 亮点与洞察

1. **思路极简但深刻**：将复杂的影响函数计算简化为"算梯度 → 异常检测"两步操作，理论推导清晰，观察 3.1 和假设 3.2 的建立逻辑自洽。
2. **Hessian-Free 的优势巨大**：完全绕开 Hessian 矩阵让方法对任意深度模型直接可用，没有凸性假设的限制，也没有 Hessian 近似带来的误差累积。
3. **通用性强**：从视觉模型（ResNet）到 NLP Transformer（RoBERTa）再到 LLM（Llama-2-13B），方法在不同规模和类型的模型上均有效。
4. **与现有方法互补**：OGA 是数据修剪方法，可与修改损失函数/模型架构的噪声学习方法组合使用，潜在地获得更大收益。

## 局限与展望

1. **修剪预算 $k$ 的选择**：异常检测算法普遍面临的超参数问题——如何自动确定最优的修剪比例，目前仍需人工设定。
2. **需要两次训练**：流程本质上需要训练两遍模型（第一遍用于获取梯度，修剪后第二遍重训练），对超大模型来说仍有一定代价。
3. **梯度降维**：论文未深入探讨当模型参数极多时（如百亿级 LLM），梯度向量的高维度是否会影响异常检测的效果；LoRA 参数的低维特性可能是 LLM 实验成功的关键因素。
4. **仅验证离散影响**：方法只判断"有害/有益"的二元标签，不能像影响函数那样给出连续的影响力排序，在需要细粒度数据估值的场景中受限。
5. **分布偏移适应初步**：虽然通过半监督 OneClassSVM 做了初步尝试，但在训练/测试分布差异显著时的表现有待更系统的验证。

## 相关工作与启发

- **Koh & Liang (2017)**：影响函数在深度学习中的开创性工作，本文的出发点。
- **DataInf (Kwon et al., 2024)**：高效影响估计方法，本文的主要比较基线之一，同样关注大模型的效率问题。
- **Pruthi et al. (2020) Gradient Tracing**：直接用梯度做影响估计的代表方法，但缺乏异常检测视角。
- **Isolation Forest (Liu et al., 2008)**：本文核心异常检测组件，其线性复杂度和子空间集成特性是方法成功的关键。
- **对目标检测的启发**：虽然论文实验主要在图像分类和 NLP 上，但 OGA 的思想可直接应用于目标检测中的数据清洗——检测标注错误的 bounding box、清理对抗样本等，特别是在大规模检测数据集（如 COCO、Objects365）中噪声标注普遍存在的情况下。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|---|---|---|
| 创新性 | ⭐⭐⭐⭐ | 视角新颖，从影响函数到异常检测的转换简洁有力 |
| 理论深度 | ⭐⭐⭐ | 观察和假设合理但偏实验驱动，缺乏严格理论证明 |
| 实验充分性 | ⭐⭐⭐⭐⭐ | 合成→视觉→NLP→LLM，覆盖面极广，消融充分 |
| 实用价值 | ⭐⭐⭐⭐ | 方法简单、通用、高效，直接可落地 |
| 写作质量 | ⭐⭐⭐⭐ | 逻辑清晰，图表丰富 |
| **综合** | **⭐⭐⭐⭐** | 扎实的工作，在数据中心学习领域有实际影响力 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] The COTe Score: A Decomposable Framework for Evaluating Document Layout Analysis Models](../../CVPR2026/object_detection/the_cote_score_a_decomposable_framework_for_evaluating_document_layout_analysis_.md)
- [\[ECCV 2024\] YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](../../ECCV2024/object_detection/yolov9_learning_what_you_want_to_learn_using_programmable_gradient_information.md)
- [\[ICML 2026\] EARL: Towards a Unified Analysis-Guided Reinforcement Learning Framework for Egocentric Interaction Reasoning and Pixel Grounding](../../ICML2026/object_detection/earl_towards_a_unified_analysis-guided_reinforcement_learning_framework_for_egoc.md)
- [\[ICML 2025\] Open-Det: An Efficient Learning Framework for Open-Ended Detection](open-det_an_efficient_learning_framework_for_open-ended_detection.md)
- [\[ICCV 2025\] Uncertainty-Aware Gradient Stabilization for Small Object Detection](../../ICCV2025/object_detection/uncertainty-aware_gradient_stabilization_for_small_object_detection.md)

</div>

<!-- RELATED:END -->
