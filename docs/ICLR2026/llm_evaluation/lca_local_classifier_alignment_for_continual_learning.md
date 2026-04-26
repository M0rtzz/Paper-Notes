---
title: >-
  [论文解读] LCA: Local Classifier Alignment for Continual Learning
description: >-
  [ICLR 2026][类增量学习] 提出 Local Classifier Alignment (LCA) 损失函数，通过在类原型高斯分布的局部区域内同时最小化分类损失和损失灵敏度，解决持续学习中 backbone 增量合并后分类器不匹配的问题，配合增量 PEFT 合并策略 (IM)，在 7 个基准数据集上达到整体 85.6% 的平均精度，大幅超越 SOTA。
tags:
  - ICLR 2026
  - 类增量学习
  - 分类器对齐
  - 模型合并
  - 鲁棒性
  - 预训练模型
---

# LCA: Local Classifier Alignment for Continual Learning

**会议**: ICLR 2026  
**arXiv**: [2603.09888](https://arxiv.org/abs/2603.09888)  
**代码**: [GitHub](https://github.com/tung-tran-kyushu/LCA)  
**领域**: 持续学习  
**关键词**: 类增量学习, 分类器对齐, 模型合并, 鲁棒性, 预训练模型

## 一句话总结

提出 Local Classifier Alignment (LCA) 损失函数，通过在类原型高斯分布的局部区域内同时最小化分类损失和损失灵敏度，解决持续学习中 backbone 增量合并后分类器不匹配的问题，配合增量 PEFT 合并策略 (IM)，在 7 个基准数据集上达到整体 85.6% 的平均精度，大幅超越 SOTA。

## 研究背景与动机

**领域现状**：基于预训练模型 (PTM) 的类增量学习 (CIL) 是持续学习的主流范式。PTM 提供强大的特征提取能力，只需轻量微调即可适应新任务，但朴素的序列微调仍会导致灾难性遗忘。

**两大路线的不足**：(1) 仅在第一个任务上微调 (如 APER)，随任务增多、分布偏移加大，性能快速衰退；(2) 逐任务微调 + backbone 合并 (如 EASE, MOS)，虽然综合能力强，但冻结的旧分类器与合并后 backbone 产生不匹配 (mismatch)。

**核心痛点**：backbone 在多任务合并后参数发生变化，之前冻结的任务特定分类器与新 backbone 的特征空间不再对齐→旧任务性能急剧下降。由于无法回访旧数据，直接重训分类器不可行。

**切入角度**：借助类的高斯原型 (Gaussian prototype) 生成合成样本，在合成样本上重新对齐所有分类器。关键创新：不仅最小化分类损失，还正则化损失对输入扰动的灵敏度→实现局部鲁棒性→更好泛化。

**模型合并的启发**：Task Arithmetic、TIES-Merge 等工作表明，独立训练的任务特定模型可以通过参数合并形成更强的统一模型。本文将此思想融入 CIL，仅合并 PEFT (LoRA) 参数，存储开销极低。

**理论缺口**：现有 CIL 方法缺乏理论分析来指导分类器对齐。本文提供了测试误差的分解定理，将 CIL 性能拆分为特征分布偏移、类损失和鲁棒性三个可控部分。

## 方法详解

### 关键设计 1：增量合并 (Incremental Merging, IM)

**功能**：逐任务微调 PEFT 模块，然后将任务向量按元素合并为统一 backbone。

**核心思路**：每个新任务从上一次合并结果初始化训练，保持参数空间邻近性。训练后计算任务向量 $\tau_{\text{curr}} = \theta_{\text{peft}_i} - \theta_{\text{peft}_0}$，与累积向量 $\tau$ 按元素比较，保留绝对值更大者：

$$
\tau^{(k)} \leftarrow \begin{cases} \tau_{\text{curr}}^{(k)} & \text{if } |\tau_{\text{curr}}^{(k)}| \geq |\tau^{(k)}| \\ \tau^{(k)} & \text{otherwise} \end{cases}
$$

最终合并结果为 $\theta_{\text{merged}} = \theta_{\text{peft}_0} + \alpha \cdot \tau$。

**设计动机**：(1) 仅保留当前合并向量和新任务向量→避免存储所有历史参数；(2) 保留绝对值大者→保持最显著的任务特定更新；(3) 从上一合并结果初始化→参数空间连续性有利于稳定合并 (Li et al., 2025)。

### 关键设计 2：LCA 损失函数

**功能**：在 backbone 合并后，利用类高斯原型生成合成样本，通过 LCA 损失重新训练所有分类器头。

**核心思路**：LCA 损失对每个类 $i$ 定义为：

$$
L_i = \underbrace{\mathbb{E}_{\boldsymbol{z} \sim \boldsymbol{D}_i}[\ell(h_t, \boldsymbol{z})]}_{\text{分类损失}} + \lambda \underbrace{\mathbb{E}_{\boldsymbol{z}, \boldsymbol{z}' \sim \boldsymbol{D}_i}[|\ell(h_t, \boldsymbol{z}) - \ell(h_t, \boldsymbol{z}')|]}_{\text{损失灵敏度正则项}}
$$

总损失为所有已见类的均值：$L(\boldsymbol{D}, h_t) = \frac{1}{C_t} \sum_{i=1}^{C_t} L_i$。

**设计动机**：
- 第一项是标准交叉熵损失，确保分类器在类原型附近的合成样本上分类正确
- 第二项衡量同一类分布中两个随机样本的损失差异→惩罚分类器对输入微小变化的敏感性→使损失面在原型邻域内"平坦"
- 这种局部鲁棒性尤其重要：高斯采样的部分样本可能远离类原型、靠近其他类原型，第二项可以降低这些"有害样本"对训练的负面影响
- $\lambda$ 控制鲁棒性惩罚强度，实验中 $\lambda = 0.1$ 在所有数据集上表现稳定

### 关键设计 3：理论误差分解

**功能**：为 CIL 分类器的泛化能力提供理论分析，分解测试误差为三个可控部分。

**核心思路**：

**定理 3.1 (固定 backbone)**：对于有界损失 $\ell$，测试误差满足：

$$
L(P, h_t) \leq L(\boldsymbol{D}, h_t) + \sum_{i=1}^{C_t} \frac{n_i}{n} \bar{\epsilon}_i(h_t) + \ell_{\max} \sqrt{\frac{C_t \ln 4 + 2\ln(1/\delta)}{n}}
$$

其中 $\bar{\epsilon}_i(h_t)$ 是类 $i$ 局部区域内的损失鲁棒性项。

**定理 3.2 (backbone 变化)**：考虑 backbone 更新导致的特征分布偏移：

$$
L(P_t, h_t) \leq 2\ell_{\max} \text{TV}(P_t, \hat{P}_t) + L(\hat{\boldsymbol{D}}, h_t) + \sum_{i=1}^{C_t} \frac{n_i}{n} \bar{\epsilon}_i(h_t) + \ell_{\max} \sqrt{\frac{C_t \ln 4 + 2\ln(1/\delta)}{n}}
$$

**设计动机**：三部分各有对应控制方法：(1) $\text{TV}(P_t, \hat{P}_t)$ 特征分布偏移→由 IM 的增量合并控制；(2) $L(\hat{\boldsymbol{D}}, h_t)$ 训练误差→由 LCA 的第一项控制；(3) $\bar{\epsilon}_i$ 鲁棒性→由 LCA 的第二项控制。理论直接支撑了 IM+LCA 的设计合理性。

### 关键设计 4：分类器架构

**功能**：每个新任务添加一个独立的 MLP 分类头，推理时拼接所有头的输出。

**核心思路**：推理为 $h(x) = \text{concat}(h(x;\theta_1^{\text{cls}}), \ldots, h(x;\theta_t^{\text{cls}}))$。每个类用高斯分布 $\mathcal{N}_i$ 表示其在特征空间中的分布，LCA 对齐时从各类高斯中采样生成合成特征。

**设计动机**：独立分类头避免旧头在新任务训练时被修改→减少遗忘；额外存储仅为每类的均值和协方差 $\mathcal{O}(n)$→比存储原始样本高效得多。

## 实验关键数据

### 表1：7个基准数据集上的平均精度对比 (ViT-B/16-IN1K)

| 方法 | CIFAR100 | IN-R | IN-A | CUB | OB | VTAB | CARS | Overall |
|------|----------|------|------|-----|----|----|------|---------|
| CODA-Prompt | 91.0 | 78.2 | 48.1 | 75.6 | 71.0 | 65.6 | 26.3 | 65.1 |
| DualPrompt | 86.7 | 74.6 | 55.3 | 78.9 | 74.4 | 84.0 | 49.4 | 71.9 |
| EASE | 91.7 | 82.4 | 67.8 | 89.5 | 80.8 | 93.3 | 48.1 | 79.1 |
| MOS | 94.3 | 83.3 | 67.6 | 92.3 | 86.1 | 92.4 | 71.4 | 83.9 |
| SLCA | 93.7 | 85.1 | 45.1 | 90.2 | 82.7 | 91.1 | 74.6 | 80.4 |
| IM (仅合并) | 92.8 | 84.3 | 66.5 | 86.7 | 81.1 | 84.6 | 70.1 | 80.9 |
| **IM+LCA** | **94.8** | **85.8** | **75.0** | 90.8 | 81.4 | **95.2** | **76.2** | **85.6** |

### 表2：鲁棒性对比 (CIFAR100-C / CIFAR100-P)

| 指标 | IM | IM+LCA | 提升 |
|------|-------|---------|------|
| CIFAR100-C 平均精度 | ~88% | ~90% | +2% |
| CIFAR100-P 平均精度 | ~86% | ~88.5% | +2.5% |
| CIFAR100-C 严重度5 | 较低 | 更高 | 显著提升 |
| 综合鲁棒性分数 | 基线 | 更优 | 全面改善 |

### LCA 作为组件嵌入其他方法

| 方法 | 原始 | +LCA | 效果 |
|------|------|------|------|
| SLCA 基础版 | 基线 | SLCA-LCA | IN-A, CUB, VTAB, CARS 均有提升 |
| MOS 基础版 | 基线 | MOS-LCA | 多个数据集改善，CIFAR100 达 93.1% |

## 关键发现

1. **分类器对齐是 CIL 的关键瓶颈**：IM→IM+LCA 的提升幅度在所有 7 个数据集上一致显著，尤其 ImageNet-A 上提升达 8.5% (66.5→75.0)，说明 backbone 合并后分类器不匹配是性能瓶颈。

2. **鲁棒性正则项的重要性**：LCA 第二项（损失灵敏度正则）在 CIFAR100-C 和 CIFAR100-P 上分别带来 +2% 和 +2.5% 的鲁棒性提升，且在所有 19 种 corruption 和多种 perturbation 类型上均一致改善。

3. **LCA 的可组合性**：LCA 可以作为 plug-in 嵌入 SLCA、MOS 等方法中。即使不调优超参数 (固定 $\lambda=0.1$)，也能在多个数据集上带来稳定提升。

4. **仅合并 PEFT 参数的有效性**：无需合并全部 backbone 参数，仅合并 LoRA 参数即可实现高效的知识整合，存储开销极低。

5. **$\lambda$ 的选择**：$\lambda=0.1$ 在所有数据集上表现稳定，过大的 $\lambda$ 会导致性能下降（过度正则化），符合理论预期。

## 亮点与洞察

- **损失灵敏度作为正则化目标**：不同于传统的权重正则化或特征对齐，LCA 直接约束损失函数在输入空间上的变化率。这种"损失面平坦化"思想与 SAM (Sharpness-Aware Minimization) 异曲同工，但应用于分类器对齐的特定场景。

- **理论驱动的方法设计**：定理 3.2 的三部分分解 (分布偏移 + 训练误差 + 鲁棒性) 直接指导了 IM+LCA 的双组件设计，每个组件负责控制一个理论误差项。这种理论指导实践的方法设计在 CIL 领域较为少见。

- **合成样本的巧妙使用**：不需要 exemplar memory 或数据回放，仅需存储每类的均值和协方差→从高斯分布采样→在特征空间中对齐分类器。这避免了隐私问题和存储开销。

- **简洁高效**：整体方法不需要扩展 backbone (如 EASE)、复杂推理过程 (如 MOS) 或额外的记忆缓冲区。合并后一次 LCA 对齐即可，实现简洁且有效。

## 局限性

1. **LCA 仅作用于分类器对齐阶段**：未将 LCA 损失集成到 backbone 训练的端到端流程中。作者自己也指出，将 LCA 融入 backbone 训练有可能进一步提升鲁棒性。

2. **高斯假设的局限**：每类用单个高斯分布表示可能无法捕捉真实特征分布的多模态性或非对称性，尤其在复杂细粒度数据集上。

3. **理论分析假设 backbone 固定**：定理 3.1 在 backbone 固定时成立；虽然定理 3.2 引入了分布偏移项，但未直接分析 backbone 训练过程中的动态。

4. **未探索其他上下文**：LCA 损失具有通用性，但本文仅在 CIL 场景中验证，未在其他持续学习设置 (如 domain-incremental, task-incremental) 或一般分类任务中测试。

5. **OB 数据集上提升有限**：IM+LCA 在 OmniBenchmark 上仅从 81.1 提升到 81.4 (0.3%)，且低于 MOS 的 86.1%，说明方法在某些分布场景下可能不占优势。

## 相关工作对比

### vs EASE (Zhou et al., 2024)
EASE 通过扩展子空间 (expandable subspace) 来整合新任务，利用语义相似度重加权旧分类器。相比之下，LCA 不需要扩展 backbone 架构，存储开销更低，且通过理论支撑的损失函数直接对齐分类器。IM+LCA (85.6%) 在整体精度上大幅超越 EASE (79.1%)，尤其在 IN-A (+7.2%)、VTAB (+1.9%)、CARS (+28.1%) 上优势明显。

### vs MOS (Sun et al., 2025b)
MOS 在推理时动态选择合适的 backbone adapter，重在推理阶段的适配。IM+LCA 则在训练后通过一次对齐步骤完成，推理更简洁。虽然 MOS 在 CUB (92.3 vs 90.8) 和 OB (86.1 vs 81.4) 上表现更好，但 IM+LCA 在 IN-A (+7.4%)、VTAB (+2.8%)、CARS (+4.8%) 上大幅领先，且整体精度 85.6% > 83.9%。

### vs SLCA (Zhang et al., 2023)
SLCA 用较小学习率训练 backbone 以减少遗忘，但 backbone 变化仍导致分类器不匹配。IM+LCA 直接解决这一问题，在 IN-A 上从 45.1% 提升到 75.0% (+29.9%)，整体 85.6% vs 80.4%。此外，LCA 可作为 SLCA 的补充组件 (SLCA-LCA) 进一步提升性能。

## 评分

- **新颖性**: ⭐⭐⭐⭐ LCA 损失的设计新颖——将损失灵敏度作为正则化目标，有理论支撑的误差分解；增量合并策略虽然和已有工作相关，但仅合并 PEFT 参数且无需修剪阶段是新贡献
- **实验充分度**: ⭐⭐⭐⭐⭐ 7 个基准数据集 + 3 种 seed 报告均标差 + LCA 作为 plug-in 组合验证 + CIFAR100-C/P 鲁棒性评估 + 超参数敏感性分析 + 多种合并策略消融
- **写作质量**: ⭐⭐⭐⭐ 理论分析清晰完整，方法描述简洁明了，算法伪代码规范；论文结构合理
- **实用价值**: ⭐⭐⭐⭐ LCA 实现简单可作为 plug-in 嵌入现有 CIL 方法，无需额外存储或复杂推理，适合实际部署

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery](../../AAAI2026/llm_evaluation/goal_geometrically_optimal_alignment_for_continual_generalized_category_discover.md)
- [\[ICLR 2026\] PlanetAlign: A Comprehensive Python Library for Benchmarking Network Alignment](planetalign_a_comprehensive_python_library_for_benchmarking_network_alignment.md)
- [\[NeurIPS 2025\] Exploiting Task Relationships in Continual Learning via Transferability-Aware Task Embeddings](../../NeurIPS2025/llm_evaluation/exploiting_task_relationships_in_continual_learning_via_transferability-aware_ta.md)
- [\[AAAI 2026\] GDBA Revisited: Unleashing the Power of Guided Local Search for Distributed Constraint Optimization](../../AAAI2026/llm_evaluation/gdba_revisited_unleashing_the_power_of_guided_local_search_for_distributed_const.md)
- [\[ICLR 2026\] In-Context Learning of Temporal Point Processes with Foundation Inference Models](in-context_learning_of_temporal_point_processes_with_foundation_inference_models.md)

<!-- RELATED:END -->
