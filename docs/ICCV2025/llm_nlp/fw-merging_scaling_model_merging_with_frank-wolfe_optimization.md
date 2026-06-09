---
title: >-
  [论文解读] FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization
description: >-
  [ICCV 2025][LLM/NLP][模型合并] 将模型合并形式化为约束优化问题，引入Frank-Wolfe优化启发的FW-Merging方法，通过迭代选择最相关模型并局部合并，实现在大规模黑盒模型池中的可扩展、鲁棒合并，合并20个ViT模型时超越数据感知方法Adamerging 8.39%。
tags:
  - "ICCV 2025"
  - "LLM/NLP"
  - "模型合并"
  - "Frank-Wolfe优化"
  - "多任务学习"
  - "可扩展性"
  - "大语言模型"
---

# FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization

**会议**: ICCV 2025  
**arXiv**: [2503.12649](https://arxiv.org/abs/2503.12649)  
**代码**: 有（论文中提到open-sourced）  
**领域**: 模型合并 / 多任务学习  
**关键词**: 模型合并, Frank-Wolfe优化, 多任务学习, 可扩展性, 大语言模型

## 一句话总结

将模型合并形式化为约束优化问题，引入Frank-Wolfe优化启发的FW-Merging方法，通过迭代选择最相关模型并局部合并，实现在大规模黑盒模型池中的可扩展、鲁棒合并，合并20个ViT模型时超越数据感知方法Adamerging 8.39%。

## 研究背景与动机

模型合并(Model Merging)作为多任务学习的数据高效替代方案日益重要。然而随着开源AI生态的快速发展，现有方法面临两个关键限制：

**缺乏对未知模型的适应性**：现有方法基于已知的模型能力信息调整合并系数，对来源多样、信息部分未知的模型效果差，无法区分高质量与低质量微调模型

**无法有效扩展**：当合并大量未知model checkpoint时，性能严重退化。作者的分析实验显示，添加16个无关模型时性能下降18.9%-64.4%

理想的合并方法应满足两个基本缩放属性：(1) 添加无关模型不影响性能；(2) 添加相关模型性能稳步提升。

## 方法详解

### 整体框架

FW-Merging包含三个核心阶段的迭代过程：
1. **相关性评估**：利用当前模型的梯度构建目标函数的线性近似，揭示最有益的改进方向
2. **模型选择**：通过最小化线性近似，从候选池中选择最相关的checkpoint
3. **知识整合**：使用正交合并方法将选中的checkpoint整合到当前模型

### 关键设计

1. **约束优化形式化**：将模型合并问题重新定义为在凸包 $\mathcal{M} = \text{conv}(\{\theta_i^*\}_{i=1}^n)$ 上最小化目标函数 $\ell(\theta)$ 的约束优化问题。通过Proposition 1证明这与传统的系数优化形式等价。核心优势在于Linear Minimization Oracle (LMO)被简化为：

$$\text{LMO}(\{\theta_i^*\}, \theta_t) = \arg\min_{s \in \{\theta_1^*,...,\theta_n^*\}} \langle \nabla\ell(\theta_t), s \rangle$$

即在有限顶点集上做内积最小化，计算高效。

2. **Hard FW vs. Soft FW**：

    - **Hard LMO**：选择线性子问题的argmin作为合并方向，简单直接
    - **Soft LMO**：选择top-k个顶点，对它们的合并系数做内部优化（投影梯度下降到单纯形上），更新公式为 $\theta_{t+1} = \theta_t + \sum_{j=1}^k \lambda_j^*(\tilde{s}_j - \theta_t)$
    - Theorem 1证明Soft FW收敛速率为 $O(1/T)$，优于vanilla的 $O(1/\sqrt{T})$

3. **Task-wise vs. Layer-wise LMO**：

    - Task-wise LMO：将整个模型权重向量化后求解LMO
    - Layer-wise LMO：约束集定义为各层凸包的笛卡尔积 $\mathcal{M} = \mathcal{M}_1 \times \cdots \times \mathcal{M}_L$，每层独立选择最佳模型，可视为块坐标Frank-Wolfe算法
    - $\text{FW}_{hard}$ 更适合layer-wise（NLP判别任务提升7.2分），$\text{FW}_{soft}$ 更适合task-wise（因为内部已做层级系数优化）

### 损失函数 / 训练策略

- 目标函数：任务相关的交叉熵损失（训练数据上最小化）
- $\text{FW}_{hard}$：NLP 10轮迭代，CV 3轮迭代，从Task Arithmetic结果初始化
- $\text{FW}_{soft}$：CV 15轮迭代，从预训练模型初始化
- 每个任务仅需100个训练样本（而传统MTL需2.9K样本/任务）
- 恒定内存开销：每次仅加载固定数量模型，无需同时存储所有模型

## 实验关键数据

### 主实验

**视觉任务（合并20个ViT-B/32）**：

| 方法 | SUN397 | Cars | GTSRB | DTD | 平均 |
|------|--------|------|-------|-----|------|
| Pretrained | 62.3 | 59.7 | 32.6 | 43.8 | 49.6 |
| Task Arithmetic | 20.4 | 12.2 | 29.8 | 22.3 | 21.2 |
| Ties-Merging | 51.0 | 36.2 | 57.7 | 40.6 | 46.4 |
| Adamerging | 66.4 | 70.1 | 95.1 | 64.0 | 73.9 |
| Surgery | 69.7 | 71.8 | 96.6 | 73.4 | 77.9 |
| **FW_soft (Ours)** | **72.9** | **74.8** | **96.8** | **76.0** | **80.1** |

**语言任务**：

| 方法 | 4 判别任务 (8模型) | 3 生成任务 (16模型) | 平均规范化得分 |
|------|---------------------|---------------------|----------------|
| Traditional MTL | 73.1 | 81.2 | 77.2 |
| Task Arithmetic | 80.8 | 75.9 | 78.4 |
| Ties-Merging | 64.3 | 78.5 | 71.4 |
| **FW_hard (Ours)** | **85.4** | **81.1** | **83.1** |

### 消融实验

**扩展性实验（添加无关/相关模型的影响）**：

| #模型 | 无关模型场景 | | 相关模型场景 | |
|-------|-------------|---|-------------|---|
| | Task Arith. | FW_soft | Task Arith. | FW_soft |
| 4 | 70.3 | 74.1 | 59.2 | 59.2 |
| 12 | 47.9 | **74.1** | 52.3 | **67.5** |
| 20 | 21.2 | **74.2** | 36.3 | **68.3** |

FW-Merging展现了理想的缩放特性：16个无关模型加入后性能不衰退（74.1→74.2），16个相关模型加入后性能提升15.3%。

**设计变体消融**：

| 系数 | 方法 | LMO | CV得分 |
|------|------|-----|--------|
| 优化 | FW_soft | Task-wise | 80.1 |
| 优化 | FW_soft | Layer-wise | 79.7 |
| 未优化 | FW_soft | Task-wise | 70.3 |
| - | FW_hard | Layer-wise | 74.0 |

优化合并系数最多提升9.9分；Task-wise LMO略优于Layer-wise（对FW_soft）。

### 关键发现

- FW-Merging仅需每任务100个样本和2分钟时间，性能超越需要2.9K样本和4.2小时的传统MTL
- 线性近似值最小的checkpoint恰好是与目标任务最相关的模型（Figure 3验证），说明梯度内积是可靠的模型相关性指标
- 对噪声模型（不同预训练起点初始化的模型）也具有很强的鲁棒性
- FW-Merging可与其他合并函数（如Ties-Merging）配合使用进一步提升

## 亮点与洞察

- **优雅的理论框架**：将模型合并与经典Frank-Wolfe优化算法对应，提供了收敛性保证和清晰的几何直觉
- **实用的可扩展性**：恒定内存开销 + 对无关模型的鲁棒性，这对HuggingFace生态下的大规模模型合并至关重要
- **极高的数据效率**：每任务仅100个样本，适合隐私敏感和数据稀缺场景
- **正交性强**：作为通用框架可与现有合并方法结合

## 局限与展望

- 内部优化（Soft FW的系数求解）可能增加总时间成本
- 仅使用Task Arithmetic作为MergeFn以保证凸包可行性，更复杂的合并函数可能违反约束但实际效果更好
- 未在更大规模模型（如70B+）上验证
- Layer-wise vs Task-wise的选择缺乏自适应机制，需要根据场景手动选择
- 目标函数设计依赖任务对齐的少量数据，完全无监督场景不适用

## 相关工作与启发

- Task Arithmetic开创了权重空间的任务向量操作思路
- TIES-Merging和DARE解决参数冲突的思路与FW-Merging的模型选择互补
- Adamerging的测试时entropy优化与FW-Merging的训练集cross-entropy优化形成对比
- Frank-Wolfe算法在深度学习约束优化中的应用前景广阔

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 将经典优化算法与模型合并创新结合，理论贡献扎实
- **实验充分度**: ⭐⭐⭐⭐ 覆盖NLP判别/生成+CV任务，扩展性分析详尽，消融完整
- **写作质量**: ⭐⭐⭐⭐⭐ 数学推导严谨，直觉图示清晰，理论与实验结合紧密
- **价值**: ⭐⭐⭐⭐⭐ 开创了可扩展模型合并的新范式，对开源模型生态极具实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Training-free LLM Merging for Multi-task Learning](../../ACL2025/llm_nlp/training-free_llm_merging_for_multi-task_learning.md)
- [\[ACL 2025\] Binary Classifier Optimization for Large Language Model Alignment](../../ACL2025/llm_nlp/bco_binary_classifier_alignment.md)
- [\[ICCV 2025\] VIM: Versatile Interactive Motion-Language Model](vim_versatile_interactive_motion_language_model.md)
- [\[ICML 2025\] TabFlex: Scaling Tabular Learning to Millions with Linear Attention](../../ICML2025/llm_nlp/tabflex_scaling_tabular_learning_to_millions_with_linear_attention.md)
- [\[ICML 2025\] Towards Universal Offline Black-Box Optimization via Learning Language Model Embeddings](../../ICML2025/llm_nlp/towards_universal_offline_black-box_optimization_via_learning_language_model_emb.md)

</div>

<!-- RELATED:END -->
