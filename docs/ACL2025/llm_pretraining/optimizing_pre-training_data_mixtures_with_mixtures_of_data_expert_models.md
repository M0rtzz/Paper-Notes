---
title: >-
  [论文解读] Optimizing Pre-Training Data Mixtures with Mixtures of Data Expert Models
description: >-
  [ACL2025][Data Mixture] 提出Mixture of Data Experts (MDE)方法，通过在各数据域上独立训练专家模型并用混合权重进行概率级集成，高效近似不同数据混合比下的语言模型损失，大幅提升预训练数据混合比例的搜索效率和预测精度。
tags:
  - ACL2025
  - Data Mixture
  - Mixture of Data Experts
  - Proxy Model
  - Regression
  - Pretraining
---

# Optimizing Pre-Training Data Mixtures with Mixtures of Data Expert Models

**会议**: ACL2025  
**arXiv**: [2502.15950](https://arxiv.org/abs/2502.15950)  
**代码**: 未公开  
**领域**: LLM预训练 / 数据混合优化  
**关键词**: Data Mixture, Mixture of Data Experts, Proxy Model, Regression, Pretraining  

## 一句话总结

提出Mixture of Data Experts (MDE)方法，通过在各数据域上独立训练专家模型并用混合权重进行概率级集成，高效近似不同数据混合比下的语言模型损失，大幅提升预训练数据混合比例的搜索效率和预测精度。

## 研究背景与动机

- **数据混合比重要性**：LLM预训练数据来自多个异构来源（如Wikipedia、GitHub、CommonCrawl等），不同来源的采样比例对模型泛化性能有显著影响
- **搜索空间巨大**：对于k个数据域，混合比例定义了k-1个实值超参数，大规模LLM通常只训练一次，无法穷举评估大量混合方案
- **代理模型方法的局限**：
    - 在线方法（DoGe、DoReMi）需要修改训练算法，不能为不同优化目标复用同一组代理模型
    - 回归方法（RegMix等）仅使用混合权重λ作为特征，预测精度有限，需要训练30-500个代理模型
- **核心问题**：能否用极少的代理模型高效预测任意数据混合比的模型泛化性能？

## 方法详解

### 整体框架

MDE方法包含三个层次：
1. 单独训练k个数据专家模型（每个域一个）
2. 用MDE集成近似任意混合比λ的模型性能
3. （可选）将MDE特征注入回归模型，进一步提升预测精度

### Mixture of Data Experts (MDE) 近似

**核心思想**：用k个分别在单独数据域上训练的专家模型θ₁*,...,θₖ*，通过概率级加权集成近似在混合数据上训练的模型：

$$P_{MDE}(x_t|x_{1\cdots t-1},\lambda) := \sum_{i=1}^{k}\lambda_i P_{\theta_i^*}(x_t|x_{1\cdots t-1})$$

对于每个候选混合方案λ，MDE损失定义为此集成模型在各验证域上的交叉熵损失。

**高效实现**：
- 预计算并缓存每个专家θᵢ*在所有验证域token上的next-token概率
- 对每个新的候选λ，只需在CPU上执行O(k)次加权求和和取对数操作
- 不需要为每个候选混合方案运行神经网络推理，计算成本可忽略

### MDE作为回归特征

将MDE近似的各域损失作为额外特征输入回归模型：

$$\hat{L}_j(\lambda) = M_j(\lambda; L_{MDE}^1, ..., L_{MDE}^m)$$

考察了三类回归模型：
- **线性模型**：正则化加权特征求和
- **梯度提升(GBM)**：回归树集成（参考RegMix）
- **多任务高斯过程(MTGP)**：利用域间任务相关性

### 理论基础

**命题3.1**：对于任意混合权重λ，最小化加权混合损失的最优分布p*λ可以表示为各数据专家最优分布的加权组合：

$$p^*_\lambda(y|x) = \sum_{i=1}^{k}\lambda'_i(x)p^*_i(y|x)$$

其中λ'ᵢ(x) ∝ Dᵢ(x)λᵢ。当各域的前缀分布相同时，λ'ᵢ = λᵢ，与MDE近似完全一致。这为概率集成提供了理论正当性。

### 优化目标

泛化性能定义为验证域损失的聚合：
- **avg-sp**：7个SlimPajama验证域的平均损失（训练域）
- **avg-et**：11个end-task验证域的平均损失（下游任务）
- **avg-et+sp**：18个验证域的联合平均损失

使用Vizier框架在k个非负参数搜索空间中优化，不要求目标可微。

## 实验

### 实验设置

- **数据集**：SlimPajama（7个训练域）
- **模型规模**：70M、150M、280M、510M、1B参数
- **训练token**：代理模型5-25B，全规模模型100B
- **代理模型选择**：280M/10K steps作为1B/200K steps的代理
- **验证域**：SlimPajama验证集(7域) + ARC/OpenBookQA/MultiRC(11域)
- **下游任务**：TriviaQA、NQ、SQuAD 2.0、LAMBADA等10个任务

### 损失预测精度

| 方法 | MSE(SP)↓ | Spearman ρ(SP)↑ | MSE(ET+SP)↓ | ρ(ET+SP)↑ |
|------|----------|-----------------|-------------|-----------|
| Empirical Mean | 0.01151 | N/A | 0.01250 | N/A |
| Linear | 0.01637 | 0.234 | 0.00655 | 0.646 |
| GBM-RegMix | 0.00242 | 0.923 | 0.00431 | 0.814 |
| DML | 0.00296 | 0.920 | 0.00116 | 0.892 |
| **MDE (仅7模型)** | 0.02809 | 0.912 | 0.00391 | 0.886 |
| **Linear+MDE** | **0.00050** | **0.976** | **0.00048** | **0.953** |
| **MTGP+MDE** | **0.00053** | **0.984** | 0.00116 | 0.935 |
| **GBM+MDE** | 0.00140 | 0.950 | **0.00089** | **0.955** |

**关键发现**：
- MDE作为独立预测器，排序能力与使用3x更多代理模型的最佳回归器相当
- MDE特征为所有回归模型带来大幅提升：Linear的Spearman从0.65提升到0.95
- 25个训练样本+MDE特征的性能超越现有方法

### 代理模型规模分析

- 对单个训练域的排序，70M和280M代理差异不大
- 对跨多域聚合指标的排序，70M和训练步数<6K的代理表现显著劣于280M
- 验证了"近似排名不变性"假设的条件性

### 学习曲线

- 低数据情况下，MDE（仅k=7个模型）持续优于所有回归方法
- 超过25个训练样本后收益递减
- 结合MDE特征的回归模型随样本增加稳步提升

### 下游任务表现

使用avg-et+sp作为优化目标选出的混合方案:
- 在generation任务和ranking任务上均优于DoGe、DoReMi的启发式方法
- 验证了将end-task验证域纳入优化目标的价值

### 验证损失与下游性能的相关性

- 单个end-task验证域与下游性能的相关性差异较大（Self相关性从0.245到0.903）
- 聚合多个验证域的损失与所有下游任务的相关性更稳定（0.77-0.85）
- 验证了使用多样化验证集定义优化目标的合理性

## 亮点与洞察

1. **极致效率**：仅需k个数据专家模型（每个域一个），即可获得与30+代理模型方案可比的混合比排序能力
2. **即插即用**：MDE特征可增强任意回归模型（Linear、GBM、MTGP），提升幅度显著且稳定
3. **理论驱动**：命题3.1从信息论角度证明了概率集成近似的合理性，并指出了改进方向（前缀依赖的权重）
4. **实用价值高**：不需要修改LM训练算法，同一组代理模型可服务于多种优化目标
5. **缓存+CPU计算**：MDE近似的核心计算几乎零成本，缓存token概率后对无穷多候选方案的评估都在CPU上完成
6. **跨尺度有效**：280M代理模型优化的混合权重可有效迁移到1B模型

## 局限性

1. **域数量限制**：实验仅在7个训练域上验证，数十或更多域时可能面临新挑战
2. **静态混合**：仅考虑固定采样比，未探索动态课程式混合
3. **SlimPajama单一数据集**：实验集中在一个数据集上，更大规模/更多样化数据集上的表现未知
4. **近似理论的理想假设**：理论结果在各域前缀分布不同时最优权重变为x-dependent，实际效果受此偏差影响
5. **交叉熵vs.解码性能**：优化的是交叉熵而非下游任务准确率，两者的关系并不完全线性

## 相关工作

- **在线方法**：DoGe（一阶双层优化）、DoReMi（最差情况gap优化）
- **回归方法**：RegMix（GBM回归）、DML（指数模型）、BiMix（幂律模型）
- **模型混合近似**：参数平均方法、从头预训练模型的合并
- **数据选择与课程**：domain-level、sample-level等不同粒度的数据选择

## 评分 ⭐⭐⭐⭐

- **创新性**：⭐⭐⭐⭐⭐ MDE思路简洁优美，理论和实践统一
- **实验充分性**：⭐⭐⭐⭐ 多尺度、多回归模型、学习曲线分析全面
- **实用价值**：⭐⭐⭐⭐ 对大规模LLM预训练的数据配方优化有直接指导意义
- **写作质量**：⭐⭐⭐⭐ 结构清晰，理论部分易读

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](metarater_a_multidimensional_data_selection_method.md)
- [\[ACL 2025\] Improving Continual Pre-training Through Seamless Data Packing](improving_continual_pre-training_through_seamless_data_packing.md)
- [\[ACL 2025\] Data-Constrained Synthesis of Training Data for De-Identification](data-constrained_synthesis_of_training_data_for_de-identification.md)
- [\[ACL 2025\] Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack](stealing_training_data_from_large_language_models_in_decentralized_training_thro.md)
- [\[NeurIPS 2025\] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](../../NeurIPS2025/llm_pretraining/nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)

<!-- RELATED:END -->
