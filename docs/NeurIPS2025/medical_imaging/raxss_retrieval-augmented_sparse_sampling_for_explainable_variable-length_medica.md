---
title: >-
  [论文解读] RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification
description: >-
  [医学图像] 提出RAxSS框架，将检索增强机制引入随机稀疏采样(SSS)流水线，通过窗口内相似度加权聚合替代均匀平均，在保持变长医学时间序列分类性能的同时提供从"哪里"到"为什么"的可解释性证据链。
tags:
  - 医学图像
---

# RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification

## 元信息
- **会议**: NeurIPS 2025
- **arXiv**: [2510.02936](https://arxiv.org/abs/2510.02936)
- **代码**: 暂无
- **领域**: 医学图像
- **关键词**: 时间序列分类, 检索增强, 稀疏采样, 可解释性, 癫痫脑电

## 一句话总结
提出RAxSS框架，将检索增强机制引入随机稀疏采样(SSS)流水线，通过窗口内相似度加权聚合替代均匀平均，在保持变长医学时间序列分类性能的同时提供从"哪里"到"为什么"的可解释性证据链。

## 研究背景与动机

医学时间序列（心率、血糖、脑电图等）的AI分析面临两大持续性障碍：临床数据的异质性和对透明可解释性的需求。具体挑战包括：

**变长序列问题**：不同患者和事件的记录长度差异巨大，但主流时间序列分类(TSC)方法基于固定长度序列设计

**均匀聚合的局限**：SSS方法通过从长记录中随机采样固定长度窗口并聚合预测来处理变长问题，但均匀聚合假设所有片段同等重要——在非平稳、不规则模式的真实数据中不合理

**可解释性不足**：SSS的可解释性仅依赖局部分数的可视化（告诉"哪里"有信号），无法回答"为什么"某区域被信任

**核心思路**：借鉴预测领域的检索增强方法(RAFT)——通过相似性检索提升稀有模式和弱时间相关性场景的表现——将其适配到分类任务中，实现选择性加权和可解释的证据追溯。

## 方法详解

### 整体框架

RAxSS在SSS基础上统一了采样、检索和聚合三个步骤。长序列被分割为固定长度窗口，按长度成比例采样（序列 $i$ 被抽取的概率 $p_i \propto T_i / \sum_j T_j$），由骨干网络 $f_\theta$ 评分。同时，通过通道内检索计算窗口间相似度，用相似度加权替代均匀平均。

### 关键设计

1. **检索感知聚合**：对序列 $i$ 中的每个窗口 $k$，在同一序列内检索 $m$ 个最相似但不相同的窗口（可因滑动窗口提取而时间重叠），使用Pearson或余弦相似度：

$$\bar{s}_k = \frac{1}{m} \sum_{j \in N_k} s_k^{(j)}, \quad s_k^{(j)} = \phi(w_k, w_j)$$

2. **温度软最大加权**：通过温度参数 $\tau$ 的softmax将支持度转换为归一化权重：

$$\alpha_k = \frac{\exp(\bar{s}_k / \tau)}{\sum_{t \in K_i} \exp(\bar{s}_t / \tau)} \in [0,1], \quad \sum_{k \in K_i} \alpha_k = 1$$

3. **概率空间凸组合**：序列级预测是窗口后验概率的凸组合（保证输出仍在概率单纯形上）：

$$\hat{p}^{(i)} = \sum_{k \in K_i} \alpha_k \, p_k$$

4. **从"哪里"到"为什么"的可解释性**：超越定位热图，为每个高影响力窗口 $k$ 提供：(i) 汇总支持度 $\bar{s}_k$；(ii) 排序邻居排行榜 $\{(w_k^{(j)}, s_k^{(j)}) : j \in N_k\}$ 及时间戳。由于 $\partial \alpha_k / \partial s_k^{(j)} = \frac{1}{m\tau} \alpha_k(1 - \alpha_k) > 0$，增加任何邻居相似度严格增加 $\alpha_k$，排行榜构成忠实的归因解释。

### 设计动机

- **选择性放大**：高支持度窗口（与邻域一致的模式）获得更高权重，噪声/异常窗口被降权
- **隐私友好**：检索严格限于同一记录/通道内，不查询外部数据
- **骨干无关**：可与任意分类器结合，包括Transformer变体

## 实验关键数据

### 主实验：多中心iEEG癫痫发作起始区定位

| 模型 | F1 | AUC | Accuracy(%) |
|------|-----|------|-----------|
| **RAxSS (cosine)** | 0.6967±0.0791 | **0.8046±0.0346** | 69.76±5.25 |
| **RAxSS (pearson)** | 0.7275±0.0489 | 0.7980±0.0537 | 70.51±3.59 |
| SSS (复现) | **0.7437±0.0537** | 0.8035±0.0686 | **71.14±6.31** |
| SSS (原文) | 0.7629 | 0.7999 | 72.35 |
| PatchTST | 0.7097 | 0.7852 | 66.83 |
| TimesNet | 0.6897 | 0.7174 | 65.98 |
| ModernTCN | 0.6938 | 0.7305 | 68.42 |
| DLinear | 0.6916 | 0.7044 | 68.41 |
| ROCKET | 0.6847 | 0.7481 | 69.27 |
| Mamba | 0.6452 | 0.7134 | 64.39 |
| GRUs | 0.6948 | 0.7340 | 65.85 |
| LSTM | 0.6709 | 0.7144 | 65.43 |

### 消融实验：相似度函数选择

| 相似度函数 | F1偏好 | AUC偏好 | 特点 |
|-----------|--------|--------|------|
| Cosine | 较低(0.697) | **最高(0.805)** | 优先判别能力 |
| Pearson | **较高(0.728)** | 略低(0.798) | F1/准确率平衡更好 |

临床实践者可根据需求选择：cosine偏重AUC判别，Pearson偏重平衡检测。

### 关键发现

1. **RAxSS与SSS性能竞争**：cosine变体在AUC上最优(0.8046)，超过SSS复现(0.8035)和所有非SSS基线
2. **大幅领先固定长度方法**：相比PatchTST(0.7852 AUC)、ROCKET(0.7481 AUC)等固定长度方法优势明显
3. **可解释性是核心附加价值**：在竞争性性能的基础上提供了SSS所缺失的"为什么"层面的解释
4. **跨中心鲁棒性**：在JHH、NIH、UMMC、UMH四个医疗中心的iEEG数据上验证

## 亮点与洞察

1. **检索从预测迁移到分类**：首次将RAFT的检索增强思路从时间序列预测任务适配到变长分类，是一个巧妙的迁移
2. **数学上保证的忠实解释**：$\alpha_k$ 关于邻居相似度严格单调递增的性质保证了邻居排行榜是"忠实的"(faithful)归因——不是事后的近似解释
3. **凸组合的理论保证**：在概率空间聚合保证输出仍是有效概率分布，避免了logit空间聚合的潜在问题
4. **可操控性**：温度参数 $\tau$ 和邻居数 $m$ 提供了调节可解释性粒度的旋钮

## 局限与展望

- 检索局限于同一通道/记录内，跨被试/跨中心的模式检索可能增强证据质量但引入隐私问题
- 相似度/温度参数目前手动设置，可通过数据学习
- F1和Accuracy未超过SSS，需要后续调优和校准
- 仅在癫痫iEEG一个任务上验证，需扩展到其他医学时间序列（心电、血糖等）
- 缺乏系统的忠实性压力测试（删除/插入测试、检索随机化、反事实探测）

## 相关工作与启发

- **SSS (Mootoo et al.)**: 随机稀疏采样框架，本文在其上增加检索加权
- **RAFT**: 检索增强时间序列预测，利用历史补丁相似性改善预测
- **PatchTST, TimesNet**: 固定长度TSC的代表方法
- **启发**: 可解释性不应是事后附加的装饰，而应内嵌于聚合机制本身——"同一证据既决定权重，又解释贡献"

## 评分
- 新颖性：⭐⭐⭐⭐☆ — 检索增强+稀疏采样的结合思路新颖，可解释性设计精巧
- 实验充分度：⭐⭐⭐☆☆ — 仅一个数据集，缺乏消融实验深度
- 写作质量：⭐⭐⭐⭐☆ — 方法叙述清晰，公式推导严谨
- 价值：⭐⭐⭐⭐☆ — 为临床时间序列分类提供了兼顾性能和可解释性的实用框架

<!-- RELATED:START -->

## 相关论文

- [MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](../../ACL2025/medical_imaging/medbiorag_semantic_search_and_retrieval-augmented_generation_for_biomedical_lite.md)
- [Thompson Sampling via Fine-Tuning of LLMs](../../ICLR2026/medical_imaging/thompson_sampling_via_fine-tuning_of_llms.md)
- [Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](../../AAAI2026/medical_imaging/hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](../../AAAI2026/medical_imaging/fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)
- [Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis](scalable_diffusion_transformer_for_conditional_4d_fmri_synthesis.md)

<!-- RELATED:END -->
