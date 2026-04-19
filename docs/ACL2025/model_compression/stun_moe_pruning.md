---
title: >-
  [论文解读] STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning
description: >-
  [ACL 2025][模型压缩][MoE剪枝] STUN 提出"先结构化后非结构化"的两阶段 MoE 剪枝范式：第一阶段利用路由权重的行为相似性聚类冗余专家，以 $O(1)$ GPU 前向传播完成专家级剪枝；第二阶段在剩余专家内做非结构化权重剪枝，两者协同在 480B Snowflake Arctic 上以 40% 稀疏度几乎无性能损失。
tags:
  - ACL 2025
  - 模型压缩
  - MoE剪枝
  - 结构化剪枝
  - 非结构化剪枝
  - 专家聚类
  - 稀疏化
---

# STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning

**会议**: ACL 2025  
**arXiv**: [2409.06211](https://arxiv.org/abs/2409.06211)  
**代码**: [https://github.com/thnkinbtfly/STUN](https://github.com/thnkinbtfly/STUN)  
**领域**: Model Compression  
**关键词**: MoE剪枝, 结构化剪枝, 非结构化剪枝, 专家聚类, 稀疏化

## 一句话总结

STUN 提出"先结构化后非结构化"的两阶段 MoE 剪枝范式：第一阶段利用路由权重的行为相似性聚类冗余专家，以 $O(1)$ GPU 前向传播完成专家级剪枝；第二阶段在剩余专家内做非结构化权重剪枝，两者协同在 480B Snowflake Arctic 上以 40% 稀疏度几乎无性能损失。

## 研究背景与动机

**领域现状**：MoE 通过稀疏激活专家子集来降低推理成本，已成为大模型的主流架构选择。但参数总量不减反增——Mixtral 56B (8 experts)、DBRX 132B (16 experts)、Snowflake Arctic 480B (128 experts)——服务部署仍需要大量 GPU 内存。

**现有方案局限**：非结构化剪枝（Wanda、SparseGPT）在权重张量内逐元素稀疏化，无法利用 MoE 天然的专家结构；行/列级结构化剪枝（LLM-Pruner）破坏模型能力严重；现有专家级剪枝方法（Lu et al.）需要穷举专家组合，复杂度为 $O(k^n/\sqrt{n})$（$k>1$, $n$ 为专家数），对 128 个专家完全不可行。

**核心矛盾**：专家级剪枝能利用 MoE 固有结构、效果好，但搜索空间随专家数指数爆炸；非结构化剪枝可扩展却在生成任务（GSM8K）上性能崩塌。单用任何一种方法均无法在高稀疏度下保持性能。

**本文切入角度**：将剪枝问题重新定义为结构化与非结构化的插值（interpolation），发现中间插值点性能最优（图1）。进一步提出基于专家行为相似性的 $O(1)$ 聚类算法解决可扩展性问题。

**核心 idea**：先用路由权重相似性以 $O(1)$ 做专家级剪枝消除 inter-expert 冗余，再在剩余专家内做非结构化剪枝捕获 intra-expert 稀疏性，两阶段互补协同。

## 方法详解

### 整体框架

原始 MoE → **阶段一：专家级结构化剪枝**（基于路由权重的行为相似性做专家聚类，每簇保留一个代表专家，$O(1)$ GPU 调用）→ **阶段二：非结构化权重剪枝**（在剩余专家内部用 Wanda/OWL 做逐元素稀疏化）→ 最终压缩模型。整个过程 training-free，无需微调或反向传播。

### 关键设计

1. **基于行为相似性的 $O(1)$ 专家聚类与剪枝**

    - **功能**：将每层 MoE 中行为相似的专家聚类为簇，每簇仅保留最接近簇中心的代表专家，其余全部移除
    - **核心思路**：从路由层权重矩阵 $W$ 中提取专家对的行为相似性 $b_{i,j} = -\|W_i - W_j\|_F + \lambda_2 a_{i,j}$（$a_{i,j}$ 为共激活统计量），路由权重相似意味着对相同输入给出相近的激活概率、暗示相似的专业分工；基于此相似度做层次聚类，将贪心逐步剪枝重构为概率分布的逐步最大化（Eq. 5-6），使得贪心决策近似等效于组合搜索
    - **设计动机**：Lu et al. 的穷举式专家组合搜索在 128 专家时需 $>10^{33}$ 次前向传播，完全不可行；本方法仅需路由权重计算（CPU 端 < 1分钟）+ 可选的 1 次前向传播收集共激活统计

2. **Taylor 近似 + 选择性专家重建**

    - **功能**：在聚类确定后，决定每簇保留哪个专家，以及是否用簇内均值重建代表专家
    - **核心思路**：用一阶 Taylor 展开近似各专家的重建损失 $\mathcal{E}_i = \|M(x;\theta) - M(x;\theta - \theta_i)\|_F$，选择离簇中心参数 $\bar{\theta_i}$ 最近的专家作为代表；当簇内专家数少于阈值 $\kappa$（取 3）时，用所有成员参数的均值重建代表专家以进一步降低重建误差
    - **设计动机**：纯选择最近专家丢失了被剪除专家的知识，小簇内均值重建可部分恢复；大簇避免均值以防模糊化

3. **Kurtosis 保持下的非结构化剪枝**

    - **功能**：在专家剪枝后的模型上执行 Wanda 或 OWL 等非结构化剪枝
    - **核心思路**：权重分布的 Kurtosis（峰度）是非结构化剪枝鲁棒性的代理指标——高峰度意味着大量可安全移除的近零权重。专家级剪枝不改变剩余专家内部的权重分布（仍近似高斯），因此 kurtosis 不降反升（实验中从 14248 升至 15623）；而直接做非结构化剪枝会移除近零权重、使分布趋向双峰（kurtosis 降低），削弱后续进一步稀疏化的空间
    - **设计动机**：理论保证"先结构化后非结构化"的顺序优于反向或单独使用

### 损失函数 / 训练策略

整个流程 training-free，无需反向传播或微调。仅需少量 C4 校准数据用于可选的共激活统计收集。对于最大的 Arctic 模型，使用 $(\lambda_1, \lambda_2) = (1, 0)$ 即纯路由权重相似性，完全零 GPU 调用完成专家剪枝。

## 实验关键数据

### 主实验：STUN vs 非结构化剪枝基线

| 模型 | 稀疏度 | 方法 | GSM8K | ARC-c | HellaSwag | MMLU | 平均 |
|------|--------|------|-------|-------|-----------|------|------|
| Arctic (480B) | 0% | 未剪枝 | 70.74 | 56.91 | 66.94 | 64.86 | 68.33 |
| Arctic (480B) | 40% | **STUN (OWL)** | **70.28** | **57.68** | **64.94** | **64.75** | **67.66** |
| Arctic (480B) | 40% | OWL | 63.76 | 56.74 | 65.08 | 63.47 | 67.35 |
| Arctic (480B) | 65% | **STUN (OWL)** | **43.97** | **51.54** | **59.91** | **59.24** | **62.67** |
| Arctic (480B) | 65% | OWL | 13.42 | 44.37 | 53.69 | 52.02 | 56.68 |
| Mixtral-8x7B-Inst | 65% | **STUN (OWL)** | **25.09** | **48.12** | **54.05** | **60.39** | **60.34** |
| Mixtral-8x7B-Inst | 65% | OWL | 1.29 | 24.15 | 49.27 | 57.60 | 45.20 |

### 专家剪枝方法对比（Mixtral-8x7B, 25% 稀疏度）

| 方法 | 计算复杂度 | ARC-C | BoolQ | HellaSwag | MMLU | WinoGrande | 平均 |
|------|-----------|-------|-------|-----------|------|------------|------|
| 未剪枝 | - | 59.4 | 84.2 | 84.0 | 67.9 | 75.6 | 71.5 |
| **Ours** | **$O(1)$** | **55.6** | **83.1** | **81.1** | **63.3** | **72.7** | **70.7** |
| Expert Drop | $O(n)$ | 53.2 | 77.7 | 80.5 | 52.2 | 76.8 | 66.0 |
| SEER-MoE | - | - | - | - | 56.7 | - | - |
| Lu et al. | $O(k^n/\sqrt{n})$ | - | - | - | - | - | 64.22 |

### 关键发现

- **插值最优**：在 Mixtral 上固定 50% 总稀疏度，纯结构化（x=1）和纯非结构化（x=0）均大幅下降，中间插值点性能最高（图1），验证了两阶段组合的必要性
- **GSM8K 生成任务优势显著**：Arctic 40% 稀疏度下 STUN GSM8K 仅降 0.46 分，而 OWL 降 6.98 分；65% 稀疏度 STUN 仍有 43.97 而 OWL 仅 13.42
- **$O(1)$ vs $O(n)$ 无性能差异**：在 Mixtral 上 25% 稀疏度时两者平均分分别为 64.34 和 63.97，$O(1)$ 甚至略优
- **小专家多专家受益更大**：STUN 与纯非结构化的性能差距随专家数增加而增大（Arctic 128 experts > Mixtral-8x22B > Mixtral-8x7B），契合 MoE 的发展趋势
- **效率极高**：Arctic 480B 的 STUN+OWL 仅需单张 H100 + 1.12 小时；Lu et al. 方法对 128 专家完全不可行（>$10^{33}$ 次前向传播）

## 亮点与洞察

- $O(1)$ 专家剪枝的关键洞见是利用路由权重的隐含结构——相似路由权重意味着相似激活模式，聚类后贪心剪枝近似等效于组合搜索，这一思路可迁移到其他有天然模块结构的模型
- Kurtosis 保持的理论分析优雅地解释了为什么阶段顺序（先结构化后非结构化）不能颠倒——先做非结构化会降低 kurtosis、削弱进一步剪枝的空间
- 在非 MoE 模型（Llama-2）上也验证了 STUN 范式的有效性，说明"先粗后细"的两阶段剪枝是通用策略

## 局限与展望

- 非结构化剪枝在当前 GPU 上难以直接加速推理（需要稀疏硬件支持），实际部署加速效果取决于硬件
- 专家行为相似性仅基于路由权重和共激活统计，未考虑专家输出的语义差异
- 高于 40% 稀疏度后性能下降明显（Arctic 65% 时 GSM8K 从 70 降到 44），极端稀疏度下的策略有待改进
- 在非 MoE 上的实验仅用了 LLM-Surgeon 5% 结构化剪枝作为第一阶段，设计空间未充分探索

## 相关工作与启发

- **vs Wanda/OWL**：纯非结构化方法，STUN 在第一阶段先移除冗余专家后再调用它们，两者互补；核心差异是 STUN 额外利用了 MoE 的 inter-expert 冗余
- **vs Lu et al.**：同为专家级剪枝但复杂度 $O(k^n/\sqrt{n})$，STUN 通过行为聚类将搜索空间从指数降到常数，且效果更好；Lu et al. 依赖校准数据而 STUN 利用模型自身结构
- **vs LLM-Pruner**：通用结构化剪枝（行/列级），在高稀疏度 MoE 上性能崩塌（65% 时 GSM8K 仅 1.29），STUN 通过专家级粒度保留关键能力
- **vs SEER-MoE / Expert Drop**：其他高效专家剪枝方法，STUN 在 MMLU 上大幅领先（63.3 vs 56.7 vs 52.2），且额外叠加非结构化剪枝进一步压缩

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将剪枝问题重定义为结构化-非结构化的插值、通过路由权重聚类实现 $O(1)$ 专家剪枝的思路清晰优雅，kurtosis 保持分析提供了良好的理论支撑
- **实验充分度**: ⭐⭐⭐⭐ 涵盖 480B Arctic + Mixtral 两个系列 + 非 MoE Llama-2，多稀疏度多任务，消融实验（$O(1)$ vs $O(n)$、插值比例）完整
- **写作质量**: ⭐⭐⭐⭐ 图1 直观展示插值优势、Table 1 清晰定位贡献，整体逻辑从动机→方法→理论→实验层层递进
- **实用价值**: ⭐⭐⭐⭐⭐ 对 480B 级 MoE 仅需 1 GPU + 1 小时即可完成 40% 无损压缩，对大模型部署有直接工程价值

<!-- RELATED:START -->

## 相关论文

- [MoNE: Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE](../../ICLR2026/model_compression/mone_replacing_redundant_experts_with_lightweight_novices_for_structured_pruning.md)
- [Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [BlockPruner: Fine-grained Pruning for Large Language Models](blockpruner_fine-grained_pruning_for_large_language_models.md)
- [Olica: Efficient Structured Pruning of Large Language Models without Retraining](../../ICML2025/model_compression/olica_efficient_structured_pruning_of_large_language_models_without_retraining.md)
- [MUSTAFAR: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference](../../NeurIPS2025/model_compression/mustafar_promoting_unstructured_sparsity_for_kv_cache_pruning_in_llm_inference.md)

<!-- RELATED:END -->
