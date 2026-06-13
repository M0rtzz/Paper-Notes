---
title: >-
  [论文解读] Beyond One-Size-Fits-All: Tailored Benchmarks for Efficient Evaluation
description: >-
  [ACL 2025][LLM评测][高效评估] 提出 TailoredBench 方法，为每个待评估的目标模型**自适应构建定制化核心集**（Native-coreset），而非使用所有模型共享的静态子集，通过自适应源模型选择、可扩展 K-Medoids 聚类和校准估计策略，在仅需 20-40 个样本的推理预算下将准确率估计的 MAE 平均降低 **31.4%**。
tags:
  - "ACL 2025"
  - "LLM评测"
  - "高效评估"
  - "基准压缩"
  - "核心集选择"
  - "预测一致性"
  - "K-Medoids聚类"
  - "模型排序"
---

# Beyond One-Size-Fits-All: Tailored Benchmarks for Efficient Evaluation

**会议**: ACL 2025  
**arXiv**: [2502.13576](https://arxiv.org/abs/2502.13576)  
**代码**: [https://github.com/marvelcell/TailoredBench](https://github.com/marvelcell/TailoredBench)  
**作者**: Peiwen Yuan, Yueqi Zhang, Shaoxiong Feng, Yiwei Li, Xinglin Wang, Jiayi Shi, Chuyi Tan, Boyuan Pan, Yao Hu, Kan Li  
**机构**: Beijing Institute of Technology, Xiaohongshu Inc  
**领域**: 高效评估 / 基准压缩  
**关键词**: 高效评估, 基准压缩, 核心集选择, 预测一致性, K-Medoids聚类, 模型排序

## 一句话总结

提出 TailoredBench 方法，为每个待评估的目标模型**自适应构建定制化核心集**（Native-coreset），而非使用所有模型共享的静态子集，通过自适应源模型选择、可扩展 K-Medoids 聚类和校准估计策略，在仅需 20-40 个样本的推理预算下将准确率估计的 MAE 平均降低 **31.4%**。

## 研究背景与动机

**评估成本危机**：在 HELM 排行榜上评估一个 10B 参数模型需花费 $1,700（API）或 1,200+ GPU 小时。当需要比较 $X$ 个配置时，成本线性增长。

**现有高效评估范式**：
   - 利用已公开的源模型（source models）评测结果构建样本嵌入
   - 聚类选取少量代表性核心集（coreset，通常 <100 样本）
   - 用目标模型（target model）在核心集上的表现估算全基准性能

**核心假设的问题**：现有方法假设源模型和目标模型的**预测一致性**（prediction consistency）很高——如果源模型在样本 a 和 b 上表现相似，目标模型也应如此。但作者发现**这一假设在实践中不成立**：
   - t-SNE 可视化（Hellaswag）显示：用目标模型嵌入后，样本到聚类中心的平均距离从 10.09 增加到 12.48
   - 源模型选定的核心集无法有效代表目标模型的行为模式

**核心动机**：摒弃"一刀切"的静态核心集，为每个目标模型量身定制评估子集。

## 方法详解

### 整体框架：Global-to-Native 评估流程

TailoredBench 包含四个紧密集成的步骤：

### Step 1: 构建 G-set（Global-coreset）

- 对每个样本 $x_k$，用所有 $|\mathcal{S}|$ 个源模型的正确率构建嵌入向量 $\dot{x}_k^{\mathcal{S}}$
- 在该嵌入空间上执行 **K-Medoids 聚类**，选取聚类中心构成 G-set（默认 10 个样本）
- G-set 作为**探针**，用于识别与目标模型最一致的源模型
- **距离度量选择**：采用**曼哈顿距离**（element-wise）而非相关距离（correlation），因为相关距离假设线性关系，不适用于离散二值嵌入

### Step 2: 自适应原生源模型选择

- 让目标模型在 G-set 上做推理
- 将所有源模型和目标模型的 G-set 预测编码为嵌入
- 计算所有模型间的平均预测一致性 $\bar{d}$ 作为阈值
- 对每个目标模型 $t_m$，选取距离小于 $\bar{d}$ 的源模型组成原生源模型集 $\mathcal{S}_{t_m}$
- 跨所有目标模型**统一原生源模型数量** $\bar{n}$，保持嵌入维度一致

### Step 3: 可扩展 K-Medoids 聚类构建 N-set

- 基于原生源模型的预测重新构建样本嵌入（更贴合目标模型视角）
- **锚定初始化**：G-set 中的样本固定为初始聚类中心，新增 $|N-set| - |G-set|$ 个随机样本
- **动态精炼**：非 G-set 聚类中心可更新（选择最小化簇内距离的样本），G-set 中心保持不变
- 迭代直至收敛 → 得到定制化 N-set $\mathcal{N}_{t_m}$

### Step 4: 校准性能估计

不直接用中心样本的表现代表整个簇，而是利用源模型预测一致性进行**校准**：

- 对簇内非中心样本 $x'$，计算缩放因子：$\text{Scale}(x') = \frac{\bar{c}_{\mathcal{S}_{t_m}, x'} + 0.5}{\bar{c}_{\mathcal{S}_{t_m}, x} + 0.5}$
- 用缩放因子校准目标模型在所有样本上的预测：$c_{t_m, x'} = (c_{t_m, x} + 0.5) \cdot \text{Scale}(x') - 0.5$
- 全基准性能：$P_{t_m} = \frac{1}{|\mathcal{D}|} \sum_{x' \in \mathcal{D}} c_{t_m, x'}$

## 实验关键数据

### 主实验（5 基准，300+ 模型）

| 基准 | 推理数 | 最佳基线 MAE | TailoredBench MAE | MAE降幅 |
|------|--------|------------|-------------------|--------|
| ARC Challenge | 30 | 0.036 | **0.028** | 22.2% |
| Hellaswag | 30 | 0.043 | **0.018** | 58.1% |
| GSM8K | 30 | 0.041 | **0.033** | 19.5% |
| Winogrande | 30 | 0.038 | **0.024** | 36.8% |
| POPE | 30 | 0.034 | **0.031** | 8.8% |

- 平均 MAE 降低 **31.4%**
- Kendall's τ 也一致提升（排序准确性更高）
- Hellaswag 上成对模型排序准确率达 **96.0%**

### 距离度量消融

| 距离类型 | Kendall's τ | MAE |
|---------|------------|-----|
| Correlation | 0.720 | 0.032 |
| Cosine | 0.736 | 0.028 |
| **Manhattan** | **0.740** | **0.027** |

- 曼哈顿距离在连续和离散正确率格式上均优于相关距离

### 校准策略消融

| 策略 | Kendall's τ | MAE |
|------|------------|-----|
| 无校准 | 0.724 | 0.030 |
| **有校准** | **0.740** | **0.027** |

### G-set 大小分析

| G-set 大小 | Kendall's τ | MAE |
|-----------|------------|-----|
| 5 | 0.734 | 0.030 |
| **10** | **0.740** | **0.027** |
| 15 | 0.736 | 0.028 |
| 25 | 0.731 | 0.029 |

- **10 个样本作为探针即足够**，过多 G-set 反而挤压 N-set 的灵活性

### 大推理预算验证（Hellaswag）

| 推理数 | 方法 | Kendall's τ | MAE |
|--------|------|------------|-----|
| 150 | Random | 0.935 | 0.030 |
| 150 | AnchorPoints | 0.940 | 0.040 |
| 150 | gp-IRT | 0.936 | **0.012** |
| 150 | **TailoredBench** | **0.943** | **0.012** |

- 在较大推理预算下仍保持优势

### 关键发现

- 原生源模型数量越多、与目标模型预测一致性越高，估计越准确
- 目标模型倾向于选择**同家族的源模型**（如 Llama 选 Llama，Qwen 选 Qwen），说明模型家族内部共享预测模式
- 所有优势通过 Z 检验验证统计显著（p < 0.05）

## 亮点与洞察

1. **直击核心假设缺陷**：通过 t-SNE 可视化直观展示了"预测一致性假设"的失败，动机清晰有力
2. **从全局到局部的优雅设计**：G-set 作为探针 → 自适应选源模型 → 定制 N-set → 校准估计，四步流程逻辑自洽
3. **10 样本探针**：仅需目标模型在 10 个样本上的推理即可识别最匹配的源模型子集，极高效
4. **跨模态泛化**：在 NLP（ARC、Hellaswag、GSM8K、Winogrande）和多模态（POPE）基准上均有效
5. **实际意义巨大**：对于模型开发者，在超参/配置搜索时可将评估成本降低数个数量级

## 局限性

1. **依赖源模型评测结果**：需要大量源模型在全基准上的完整评测结果，对新/私有基准的冷启动成本仍然存在
2. **假设源模型表征足够**：若目标模型的行为模式与所有源模型差异极大，方法可能失效
3. **仅验证选择题/分类基准**：对开放式生成任务（如摘要、对话）的适用性未探索
4. **静态 G-set**：G-set 对所有目标模型共享，若目标模型群体分化极大，单一 G-set 可能不够
5. **校准策略简单**：线性缩放因子假设源-目标模型行为关系是线性的

## 相关工作

- **AnchorPoints (Vivek et al. 2024)**：基于源模型预测的 K-Medoids 聚类选静态核心集
- **gp-IRT (Polo et al. 2024)**：基于项目反应理论（IRT）为样本提取潜在表示
- **Flash-HELM (Perlitz et al. 2023)**：动态调整随机子集大小
- **Sort & Search (Prabhu et al. 2024)**：利用题目难度和动态规划

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**：⭐⭐⭐⭐ 定制化核心集的思路清晰且新颖，对预测一致性假设的质疑有充分实验支撑
- **实验充分性**：⭐⭐⭐⭐⭐ 5 基准 300+ 模型，多种消融和分析，极其充分
- **写作质量**：⭐⭐⭐⭐ 公式推导清楚，但符号较多需要仔细阅读
- **实用性**：⭐⭐⭐⭐ 对排行榜维护者和模型开发者有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ChatBench: From Static Benchmarks to Human-AI Evaluation](chatbench_from_static_benchmarks_to_human-ai_evaluation.md)
- [\[ACL 2025\] Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks](somethings_fishy_in_the_data_lake_a_critical_re-evaluation_of_table_union_search.md)
- [\[ACL 2026\] Beyond Static Benchmarks: Synthesizing Harmful Content via Persona-based Simulation for Robust Evaluation](../../ACL2026/llm_evaluation/beyond_static_benchmarks_synthesizing_harmful_content_via_persona-based_simulati.md)
- [\[ACL 2025\] ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities](onebench_to_test_them_all_sample-level_benchmarking_over_open-ended_capabilities.md)
- [\[ACL 2025\] Revisiting 3D LLM Benchmarks: Are We Really Testing 3D Capabilities?](revisiting_3d_llm_benchmarks_are_we_really_testing_3d_capabilities.md)

</div>

<!-- RELATED:END -->
