---
title: >-
  [论文解读] Verifying Chain-of-Thought Reasoning via Its Computational Graph
description: >-
  [ICLR 2026][LLM推理][Chain-of-Thought] 提出 CRV（Circuit-based Reasoning Verification），通过将 LLM 的 MLP 替换为 transcoder 构建可解释归因图，从图的结构特征中提取推理错误的"指纹"，实现白盒 CoT 推理验证，并可通过因果干预修正错误推理。
tags:
  - ICLR 2026
  - LLM推理
  - Chain-of-Thought
  - 归因图
  - Transcoder
  - 推理验证
  - 因果干预
---

# Verifying Chain-of-Thought Reasoning via Its Computational Graph

**会议**: ICLR 2026  
**arXiv**: [2510.09312](https://arxiv.org/abs/2510.09312)  
**代码**: [有](https://github.com/facebookresearch/CRV)  
**领域**: LLM Reasoning / Mechanistic Interpretability  
**关键词**: Chain-of-Thought, 归因图, Transcoder, 推理验证, 因果干预

## 一句话总结

提出 CRV（Circuit-based Reasoning Verification），通过将 LLM 的 MLP 替换为 transcoder 构建可解释归因图，从图的结构特征中提取推理错误的"指纹"，实现白盒 CoT 推理验证，并可通过因果干预修正错误推理。

## 研究背景与动机

现有 CoT 验证方法分为两类：**黑盒方法**（分析输出文本或 logit 分布）和**灰盒方法**（利用隐层激活或隐状态轨迹的探针）。这些方法能检测到错误的"相关性"，但无法揭示推理**为何**出错——即无法深入到模型的计算过程层面理解失败的原因。

作者的核心假设是：模型内部实现了特定的"潜在算法电路"来完成推理任务，推理失败本质上是电路执行中的缺陷。通过构建归因图（类似软件调试中的执行追踪），可以从计算图的结构属性中检测到错误的可辨识信号。

## 方法详解

### 整体框架

CRV 是一个四阶段流水线：

1. **替换 MLP 为 Transcoder**：为模型每一层的 MLP 训练对应的 transcoder（稀疏过完备表示），用 TopK 激活函数强制稀疏，使内部计算在可解释的特征基上进行

2. **构建步级归因图**：对每个推理步骤 $s_i$，使用贪心路径查找算法从最终 logit 反向追踪高归因连接，得到稀疏有向图 $G_i = (\mathcal{V}, \mathcal{E})$，节点包括输入 token、transcoder 特征和输出 logit

3. **提取结构特征向量**：从归因图中提取固定维度的"结构指纹" $\mathbf{x}_i = \phi(G_i)$

4. **训练诊断分类器**：使用梯度提升分类器 (GBC) 预测推理步骤的正确性 $\hat{y}_i = f_\theta(\mathbf{x}_i)$

### 关键设计

1. **Transcoder 可解释化改造**

    功能：将目标模型每层的 MLP 替换为训练好的 transcoder，使前向传播通过稀疏、可解释的瓶颈层。

    核心思路：Transcoder 的训练目标是 $f(x) \approx \text{MLP}(x)$，即用稀疏过完备基来拟合 MLP 的输入-输出函数，而非自编码器式的自重构。输出的特征向量维度 $D \gg d$，但大部分为零，每个非零元素对应一个可解释概念。

    设计动机：标准 SAE 只是重构自身输入，而 transcoder 是 MLP 的功能替代品，能以可解释方式完成同等计算，为后续归因图分析提供语义基础。

2. **三层次结构指纹提取**

    功能：从修剪后的归因图（保留贡献前 80% 影响力的节点/边）中提取三个层次的特征。

    核心思路：
    - **全局图统计**：活跃特征节点数、logit 概率与熵——衡量计算复杂度和不确定性
    - **节点影响力统计**：激活值和影响力分数的均值/最大值/标准差，以及按层的活跃特征直方图——区分"少数高激活特征驱动"与"大量弱特征扩散"两种计算模式
    - **拓扑与路径特征**：图密度、度中心性、介数中心性、连通性——分析信息流结构

    设计动机：不同层次的特征互补，组合使用才能达到最优检测性能。消融实验证实节点统计最关键（移除后 FPR@95 上升 12 个百分点）。

3. **因果干预验证**

    功能：利用 CRV 发现的错误特征指导针对性的模型修复——抑制或放大特定 transcoder 特征以纠正推理错误。

    核心思路：当 CRV 检测到某推理步骤错误时，追溯到高重要性的 transcoder 特征（如"乘法"概念的特征），通过 forward hook 将其激活值钳制为零，从而改变模型的计算路径。

    设计动机：这一闭环验证了 CRV 发现的结构指纹与推理错误之间存在**因果关系**，而非仅仅是相关性，为可解释的模型调试开辟了新方向。

### 损失函数 / 训练策略

- Transcoder 使用 L2 重构损失 + TopK 激活函数训练
- 诊断分类器使用梯度提升分类器 (GBC)，直接在提取的表格化特征上训练
- 数据集构建：合成任务（布尔/算术）通过解析器自动标注；GSM8K 使用 Llama 3.3 70B Instruct 作为半自动评注器，并经人工审核

## 实验关键数据

### 主实验（表格）

| 方法 | 范式 | Boolean AUROC↑ | Arithmetic AUROC↑ | GSM8K AUROC↑ |
|------|------|----------------|-------------------|-------------|
| MaxProb | Black-box | 58.81 | 61.87 | 54.91 |
| Energy | Black-box | 51.08 | 76.45 | 62.55 |
| CoE-C | Gray-box | 51.03 | 69.39 | 53.57 |
| MLP Probe | Gray-box | 53.63 | 54.41 | 56.02 |
| **CRV (Ours)** | **White-box** | **75.87** | **92.47** | **70.17** |

CRV 在所有数据集上全面超越黑盒和灰盒基线。在算术任务上 AUROC 达 92.47，FPR@95 降至 37.09%（最强基线为 63.33%）。

### 消融实验（表格）

| 特征集 | Arithmetic AUROC↑ | Arithmetic FPR@95↓ |
|--------|-------------------|--------------------|
| CRV（全部三类） | 92.47 | 37.09 |
| w/o 全局统计 | 89.62 | 44.54 |
| w/o 节点统计 | 88.31 | 49.07 |
| w/o 拓扑统计 | 90.89 | 39.19 |

节点影响力统计是最关键的特征类别。

### 关键发现

- **错误指纹具有领域特异性**：不同推理任务（布尔逻辑 vs 算术 vs 自然语言数学）的错误在计算图上表现为不同的结构模式。单独在算术上训练的分类器迁移到 GSM8K 仅获得 57.04 AUROC。
- **联合训练可恢复性能**：用三个任务的联合数据训练的分类器在 GSM8K 上达到 70.62 AUROC，略超专用模型（70.17）。
- **因果干预成功**：在算术任务中，通过抑制一个"乘法"概念的 transcoder 特征（ID 91814），成功将错误的运算顺序（先乘后加）修正为正确顺序（先加后乘），答案从 105 修正为 147。

## 亮点与洞察

- 首次将归因图作为"推理执行追踪"用于自动化验证，在检测与理解之间架起桥梁
- 揭示了"计算完整性区域"的存在——正确推理占据了错误推理不可达的结构空间
- 因果干预的闭环设计——从检测到诊断到修复的完整链路——是传统探针方法做不到的

## 局限与展望

- 计算开销大：需要训练每层 transcoder + 构建归因图 + 训练分类器，不适合作为即插即用的验证器
- 仅在标准指令微调模型上验证，未测试搜索/回溯等高级推理模型（如 o1）
- 跨域泛化有限，需要为新任务收集标注数据重新训练分类器
- 实验模型仅为 Llama 3.1 8B Instruct，更大模型上的表现未知

## 相关工作与启发

- 与 PRM（过程奖励模型）互补：PRM 是黑盒训练的步骤级判别器，CRV 提供白盒可解释诊断
- 基于 transcoder 归因图技术 (Ameisen et al., 2025)，但从定性可视化推进到定量自动化验证
- 启发方向：可结合 CRV 的诊断能力与 PRM 的可扩展性，构建混合验证系统

## 评分

⭐⭐⭐⭐ 方法新颖度高，白盒归因图验证是全新视角，因果干预验证了因果性而非仅相关性，但计算开销限制了实用性。

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)
- [\[ICLR 2026\] Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)
- [\[ICLR 2026\] SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes](scenecot_eliciting_grounded_chain-of-thought_reasoning_in_3d_scenes.md)
- [\[ICLR 2026\] CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos](cot-rvs_zero-shot_chain-of-thought_reasoning_segmentation_for_videos.md)
- [\[AAAI 2026\] Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs](../../AAAI2026/llm_reasoning/graph_of_verification_structured_verification_of_llm_reasoning_with_directed_acy.md)

<!-- RELATED:END -->
