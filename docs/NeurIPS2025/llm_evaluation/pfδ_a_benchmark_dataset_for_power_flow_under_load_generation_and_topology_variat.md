---
title: >-
  [论文解读] PFΔ: A Benchmark Dataset for Power Flow under Load, Generation, and Topology Variations
description: >-
  [NeurIPS 2025][LLM评测][电力潮流] PFΔ 是首个同时涵盖负荷、发电机出力和拓扑变化的电力潮流基准数据集，包含 859,800 个求解实例、六种电网规模和接近不可行的极端工况，并提出标准化评估任务来系统评测 ML 方法在多种运行条件下的表现。
tags:
  - "NeurIPS 2025"
  - "LLM评测"
  - "电力潮流"
  - "基准数据集"
  - "图神经网络"
  - "拓扑扰动"
  - "电网仿真"
---

# PFΔ: A Benchmark Dataset for Power Flow under Load, Generation, and Topology Variations

**会议**: NeurIPS 2025  
**arXiv**: [2510.22048](https://arxiv.org/abs/2510.22048)  
**代码**: [GitHub](https://github.com/MOSSLab-MIT/pfdelta)  
**领域**: 电力系统 / 图神经网络基准  
**关键词**: 电力潮流, 基准数据集, 图神经网络, 拓扑扰动, 电网仿真

## 一句话总结

PFΔ 是首个同时涵盖负荷、发电机出力和拓扑变化的电力潮流基准数据集，包含 859,800 个求解实例、六种电网规模和接近不可行的极端工况，并提出标准化评估任务来系统评测 ML 方法在多种运行条件下的表现。

## 研究背景与动机

电力潮流（Power Flow, PF）计算是电网实时运行的核心基础，几乎所有关键决策流程都依赖它：
- **N-k 安全分析**：需要在每 5 分钟的操作窗口内对数千种故障场景求解 PF，评估电网安全性
- **拓扑优化**：在组合爆炸的动作空间中搜索最优拓扑（例如 118 母线系统的单个变电站可有 65,000 种配置），每步都需 PF 求解
- **极端天气应对**：气候变化导致更频繁的设备故障，需要更大规模、更快速的 PF 仿真

传统 Newton-Raphson (NR) 求解器虽精度高但计算代价大，难以满足实时大规模需求。ML 方法（尤其是 GNN）作为快速近似器具有潜力，但当前研究面临两大瓶颈：

**缺乏标准化基准**：不同论文使用不同的数据生成和评估方式，难以公平对比

**现有数据集变化维度不足**：大多只涵盖负荷变化的子集，未能同时捕捉发电机出力多样性和拓扑扰动，远离真实电网面临的复杂运行条件

与已有数据集（如 OPFData 仅考虑 N-1 拓扑 + 负荷扰动，OPF-Learn 仅关注负荷多样性）不同，PFΔ 首次在单一基准中集成了负荷、发电机、拓扑三个维度的扰动，并引入了接近不可行的极端工况。

## 方法详解

### 整体框架

PFΔ 的核心贡献包括三部分：（1）多维度扰动的数据生成流程；（2）包含极端工况的数据集构建；（3）涵盖四大类场景的标准化评估任务体系。

### 关键设计

1. **三维扰动数据生成**：对每个数据样本同时引入三种扰动：

    - **负荷扰动**：采用 OPF-Learn 的凸集采样方法，从包含 ACOPF 可行空间的凸集中均匀采样负荷配置。当采到不可行点时缩小凸集。相比简单的 ±20% 均匀采样，该方法能产生更丰富多样的负荷分布
    - **拓扑扰动**：模拟 N-1 和 N-2 故障场景，等概率执行以下操作之一：移除最多 2 个发电机、移除最多 2 条线路、各移除 1 个、或保持原拓扑不变
    - **发电机出力扰动**：通过随机置换发电机成本参数来产生不同的有功功率和电压设定值，经修改版 ACOPF 求解得到多样的发电机出力
   
   对扰动后的输入，使用 PowerModels.jl + Ipopt 求解修改版 ACOPF（移除了输出变量的约束使其等价于 PF 方程），仅保留收敛的可行解。

2. **接近不可行工况（Close-to-Infeasible, C2I）生成**：通过逐步增加功率注入/抽取并重复求解 PF，找到稳态电压稳定极限处的负荷能力边界。在此边界点，PF 雅可比矩阵变为奇异，传统求解器常无法收敛。同时生成"趋近不可行"样本作为数据增强。这种极端工况对测试 ML 模型的鲁棒性至关重要。

3. **标准化评估体系**：设计四组共 12 个评估任务：

    - **组 1（分布内泛化）**：训练不同拓扑组合（N / N+N-1 / N+N-1+N-2），测试是否能泛化到未见拓扑
    - **组 2（数据效率）**：从 54,000 逐步减少到 18,000 训练样本
    - **组 3（分布外泛化）**：在一种电网规模上训练，测试不同规模（如 118 训练 → 57/500 测试）
    - **组 4（极端工况）**：加入 C2I 样本训练，评估对极端场景的适应性
   
   核心评估指标为无监督的功率平衡失配 $|\Delta S_i| = \sqrt{(\Delta p_i)^2 + (\Delta q_i)^2}$，避免了多解问题带来的求解器偏差。

### 损失函数 / 训练策略

评测的三种 GNN 模型各有不同训练策略：
- **CANOS-PF**（改编自 ACOPF 模型）：交互网络 + L2 + 约束违反损失，解析计算支路潮流
- **PowerFlowNet**：母线类型嵌入 + TAGConv 消息传递，L2 监督损失
- **GNS-S**（自监督）：直接最小化功率平衡方程，迭代更新电压，无需标签

## 实验关键数据

### 主实验

以 IEEE 118 母线系统为主，训练 Task 1.3（包含 N+N-1+N-2 拓扑），评估多种场景。

| 模型 | 可行样本 Mean PBL | 可行样本 Max PBL | C2I Mean PBL | 运行时/NR对比 |
|------|------------------|-----------------|-------------|-------------|
| Newton-Raphson | ~$10^{-6}$ | ~$10^{-5}$ | 常不收敛 | 1× (基准) |
| CANOS-PF | 最低（总体最佳） | 最低 | 中等 | ~5× 加速 |
| PowerFlowNet | 中等 | 中等 | 最差 | ~5× 加速 |
| GNS-S | 高方差 | 高方差 | **最低** | ~5× 加速 |

分布外泛化（118→57/500 测试）:

| 测试规模 | CANOS-PF | PowerFlowNet | GNS-S | NR 收敛率 |
|---------|---------|-------------|-------|----------|
| 57-bus | 中等误差 | 较大误差 | 中等误差 | 95.2% |
| 118-bus | 最佳 | 中等 | 高方差 | 65.7% |
| 500-bus | 均较大误差 | 均较大误差 | 均较大误差 | 43.4% |

### 消融实验

训练拓扑对泛化的影响（Task 1.1 vs 1.2 vs 1.3）：

| 训练拓扑 | N-1 测试表现 | N-2 测试表现 | 关键发现 |
|---------|------------|------------|---------|
| 仅 N | 所有模型在 N-1/N-2 上急剧退化 | 更差 | CANOS-PF 退化最严重（因解析公式放大误差） |
| N + N-1 | 显著改善 | 也有提升 | N-1 训练数据对泛化至关重要 |
| N + N-1 + N-2 | 最佳 | 最佳 | 拓扑多样性比数据量更重要 |

数据效率实验（Task 2.1–2.3, 18k/36k/54k 样本）：

| 模型 | 低数据表现 | 高数据表现 | 说明 |
|------|----------|----------|------|
| CANOS-PF | 基本稳定 | 基本稳定 | 数据量影响小 |
| PowerFlowNet | 基本稳定 | 基本稳定 | 类似 |
| GNS-S | 退化至仅 N 训练水平 | 正常 | 低数据下对拓扑多样性更敏感 |

### 关键发现

- **所有 GNN 模型都无法达到 NR 求解器 $10^{-6}$ 量级的精度**，即便包含物理信息组件
- 拓扑多样性对模型泛化能力的影响远大于数据量
- 跨规模泛化是重大挑战：所有模型在不同于训练规模的电网上表现大幅下降
- GNS-S 的自监督物理损失使其在 C2I 极端工况下表现最好
- NR 求解器在大规模电网上收敛率急剧下降（2000 母线完全不收敛），GNN 可填补此空白
- GNN 方法实现约 5× 的推理加速

## 亮点与洞察

- **全面性**：首次在负荷、发电机出力、拓扑三个维度同时构建基准，更贴近真实电网运行
- **C2I 设计**：极端工况的引入是亮点——这是传统求解器的薄弱环节，恰好是 ML 可以发挥优势的方向
- **无监督指标**：采用功率平衡失配作为评估指标，回避了多解问题带来的偏差
- 标准化的任务框架使得未来不同方法的公平对比成为可能

## 局限与展望

- GOC-2000 规模数据量仅为其他规模的一半，大规模电网的数据生成仍受计算成本限制
- 未评估模型识别不可行工况的能力（即判断 PF 无解的能力）
- 未覆盖 N-k（k>2）的更高阶故障场景
- 负荷采样方法在大规模网络中难以准确表示完整的可行空间
- 所有模型距实际部署所需的精度仍有差距

## 相关工作与启发

- 与 OPFData、OPF-Learn 形成互补：PFΔ 专注于 PF 而非 ACOPF，变化维度更丰富
- PowerGraph 仅支持 118 母线且无拓扑扰动，PFΔ 在规模和多样性上全面超越
- 物理信息组件（如 GNS-S 的功率平衡损失）对极端工况至关重要，提示未来架构设计应更深度融合物理约束

## 评分

- **新颖性**: ⭐⭐⭐⭐ 基准数据集论文，关键创新在 C2I 生成和多维扰动设计
- **实验充分度**: ⭐⭐⭐⭐⭐ 12 个标准化任务，3 种 GNN + NR 对比，分析非常全面
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，电力系统背景介绍详尽，对非领域读者友好
- **价值**: ⭐⭐⭐⭐ 填补了电力潮流 ML 基准的空白，对电力系统 AI 社区有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] StructFlowBench: A Structured Flow Benchmark for Multi-turn Instruction Following](../../ACL2025/llm_evaluation/structflowbench_a_structured_flow_benchmark_for_multi-turn_instruction_following.md)
- [\[NeurIPS 2025\] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)
- [\[NeurIPS 2025\] CodeAssistBench (CAB): Dataset & Benchmarking for Multi-turn Chat-Based Code Assistance](codeassistbench_cab_dataset_benchmarking_for_multi-turn_chat-based_code_assistan.md)
- [\[ACL 2025\] NorEval: A Norwegian Language Understanding and Generation Evaluation Benchmark](../../ACL2025/llm_evaluation/noreval_a_norwegian_language_understanding_and_generation_evaluation_benchmark.md)
- [\[NeurIPS 2025\] OptiTree: Hierarchical Thoughts Generation with Tree Search for LLM Optimization Modeling](optitree_hierarchical_thoughts_generation_with_tree_search_for_llm_optimization_.md)

</div>

<!-- RELATED:END -->
