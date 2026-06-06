---
title: >-
  [论文解读] NADER: Neural Architecture Design via Multi-Agent Collaboration
description: >-
  [CVPR 2025][LLM评测][神经架构设计] NADER 将神经架构设计建模为多 LLM Agent 协作任务——Reader 读论文提炼知识、Proposer 生成改进方案、Modifier 用 DAG 图实现修改、Reflector 从失败中学习经验…
tags:
  - "CVPR 2025"
  - "LLM评测"
  - "神经架构设计"
  - "多Agent协作"
  - "LLM驱动NAS"
  - "DAG图表示"
  - "经验学习"
---

# NADER: Neural Architecture Design via Multi-Agent Collaboration

**会议**: CVPR 2025  
**arXiv**: [2412.19206](https://arxiv.org/abs/2412.19206)  
**代码**: 待确认  
**领域**: LLM评测  
**关键词**: 神经架构设计、多Agent协作、LLM驱动NAS、DAG图表示、经验学习

## 一句话总结
NADER 将神经架构设计建模为多 LLM Agent 协作任务——Reader 读论文提炼知识、Proposer 生成改进方案、Modifier 用 DAG 图实现修改、Reflector 从失败中学习经验，仅 10 次试验即突破 NAS-Bench-201 搜索空间的准确率上限，在 CIFAR-100 上达 74.51%（搜索空间最优 73.51%）。

## 研究背景与动机

**领域现状**：神经架构搜索（NAS）一直局限在预定义搜索空间中，近期有工作尝试用 LLM 生成代码来做架构设计（如 LeMo-NADe、GENIUS），但 LLM 生成的代码容易出错、不可执行，且效率低下。

**现有痛点**：(1) 传统 NAS 受限于搜索空间，无法发现空间外的更优架构；(2) LLM 直接生成代码问题多——代码冗余导致 token 浪费、注意力分散在实现细节上而非架构逻辑、生成的代码经常不可执行；(3) LLM 会反复犯同样的设计错误，缺乏从失败中学习的机制；(4) 现有 LLM NAS 方法（如 LeMo-NADe）效果差（ImageNet16-120 仅 31%，搜索空间最优 47%）。

**核心矛盾**：LLM 拥有丰富的架构知识（来自论文），但缺乏将知识转化为有效架构的可靠执行能力和迭代改进能力。

**本文目标** 如何让多个 LLM Agent 分工协作，系统性地设计出超越人工搜索空间的神经架构？

**切入角度**：模仿真实 AI 研究团队的工作流程——有人读论文（Reader）、有人提方案（Proposer）、有人写代码（Modifier）、有人做 review 和 debug（Reflector），各司其职、协同迭代。

**核心 idea**：用四个 LLM Agent 模拟 AI 研究团队——Reader 读论文提炼创新点，Proposer 生成改进方案，Modifier 用 DAG 图实现架构修改，Reflector 从历史失败/成功中学习经验指导后续设计。

## 方法详解

### 整体框架
分为研究团队（Reader + Proposer）和开发团队（Modifier + Reflector）。迭代流程：Reader 读论文→提炼知识入数据库 𝒦 → Proposer 从网络修改树中选候选+从 𝒦 检索相关建议→生成改进方案 → Modifier 用 DAG 图实现修改 → Reflector 验证并从经验数据库 ℰ 中检索经验辅助修复。训练、评估后更新修改树，循环迭代。

### 关键设计

1. **DAG 图表示（替代代码）**:

    - 功能：将神经网络架构表示为简洁的有向无环图，而非冗长代码
    - 核心思路：节点=操作（卷积、归一化等），边=信息流。LLM Agent 在图空间上操作——添加/删除/替换节点或边。图表示的优势：token 使用量减少 74-77%，可以做同构图检测避免重复训练，让 Agent 专注于架构逻辑而非代码实现细节。最终由 Graph-to-Code 转换模块将有效图翻译为可执行代码
    - 设计动机：消融实验显示图表示让设计质量（Quality）从 0.63→0.78（macro级），同时 token 从 2.23K→0.58K，证明去掉代码细节的干扰后 LLM 的架构设计能力大幅提升

2. **Reflector 的双重学习机制**:

    - 功能：从即时反馈和历史经验中学习，避免重复犯错
    - 核心思路：(a) Learn from Immediate Feedback (LIF)——用计算图流验证工具检查生成的架构是否合法，如果有错就把错误信息反馈给 Modifier 重试。(b) Learn from Design Experience (LDE)——维护设计经验数据库 ℰ，记录三类经验：失败记录、从失败到成功的修复记录、成功记录。每次新设计前检索 5 个最相关的历史经验给 Modifier 参考
    - 设计动机：LIF 将可执行率从 54%→64%（macro），LDE 将成功率从 0.62→0.88（micro），提升极其显著。经验积累让系统越用越好

3. **网络修改树 + 知识数据库**:

    - 功能：系统性管理架构搜索的探索-利用平衡和外部知识来源
    - 核心思路：修改树以基网络为根，每次迭代在树中选候选节点（DFS+BFS 结合，优先高性能节点），从论文知识库 𝒦 中检索相关的架构创新作为修改建议。Reader 自动下载评估论文、用 LLM 提取方法创新点存入 𝒦
    - 设计动机：树结构避免盲目搜索，知识数据库引入人类研究的最新进展。消融显示 Reader+Proposer 的组合在困难数据集（ImageNet16-120）上提升 1.58%

### 损失函数 / 训练策略
架构设计过程不涉及 loss 训练——全部通过 LLM prompt 驱动。发现的架构在标准设置下训练评估（遵循 NAS-Bench-201 的训练/验证/测试划分）。约束条件：FLOPs ≤0.2G（CIFAR）/ ≤0.05G（ImageNet16-120），参数 ≤1.5M。每个架构的 LLM 调用成本约 $0.046。

## 实验关键数据

### 主实验

| 方法 | 试验数 | CIFAR-10 Test | CIFAR-100 Test | ImageNet16-120 Test |
|--------|------|------|----------|------|
| 搜索空间最优 | - | 94.37 | 73.51 | 47.31 |
| LeMo-NADe (GPT-4) | 30 | 89.41 | 67.90 | 27.70 |
| GENIUS | 10 | 93.79 | 70.91 | 44.96 |
| LLMatic | 2000 | 94.26 | 71.62 | 45.87 |
| **NADER (Random, 10)** | **10** | **94.40** | **74.51** | **49.63** |
| **NADER (ResNet, 500)** | **500** | **94.62** | **76.00** | **50.52** |

### 消融实验（NAD Benchmark, micro-level）

| 配置 | Token (K) | 可执行率 | 质量 | 成功率 |
|------|---------|---------|---------|---------|
| Baseline (纯代码) | 2.10 | 0.76 | 0.65 | 0.49 |
| + Graph 表示 | 0.49 | 0.62 | 0.97 | 0.60 |
| + LIF | 0.48 | 0.70 | 0.89 | 0.62 |
| + LDE | **0.31** | **0.92** | **0.96** | **0.88** |

### 关键发现
- **突破搜索空间上限**：仅 10 次试验，CIFAR-100 达 74.51% 超过搜索空间最优 73.51%（+1%），ImageNet16-120 达 49.63% 超过 47.31%（+2.32%）。500 次试验时差距更大（+2.49%/+3.21%）
- **图表示是基础**：token 减少 77%（2.10K→0.49K），质量从 0.65→0.97，证明代码细节严重干扰了 LLM 的架构设计
- **LDE 经验学习效果最显著**：micro-level 成功率翻倍（0.49→0.88），可执行率从 0.70→0.92。历史经验的积累是越用越好的关键
- **Reader 和 Proposer 缺一不可**：单独用 Reader 或 Proposer 都不如两者结合（尤其 ImageNet16-120 差 0.8%），说明论文知识和搜索策略需要配合

## 亮点与洞察
- **"AI 研究团队"的 Agent 协作范式**：把架构设计链路分解为读论文→提方案→实现→反思四个角色的Agent系统，模拟了真实研究工作流。这种模式可以推广到其他需要创造性设计的工程问题
- **DAG 图替代代码是关键洞察**：让 LLM 在更抽象的图空间而非代码空间操作，大幅提升效率和质量。这对所有"LLM 写代码解决结构化问题"的工作都有启示——也许该让 LLM 在更高层级的表示上操作
- **经验数据库的可积累性**：设计经验越积越多，系统性能持续提升，有类似持续学习的特性。这比单次 prompt 的 LLM 应用有质的飞跃

## 局限与展望
- 为了公平比较，实验仍受 FLOPs/参数量约束，限制了架构创新的自由度
- 仅在图像分类（CIFAR-10/100、ImageNet16-120）上验证，未拓展到检测、分割等任务
- 依赖 GPT-4 的能力，Agent 质量受 LLM 能力上限影响
- Reader 读论文的范围和质量取决于检索到的论文列表，可能遗漏新兴方向
- 每个架构仍需实际训练评估（虽然代价比传统 NAS 小），完全无训练的架构质量预估尚未实现

## 相关工作与启发
- **vs GENIUS**: GENIUS 也用 LLM 做 NAS，但只做搜索空间内的搜索；NADER 突破搜索空间限制，10 次试验 ImageNet16-120 上超 GENIUS 4.67%
- **vs LeMo-NADe**: LeMo-NADe 用 GPT-4/Gemini 直接生成代码，效果很差（ImageNet16-120 仅 27.70%/31.02%）；NADER 的图表示 + 多Agent协作显著更强
- **vs LLMatic**: LLMatic 需要 2000 次试验才接近搜索空间最优；NADER 10 次试验就超越，效率高 200 倍

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 多Agent协作做架构设计的范式非常新，图表示+经验学习的设计巧妙
- 实验充分度: ⭐⭐⭐⭐ NAS-Bench-201 对比充分，消融详细，但数据集范围有限
- 写作质量: ⭐⭐⭐⭐ 框架和Agent角色描述清晰
- 价值: ⭐⭐⭐⭐ 首次用多Agent系统突破搜索空间限制，为 AI4Science 和 AutoML 提供新思路

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] GLGENN: A Novel Parameter-Light Equivariant Neural Networks Architecture Based on Clifford Geometric Algebras](../../ICML2025/llm_evaluation/glgenn_a_novel_parameter-light_equivariant_neural_networks_architecture_based_on.md)
- [\[CVPR 2025\] TraF-Align: Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception](traf-align_trajectory-aware_feature_alignment_for_asynchronous_multi-agent_perce.md)
- [\[NeurIPS 2025\] Thought Communication in Multiagent Collaboration](../../NeurIPS2025/llm_evaluation/thought_communication_in_multiagent_collaboration.md)
- [\[AAAI 2026\] MAPS: Multi-Agent Personality Shaping for Collaborative Reasoning](../../AAAI2026/llm_evaluation/maps_multi-agent_personality_shaping_for_collaborative_reaso.md)
- [\[NeurIPS 2025\] Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks](../../NeurIPS2025/llm_evaluation/belief-calibrated_multi-agent_consensus_seeking_for_complex_nlp_tasks.md)

</div>

<!-- RELATED:END -->
