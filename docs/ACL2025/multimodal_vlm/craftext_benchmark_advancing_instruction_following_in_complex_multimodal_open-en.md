---
title: >-
  [论文解读] CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World
description: >-
  [ACL 2025][多模态][指令跟随] 提出 CrafText，一个基于 Craftax 开放世界环境的多模态指令跟随基准，包含 3,924 条指令和 3,423 个独特词汇，覆盖定位、条件、建造和成就四类任务，并设计双重评估协议测试智能体的语言泛化和目标泛化能力。
tags:
  - ACL 2025
  - 多模态
  - 指令跟随
  - 多模态基准
  - 强化学习
  - 开放世界
  - 语言接地
---

# CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World

**会议**: ACL 2025  
**arXiv**: [2505.11962](https://arxiv.org/abs/2505.11962)  
**代码**: [GitHub](https://anonymous.4open.science/r/CrafText-D217/)  
**领域**: 多模态VLM  
**关键词**: 指令跟随, 多模态基准, 强化学习, 开放世界, 语言接地

## 一句话总结

提出 CrafText，一个基于 Craftax 开放世界环境的多模态指令跟随基准，包含 3,924 条指令和 3,423 个独特词汇，覆盖定位、条件、建造和成就四类任务，并设计双重评估协议测试智能体的语言泛化和目标泛化能力。

## 研究背景与动机

现实世界中的指令跟随面临两大核心挑战：（1）在动态变化的环境中进行决策——环境是不可预测的，状态会独立于智能体的行为而演变；（2）在多样化的任务和指令表述中进行泛化——智能体需要正确解释各种措辞的指令并将其与观察关联。

现有基准存在显著不足：
- 大多数环境是静态的（如 Alfred、Touchdown），缺乏环境动态性
- 指令通常通过模板程序化生成，词汇量有限（如 BabyAI、HomeGrid）
- 即使词汇丰富的环境（如 Alfred）也缺乏多样的物体交互
- 没有环境同时提供"语言泛化"和"目标泛化"的双重评估协议

CrafText 旨在填补这一空白，构建一个同时具备环境动态性、语言多样性、丰富交互和双重评估的综合基准。

## 方法详解

### 整体框架

CrafText 基于 Craftax（一个类 Minecraft 的开放世界 RL 环境）构建，扩展了自然语言指令接口。整体框架包括三部分：数据集设计、指令生成管线、环境扩展。

### 关键设计

1. **层次化数据集结构**：采用"场景（Scenario）→ 目标（Goal）→ 指令（Instruction）"的三层结构。场景定义抽象任务类（如"建造正方形"），目标参数化为具体实例（如"建造 2×2 木质正方形"），指令是目标的多种自然语言表述（每个目标约 5-6 种表述）。

2. **四大任务类别**：

    - **建造（Building）**：要求智能体构建指定结构，需记住起点并可能需要离开收集额外资源
    - **条件（Conditional）**：测试指令理解，如"采集两块石头后制造剑"vs"制造剑之前采集两块石头"
    - **定位（Localization）**：评估空间指令理解，包括罗盘方向（南、东、西、北）和相对方向（右边、上方）
    - **成就（Achievement）**：执行游戏内标准任务及其组合，如收集木材、开采钻石

3. **三级难度分层**：基于完成任务所需的前置动作序列长度：

    - Easy：成就类场景，完成游戏内成就及其组合
    - Medium：所有场景类型，但动作序列较短（<10步）
    - Hard：复杂目标或长动作序列

4. **指令生成管线（Instruction Generation Pipeline）**：结合程序化目标生成与 GPT-4 语言生成。首先由专家定义场景检查函数和参数范围，枚举组合生成大量目标模板；然后使用 GPT-4 为每个目标生成多样化的自然语言指令和释义，确保语言复杂性和多样性。

5. **双重评估协议**：

    - **Paraphrased 测试集**：与训练集相同的目标，但指令被重新措辞，测试语言泛化能力
    - **New Objects 测试集**：引入训练中未见过的物体组合（但所有物体在训练中都出现过），测试目标级别的泛化能力

6. **JAX 加速环境**：所有检查函数用 JAX 实现，支持 JIT 编译和 GPU 加速，实现高度并行化的大规模训练。

### 奖励系统

- 完成指令获得奖励 1
- Craftax 环境提供的成就发现奖励，缩放比例为 1/50
- 每一步运行对应的场景检查函数验证完成状态

## 实验关键数据

### 主实验（Medium 任务，50 seeds）

| 算法 | 条件 | 建造 | 定位 | 成就 | 总计 |
|------|------|------|------|------|------|
| PPO-T | 0.15 | 0.25 | 0.33 | 0.55 | 0.40 |
| PPO-T+ | 0.17 | 0.24 | 0.30 | 0.70 | **0.45** |
| Dynalang | 0.00 | 0.12 | 0.15 | 0.17 | 0.15 |
| FiLM | 0.07 | 0.38 | 0.29 | 0.76 | 0.43 |

### 泛化实验

| 测试集 | PPO-T | PPO-T+ | Dynalang | FiLM |
|--------|-------|--------|----------|------|
| Train | 0.40 | 0.45 | 0.15 | 0.43 |
| Paraphrased | 0.36 | 0.35 | 0.05 | 0.35 |
| New Objects | 0.22 | **0.28** | 0.10 | 0.26 |

### 关键发现

- **Dynalang 表现远低于预期**：尽管在 Crafter 环境有优异表现，但在 CrafText 中仅达到 0.15 的成功率，说明复杂语言指令+动态环境的组合极大增加了学习难度
- **所有方法成功率都很低**：最好的 PPO-T+ 也只有 0.45，说明 CrafText 确实具有高难度
- **释义导致显著性能下降**：PPO-T+ 从 0.45 下降到 0.35，说明现有方法对语言变化的鲁棒性不足
- **PPO-T+（带规划）在新目标泛化上表现最佳**：成功率 0.28，表明将指令分解为结构化计划有助于目标级泛化
- **FiLM 在建造任务上表现最佳**（0.38），其特征级调制机制在处理视觉-语言交互上更灵活
- **条件任务对所有方法都极其困难**：最高仅 0.17-0.20，说明条件逻辑推理是当前方法的强对立面

## 亮点与洞察

- **全面性**：同时满足环境动态性、语言多样性、丰富交互、GPU 加速和双重评估协议，是对比表中唯一全部满足的基准
- **揭示问题本质**：实验清楚表明静态环境中表现良好的方法（Dynalang）在动态+复杂语言条件下彻底失效
- **JAX 实现**：支持大规模并行训练，解决了 RL 训练效率的实际瓶颈
- **规划增强的价值**：PPO-T+ 的 GPT-4 规划步骤虽简单但有效，暗示未来方向应更多利用 LLM 进行任务分解

## 局限与展望

- 数据集全部使用 AI 生成的指令，缺乏人类编写的指令，可能无法完全捕捉人类语言的细微差别
- 缺乏真实世界的交互元素，如指令协商、澄清和动态对话
- 当前基线方法的成功率整体偏低，需要更强的方法来验证基准的区分度
- 虽然基于 Craftax，但仍是 2D 像素环境，与真实 3D 世界存在差距
- 仅使用 DistilBERT 和 T5 的语言编码，未探索更强大的语言模型（如使用 VLM 本身作为策略网络）

## 相关工作与启发

- 与 BabyAI、HomeGrid 等模板指令环境相比，CrafText 提供了更丰富的词汇和语言复杂性
- 与 MineDojo 相比，CrafText 提供了精确的目标验证函数和双重评估协议
- 启发：结合 LLM 进行指令预处理/规划（如 PPO-T+）是提升指令跟随能力的有前景方向
- 环境动态性对指令跟随的挑战尚未被充分研究，这是一个重要的开放问题

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个同时满足多项关键属性的指令跟随基准，双重评估协议设计新颖
- 实验充分度: ⭐⭐⭐ 基线方法有限（仅4种），缺少 VLM-based 方法和更多 RL 算法的对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，对比表完整，但部分环境描述可更精炼
- 价值: ⭐⭐⭐⭐ 填补了动态环境+复杂语言指令跟随基准的空白，对 RL+NLP 社区有重要价值

<!-- RELATED:START -->

## 相关论文

- [MM-IFEngine: Towards Multimodal Instruction Following](../../ICCV2025/multimodal_vlm/mm-ifengine_towards_multimodal_instruction_following.md)
- [OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation](../../CVPR2025/multimodal_vlm/opening_a_comprehensive_benchmark_for_judging_open-ended_interleaved_image-text_.md)
- [REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark](real-mm-rag_a_real-world_multi-modal_retrieval_benchmark.md)
- [Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents](agent_rewardbench.md)
- [MMBoundary: Advancing MLLM Knowledge Boundary Awareness through Reasoning Step Confidence Calibration](mmboundary_reasoning_step_confidence.md)

<!-- RELATED:END -->
