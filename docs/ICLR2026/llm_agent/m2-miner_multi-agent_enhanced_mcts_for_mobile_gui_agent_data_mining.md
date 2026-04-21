---
title: >-
  [论文解读] M²-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining
description: >-
  [ICLR 2026][LLM Agent][GUI Agent] 提出 M²-Miner，首个基于 MCTS 的移动端 GUI agent 自动数据挖掘框架，通过 InferAgent/OrchestraAgent/JudgeAgent 三智能体协作将挖掘效率提升 64 倍，结合 intent recycling 策略丰富意图多样性，训练的 GUI agent 在多个 benchmark 上达到 SOTA。
tags:
  - ICLR 2026
  - LLM Agent
  - GUI Agent
  - MCTS
  - 数据挖掘
  - 多智能体协作
  - 移动端交互
---

# M²-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining

**会议**: ICLR 2026  
**arXiv**: [2602.05429](https://arxiv.org/abs/2602.05429)  
**代码**: 即将开源 (有)  
**领域**: LLM Agent  
**关键词**: GUI Agent, MCTS, 数据挖掘, 多智能体协作, 移动端交互

## 一句话总结
提出 M²-Miner，首个基于 MCTS 的移动端 GUI agent 自动数据挖掘框架，通过 InferAgent/OrchestraAgent/JudgeAgent 三智能体协作将挖掘效率提升 64 倍，结合 intent recycling 策略丰富意图多样性，训练的 GUI agent 在多个 benchmark 上达到 SOTA。

## 研究背景与动机
**领域现状**：GUI agent 通过理解用户意图并在图形界面上执行动作序列来自动化操作软件应用，是学界和业界的热门方向。当前 GUI agent 的核心依赖是高质量的 intent-trajectory（意图-轨迹）训练数据。

**现有痛点**：
   - **高成本**：人工标注（如 AITW、AndroidControl）每条数据需要数小时，成本高达 $0.36/image
   - **低质量**：人工标注和自动挖掘的数据常含冗余步骤、模糊意图描述、偏差操作路径
   - **低多样性**：现有数据集采用 intent-to-flat-trajectory 结构，每个意图仅记录单一成功路径，意图类型单调

**核心矛盾**：手动标注质量可控但无法规模化，现有自动挖掘方法（如 AgentQ 基于原生 MCTS、OS-Genesis 基于规则探索）效率低、只适用于 web 环境或仅产生单一轨迹。

**本文目标**：如何低成本、自动化地挖掘高质量、高多样性的移动端 GUI 交互轨迹数据？

**切入角度**：将 MCTS 引入移动端 GUI 数据挖掘，但原生 MCTS 随机扩展效率极低。作者观察到：(a) 扩展阶段需要智能引导而非随机探索；(b) 模拟阶段可以用过程奖励替代 rollout；(c) 搜索树中非主路径蕴含额外有价值的意图-轨迹对。

**核心idea**：MCTS + 三智能体协作（引导扩展 + 加速排序 + 过程评估）+ intent recycling = 高效、高质量、高多样性的 GUI 数据挖掘。

## 方法详解

### 整体框架
M²-Miner 以 MCTS 为骨架，输入为初始意图 $I_0$ 和起始 GUI 状态 $s_0$，输出为包含有效交互轨迹 $\tau=(s_0,a_0,s_1,\ldots)$ 的意图-轨迹树 $\mathcal{T}=(\mathcal{V},\mathcal{A},\mathcal{P},\mathcal{I})$。树中每个节点包含截图、动作描述、Q值、访问次数和任务完成状态。MCTS 的四个阶段（选择→扩展→模拟→回溯）循环执行直到挖掘到匹配意图的轨迹。

### 关键设计

1. **InferAgent（扩展阶段 - 动作生成）**:

    - 功能：为选中节点生成 $K$ 个候选动作
    - 核心思路：基于当前 GUI 截图和目标意图推理最可能正确的动作。使用多个不同 MLLM 生成以确保动作空间多样性，同时将已生成的动作加入 prompt 避免重复
    - 设计动机：替代原生 MCTS 的随机扩展，大幅提高正确动作的命中率

2. **OrchestraAgent（扩展阶段 - 动作排序与去重）**:

    - 功能：合并语义等价的动作（如点击同一按钮的不同坐标），按达成目标意图的可能性排序
    - 核心思路：通过多选题方式（multi-choice question）让 MLLM 在每次迭代中选出最有希望的动作，经 $K-1$ 次查询得到排序队列。排序后的动作按顺序赋予递减的初始 UCT 值
    - 设计动机：避免冗余扩展，确保搜索优先探索最有前景的分支

3. **JudgeAgent（模拟阶段 - 过程奖励估计）**:

    - 功能：分析新扩展节点的 GUI 截图，判断任务完成状态并计算奖励
    - 核心思路：终端节点（成功/失败）奖励为 1/0。中间节点使用 MLLM head 输出 "valid"/"invalid" 的 logits，通过 softmax 归一化为 $[0,1]$ 的概率作为奖励：$r_{\text{intermediate}} = \frac{\exp(logits_{\text{valid}})}{\exp(logits_{\text{valid}}) + \exp(logits_{\text{invalid}})}$
    - 设计动机：替代原生 MCTS 需要完整 rollout 才能评估奖励的方式，节点 Q 值更新为：$Q_i = \frac{Q_{i-1} \times N_{i-1} + R_i}{N_{i-1}+1}$，极大减少模拟阶段的计算开销

4. **Intent Recycling 策略**:

    - 功能：从完成挖掘的轨迹树中提取非主路径的额外意图-轨迹对
    - 核心思路：遍历树中从根到每个节点的路径，用 intent recycling filter（MLLM 实现）评估路径质量，对通过的路径用 MLLM 生成新意图，再由 JudgeAgent 验证意图与轨迹的一致性
    - 设计动机：将原来"一棵树一个意图"的结构进化为"一棵树多个意图"，不需重新挖掘即可获得更多样的数据。例如，挖掘"查询路线"意图时，误点"打车"按钮可能产生了一条有效的打车轨迹

5. **Progressive Model-in-the-loop 训练策略**:

    - 功能：迭代提升 InferAgent 和 JudgeAgent 的能力
    - 核心思路：三阶段渐进训练——Stage 1 基础意图（常用服务 + 条件改写）→ Stage 2 复杂意图（功能组合 + 失败意图重试）→ Stage 3 回收意图（对历史树执行 intent recycling）。每阶段挖掘的数据用于持续训练两个 agent
    - 设计动机：形成正反馈循环，agent 能力和数据复杂度同步增长

### 损失函数 / 训练策略
- InferAgent 和 JudgeAgent 基于 Qwen2.5-VL-7B 微调
- OrchestraAgent 和 intent recycling filter 使用 Qwen2.5-VL-72B
- 训练数据还包含描述信息（description）和偏好数据（preference data，从正/负路径构建）

## 实验关键数据

### 主实验

| 模型 | AC-Low TP/SR | AC-High TP/SR | AITZ TP/SR | GUI-Odyssey TP/SR | CAGUI TP/SR |
|------|-------------|--------------|-----------|-------------------|-------------|
| GPT-4o | 74.3/19.4 | 66.3/20.8 | 70.0/35.3 | - | 3.67/3.67 |
| UI-TARS-7B* | 98.0/90.8 | 83.7/72.5 | 80.4/65.8 | 90.1/87.0 | 88.6/70.0 |
| OS-Genesis-7B | 90.7/74.2 | 66.2/44.5 | 20.0/8.5 | 11.7/3.6 | 38.1/14.5 |
| GUI-Owl-7B | 93.8/90.0 | 81.5/72.8 | 78.9/65.1 | 83.4/60.7 | 80.0/59.2 |
| **M²-Miner-7B** | **97.5/93.5** | 81.8/**72.9** | **81.3/69.4** | **90.5/79.3** | **88.8/70.2** |

*UI-TARS-7B 使用大规模私有人工标注数据

### 消融实验

| 配置 | TP | SR | 说明 |
|------|----|----|------|
| Warm-up | 85.0 | 64.2 | 仅公开数据预训练 |
| + Stage 1 (基础意图) | 86.5 | 67.3 | +3.1% SR |
| + Stage 2 (复杂意图) | 87.6 | 69.1 | +1.8% SR |
| + Stage 3 (回收意图) | 88.2 | 69.9 | +0.8% SR，累计 +5.7% |
| Act only | 85.2 | 66.8 | 仅动作标签 |
| Act + Description | 88.2 | 69.9 | +3.1% SR |
| Act + Des + Preference | **88.8** | **70.2** | +3.4% SR |

### 关键发现
- **效率提升指数级增长**：与原生 MCTS 相比，M²-Miner 在任务长度为 9 时效率提升 64 倍。OrchestraAgent 在扩展阶段减少冗余节点，JudgeAgent 在模拟阶段省去 rollout
- **成本降低 18 倍**：M²-Miner-Agent 数据集每张图片成本仅 $0.02，而人工标注数据集为 $0.36
- **数据质量更高**：随机抽取 100 条数据人工检查，M²-Miner 的数据质量准确率（DQA）高于人工标注数据集（AC 和 AITZ）
- **描述和偏好数据有用**：相比仅使用动作标签，加入描述信息提升 SR 3.1%，再加偏好数据又提升 0.3%
- **对未见场景泛化良好**：在无训练数据的 CAGUI 上，Qwen2.5-VL-7B 经 M²-Miner 数据训练后 SR 从 55.2% 提升至 70.2%

## 亮点与洞察
- **MCTS + 多智能体的范式非常巧妙**：三个 agent 各司其职——生成、排序、评估——精确地解决了原生 MCTS 在 GUI 领域的三个瓶颈（随机扩展、冗余节点、昂贵 rollout）
- **Intent Recycling 是极具创意的设计**：将"失败"的探索路径转化为额外数据源，一举解决了意图多样性和效率两个问题。这个思路可以迁移到其他 agent 数据收集场景
- **过程奖励替代 rollout**：用 MLLM 的 logits 概率作为中间奖励，既保留了 MCTS 的理论优势，又避免了完整模拟的高成本。这种设计对所有 MCTS+LLM 的系统都有参考价值

## 局限与展望
- 当前仅在移动端验证，桌面端和 web 端的适用性未探索
- OrchestraAgent 使用 72B 模型（Qwen2.5-VL-72B），部署成本较高
- Intent recycling 的过滤和意图生成质量依赖 MLLM 能力，对于复杂应用可能效果下降
- 数据规模（20k 图片、2565 轨迹）相比 UI-TARS 的私有数据仍有差距
- Model-in-the-loop 的三阶段训练需要多轮挖掘→训练循环，实际工程成本未详细报告

## 相关工作与启发
- **vs AgentQ**：AgentQ 首创 MCTS 进行 web 环境的数据挖掘，但局限于可解析的 HTML 环境，且效率低。M²-Miner 扩展到视觉驱动的移动端，效率通过多智能体提升数个数量级
- **vs OS-Genesis**：OS-Genesis 采用无监督的规则交互+逆向任务合成，无需预定义任务但数据质量不高（AITZ 上 SR 仅 8.5%）。M²-Miner 的 MCTS 结构化搜索产生更高质量的轨迹
- **vs UI-TARS**：UI-TARS 使用大规模私有人工数据，M²-Miner 在几乎所有 SR 指标上超越它，说明自动挖掘框架的潜力已超过人工标注

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ MCTS+多智能体+intent recycling 的组合在 GUI 数据挖掘领域是首创，每个组件都有明确的技术贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个 benchmark、多组消融（多智能体/训练策略/数据结构/训练数据）、效率分析、成本对比，非常完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示丰富，但部分内容重复，论文较长
- 价值: ⭐⭐⭐⭐⭐ 对 GUI agent 社区提供了极有价值的数据生产范式，且实际证明了自动挖掘可以超越人工标注

<!-- RELATED:START -->

## 相关论文

- [M2-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining](m2-miner_multi-agent_enhanced_mcts_for_mobile_gui_agent.md)
- [FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)
- [ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning](tooltree_efficient_llm_agent_tool_planning_via_dual-feedback_monte_carlo_tree_se.md)
- [Efficient Agent Training for Computer Use](efficient_agent_training_for_computer_use.md)
- [CoMind: Towards Community-Driven Agents for Machine Learning Engineering](comind_towards_community-driven_agents_for_machine_learning_engineering.md)

<!-- RELATED:END -->
