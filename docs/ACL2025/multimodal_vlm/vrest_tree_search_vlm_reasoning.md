---
title: >-
  [论文解读] VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism
description: >-
  [ACL 2025 (Long Paper)][多模态][MCTS] 提出VReST，首次将蒙特卡洛树搜索（MCTS）应用于多模态CoT推理：每个节点是一个推理步骤，通过多模态自奖励机制（sub-question有用性+答案正确性+视觉-语言线索相关性）评估推理质量，无需训练即在MathVista上达到64.50%（超越CoT的54.60%和ToT的60.20%），并展示出多模态测试时缩放定律。
tags:
  - ACL 2025 (Long Paper)
  - 多模态
  - MCTS
  - 视觉推理
  - 自奖励
  - 测试时缩放
  - 多模态CoT
---

# VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism

**会议**: ACL 2025  
**arXiv**: [2506.08691](https://arxiv.org/abs/2506.08691)  
**代码**: [GitHub](https://github.com/GaryJiajia/VReST)  
**领域**: 多模态VLM  
**关键词**: MCTS, 多模态CoT, 自奖励机制, 测试时缩放, 视觉数学推理

## 一句话总结

首次将蒙特卡洛树搜索(MCTS)引入多模态CoT推理，配合无需额外模型的多模态自奖励机制系统性探索推理空间，在三个视觉数学推理基准上实现SOTA并验证了多模态测试时缩放定律。

## 研究背景与动机

**领域现状**：大型视觉-语言模型(LVLM)在多模态任务上表现出色，但在复杂视觉推理上仍受限。现有的多模态CoT方法（如DDCoT、Cantor）试图通过分解问题来增强推理，但效果有限。一个令人意外的现象是：在MathVista等更复杂的视觉数学任务上，多模态CoT推理的准确率甚至不如直接回答（Direct QA 55.70% vs CoT 54.60%）。

**现有痛点**：现有多模态CoT方法有两个根本性缺陷：(1) 生成的中间推理步骤有限，依赖贪心策略只能获得次优解，无法充分探索推理空间；(2) 缺乏对已生成推理链质量的评估和修正机制，一旦某一步推理出错就无法纠正。

**核心矛盾**：训练LVLM推理数据集代价高昂且难以扩展，但现有的training-free方法（如ToT使用启发式贪心选择）在探索能力上不足以应对复杂多模态推理。

**本文目标** 在不训练的前提下，如何让LVLM系统性地探索推理空间并可靠地评估推理质量？

**切入角度**：借鉴LLM领域中MCTS在NLP推理任务上的成功经验（如RAP、LLaMA-Berry），但这些方法都只处理纯文本，尚无将MCTS用于多模态推理的工作。作者的关键观察是：视觉推理需要同时利用图像和文本信息来评估推理步骤，这为设计多模态自奖励机制提供了独特的设计空间。

**核心 idea**：用MCTS构建推理搜索树（节点=推理步骤，路径=推理链），通过整合视觉和文本线索的自奖励机制评估节点质量，系统性迭代探索后选择最优推理路径。

## 方法详解

### 整体框架

VReST将MCTS与LVLM结合形成推理搜索框架。给定图像 $I$ 和问题 $Q$，目标是找到最优推理链 $\mathcal{P}^* = \{Q, S_1, S_2, ..., S_n\}$，其中每个推理步骤 $S_i$ 包含子问题 $Q_i$ 和子答案 $A_i$。整个过程执行 $K$ 次MCTS迭代（默认 $K=10$），每次迭代依次执行Selection→Expansion→Rewarding→Backpropagation四步。迭代完成后，根据累积奖励选择最优推理链。

### 关键设计

1. **UCT引导的节点选择与LVLM驱动的扩展**:

    - 功能：从根节点递归选择子节点到叶节点，然后在叶节点处生成新的推理步骤
    - 核心思路：选择阶段使用UCB公式平衡探索和利用 $UCT(v) = R(v) + c\sqrt{\frac{\ln N(p(v))}{N(v)}}$；扩展阶段通过提高LVLM温度参数（0.7）生成 $w=5$ 个不同推理步骤，用自奖励机制选出最佳节点继续扩展，直到达到终止条件或最大深度 $D_{\max}=8$
    - 设计动机：UCT确保搜索不过度集中于局部最优；多样化生成+奖励筛选兼顾探索广度和质量控制，本质上比ToT的贪心选择更全局

2. **多模态自奖励机制(Self-Reward)**:

    - 功能：不依赖额外模型，利用LVLM自身评估每个推理步骤的质量
    - 核心思路：综合两个维度——$R_1 = P(\text{"Yes"} | [\mathcal{P}_t, \mathcal{P}_Q], I)$ 评估所有子问题是否有用，$R_2 = P(\text{"Yes"} | [\mathcal{P}_t, \mathcal{P}_A], I)$ 评估最后一步答案是否正确。最终奖励为几何均值 $R = \sqrt{R_1 \cdot R_2}$。注意两个评估都以图像 $I$ 为输入，整合了视觉信息
    - 设计动机：(1) 使用LVLM生成"Yes"的token概率作为奖励信号，无需额外训练奖励模型；(2) 几何均值比算术均值更严格——要求两个指标都不能太低；(3) 图像 $I$ 参与奖励计算，使奖励机制能感知视觉线索

3. **推理链选择策略(Best-Trace & Trace-Vote)**:

    - 功能：$K$ 次迭代完成后，从所有探索过的推理链中选出最终答案
    - 核心思路：Best-Trace选择平均奖励最高的路径 $\mathcal{P}^* = \arg\max_{\mathcal{P}} \text{Avg}(\{R(S_t) | S_t \in \mathcal{P}\})$；Trace-Vote选出top-n推理链后对最终答案投票取多数
    - 设计动机：消融实验表明Trace-Vote > Best-Trace > Greedy-Trace，投票机制通过聚合多条高奖励推理链降低了单条路径的偶然误差

### 损失函数 / 训练策略

VReST完全无需训练，所有组件都基于推理时的LVLM前向传播。反向传播阶段将终止节点的奖励沿路径回传更新祖先节点统计信息：$R(S_t) = \text{Avg}(\{R(S_i)\}_{i=t}^{T})$，$N(S_t) = N(S_t) + 1$。基座模型使用Qwen2-VL-7B-Instruct（LVLM）和Qwen2.5-7B-Instruct（纯文本LLM，仅用于消融和答案评判）。

## 实验关键数据

### 主实验

MathVista testmini上使用Qwen2-VL-7B-Instruct（%准确率）：

| 方法 | MWP | GPS | VQA | SCI | ALL |
|------|-----|-----|-----|-----|-----|
| Direct QA | 60.75 | 48.56 | 50.28 | 59.84 | 55.70 |
| CoT | 56.99 | 40.87 | 48.04 | 59.02 | 54.60 |
| CoT-Vote | 69.89 | 48.08 | 56.98 | 60.66 | 62.30 |
| ToT | 63.44 | 53.37 | 54.19 | 57.38 | 60.20 |
| **VReST** | **72.04** | **56.73** | 58.10 | **67.21** | **64.50** |
| **VReST-Vote** | **75.81** | 51.44 | **64.25** | 68.03 | **65.40** |

CharXiv验证集上VReST-Vote达到38.10%，大幅超越ToT的32.10%和CoT-Vote的30.90%。

### 消融实验

| 配置 | MathVista | MathVision | CharXiv |
|------|-----------|------------|---------|
| 全部用LVLM (V,V,V) | 64.50 | 26.64 | 33.10 |
| 推理步骤改用纯文本LLM | ~60.80 | ~21.71 | ~29.50 |
| w/o R1 (去掉问题有用性) | ~62.30 | ~25.33 | ~31.40 |
| w/o R2 (去掉答案正确性) | ~61.70 | ~24.67 | ~30.80 |
| w/o PRM (去掉过程奖励) | ~59.60 | ~22.37 | ~29.30 |
| Trace-Vote | **65.40** | **28.29** | **38.10** |
| Best-Trace | 64.50 | 26.64 | 33.10 |
| Greedy-Trace | 60.00 | 23.03 | 31.30 |

### 关键发现

- **视觉信息不可或缺**：任何一个组件替换为纯文本LLM都导致显著下降，证明多模态自奖励需要视觉感知
- **过程奖励(PRM)贡献最大**：去掉中间节点奖励导致最严重的性能下降，说明逐步评估比仅看终点更重要
- **测试时缩放定律**：随MCTS迭代次数从2增到10，VReST-Vote性能持续上升且斜率大于其他方法
- **计算代价**：VReST每样本平均108-158秒，约为CoT的15倍、ToT的3倍

## 亮点与洞察

- **首个多模态MCTS推理框架**：将MCTS从NLP推理扩展到视觉-语言多模态场景，概念上具有开创性。巧妙之处在于利用LVLM同时承担推理步骤生成和奖励评估两个角色，整个系统不引入任何额外模型
- **自奖励机制设计精巧**：用LVLM生成"Yes"的token概率作为奖励信号，既简洁又有效。这个trick可以迁移到其他需要过程奖励但没有现成奖励模型的场景
- **多模态测试时缩放**：明确验证了在多模态推理中也存在test-time scaling law，增加推理计算可以换取性能提升

## 局限与展望

- **计算开销过大**：每样本~2分钟（CoT仅~7秒），实际部署困难。可引入剪枝策略或早停机制减少不必要的迭代
- **模型依赖性未验证**：仅在Qwen2-VL-7B上实验，不同架构/规模的LVLM效果可能不同
- **自奖励偏见风险**：模型用自身判断做奖励，可能放大已有偏见。训练专门的多模态奖励模型是更可靠的方向
- **任务范围有限**：仅验证视觉数学推理，对视觉常识推理、空间关系等其他推理类型是否有效未知

## 相关工作与启发

- **vs ToT**: ToT使用启发式贪心选择每步最优节点，本质上是beam search；VReST通过UCT+反向传播实现真正的全局搜索，能回溯并重新探索之前被忽略的分支
- **vs CoT-Vote/Self-Consistency**: 两者都生成多条推理链后投票，但CoT-Vote的推理链是独立生成的；VReST的多条推理链共享搜索树，信息在迭代间传递，探索更高效
- **vs ReST-MCTS***: ReST-MCTS*需要训练奖励模型，不是training-free的。VReST在公平比较条件下（同为training-free）有明确优势

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将MCTS引入多模态CoT推理，方向有开拓性
- 实验充分度: ⭐⭐⭐⭐ 3个数据集+6个基线+充分消融+缩放实验+时间分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详尽，消融实验有说服力
- 价值: ⭐⭐⭐ 计算开销限制实际应用，但提供了好的research方向
---
title: >-
  [论文解读] VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism
description: >-
  [ACL 2025 (Long Paper)][多模态][MCTS] 提出VReST，首次将蒙特卡洛树搜索（MCTS）应用于多模态CoT推理：每个节点是一个推理步骤，通过多模态自奖励机制（sub-question有用性+答案正确性+视觉-语言线索相关性）评估推理质量，无需训练即在MathVista上达到64.50%（超越CoT的54.60%和ToT的60.20%），并展示出多模态测试时缩放定律。
tags:
  - ACL 2025 (Long Paper)
  - 多模态
  - MCTS
  - 视觉推理
  - 自奖励
  - 测试时缩放
  - 多模态CoT
---

# VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism

**会议**: ACL 2025  
**arXiv**: [2506.08691](https://arxiv.org/abs/2506.08691)  
**代码**: [GitHub](https://github.com/GaryJiajia/VReST)  
**领域**: 多模态VLM / 视觉推理  
**关键词**: MCTS, 多模态CoT, 自奖励机制, 测试时缩放, 视觉数学推理

## 一句话总结

首次将蒙特卡洛树搜索(MCTS)引入多模态CoT推理，配合多模态自奖励机制系统性探索推理空间，无需训练即在三个视觉数学推理基准上实现SOTA，并验证了多模态任务中的测试时缩放定律。

## 研究背景与动机

**领域现状**: 大型视觉-语言模型(LVLM)在多模态任务中表现出色，但在复杂视觉推理上仍受限——特别是Chain-of-Thought(CoT)提示技术在LVLM中的效果不及预期。在MathVista等更复杂的视觉数学任务上，多模态CoT推理甚至不如直接问答(Direct QA)。

**现有痛点**: 现有多模态CoT方法(如DDCoT、Cantor)存在两个关键缺陷：(1) 生成的中间推理步骤有限，缺乏充分探索推理空间的能力；(2) 无法评估和优化已生成的推理链，只能通过贪心算法获得次优解。

**核心矛盾**: 训练LVLM推理数据集代价高昂且难以扩展，但现有的training-free方法又无法充分释放LVLM的推理潜力。

**本文目标**: 如何在不训练的前提下，让LVLM系统性地探索推理空间、评估推理质量，从而找到最优推理路径？

**切入角度**: 借鉴LLM领域中MCTS在推理扩展上的成功经验，将其扩展到多模态场景。

**核心idea**: 用MCTS构建推理树(节点=推理步骤，路径=推理链)，通过多模态自奖励机制评估节点质量，实现对推理空间的系统性、迭代式探索。

## 方法详解

### 整体框架

VReST将MCTS与LVLM结合，构建推理搜索树。给定图像 $I$ 和问题 $Q$，目标是找到最优推理链 $\mathcal{P}^*$。每次MCTS迭代包含四个步骤：Selection → Expansion → Rewarding → Backpropagation，迭代 $K$ 次后选择最优推理链。

### 关键设计

1. **UCT引导的节点选择(Selection)**:
    - 功能：从根节点出发，递归选择子节点直至叶节点
    - 核心思路：使用UCB算法平衡探索与利用，$UCT(v) = R(v) + c\sqrt{\frac{\ln N(p(v))}{N(v)}}$，其中 $R(v)$ 是奖励值，$N(v)$ 是访问次数，$c$ 是探索常数
    - 设计动机：确保搜索不会过度集中于局部最优，而是有策略地广泛探索推理空间

2. **LVLM驱动的推理扩展(Expansion)**:
    - 功能：在选定的叶节点处生成新的推理步骤
    - 核心思路：通过提高LVLM温度参数生成 $w$ 个不同推理步骤 $\{S_{t,j}|j=1,...,w\} = \text{LVLM}(\mathcal{P}_{t-1}, I)$，选择奖励值最高的节点继续扩展，直到到达终止节点或最大深度 $D_{\max}$
    - 设计动机：在每一步都保持多样性，避免单一推理路径的局限；提升温度增加多样性同时通过奖励机制筛选质量

3. **多模态自奖励机制(Self-Reward)**:
    - 功能：不依赖额外模型，利用LVLM自身评估推理步骤质量
    - 核心思路：综合两个维度计算奖励值：$R_1 = P(\text{"Yes"}|[\mathcal{P}_t, \mathcal{P}_Q], I)$（子问题有用性）和 $R_2 = P(\text{"Yes"}|[\mathcal{P}_t, \mathcal{P}_A], I)$（答案正确性），最终奖励为几何均值 $R = \sqrt{R_1 \cdot R_2}$
    - 设计动机：(1) 几何均值要求两个指标都不能太低，比算术均值更严格；(2) 利用LVLM生成"Yes"的概率作为奖励信号，既整合了视觉信息又避免了引入额外模型

### 损失函数 / 训练策略

- **完全无需训练(Training-free)**：VReST的所有组件都基于推理时的LVLM前向传播
- **反向传播更新**：每次迭代到达终止节点后，将奖励值沿路径反向传播更新所有节点的统计信息：$R(S_t) = \text{Avg}(\{R(S_i)\}_{i=t}^{T})$，同时更新访问次数 $N(S_t) = N(S_t) + 1$
- **推理链选择**：支持三种策略——Greedy Trace（贪心选择）、Best Trace（全局最优路径）、Trace Vote（多路径投票）

## 实验关键数据

### 主实验

在MathVista testmini上使用Qwen2-VL-7B-Instruct：

| 方法 | FQA | GPS | MWP | ALL |
|------|-----|-----|-----|-----|
| QA | 60.59 | 48.56 | 60.75 | 55.70 |
| CoT | 63.57 | 40.87 | 56.99 | 54.60 |
| CoT-Vote | 70.63 | 48.08 | 69.89 | 62.30 |
| ToT | 66.54 | 53.37 | 63.44 | 60.20 |
| **VReST** | **68.03** | **56.73** | **72.04** | **64.50** |
| **VReST-Vote** | 69.14 | 51.44 | **75.81** | **65.40** |

### 消融实验

| 消融设置 | MathVista | MathVision | CharXiv |
|----------|-----------|------------|---------|
| 全部用LVLM (V,V,V) | **64.50** | **26.64** | **33.10** |
| 推理用LLM (T,V,V) | 60.80 | 21.71 | 29.50 |
| w/o R1 | 62.30 | 25.33 | 31.40 |
| w/o R2 | 61.70 | 24.67 | 30.80 |
| w/o PRM | 59.60 | 22.37 | 29.30 |

### 关键发现

- **视觉信息不可或缺**：LVLM替换为纯文本LLM都会显著降低性能
- **过程奖励至关重要**：去掉中间节点奖励评估，性能下降最严重
- **测试时缩放定律**：随MCTS迭代次数增加，VReST-Vote的性能提升幅度持续大于其他方法
- **计算开销**：每个样本平均需108-158秒，约为CoT的15倍

## 亮点与洞察

- **首个多模态MCTS推理框架**：将MCTS从LLM推理扩展到LVLM，概念上有开拓性
- **自奖励机制设计优雅**：利用LVLM自身概率输出作为奖励信号，无需外部模型
- **多模态测试时缩放验证**：明确展示了VReST在多模态推理上更好地利用额外计算资源
- **即插即用**：不修改模型参数，可直接应用于任何LVLM

## 局限与展望

- 计算开销大（每样本~2分钟，CoT仅~7秒），限制实际部署
- 仅在Qwen2-VL-7B上验证，对不同架构和规模的LVLM泛化性存疑
- 仅在视觉数学推理上验证，其他推理类型未探索
- 自奖励机制依赖模型自身判断，存在偏见传播风险

## 相关工作与启发

- 与ToT对比：ToT使用启发式贪心选择，VReST通过UCT+反向传播实现更全局的搜索
- 可尝试训练专门的多模态奖励模型以降低偏见风险
- 引入剪枝策略或早停技术可大幅减少计算开销

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将MCTS引入多模态CoT推理，方向有开拓性
- 实验充分度: ⭐⭐⭐⭐ 3个数据集+6个基线+消融+缩放实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法和实验详尽
- 价值: ⭐⭐⭐ 计算开销限制实际应用，但提供了好的research方向
---
title: >-
  [论文解读] VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism
description: >-
  [ACL 2025 (Long Paper)][多模态][MCTS] 提出VReST，首次将蒙特卡洛树搜索（MCTS）应用于多模态CoT推理：每个节点是一个推理步骤，通过多模态自奖励机制（sub-question有用性+答案正确性+视觉-语言线索相关性）评估推理质量，无需训练即在MathVista上达到64.50%（超越CoT的54.60%和ToT的60.20%），并展示出多模态测试时缩放定律。
tags:
  - ACL 2025 (Long Paper)
  - 多模态
  - MCTS
  - 视觉推理
  - 自奖励
  - 测试时缩放
  - 多模态CoT
---

# VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2506.08691](https://arxiv.org/abs/2506.08691)  
**代码**: [https://github.com/GaryJiajia/VReST](https://github.com/GaryJiajia/VReST)  
**领域**: 多模态VLM / LLM推理  
**关键词**: MCTS, 视觉推理, 自奖励, 测试时缩放, 多模态CoT  

## 一句话总结
提出VReST，首次将蒙特卡洛树搜索（MCTS）应用于多模态CoT推理：每个节点是一个推理步骤，通过多模态自奖励机制（sub-question有用性+答案正确性+视觉-语言线索相关性）评估推理质量，无需训练即在MathVista上达到64.50%（超越CoT的54.60%和ToT的60.20%），并展示出多模态测试时缩放定律。

## 背景与动机
LVLM的CoT推理在复杂视觉数学任务上效果有限——在MathVista等基准上，CoT甚至不如直接回答（Direct QA: 55.70% vs CoT: 54.60%）。原因：(1) 推理步骤有限，无法充分探索解空间；(2) 缺乏对推理链质量的评估和修正机制。已有的Tree-of-Thoughts等方法使用启发式选择，容易陷入局部最优。

## 核心问题
如何在测试时无需训练地充分探索LVLM的推理空间，并可靠地评估和选择最优推理链？

## 方法详解

### 整体框架
构建推理搜索树，每个节点是一个推理步骤（sub-question + sub-answer），通过MCTS的Selection→Expansion→Rewarding→Backpropagation四步迭代K次，最终选择累积奖励最高的推理路径。

### 关键设计

1. **MCTS推理搜索**: 

    - **Selection**: 使用UCT（Upper Confidence Bound for Trees）平衡探索和利用：UCT(v) = R(v) + c·√(ln N(parent)/N(v))
    - **Expansion**: 对选中的叶节点，用LVLM在较高temperature下生成w个候选推理步骤（宽度w的树扩展）
    - **Rewarding**: 用自奖励机制评估每个新推理步骤
    - **Backpropagation**: 将奖励值沿路径回传更新祖先节点

2. **多模态自奖励机制（Self-Reward）**: 不引入额外模型，用LVLM自身评估推理质量，融合两个维度：

    - **R₁**: sub-question的有用性——"这些子问题对解决原始问题有用吗？"（Yes/No概率）
    - **R₂**: 最后一步答案的正确性——"这个答案正确吗？"（Yes/No概率）
    - 最终奖励：R = √(R₁ × R₂)，用几何平均确保两者都要高

3. **推理链选择**: 搜索完成后，有两种策略：

    - **VReST**: 选择累积奖励最高的推理链
    - **VReST-Vote**: 将top-k推理链的最终答案做投票，多数决

### 损失函数 / 训练策略
- 完全无需训练（training-free），纯推理时方法
- 默认K=8次MCTS迭代，w=3宽度，D_max=10最大深度
- 基于InternVL2-8B和Qwen2-VL-7B进行实验

## 实验关键数据

**MathVista (testmini)**:

| 方法 | ALL | MWP | VQA | SCI | STA |
|------|-----|-----|-----|-----|-----|
| Direct QA | 55.70 | 60.75 | 50.28 | 59.84 | 67.44 |
| CoT | 54.60 | 56.99 | 48.04 | 59.02 | 70.43 |
| CoT-Vote | 62.30 | 69.89 | 56.98 | 60.66 | 79.07 |
| ToT | 60.20 | 63.44 | 54.19 | 57.38 | 74.09 |
| **VReST** | **64.50** | **72.04** | 58.10 | **67.21** | 75.75 |
| **VReST-Vote** | **65.40** | **75.81** | **64.25** | **68.03** | 77.74 |

**MATH-Vision (testmini)**: VReST: 26.64% vs ToT: 20.39% vs CoT: 14.47%

**测试时缩放**: 随着MCTS迭代次数从1增加到16，性能持续提升且不饱和——展示了多模态测试时缩放定律。

### 消融实验要点
- **奖励函数**: R₁和R₂都不可或缺，几何平均优于算术平均
- **MCTS K次迭代**: K=1时已略优于基线，K=8时显著领先
- **树宽度w**: w=3是效率和效果的最佳平衡
- **VReST vs VReST-Vote**: Vote在答案多样性高的任务更有效
- **CoT反而不如Direct QA**: 验证了复杂推理中简单CoT的局限性

## 亮点
- **首次MCTS用于多模态推理**: 填补了MCTS在视觉推理中的空白
- **优雅的自奖励设计**: 不需要额外reward model，LVLM自评即可——用Sub-Q有用性和答案正确性两个维度
- **测试时缩放定律**: 在多模态任务中首次展示：增加推理时计算=持续提升性能
- **CoT < Direct QA的揭示**: 证明简单CoT在复杂视觉推理中确实不够用

## 局限与展望
- 计算成本高：K=8的MCTS需要数十次LVLM推理，延迟增加10x+
- 自奖励可能不够可靠——LVLM评估自己生成内容时可能有偏差
- 仅在数学/视觉推理任务验证，通用VQA或开放式生成可能需要不同的奖励设计
- 树搜索假设推理可以分解为离散步骤，连续推理场景不适用
- 未与OpenAI o1/o3等内置推理模型对比

## 与相关工作的对比
- **vs CoT/CoT-SC**: VReST是系统性搜索 vs CoT的线性/采样策略
- **vs ToT**: ToT用启发式评估导致局部最优，VReST用MCTS+UCT做全局探索
- **vs Cantor**: Cantor用多角色分步推理但无搜索机制
- **vs Improve VLM CoT Reasoning (本批次)**: 那篇用SFT+DPO改进CoT质量，VReST用测试时搜索——两者正交互补

## 启发与关联
- MCTS的推理时缩放定律可以指导VLM部署策略——简单问题用少迭代、难问题用多迭代（自适应MCTS）
- 自奖励机制可以与VHR（视觉注意力头增强）结合——增强视觉感知后再做树搜索
- VReST的思路可以推广到Agent任务——GUI操作也可以用MCTS搜索最优动作序列

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次在多模态推理中应用MCTS，自奖励机制设计合理
- 实验充分度: ⭐⭐⭐⭐ 3个benchmark、详细消融、缩放分析
- 写作质量: ⭐⭐⭐⭐⭐ 图2的框架图极其清晰，方法描述精确
- 价值: ⭐⭐⭐⭐ 展示了测试时缩放在多模态中的潜力，启发后续研究

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search](visuothink_empowering_lvlm_reasoning_with_multimodal_tree_search.md)
- [\[ACL 2025\] Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](mcts_video_captioning_eval.md)
- [\[ICML 2025\] Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger](../../ICML2025/multimodal_vlm/re-ranking_reasoning_context_with_tree_search_makes_large_vision-language_models.md)
- [\[NeurIPS 2025\] Enhancing Outcome Reward-Based RL Training of MLLMs with Self-Consistency Sampling](../../NeurIPS2025/multimodal_vlm/enhancing_the_outcome_reward-based_rl_training_of_mllms_with_self-consistency_sa.md)
- [\[NeurIPS 2025\] Recognition through Reasoning: Reinforcing Image Geo-localization with Large Vision-Language Models](../../NeurIPS2025/multimodal_vlm/recognition_through_reasoning_reinforcing_image_geo-localization_with_large_visi.md)

<!-- RELATED:END -->
