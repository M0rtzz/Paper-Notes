---
title: >-
  [论文解读] InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization
description: >-
  [AAAI 2026][GUI定位] 针对GUI定位中语义对齐的探索瓶颈，提出Adaptive Exploration Policy Optimization (AEPO)框架，通过多答案生成策略强制广泛探索、自适应探索奖励函数动态引导以及共线惩罚机制确保探索质量，显著提升多模态大模型在复杂GUI定位任务上的表现。
tags:
  - AAAI 2026
  - GUI定位
  - 多模态大模型
  - 自适应探索
  - 策略优化
  - 多答案生成
---

# InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization

**会议**: AAAI 2026  
**arXiv**: [2508.05731](https://arxiv.org/abs/2508.05731)  
**代码**: [github.com/InfiXAI/InfiGUI-G1](https://github.com/InfiXAI/InfiGUI-G1)  
**领域**: 强化学习  
**关键词**: GUI定位, 多模态大模型, 自适应探索, 策略优化, 多答案生成

## 一句话总结

针对GUI定位中语义对齐的探索瓶颈，提出Adaptive Exploration Policy Optimization (AEPO)框架，通过多答案生成策略强制广泛探索、自适应探索奖励函数动态引导以及共线惩罚机制确保探索质量，显著提升多模态大模型在复杂GUI定位任务上的表现。

## 研究背景与动机

GUI定位（GUI Grounding）是自主GUI代理的核心感知任务，要求将自然语言指令精确映射到屏幕上的特定交互元素。这个任务可分解为两个正交维度：

**空间对齐（Spatial Alignment）**：精确定位元素的坐标——"指向"的准确性

**语义对齐（Semantic Alignment）**：识别正确的交互元素——"指向"正确的目标

**现有方法的困境：**

- **SFT方法**：数据密集，难以泛化到未见过的UI布局
- **RLVR方法**（如GRPO）：通过优化坐标生成有效提升了空间对齐，但存在**探索瓶颈**

**"信心陷阱"问题**是本文的核心动机。以一个具体例子说明：当指令为"使用相机搜索物体"时，屏幕上同时有"Camera"按钮和"Google Lens"图标。模型可能反复高置信度地选择"Camera"按钮（语义错误），而标准RLVR持续从这个高置信度但错误的选择中采样，极少偶然触碰到正确的"Google Lens"，因此无法获得纠正语义误解所需的学习信号。

这揭示了一个根本问题：**标准RL的单答案范式导致采样效率低下，无法突破策略的"信心陷阱"**。

## 方法详解

### 整体框架

AEPO框架包含三个协同组件：
1. **多答案生成机制**：强制模型在单次前向传播中生成 $N$ 个候选点
2. **自适应探索奖励（AER）**：基于效率第一性原理 $\eta = U/C$ 设计的非线性奖励信号
3. **共线惩罚**：防止退化的线性扫描策略

### 关键设计

#### 1. **问题建模与多答案生成**

将GUI定位形式化为策略优化问题：
- **上下文** $c = (\mathcal{S}, \mathcal{I})$：截图+自然语言指令
- **动作** $a$：坐标点 $p=(x,y)$
- **策略** $\pi_\theta(a|c)$：给定上下文的动作概率分布
- **目标**：$\theta^* = \arg\max_\theta \mathbb{E}_{c\sim\mathcal{D}, a\sim\pi_\theta(\cdot|c)}[R(a, B)]$

多答案生成的关键创新在于：prompting模型在单次前向传播中生成 $N$ 个候选点 $\mathcal{A} = \{p_1, p_2, \ldots, p_N\}$。这迫使模型超越其最高置信度的单一预测，显著增加从策略分布尾部采样到正确动作的概率——尤其对语义挑战性高的样本至关重要。

#### 2. **自适应探索奖励（AER）**

AER基于效率第一性原理 $\eta = U/C$ 推导而来：

**效用 $U$**：
- 探索成功（任一候选点落入ground truth）：$U = +1$
- 探索失败：$U = -1$

**代价 $C$**：用两个成本分量的几何均值建模：
- 提议成本 $C_p = N$（生成 $N$ 个候选的代价）
- 验证成本 $C_v$：成功时为第一个正确点的排名 $k$，失败时为 $N$
- $C = \sqrt{C_p \cdot C_v}$（几何均值捕捉递减边际收益）

最终的准确性奖励：

$$R_{\text{accuracy}}(\mathcal{A}, B) = \begin{cases} 1/\sqrt{N \cdot k} & \text{若存在} p_i \in B \\ -1/N & \text{否则} \end{cases}$$

**动态行为**：
- **失败时**：惩罚仅为 $-1/N$，随 $N$ 增大而减小，鼓励更广泛的探索
- **成功时**：奖励 $1/\sqrt{Nk}$ 在 $k$ 小时更大，奖励高效的准确预测
- 这种非对称设计在失败时促进探索、成功时促进收敛

#### 3. **共线惩罚机制**

如果生成的 $N$ 个候选点近似共线（通过检查任意三点构成的三角形面积是否接近零），则将准确性奖励覆盖为大的负值 $R_{\text{accuracy}} = -1$。这防止模型采用简单但低效的线性扫描策略，激励在几何空间中进行真正多样化的语义探索。

### 损失函数 / 训练策略

总奖励信号结合格式奖励和准确性奖励：

$$R_{\text{total}} = R_{\text{format}} + R_{\text{accuracy}}$$

格式奖励 $R_{\text{format}}$ 为输出格式正确时+1，否则为0，作为后续奖励评估的前提。总奖励用于计算优势估计 $\hat{A}$，直接指导策略参数更新。

训练使用GRPO/PPO/RLOO等标准策略梯度算法。

## 实验关键数据

### 主实验

**MMBench-GUI基准（Top-1准确率%）**：

| 模型 | Windows | MacOS | Linux | iOS | Android | Web | Avg |
|------|---------|-------|-------|-----|---------|-----|-----|
| UI-TARS-1.5-7B | 68.3/39.0 | 69.0/44.5 | 64.4/37.8 | 88.5/69.4 | 90.5/69.3 | 81.0/56.5 | 64.3 |
| Naive RLVR-7B | 79.3/58.1 | 82.3/62.7 | 64.4/44.9 | 94.9/89.1 | 95.5/84.2 | 92.9/79.5 | 79.3 |
| **InfiGUI-G1-7B** | **82.7/61.8** | **83.8/63.9** | **72.3/52.0** | 94.9/89.4 | 95.2/85.6 | 93.5/76.3 | **80.8** |
| G1-7B+探索成功 | 87.1/69.1 | 87.2/76.3 | 78.5/58.2 | 98.1/92.4 | 98.0/91.8 | 97.1/85.7 | **86.4** |

**ScreenSpot-Pro基准（Top-1准确率%）**：

| 模型 | CAD | Dev. | Creative | Scientific | Office | OS | Avg |
|------|-----|------|----------|-----------|--------|-----|-----|
| Naive RLVR-7B | 53.8/17.2 | 71.4/15.9 | 60.6/11.9 | 76.4/26.4 | 74.6/34.0 | 54.2/20.2 | 47.6 |
| **InfiGUI-G1-3B** | 50.8/25.0 | 64.9/20.0 | 51.5/16.8 | 68.8/32.7 | 70.6/32.1 | 49.5/15.7 | 45.2 |

### 消融实验

通过与Naive RLVR的对比体现各组件作用：

| 配置 | MMBench Avg | 说明 |
|------|------------|------|
| Naive RLVR-3B | 70.9 | 基线单答案RL |
| InfiGUI-G1-3B (Top-1) | 73.4 | +多答案+AER+共线惩罚的综合提升 |
| G1-3B 探索成功率 | 81.6 | 利用多答案的全部候选后的上界 |
| Naive RLVR-7B | 79.3 | 更大模型基线 |
| InfiGUI-G1-7B (Top-1) | 80.8 | 仅评估第一个答案即超越RLVR |
| G1-7B 探索成功率 | 86.4 | 多答案全候选上界 |

关键洞察：
- 3B模型中，Top-1即比RLVR好2.5%，探索上界提升10.7%
- 7B模型中，对比naive RLVR在Advanced类别（语义挑战大的项）提升最明显
- 探索成功率Avg N=1.6~2.0，说明模型学会了高效探索

### 关键发现

1. **语义对齐是关键瓶颈**：在Advanced类别上，InfiGUI-G1的提升远大于Basic类别，验证了AEPO有效解决了语义对齐问题
2. **探索效率**：InfiGUI-G1-7B平均仅生成1.6个候选即达到86.4%的探索成功率，说明模型不仅学会了探索，还学会了高效探索
3. **跨平台泛化**：在Windows、MacOS、Linux、iOS、Android、Web全平台上均获得一致提升
4. **3B vs 7B规模效应**：7B模型在绝对性能和AEPO带来的增量上均优于3B
5. **模型稳定性**：5次运行的标准差σ仅0.11-0.41，表明训练非常稳定

## 亮点与洞察

- **问题定义精准**：将GUI定位分解为空间对齐和语义对齐，并精确诊断出语义对齐的探索瓶颈，问题motivation非常清晰
- **AER的理论推导**：从效率第一性原理 $\eta=U/C$ 出发推导奖励函数，而非ad-hoc设计，具有理论优雅性
- **多答案范式的普适性**：多答案生成+自适应奖励的框架不仅适用于GUI定位，对其他需要空间探索的视觉定位任务（如referring expression comprehension）也可能有效
- **共线惩罚的巧妙设计**：用简单的几何约束防止退化策略，计算开销极小但效果显著

## 局限与展望

- 多答案生成增加了推理时的计算开销（尽管平均N较小），实际部署时需要权衡性能和效率
- 共线惩罚基于三角形面积的简单启发式，可能在高维或更复杂的空间探索场景中需要改进
- 当前框架假设ground truth为bounding box，对于非矩形或像素级目标可能需要适配
- 未探讨在Agent级别（多步决策）任务中AEPO是否保持优势
- AER中效用函数仅考虑二值（成功/失败），未利用IoU等渐进指标

## 相关工作与启发

- 与UI-R1、GUI-R1等GUI RL方法共享RLVR基础，但突破了单答案范式的限制
- 多答案生成的思路与Best-of-N采样有相似之处，但AER提供了更精细的学习信号
- 探索-利用平衡的自适应奖励设计可迁移到其他RL场景（如机器人操作中的多目标探索）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 多答案+自适应奖励的框架设计有创新，但各组件的novelty中等
- **实验充分度**: ⭐⭐⭐⭐⭐ — 多基准全平台评测，标准差报告，探索成功率分析
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，Motivation图示非常直观
- **价值**: ⭐⭐⭐⭐ — 在GUI Agent这个快速增长领域建立了新SOTA

<!-- RELATED:START -->

## 相关论文

- [Start Small, Think Big: Curriculum-based Relative Policy Optimization for Visual Grounding](start_small_think_big_curriculum-based_relative_policy_optimization_for_visual_g.md)
- [MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization](mars_multi-agent_adaptive_reasoning_with_socratic_guidance_f.md)
- [Reasoning with Exploration: An Entropy Perspective](reasoning_with_exploration_an_entropy_perspective.md)
- [Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning](behaviour_policy_optimization_provably_lower_variance_return_estimates_for_off-p.md)
- [DiffOP: Reinforcement Learning of Optimization-Based Control Policies via Implicit Policy Gradients](diffop_reinforcement_learning_of_optimization-based_control_policies_via_implici.md)

<!-- RELATED:END -->
