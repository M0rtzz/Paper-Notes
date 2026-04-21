---
title: >-
  [论文解读] SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing
description: >-
   提出 SEED-SET 框架，将自主系统的伦理评估建模为层次化贝叶斯实验设计问题，同时整合客观指标和主观价值判断，在有限预算下高效生成高伦理对齐度的测试用例。
tags:

---

# SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing

## 基本信息

- **会议**: ICLR 2026
- **arXiv**: [2603.01630](https://arxiv.org/abs/2603.01630)
- **代码**: [项目主页](https://anjaliparashar.github.io/seed-site/)
- **领域**: AI 安全 / 自主系统评估
- **关键词**: Ethical Testing, Bayesian Experimental Design, Gaussian Process, LLM Evaluator, Autonomous Systems

## 一句话总结

提出 SEED-SET 框架，将自主系统的伦理评估建模为层次化贝叶斯实验设计问题，同时整合客观指标和主观价值判断，在有限预算下高效生成高伦理对齐度的测试用例。

## 研究背景与动机

### 问题背景
自主系统（无人机、电网分配等）在高风险领域部署日益增多，其伦理对齐性评估变得至关重要。然而，伦理评估面临三大挑战：

**度量困难**：伦理行为（公平性、社会接受度）缺乏 ground-truth 标签；

**主观依赖**：价值对齐因利益相关者而异，且随时间演化，静态基准需不断修订；

**评估昂贵**：真实系统的评估受预算约束，无法大规模采集人类反馈。

### 现有方法的局限
- 规则式伦理基准依赖既定准则，不够具体；
- 基于 RL/RLHF 的方法假设充足的模拟或专家标注，样本需求大；
- 偏好式方法和大规模人类研究仅关注单一维度。

### 核心思路
同时建模**客观指标**（如火灾损失、电网成本）和**主观偏好**（利益相关者的伦理判断），通过层次化高斯过程和贝叶斯实验设计高效生成测试场景。

## 方法详解

### 整体框架

SEED-SET = Scalable Evolving Experimental Design for System-level Ethical Testing，包含三大组件：

1. **层次化变分高斯过程（HVGP）** 作为代理模型
2. **联合获取策略** 用于自适应测试用例生成
3. **LLM 代理** 替代人类进行偏好评估

### 1. 问题形式化

给定黑盒自主系统 $\mathcal{S}_\pi$、场景空间 $\mathcal{X}$，伦理合规函数分解为：

- **客观层**：$f_{\text{obj}}: \mathcal{X} \to \mathcal{Y}$，将场景参数映射到可度量指标（成本、韧性等）
- **主观层**：$f_{\text{subj}}: \mathcal{Y} \to \mathbb{R}$，根据客观指标给出伦理效用评分

### 2. 层次化变分高斯过程（HVGP）

将伦理评估分为两级 VGP 建模：

**Objective GP**：学习代理模型 $g: x \to y$，预测场景的客观指标
$$
p(f(x)|\mathcal{D}) = \mathcal{N}(\mu(x), k(x, x'))
$$

**Subjective GP**：学习偏好模型 $h: y \to z$，从客观指标映射到主观伦理评分

由于主观评估无 ground-truth，采用**成对偏好引出**：oracle $\mathcal{T}: (y, y') \to \{1, 2\}$ 比较两个场景的伦理优劣。

层次化结构的**两大优势**：
- **可解释性**：伦理偏好锚定在可观察的系统行为上
- **数据效率**：利用主观对客观的依赖关系，减少所需评估次数

### 3. 联合获取策略

核心创新——同时平衡客观探索和主观利用的获取函数：

$$
V(x) = \underbrace{I(g_x; y|\mathcal{D})}_{\text{客观信息增益}} + \mathbb{E}_{q_\phi(y|x)}\left[\underbrace{I(h_y; z|\mathcal{D})}_{\text{主观信息增益}} + \underbrace{\mathbb{E}_{q_\psi(h_y)}[h_y]}_{\text{偏好利用}}\right]
$$

三项的作用：
- **第一项**：降低客观指标空间中的不确定性（探索新场景）
- **第二项**：改善主观效用函数的估计（理解偏好）
- **第三项**：趋向高伦理效用区域（利用已知偏好）

### 4. LLM 代理评估器

使用 GPT-4o 作为利益相关者代理进行成对偏好评估，prompt 包含：

1. **任务描述**：特定领域的上下文
2. **客观指标**：两个场景的可度量结果
3. **主观准则**：用自然语言编码的伦理偏好

## 实验

### 案例 1：电网资源分配（IEEE 5/30-Bus）

| 方法 | 5-Bus 偏好得分 (↑) | 30-Bus 偏好得分 (↑) |
|------|-------------------|-------------------|
| Random | 低 | 低 |
| Single GP | 中等 | 失败 |
| VS-AL-1 | 失败 | 失败 |
| VS-AL-2 | 失败 | 失败 |
| **HVGP (SEED-SET)** | **最高** | **最高** |

### 案例 2：消防救援（无人机导航）

| 方法 | 偏好得分 (↑) | 覆盖率 (↑) |
|------|-------------|-----------|
| Random | 低 | 低 |
| Single GP | 中等 | 中等 |
| HVGP (MI1+MI2 仅探索) | 中高 | 中高 |
| HVGP (Pref 仅利用) | 较高 | 中等 |
| **HVGP (完整获取)** | **最高** | **最高** |

### 消融实验：获取策略组件

| 获取策略 | 生成最优测试比例 (↑) | 空间覆盖 (↑) |
|---------|-------------------|------------|
| 随机采样 | 1× | 1× |
| 仅 MI1+MI2 | 1.4× | 1.1× |
| 仅 Pref | 1.6× | 0.9× |
| **完整 V(x)** | **2×** | **1.25×** |

### 关键发现

1. **SEED-SET 生成最优测试用例数量是基线的 2 倍**，搜索空间覆盖率提升 1.25 倍；
2. **高维场景优势显著**：在 30-Bus（40 维设计空间）中，Single GP 完全失败，而 HVGP 仍保持高效；
3. **层次化建模关键**：将 $f$ 分解为 $f_{\text{obj}} + f_{\text{subj}}$ 比直接建模 $f(x) \to z$ 更准确；
4. **三项获取缺一不可**：去除任何一项都导致性能下降；
5. **LLM 代理可靠**：TrueSkill 评分验证 GPT-4o 的偏好判断与手工偏好函数趋势一致；
6. **适应不同利益相关者**：切换 prompt 中的主观准则可快速适配不同伦理标准。

## 亮点

- 首个同时考虑客观指标和主观价值判断的自主系统伦理测试框架
- 层次化 HVGP 设计使主观偏好锚定在可观察行为上，提升可解释性
- 联合获取策略优雅平衡探索-利用，三项设计各有明确功能
- LLM 作为代理评估器降低了对人类专家的依赖
- 框架领域无关，可适用于电网、消防、交通等多种场景

## 局限性

- 假设利益相关者如实报告偏好（假设 A2），未处理策略性误报
- 假设客观指标集完全已知且固定（假设 A3），动态指标扩展未涉及
- LLM 代理可能继承 GPT-4o 的偏见，不同 LLM 的偏好一致性需进一步验证
- VGP 在极高维场景下的可扩展性仍受限于诱导点数量
- 手工偏好评分函数的设计依赖领域知识

## 相关工作

- **AI 伦理框架**: NIST AI RMF 1.0 (2023), IEEE 标准
- **贝叶斯实验设计**: Rainforth et al. (2024), Chaloner & Verdinelli (1995)
- **偏好学习**: RLHF (Christiano et al., 2017), 成对比较 GP (Chu & Ghahramani, 2005)
- **主动学习**: 偏好引出 (Keswani et al., 2024)
- **LLM 评估器**: Huang et al. (2025) 使用 LLM 进行偏好评估

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 首次将层次化贝叶斯实验设计应用于伦理测试
- 技术深度：⭐⭐⭐⭐ — HVGP + 联合获取 + LLM 评估器三位一体
- 实验充分度：⭐⭐⭐⭐ — 三个案例研究 + 多维消融 + 利益相关者分析
- 实用价值：⭐⭐⭐⭐ — 领域无关框架，但实际部署需与真实利益相关者验证

<!-- RELATED:START -->

## 相关论文

- [Time-Evolving Dynamical System for Learning Latent Representations of Mouse Visual Cortex](../../NeurIPS2025/interpretability/time-evolving_dynamical_system_for_learning_latent_representations_of_mouse_visu.md)
- [MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning](mata_a_trainable_hierarchical_automaton_system_for_multi-agent_visual_reasoning.md)
- [Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective](../../ICML2025/interpretability/evolving_prompts_in-context_an_open-ended_self-replicating_perspective.md)
- [Beyond the Fold: Quantifying Split-Level Noise and the Case for Leave-One-Dataset-Out AU Evaluation](../../CVPR2026/interpretability/beyond_the_fold_quantifying_split-level_noise_and_the_case_for_leave-one-dataset.md)
- [VL-SAE: Interpreting and Enhancing Vision-Language Alignment with a Unified Concept Set](../../NeurIPS2025/interpretability/vlsae_interpreting_and_enhancing_visionlanguage_alignment_wi.md)

<!-- RELATED:END -->
