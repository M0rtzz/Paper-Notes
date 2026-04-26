---
title: >-
  [论文解读] PerfGuard: A Performance-Aware Agent for Visual Content Generation
description: >-
  [ICLR 2026][LLM Agent][visual content generation] 提出 PerfGuard，一个性能感知的 agent 框架用于视觉内容生成，通过多维性能评分矩阵替代文本描述来建模工具能力边界，结合自适应偏好更新和能力对齐规划优化，显著提升工具选择准确率（错误率从 77.8% 降至 14.2%）和视觉生成质量。
tags:
  - ICLR 2026
  - LLM Agent
  - visual content generation
  - agent
  - tool selection
  - performance-aware
  - AIGC
  - 偏好优化
  - 图像生成
  - 图像编辑
---

# PerfGuard: A Performance-Aware Agent for Visual Content Generation

**会议**: ICLR 2026  
**arXiv**: [2601.22571](https://arxiv.org/abs/2601.22571)  
**代码**: [GitHub](https://github.com/FelixChan9527/PerfGuard)  
**领域**: llm_agent  
**关键词**: visual content generation, agent, tool selection, performance-aware, AIGC, 偏好优化, 图像生成, 图像编辑

## 一句话总结

提出 PerfGuard，一个性能感知的 agent 框架用于视觉内容生成，通过多维性能评分矩阵替代文本描述来建模工具能力边界，结合自适应偏好更新和能力对齐规划优化，显著提升工具选择准确率（错误率从 77.8% 降至 14.2%）和视觉生成质量。

## 研究背景与动机

LLM agent 在自动化任务处理中展现出强大潜力，但在视觉内容生成（AIGC）领域存在关键缺陷：

**工具能力描述模糊**：现有系统依赖通用文本描述（如"能生成与文本语义对齐的图像"），无法区分不同模型在不同维度上的性能差异

**理想化假设**：多数框架假设"工具调用总是成功的"，缺乏对工具实际成功率的系统评估

**静态工具选择**：基准测试分数可能偏离实际任务表现，且无法适应工具更新

**规划与执行脱节**：任务规划过程未考虑工具性能边界，导致生成子任务超出工具能力

CompAgent、GenArtist 等系统虽能通过多模型调度增强生成效果，但工具描述粗粒度、无性能意识。

## 方法详解

### 整体框架

PerfGuard 基于标准化 agent 系统（Analyst→Planner→Worker→Self-Evaluator），由用户输入驱动迭代式视觉生成：

1. **Analyst** 解析多模态输入 → 任务摘要 $\tau^*$、目标图像语义 $s^*$、评估目标 $g$
2. **Planner** 基于 $\tau^*$, $s^*$ 和工具性能档案 $\mathcal{B}$ 分解为子任务 $u_t$
3. **Worker** 从工具库选择最适合的工具执行 $u_t$，生成图像输出 $o_t$
4. **Self-Evaluator** 多维度评估 $o_t$ 与目标 $g$ 的对齐程度，反馈给 Planner

### 关键设计

#### 1. Performance-Aware Selection Modeling (PASM)

用多维评分矩阵替代文本描述来精确定义工具性能边界：

**图像生成工具**：基于 T2I-CompBench 的 7 个维度（color, shape, texture, 2D spatial, 3D spatial, non-spatial, numeracy）

**图像编辑工具**：基于 ImgEdit-Bench 的 7 个维度（addition, removal, replacement, attribute alteration, motion change, style transfer, background change）

Worker 根据子任务 $u_t$ 生成偏好权重 $\mathcal{W}_{task} \in \mathbb{R}^{1 \times d}$，计算工具适配分数：

$$S_{tools} = \mathcal{W}_{task} \cdot \text{Normalize}(M_p)^\top$$

$$\mathcal{R} = \text{argsort}(S_{tools}, \text{descending})$$

其中 $M_p \in \mathbb{R}^{d \times l}$ 为 $l$ 个工具在 $d$ 个维度上的性能边界矩阵。

#### 2. Adaptive Preference Updating (APU)

通过比较理论排名与实际执行排名来迭代更新性能边界矩阵：

$$\mathcal{R}_{theory} = \text{top}_m(S_{tools}) \cup \text{rand}_n(S_{tools}[m+1:l])$$

$$M_p^{new} = \text{Normalize}\left(M_p + \mathcal{W}_{task} \cdot \eta \cdot \Delta\right)$$

$$\Delta = \frac{\mathcal{R}_{theory} - \mathcal{R}_{actual}}{m+n}$$

采用 exploration-exploitation 策略：选择 top-m 高分工具 + 随机 n 个低分工具，增加发现潜在高性能工具的概率。新工具以同类工具的平均分初始化。

#### 3. Capability-Aligned Planning Optimization (CAPO)

扩展 Step-aware Preference Optimization（SPO）到 Planner 的自回归规划：

每步生成 $k$ 个候选子任务 $\{u_t^1, ..., u_t^k\}$，经 Self-Evaluator 评估后选出最优 $u_t^w$ 和最差 $u_t^l$，优化 Planner：

$$\mathcal{L}(\theta) = -\mathbb{E}\left[\log\sigma\left(\alpha\left(\log\frac{p_\theta(u_t^w|ctx)}{p_{ref}(u_t^w|ctx)} - \log\frac{p_\theta(u_t^l|ctx)}{p_{ref}(u_t^l|ctx)}\right)\right)\right]$$

其中 $ctx = (\tau^*, s^*, \mathcal{B}, h_{t-1})$，$h_{t-1}$ 为历史子任务与评估结果。

### 损失函数

**Self-Evaluator 评分**：

$$e_t = \sum_{i=0}^L \gamma_i^{local} \pi_{Eval}(o_t, g_i^{local}) + \gamma^{global} \pi_{Eval}(o_t, g^{global})$$

全局语义 + 局部语义的加权评估，指导 CAPO 的 winning/losing sample 选择。

## 实验关键数据

### 主实验

**基础图像生成（T2I-CompBench）**：

| 方法 | Color↑ | Shape↑ | Texture↑ | Spatial↑ | Non-Spatial↑ | Complex↑ |
|------|--------|--------|----------|----------|-------------|----------|
| FLUX | 0.7407 | 0.5718 | 0.6922 | 0.2863 | 0.3127 | 0.3771 |
| SD3 | 0.8132 | 0.5885 | 0.7334 | 0.3200 | 0.3140 | 0.3703 |
| GenArtist | 0.8482 | 0.6948 | 0.7709 | 0.5437 | 0.3346 | 0.4499 |
| T2I-Copilot | 0.8039 | 0.6120 | 0.7604 | 0.3228 | 0.3379 | 0.3985 |
| **PerfGuard** | **0.8753** | **0.7366** | **0.8148** | **0.6120** | **0.3754** | **0.5007** |

PerfGuard 在所有 6 个维度上均取得最优。

**高级图像生成（OneIG-Bench）**：

| 方法 | 类型 | Alignment↑ | Text↑ | Reasoning↑ | Style↑ |
|------|------|-----------|-------|-----------|--------|
| SD3 | Diffusion | 0.801 | 0.648 | 0.279 | 0.361 |
| T2I-Copilot | Agent | 0.821 | 0.679 | 0.318 | 0.386 |
| **PerfGuard** | Agent | **0.834** | **0.684** | **0.350** | **0.395** |

**复杂图像编辑（Complex-Edit Level-3）**：

| 方法 | IF↑ | PQ↑ | IP↑ | Overall↑ |
|------|-----|-----|-----|----------|
| Step1X_Edit | 7.95 | 8.66 | 7.70 | 8.10 |
| OmniGen | 7.52 | 8.86 | 8.01 | 8.13 |
| **PerfGuard** | **8.95** | **9.02** | **8.56** | **8.84** |

### 消融实验

**模块消融（T2I-CompBench）**：

| CAPO | PASM | APU | Color↑ | Spatial↑ | Complex↑ |
|------|------|-----|--------|----------|----------|
| ✗ | ✗ | ✗ | 0.8239 | 0.5600 | 0.4327 |
| ✓ | ✗ | ✗ | 0.8466 | 0.5756 | 0.4493 |
| ✗ | ✓ | ✗ | 0.8521 | 0.5919 | 0.4412 |
| ✗ | ✓ | ✓ | 0.8596 | 0.6005 | 0.4738 |
| ✓ | ✓ | ✓ | **0.8753** | **0.6120** | **0.5007** |

PASM 贡献最大（Color +3.42%, Texture +5.7%），APU 进一步精调（Complex 0.4412→0.4738），CAPO 提供整体优化叠加。

**工具选择错误率对比**：

| 方法 | 错误率 |
|------|--------|
| 纯文本描述 + QWen3-14B | 77.8% |
| 纯文本描述 + GPT-4o | 72.2% |
| 外部经验模块 + QWen3-14B | 68.1% |
| PASM（基准分数矩阵）+ QWen3-14B | 30.5% |
| PASM + APU (η=0.13, 800步) | **14.2%** |

**更新步长 η 消融**：η=0.10 收敛慢；η=0.15 初期快但后期振荡严重；**η=0.13** 最优平衡点。

### 关键发现

1. 纯文本描述的工具选择错误率高达 77.8%，即使 GPT-4o 也仅降至 72.2%
2. 性能感知矩阵将错误率降至 30.5%，自适应更新进一步降至 14.2%（5.5× 改进）
3. CAPO 训练后的 Planner 能感知工具性能边界，理解操作顺序对结果的影响
4. PerfGuard 的 token 消耗不随工具数增长，而传统方法呈灾难性增长

## 亮点与洞察

1. **解决真实痛点**：精准建模了 AIGC 领域工具能力边界模糊的核心问题，方法直觉且有效
2. **高效的工具管理**：PASM 的维度匹配方式使 token 消耗与工具数量无关，在大规模工具库（200+ 工具）场景下优势巨大
3. **自适应闭环**：APU 通过实际执行排名反馈不断校正性能矩阵，避免了静态基准的偏差
4. **Planner 训练**：CAPO 让 Planner 学习到工具局限会反向影响规划准确度（如先编辑背景可能降低后续步骤成功率）
5. **工程实用性**：框架模块化，PASM 可直接应用于任何带有基准测试分数的工具库

## 局限性

1. 性能边界维度依赖特定基准（T2I-CompBench, ImgEdit-Bench），对新任务类型需重新设计维度
2. CAPO 需要多候选生成和评估，增加了推理成本（虽然论文展示比 GenArtist 快，但绝对时间未给出）
3. 图像编辑中 Identity Preservation (IP) 指标上不如 AnySD，因为 AnySD 做最小编辑
4. 工具库限制了上限——PerfGuard 在 alignment/text 上未大幅领先 T2I-Copilot
5. APU 的收敛依赖足够的工具使用次数，冷启动问题通过平均分初始化只是部分解决

## 相关工作与启发

与 GenArtist 的区别：GenArtist 缺乏性能感知工具选择策略，导致规划错误和元素缺失。与 T2I-Copilot 的区别：T2I-Copilot 通过多模块语义分解性能优异但工具多样性有限。与 CLOVA 的区别：CLOVA 通过自反思和 prompt tuning 提升成功率，但未建模工具性能边界。

**核心启发**：在 agent 系统中，工具选择是被严重低估的瓶颈。用结构化的性能矩阵替代自然语言描述是一个简单但高效的思路，可推广到代码生成、数据分析等任何需要多工具协作的 agent 场景。

## 评分

- 新颖性: ⭐⭐⭐⭐ (性能感知的工具选择建模是新颖且实用的思路)
- 实验充分度: ⭐⭐⭐⭐ (三个基准、详细消融、效率分析、工具规模扩展实验)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，方法表述规范，可视化丰富)
- 价值: ⭐⭐⭐⭐ (对 agent 工具选择有直接指导意义，框架具有通用性)

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents](simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_llm_agents.md)
- [\[ICLR 2026\] AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents](agentsynth_scalable_task_generation_for_generalist_computer-use_agents.md)
- [\[ICLR 2026\] Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](exploratory_memory-augmented_llm_agent_via_hybrid_on-_and_off-policy_optimizatio.md)
- [\[ICLR 2026\] HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre](hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat.md)
- [\[ICLR 2026\] Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents](your_agent_may_misevolve_emergent_risks_in_self-evolving_llm_agents.md)

<!-- RELATED:END -->
