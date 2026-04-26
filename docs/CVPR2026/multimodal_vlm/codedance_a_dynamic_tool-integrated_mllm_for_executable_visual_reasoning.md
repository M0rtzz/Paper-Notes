---
title: >-
  [论文解读] CodeDance: A Dynamic Tool-integrated MLLM for Executable Visual Reasoning
description: >-
  [CVPR 2026][多模态][可执行视觉推理] 提出CodeDance，将可执行代码作为视觉推理的通用求解器——MLLM生成代码来定义、组合和执行多种工具，渲染中间视觉结果(bbox/线/图表)支持可审查的推理链，通过平衡探索与效率的工具调用奖励做RL训练，在RL中涌现出未见过的工具调用组合和跨任务迁移行为，7B模型在计数/视觉搜索/图表QA上超越GPT-4o。
tags:
  - CVPR 2026
  - 多模态
  - 可执行视觉推理
  - 工具集成
  - 代码生成
  - 强化学习
  - 涌现行为
---

# CodeDance: A Dynamic Tool-integrated MLLM for Executable Visual Reasoning

**会议**: CVPR 2026  
**arXiv**: [2512.17312](https://arxiv.org/abs/2512.17312)  
**代码**: [CodeDance-VL.github.io](https://CodeDance-VL.github.io) (有)  
**领域**: 多模态VLM  
**关键词**: 可执行代码推理, 工具调用, 多模态推理, 强化学习, 涌现行为

## 一句话总结

提出 CodeDance，将可执行代码作为视觉推理的统一媒介，通过 SFT 教授原子能力 + RL 中的难度自适应工具调用奖励（BAT），实现动态工具编排与自检推理，7B 模型在计数/视觉搜索/图表 QA 等任务上超越 GPT-4o。

## 研究背景与动机

当前多模态大模型（MLLMs）在视觉推理中面临几个核心瓶颈：

**文本 CoT 的信息瓶颈**：传统 Chain-of-Thought 仅依赖静态文本上下文，无法在推理中间步骤动态交互视觉输入或引入新观测信息，导致多轮聚焦和验证能力受限。

**封闭系统不可复现**：OpenAI o3 展示了强大的"用图像思考"能力，但其内部机制完全不透明，无法被研究社区学习和复现。

**开源方案依赖固定 schema**：现有开源工具集成方法（DeepEyes、PixelReasoner 等）多依赖预定义的视觉操作模板（如仅预测 bounding box 坐标用于裁剪），灵活性差，难以迁移至新工具和新任务。

**缺乏工具调用时机控制**：现有方法没有考虑模型何时应该调用工具、何时不需要，导致简单问题工具过度使用（tool spamming）或困难问题工具使用不足。

CodeDance 的核心思路：**用可执行代码替代固定 schema**，让模型能自由定义、组合和执行代码来编排工具，产生可验证的中间产物（裁剪区域、标注图、计算图表等），实现透明、可自检的推理。

## 方法详解

### 整体框架

CodeDance 以 **think-execute-feedback 循环**作为推理单元。给定多模态查询（文本 + 图像），MLLM 交替产生自然语言推理和可执行代码，代码在沙盒中执行，视觉产物拼接回上下文驱动下一轮推理。一条推理轨迹为：

$$\tau = \big((s_{t_1}, a_{t_1}, s'_{t_1}), \ldots, (s_{t_M}, a_{\text{ans}})\big)$$

其中状态 $s_t = (x, o_t, \epsilon_t)$ 包含原始查询 $x$、累计推理痕迹 $o_t$ 和解释器反馈 $\epsilon_t$。动作空间包括工具调用（代码片段）和终止回答。

训练分两阶段：(1) SFT 冷启动教授原子代码能力；(2) RL 进一步优化工具调用策略。

### 关键设计

1. **原子能力 SFT 数据集（34K 轨迹）**：覆盖三类原子操作——基础图像变换（crop, resize）、数学计算（测量、代数、聚合）、开放式视觉编辑（绘图、标注等）。采用**弱到强过滤**策略：先用 Qwen2.5-VL-7B 自动过滤和难度分级，再用 Qwen2.5-VL-72B 交叉验证正确性。简单问题鼓励直接回答（不调用工具），困难问题启用多轮代码推理轨迹。这种设计天然为后续 RL 中的自适应工具调用打下基础。

2. **BAT 奖励机制（Balanced Adaptive Tool Calling）**：RL 阶段使用 GRPO 优化，总奖励分三部分：

    $r(\tau) = R_{\text{acc}}(\tau) + R_{\text{format}}(\tau) + R_{\text{BAT}}(\tau)$

   BAT 奖励进一步分解为序列级和轮次级：

    - **序列级自适应奖励 $R_{\text{seq}}$**：核心洞察是利用组内正确率 $\mu_{\text{acc}}$ 作为题目难度的代理——大多数 rollout 都答对说明题目简单，此时抑制工具调用奖励；反之鼓励探索。缩放因子为 $d = \sigma(\gamma(0.5 - \mu_{\text{acc}})) - \delta$，其中 $\gamma=4, \delta=0.2$。

    - **轮次级执行奖励 $R_{\text{turn}}$**：对代码执行失败施加即时惩罚 $-0.5$，并用折扣因子 $\beta=0.2$ 递归累计，提供密集纠正信号。实验证明此设计有效防止 entropy collapse。

3. **涌现行为（Emergent Behaviors）**：RL 训练中观察到三类超越 SFT 监督的行为：

    - **跨域工具迁移**：chart 任务中学到的 bounding box 操作自发迁移用于 counting 任务（先定位所有候选物体画框，再验证和计数）
    - **新颖工具组合**：将 pointing + crop + zoom-in 多个原子操作组合，解决单一工具无法完成的验证任务
    - **未见代码生成**：模型自发调用预训练知识中的 OpenCV 函数（如 `cv2.rectangle` 画网格辅助计数），尽管 SFT 数据中从未出现

### 损失函数 / 训练策略

- **SFT 阶段**：标准交叉熵损失训练，使用 SWIFT 框架
- **RL 阶段**：GRPO + BAT 奖励，使用 VeRL 框架；训练最大轮次 6，推理最大轮次 10
- 基座模型：Qwen2.5-VL-7B

## 实验关键数据

### 主实验

| 数据集 | 指标 | CodeDance-7B | Qwen2.5-VL-7B（基座） | GPT-4o | 提升(vs基座) |
|--------|------|-------------|---------------------|--------|------------|
| CountBench | Acc | 91.2 | 76.5 | 87.9 | ↑19.2% |
| PixmoCount | Acc | 77.1 | 50.4 | - | ↑53.0% |
| V* Bench | Acc | 84.8 | 76.4 | 67.5 | ↑11.0% |
| HR-Bench 4K | Acc | 75.2 | 69.0 | 65.0 | ↑9.0% |
| ChartQA | Acc | 87.5 | 86.3 | 86.7 | ↑1.4% |
| MathVision | Acc | 29.6 | 25.0 | 36.5 | ↑18.4% |
| MathVista | Acc | 70.3 | 68.1 | 63.4 | ↑3.2% |
| WeMath | Acc | 39.6 | 35.4 | 44.2 | ↑11.9% |

7B 模型超越 GPT-4o（CountBench, V* Bench, HR-Bench 等），也超越 Qwen2.5-VL-32B 等更大模型。

### 消融实验

| 奖励配置 | 说明 | 效果 |
|---------|------|------|
| GRPO（仅结果奖励） | 只看最终答案正确性 | 缩短交互轮次但工具使用不足，复杂任务性能差 |
| DeepEyes 奖励 | 每次成功工具调用都给正奖励 | 工具过度使用，简单题轮次膨胀，准确率不稳定 |
| BAT 无 $R_{\text{turn}}$ | 去掉轮次级奖励 | 执行成功率严重退化 |
| BAT 完整（Ours） | 序列级 + 轮次级 | 最佳准确率 + 适中轮次 + 最高执行成功率 |

扩展性消融：SFT 数据 5K→34K 持续提升；模型 3B→7B 显著提升；RL 训练 240 步仍有收益无过拟合。

### 关键发现

- 代码媒介让 7B 模型在感知密集型任务上突破了基座模型的感知上限（如 PixmoCount 提升 53%）
- 难度自适应奖励是关键：既避免简单题上的 tool spamming，又保证困难题上的充分探索
- 涌现行为表明原子级工具训练可催生组合推理能力，这是可扩展的

## 亮点与洞察

- **代码 > 固定 schema**：代码作为推理媒介比预定义操作模板更灵活、可组合、可扩展
- **难度自适应**：用组内正确率 $\mu_{\text{acc}}$ 作为难度代理是一个简洁有效的设计
- **涌现行为令人兴奋**：原子工具训练 → RL 中组合涌现 → 跨任务迁移，暗示代码执行可能是 agentic 多模态推理的可扩展路径

## 局限与展望

- 仅在 Qwen2.5-VL-7B 上验证，更大规模模型效果有待探索
- 代码执行沙盒的安全性和延迟开销未详细分析
- 涌现行为的可控性和可预测性有限，缺乏理论解释
- 34K SFT 数据构建依赖强模型蒸馏，成本较高

## 相关工作与启发

- 与 ViperGPT 思路类似但更进一步：ViperGPT 将查询编译为程序，CodeDance 让模型在 RL 中学会自主编写和迭代代码
- BAT 奖励设计优于 DeepEyes 的简单工具奖励，为 RL 训练中的工具使用提供了更好的范式
- 涌现行为的观察可启发对 LLM agentic 能力涌现机制的进一步研究

## 评分

- **新颖性**: ⭐⭐⭐⭐ 代码作为统一推理媒介 + BAT 奖励有新意，但框架整体非颠覆性
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖 10+ benchmark，消融全面，涌现行为分析深入
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，故事线完整，涌现行为的展示很有说服力
- **价值**: ⭐⭐⭐⭐⭐ 为开源社区提供了可复现的"think with images"方案，实用价值高
# CodeDance: A Dynamic Tool-integrated MLLM for Executable Visual Reasoning

**会议**: CVPR 2026  
**arXiv**: [2512.17312](https://arxiv.org/abs/2512.17312)  
**代码**: https://CodeDance-VL.github.io (有)  
**领域**: 多模态VLM / 工具使用  
**关键词**: 可执行视觉推理, 工具集成, 代码生成, 强化学习, 涌现行为

## 一句话总结
提出CodeDance，将可执行代码作为视觉推理的通用求解器——MLLM生成代码来定义、组合和执行多种工具，渲染中间视觉结果(bbox/线/图表)支持可审查的推理链，通过平衡探索与效率的工具调用奖励做RL训练，在RL中涌现出未见过的工具调用组合和跨任务迁移行为，7B模型在计数/视觉搜索/图表QA上超越GPT-4o。

## 研究背景与动机

1. **领域现状**：o3展示了"用工具思考"的能力——交替推理和工具使用。但现有开源方法要么仅用文本CoT、要么用固定schema(仅预测bbox坐标)、要么是单步pipeline。
2. **关键gap**：(1) 纯文本CoT无法动态与视觉输入交互或验证中间结果；(2) 固定schema限制了灵活性和可组合性；(3) o3是黑箱闭源系统。
3. **核心idea**：代码是最通用的"工具调用语言"——CodeDance让MLLM生成和执行Python代码来编排多种工具、计算中间结果、渲染视觉产物。通过RL训练发现**涌现行为**（训练中未见的新工具调用方式、组合和跨任务迁移）。

## 方法详解

### 关键设计
1. **可执行代码推理**：模型生成Python代码→执行→获取工具输出(裁剪/检测/OCR结果)→继续推理或生成新代码→最终输出答案。支持交替"思考(文本)"和"执行(代码)"
2. **工具调用奖励(RL)**：设计平衡探索与效率的奖励——鼓励适度使用工具(太少→信息不足，太多→过度使用/效率低)
3. **涌现行为**：RL训练中观察到——模型发明了训练数据中未出现的工具调用模式、将不同任务的工具组合应用到新任务、跨任务迁移

### 训练策略
原子监督(单工具使用示例) → SFT初始化 → RL(工具调用奖励+任务正确性奖励)

## 实验关键数据

### 主实验

| 模型 | CountBench | PixmoCount | V*Bench | ChartQA |
|------|:---:|:---:|:---:|:---:|
| GPT-4o | 87.9 | - | 67.5 | 86.7 |
| Qwen2.5-VL-7B | 76.5 | 50.4 | 76.4 | 86.3 |
| Deepeyes-7B | 80.4 | 57.2 | 90.4 | 78.2 |
| **CodeDance-7B** | **91.2** | **77.1** | 84.8 | **87.5** |

CountBench +19.2%, PixmoCount +53.0% vs Qwen2.5-VL-7B基线。

### 涌现行为案例
- 未见过的工具组合（如zoom+count+compare的链式调用）
- 跨任务迁移（计数任务中迁移用于图表分析的区域检测）
- 自发生成验证代码（画出检测结果进行视觉核查）

### 关键发现
- 代码比固定schema表达力强得多——同一模型大小下性能显著提升
- 涌现行为是RL的关键产出——SFT阶段不出现这些行为
- 工具不是越多越好——平衡奖励比"总是用工具"的奖励效果更好

## 亮点与洞察
- **代码作为通用推理媒介**：比文本CoT更有执行力，比固定schema更灵活——代码天然支持变量、循环、条件、函数定义
- **RL训练的涌现性**：工具使用的新颖组合和跨任务迁移——从原子能力到创造性组合，类似语言模型的涌现能力
- **可审查推理**：代码+渲染的视觉中间结果→推理链完全可追溯可验证

## 局限与展望
- 代码执行需要沙盒环境→部署复杂度高于纯文本模型
- 工具集是预定义的——如何动态发现和接入新工具？

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 可执行代码推理+RL涌现行为
- 实验充分度: ⭐⭐⭐⭐ 计数/搜索/图表多基准+涌现分析
- 写作质量: ⭐⭐⭐⭐ 涌现行为的案例展示直观
- 价值: ⭐⭐⭐⭐⭐ 对VLM工具使用和推理范式有重要推动

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing](from_intuition_to_investigation_a_tool-augmented_reasoning_mllm_framework_for_ge.md)
- [\[CVPR 2026\] Proof-of-Perception: Certified Tool-Using Multimodal Reasoning with Compositional Conformal Guarantees](pop_proof_of_perception_conformal_reasoning.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](docseeker_long_document_understanding.md)
- [\[CVPR 2026\] Generate, Analyze, and Refine: Training-Free Sound Source Localization via MLLM Meta-Reasoning](generate_analyze_and_refine_training-free_sound_source_localization_via_mllm_met.md)
- [\[CVPR 2026\] MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding](msjoe_jointly_evolving_mllm_and_sampler_for_efficient_long-form_video_understand.md)

<!-- RELATED:END -->
