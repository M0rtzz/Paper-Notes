---
title: >-
  [论文解读] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation
description: >-
  [CVPR 2026][图像生成][compositional T2I generation] 提出coDrawAgents交互式多智能体对话框架，Interpreter、Planner、Checker、Painter四个专业智能体闭环协作，以分治策略按语义优先级逐组增量规划布局，基于画布视觉上下文接地推理并显式纠错，在GenEval上以0.94 Overall Score大幅领先GPT Image 1（0.84），在DPG-Bench上达85.17 SOTA。
tags:
  - CVPR 2026
  - 图像生成
  - compositional T2I generation
  - multi-agent dialogue
  - layout planning
  - visual context grounding
  - error correction
---

# coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation

**会议**: CVPR 2026  
**arXiv**: [2603.12829](https://arxiv.org/abs/2603.12829)  
**代码**: 待发布  
**领域**: 图像生成 / 多智能体系统  
**关键词**: compositional T2I generation, multi-agent dialogue, layout planning, visual context grounding, error correction

## 一句话总结

提出coDrawAgents交互式多智能体对话框架，Interpreter、Planner、Checker、Painter四个专业智能体闭环协作，以分治策略按语义优先级逐组增量规划布局，基于画布视觉上下文接地推理并显式纠错，在GenEval上以0.94 Overall Score大幅领先GPT Image 1（0.84），在DPG-Bench上达85.17 SOTA。

## 研究背景与动机

**领域现状**：文本到图像（T2I）生成在多对象复杂场景中面临组合保真度问题。现有探索包括：LLM辅助布局生成（LayoutLLM-T2I、LMD）、扩散注意力引导（Attend-and-Excite）、生成思维链（GoT）、和早期多智能体框架（MCCD、T2I-Copilot）。

**现有痛点**：

1. 单智能体方法将解析/规划/验证交给一个模型，早期空间错误难以检测修复
2. 现有多智能体框架本质是固定流水线，缺乏协商和视觉反馈，错误仍然传播
3. 全局布局规划面临对象间关系的二次复杂度 $O(N^2)$，N个对象同时规划极困难
4. 绝大多数方法在无视觉上下文下预测布局，只能"想象"场景

**核心矛盾**：复杂场景需要的布局推理能力随对象数量呈二次增长，但单次规划和固定流水线均无法有效处理这种复杂度爆炸。

**本文目标** 在复杂多对象场景中实现忠实的组合T2I生成，同时解决布局复杂度、缺乏视觉感知、早期错误无法纠正三大挑战。

**切入角度**：四智能体闭环对话式协作——分治降低复杂度 + 画布视觉接地 + 显式检查纠错。

**核心 idea**：让Planner看着正在生成的画面来规划下一步、让Checker回溯所有历史布局来纠错、按语义优先级分组来降低单轮复杂度。

## 方法详解

### 整体框架

Interpreter判断文本复杂度 → 简单场景直接调用T2I（layout-free模式）→ 复杂场景激活layout-aware模式：Interpreter解析文本为带属性的对象描述 + 语义优先级排序分组 → 按优先级逐轮迭代 → 每轮：Planner用VCoT增量规划布局 → Checker两阶段检查纠正 → Painter在画布上增量渲染 → 画布作为下一轮视觉上下文 → N轮后输出最终图像。

### 关键设计

1. **Interpreter + 分治策略**

    - 功能：决定生成模式，将文本分解为属性丰富的对象描述，按语义显著性排序分组
    - 核心思路：LLM + CoT提示执行三步：(i) 识别并分解文本为语义单元 (ii) 按语义显著性排序并分组同优先级对象 (iii) CoT引导的属性增强和背景描述生成
    - 设计动机：分组后每轮仅处理同优先级对象，将$O(N^2)$全局规划降为多轮$O(k^2)$局部规划（$k \ll N$）。DPG-Bench平均2.79个对象仅需1.52轮Planner调用

2. **Planner + Visualization Chain-of-Thought (VCoT)**

    - 功能：基于当前画布视觉上下文增量规划当前优先级对象的布局
    - 核心思路：GPT-5作为MLLM执行三步VCoT：(1) Canvas State Analysis——接收画布图像 $I_{i-1}$ 和已有布局，分析现有对象空间状态 (2) Context-Aware Planning——基于世界知识推理新对象与现有场景的合理交互 (3) Physics Constraint Enforcement——确保物理合理性（无漂浮、合理接触面）。对象接地（grounding）建立文本实体与画布区域的对应，弥补LLM对坐标的不敏感性
    - 设计动机：基于实际画布规划而非"想象式"规划，从根本上解决布局-视觉不一致问题

3. **Checker两阶段检查-修正**

    - 功能：验证布局的空间一致性和属性对齐，修正错误
    - 核心思路：第一阶段对当前布局 $L_i$ 做对象级（尺寸/比例/覆盖）+ 全局级（相对位置/关系）检查并修正。第二阶段回溯所有历史布局 $\{L_1, ..., L_i\}$，检测跨对象冲突（重叠/遮挡/尺度漂移），逐步修复并传播修正
    - 设计动机：扩散模型在早期步确定粗结构后难以修正，Checker在布局阶段就进行显式纠错，避免错误被"bake in"

4. **Painter即插即用渲染**

    - 功能：每轮增量渲染画布，为后续迭代提供视觉上下文
    - Layout-free模式用Flux（T2I），Layout-aware模式用3DIS（L2I），不需额外训练
    - 设计动机：解耦绘制能力与规划/验证逻辑，可随底层模型升级而自然受益

### 损失函数 / 训练策略

无需额外训练。全部利用预训练LLM（GPT-5）和现有T2I（Flux）/L2I（3DIS）模型，属于training-free和plug-and-play框架。

## 实验关键数据

### 主实验

**GenEval基准对比**

| 模型 | Single | Two Obj. | Counting | Colors | Position | Color Attr. | Overall↑ |
|------|--------|----------|----------|--------|----------|-------------|----------|
| DALL-E 3 | 0.96 | 0.87 | 0.47 | 0.83 | 0.43 | 0.45 | 0.67 |
| FLUX.1-dev | 0.99 | 0.81 | 0.79 | 0.74 | 0.20 | 0.47 | 0.67 |
| GoT | 0.99 | 0.69 | 0.67 | 0.85 | 0.34 | 0.27 | 0.64 |
| UniWorld-V1 | 0.99 | 0.93 | 0.79 | 0.89 | 0.49 | 0.70 | 0.80 |
| GPT Image 1 [High] | 0.99 | 0.92 | 0.85 | 0.92 | 0.75 | 0.61 | 0.84 |
| **coDrawAgents** | **1.00** | **0.96** | **0.94** | **0.97** | **0.95** | **0.81** | **0.94** |

**DPG-Bench对比**

| 模型 | Global | Entity | Relation | Overall↑ |
|------|--------|--------|----------|----------|
| DALL-E 3 | 90.97 | 89.61 | 90.58 | 83.50 |
| SD3-Medium | 87.90 | 91.01 | 80.70 | 84.08 |
| OmniGen2 | 88.81 | 88.83 | 89.37 | 83.57 |
| **coDrawAgents** | 84.78 | **90.15** | **92.92** | **85.17** |

### 消融实验

| 配置 | DPG Overall↑ | 说明 |
|------|-------------|------|
| Layout-free baseline | 77.60 | 仅直接T2I |
| + Layout-aware | 82.61 (+5.01) | 分治策略降低复杂度 |
| + Visual context | 84.51 (+1.90) | 画布接地增强空间一致性 |
| + Checker (完整) | **85.17** (+0.66) | 显式纠错提升忠实度 |

**效率统计（DPG-Bench 1074图）**

| 智能体 | 平均调用次数/图 |
|--------|--------------|
| Interpreter | 1.00 |
| Planner | 1.52 |
| Checker | 1.62 |
| Painter | 1.95 |
| 场景平均对象数 | 2.79 |

### 关键发现

- GenEval Overall从GPT Image 1的0.84跃升到0.94（+11.9%），全子指标均为最高
- Position指标从0.75暴涨到0.95，说明画布视觉接地+分治策略极大增强了空间推理能力
- Counting从0.85→0.94，分组生成有效解决了计数问题
- 智能体平均调用次数远少于场景对象数（1.52 vs 2.79），因分组策略减少迭代轮次

## 亮点与洞察

- 分治策略将N对象全局布局分解为按语义优先级逐组规划，优雅降低复杂度
- 画布视觉上下文作为Planner输入是核心创新——让布局推理从"想象"变为"看着画"
- Checker的跨迭代回溯修正可处理早期错误的级联效应，这在固定流水线中不可能实现
- VCoT三步推理（状态分析→上下文规划→物理约束）结构清晰，可推广到其他需要空间推理的生成任务

## 局限与展望

- 多智能体调用引入计算开销（多次LLM推理 + 多次图像生成），推理时间比单次方法长
- Painter性能依赖底层T2I/L2I模型能力，属性渲染不完美（如"黑皮萝卜"）会传播
- Planner和Checker依赖GPT-5 MLLM，存在幻觉和过度自信风险
- 仅支持2D合成，未扩展到3D场景生成
- DPG-Bench Global指标（84.78）低于部分单模型（如DALL-E 3的90.97），分步生成可能损失全局一致性

## 相关工作与启发

- **vs GoT**：GoT一次性推理所有bbox且无视觉反馈（Overall 0.64 vs 0.94），验证了闭环交互式协作的根本优势
- **vs T2I-Copilot**：固定流水线无对话协商和视觉接地（Overall 74.34 vs 85.17）
- **vs MCCD**：仅做文本分解无画布感知，本质是并行生成再融合
- **启发**：闭环多智能体协作范式可推广到视频生成（逐帧规划+一致性检查）、3D场景构建（逐物体放置+碰撞检测）等需要增量式组合的任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 闭环多智能体对话框架和VCoT视觉接地规划有创新，但核心技术是LLM/MLLM的prompt工程
- 实验充分度: ⭐⭐⭐⭐ GenEval和DPG-Bench全面对比+消融+效率分析，定性比较清晰
- 写作质量: ⭐⭐⭐⭐ 四智能体定位和分工描述清晰，框架图直观
- 价值: ⭐⭐⭐ 组合生成效果惊艳但工程性强，依赖GPT-5的成本和可复现性是主要顾虑
